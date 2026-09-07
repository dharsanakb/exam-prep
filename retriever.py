from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "exam_notes"
TOP_K = 8
DISTANCE_THRESHOLD = 1.8  # Loosened to not miss valid chunks

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


def keyword_match_score(chunk: str, query: str) -> int:
    """
    Count how many significant words from the query appear in the chunk.
    Used to re-rank and filter out chunks that matched only on generic terms.
    """
    # Split query into meaningful words (ignore short stop words)
    stopwords = {"what", "is", "the", "a", "an", "of", "and", "in", "for", "to", "how", "does", "explain"}
    query_words = [w.lower() for w in query.split() if w.lower() not in stopwords and len(w) > 2]
    chunk_lower = chunk.lower()
    return sum(1 for word in query_words if word in chunk_lower)


def retrieve(query: str) -> tuple[list[str], list[str]]:
    try:
        collection = get_collection()
    except Exception:
        return [], []

    if collection.count() == 0:
        return [], []

    # Only use the original query + one specific variation.
    # Too many expanded queries was pulling in loosely related chunks.
    expanded_queries = [
        query,
        f"what is {query}",
    ]

    seen_ids = set()
    candidates = []  # (chunk, source, distance, keyword_score)

    for q in expanded_queries:
        embedding = model.encode([q]).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=min(TOP_K, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk_id = meta.get("chunk_index", doc[:30])
            source = meta["source"]
            unique_key = f"{source}_{chunk_id}"

            if unique_key not in seen_ids and dist < DISTANCE_THRESHOLD:
                seen_ids.add(unique_key)
                score = keyword_match_score(doc, query)
                print(f"[RETRIEVER] dist={dist:.3f} score={score} | {doc[:80]}")
                candidates.append((doc, source, dist, score))

    if not candidates:
        # Nothing matched tightly — fall back with relaxed threshold
        embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=min(5, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            source = meta["source"]
            score = keyword_match_score(doc, query)
            candidates.append((doc, source, dist, score))

    # Sort: prioritize chunks that contain the actual query keywords
    # Higher keyword score = more relevant = sort first
    candidates.sort(key=lambda x: (-x[3], x[2]))

    # Keep only top 6 after re-ranking
    candidates = candidates[:6]

    all_chunks = [c[0] for c in candidates]
    all_sources = [c[1] for c in candidates]

    return all_chunks, all_sources