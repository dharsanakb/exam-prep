import os
import re
import fitz  # PyMuPDF
from pptx import Presentation
from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "exam_notes"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_collection():
    """Always create a fresh client and collection — avoids stale references."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


def clean_text(text: str) -> str:
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove standalone page numbers
    text = re.sub(r'\b\d{1,3}\b', ' ', text)
    # Remove bullet point symbols
    text = re.sub(r'[●•■▪►✓◆→\-–—]\s*', ' ', text)
    # Discard fragment if fewer than 5 real words (catches formula-only lines)
    words = [w for w in text.split() if re.search(r'[a-zA-Z]{2,}', w)]
    if len(words) < 5:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_sentences(text: str) -> list[str]:
    sentences = split_into_sentences(text)
    chunks = []
    current_words = []
    overlap_buffer = []

    for sentence in sentences:
        words = sentence.split()
        current_words.extend(words)

        if len(current_words) >= CHUNK_SIZE:
            chunk = " ".join(current_words)
            if chunk.strip():
                chunks.append(chunk)
            overlap_buffer = current_words[-CHUNK_OVERLAP:]
            current_words = overlap_buffer.copy()

    if current_words:
        chunk = " ".join(current_words)
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def parse_pdf(file_path: str) -> list[str]:
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        text = page.get_text()
        cleaned = clean_text(text)
        if cleaned:
            pages.append(cleaned)
    return pages


def parse_pptx(file_path: str) -> list[str]:
    prs = Presentation(file_path)
    slides = []
    for i, slide in enumerate(prs.slides):
        texts = []
        title_text = ""
        if slide.shapes.title and slide.shapes.title.text.strip():
            title_text = slide.shapes.title.text.strip()
            texts.append(f"[Slide {i+1}: {title_text}]")
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = para.text.strip()
                    if line and line != title_text:
                        texts.append(line)
        slide_text = " ".join(texts)
        cleaned = clean_text(slide_text)
        if cleaned:
            slides.append(cleaned)
    return slides


def already_ingested(filename: str) -> bool:
    try:
        collection = get_collection()
        results = collection.get(where={"source": filename})
        return len(results["ids"]) > 0
    except Exception:
        return False


def ingest_file(file_path: str, filename: str) -> int:
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        page_texts = parse_pdf(file_path)
    elif ext == ".pptx":
        page_texts = parse_pptx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    all_chunks = []
    for page_text in page_texts:
        chunks = chunk_by_sentences(page_text)
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    collection = get_collection()
    embeddings = model.encode(all_chunks, show_progress_bar=False).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=[f"{filename}_chunk_{i}" for i in range(len(all_chunks))],
        metadatas=[{"source": filename, "chunk_index": i} for i in range(len(all_chunks))],
    )

    return len(all_chunks)