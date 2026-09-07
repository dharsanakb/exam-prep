# Exam Prep Bot

A RAG-based study assistant that lets you upload your own notes or slides and ask it to explain any concept from them. It only uses your material — no hallucinated answers from general knowledge.


## What It Does

1. You upload a PDF or PPTX file (your lecture notes, textbook chapters, slides)
2. The app breaks the content into chunks and stores them in a local vector database
3. You type a concept you want explained
4. The app finds the most relevant chunks from your notes and sends them to an LLM
5. The LLM explains the concept using only your material

---

## Tech Stack

| Tool | Role | Why This Tool |
|---|---|---|
| **Streamlit** | UI framework | Fastest way to build a Python web app with file upload and interactive interface |
| **PyMuPDF (fitz)** | PDF parsing | Reliable, fast text extraction from PDFs including multi-column layouts |
| **python-pptx** | PPTX parsing | Native Python library for reading PowerPoint slide text with title preservation |
| **sentence-transformers** | Embedding model | Converts text chunks into vectors for semantic similarity search. Uses `all-MiniLM-L6-v2` — small, fast, and accurate enough for study notes |
| **ChromaDB** | Vector database | Stores embeddings locally with no setup or cloud account needed. Persists between sessions |
| **Groq + GPT-OSS-120b** | LLM | Groq provides extremely fast inference. GPT-OSS-120b gives high-quality explanations via Groq's API |
| **python-dotenv** | API key management | Loads the Groq API key from a `.env` file so it is never hardcoded in source code |

---

## How RAG Works Here

RAG stands for **Retrieval-Augmented Generation**. Instead of asking an LLM to answer from its training data (which can hallucinate), RAG first retrieves relevant content from your own documents and feeds that as context to the LLM.

```
Your Notes (PDF/PPTX)
        ↓
   Parse text (page by page / slide by slide)
        ↓
   Clean text — remove bullets, symbols, fragments
        ↓
   Split into sentence-aware chunks (~500 words, 100-word overlap)
        ↓
   Embed each chunk → vector (all-MiniLM-L6-v2)
        ↓
   Store in ChromaDB with metadata (filename, chunk index)
        ↓
   User asks a question
        ↓
   Embed query → vector
        ↓
   Find top 8 similar chunks (cosine distance < 1.8)
        ↓
   Re-rank by keyword match score
        ↓
   Send top 6 chunks + question to Groq (openai/gpt-oss-120b)
        ↓
   LLM explains using only the retrieved chunks
```

---

## Project Structure

```
exam-prep-bot/
├── ingest.py        # Parse files, clean text, chunk, embed, store in ChromaDB
├── retriever.py     # Query ChromaDB, re-rank by keyword score, return top chunks
├── app.py           # Streamlit UI + Groq API call
├── requirements.txt
├── .env             # Your API keys (never commit this)
└── chroma_db/       # Auto-created local vector store
```

### File Responsibilities

**`ingest.py`**

Handles everything from raw file to stored embeddings:
- Parses PDF page by page and PPTX slide by slide (preserves structure)
- Slide titles are prepended to slide body text so the LLM knows the topic context
- `clean_text()` strips bullet symbols (`●`, `•`, `■`, etc.), page numbers, and discards fragments with fewer than 5 real words — prevents formula-only lines from polluting the index
- Splits text on sentence boundaries (not raw word count) so chunks never cut off mid-sentence
- Chunk size: 500 words with 100-word overlap so concepts spanning two chunks are not lost
- Checks if a file is already indexed before re-ingesting to avoid duplicates
- `get_collection()` always creates a fresh ChromaDB client to avoid stale references after index deletion

**`retriever.py`**

Takes a plain-text query and returns the most relevant chunks:
- Embeds two query variations (`query` and `"what is {query}"`) to improve recall when note wording differs from the question
- Filters chunks by cosine distance threshold (1.8) — chunks too far from the query are discarded
- `keyword_match_score()` counts how many significant words from the query actually appear in each chunk, ignoring stop words
- Re-ranks candidates by keyword score (higher = more relevant) then by distance — ensures chunks that literally mention the asked concept rise to the top
- Falls back to a relaxed search if nothing passes the threshold, so the app never returns empty-handed
- Always fetches a fresh ChromaDB client to avoid stale collection references

**`app.py`**

The main entry point:
- Loads API key from `.env` via `python-dotenv`
- Handles file upload and triggers ingestion
- Calls `retriever.py` to fetch ranked context chunks
- Sends context + question to Groq with a strict system prompt: explain only the exact concept asked, do not include related topics
- Debug expander shows the raw retrieved chunks so you can verify what the LLM is working with
- Sidebar "Clear index" button deletes `chroma_db/` folder so you can re-index cleanly without restarting

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Groq API key

Add your API Key in a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com)

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Windows (PowerShell) note:** To clear the index manually, use:
> ```powershell
> Remove-Item -Recurse -Force chroma_db
> ```

---

## Usage

1. Upload a PDF or PPTX file using the file uploader
2. Wait for indexing to finish (success message shows chunk count)
3. Type any concept from your notes in the text box
4. Click **Explain**
5. Read the explanation — source file is shown below the answer
6. Use the ** Debug** expander to see which chunks were retrieved if the answer seems off

---

## Re-indexing

If you update your notes file or the explanation quality seems wrong, clear the index and re-upload:

- Use the ** Clear index and re-upload** button in the sidebar, or
- Delete the `chroma_db/` folder manually and restart the app

---

## Limitations

- Only works with text-based PDFs. Scanned PDFs (image-only) will not extract text correctly
- Explanation quality depends on how thoroughly your notes cover the topic
- The app does not have conversation memory — each question is independent
- Debug mode is currently always visible; remove the expander block in `app.py` before sharing or deploying

---

## Dependencies

```
streamlit
groq
chromadb
sentence-transformers
pymupdf
python-pptx
python-dotenv
```

---

## Built With

- [Groq](https://groq.com) — LLM inference
- [ChromaDB](https://www.trychroma.com) — vector database
- [Sentence Transformers](https://www.sbert.net) — text embeddings
- [Streamlit](https://streamlit.io) — UI
