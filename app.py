import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from ingest import ingest_file, already_ingested
from retriever import retrieve

load_dotenv()

st.set_page_config(page_title="Exam Prep", layout="centered")

st.title("Exam Prep")
st.caption("Upload your notes and ask it to explain any concept from them.")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def explain(query: str) -> tuple[str, list[str]]:
    chunks, sources = retrieve(query)

    if not chunks:
        return "No notes have been indexed yet. Please upload a file first.", []

    context = "\n\n".join(chunks)
    unique_sources = list(set(sources))

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a study assistant. "
                    "You will be given excerpts from a student's notes and a specific concept to explain. "
                    "Explain ONLY that exact concept using ONLY information from the notes that directly describes it. "
                    "Do NOT explain related concepts, similar algorithms, or other topics even if they appear in the notes. "
                    "Structure: Definition, How it works, Key points. "
                    "If the notes do not contain information specifically about the asked concept, say: "
                    "'Your notes do not contain information about [concept].'"
                ),
            },
            {
                "role": "user",
                "content": f"Notes:\n{context}\n\nConcept to explain: {query}",
            },
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    return answer, unique_sources


#File upload 
st.subheader("1. Upload your notes")
uploaded_file = st.file_uploader("Supported formats: PDF, PPTX", type=["pdf", "pptx"])

if uploaded_file is not None:
    filename = uploaded_file.name

    if already_ingested(filename):
        st.info(f" **{filename}** is already indexed.")
    else:
        ext = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner(f"Reading and indexing {filename}… this may take a moment."):
            try:
                num_chunks = ingest_file(tmp_path, filename)
                st.success(f" Indexed **{filename}** → {num_chunks} chunks stored.")
            except Exception as e:
                st.error(f"Failed to index file: {e}")
            finally:
                os.unlink(tmp_path)

#Explain
st.subheader("2. Ask a concept to explain")
query = st.text_input(
    "What concept do you want explained?"
)

if st.button("Explain", type="primary") and query.strip():
    with st.spinner("Finding relevant sections from your notes…"):
        # Debug: show raw retrieved chunks
        from retriever import retrieve as raw_retrieve
        debug_chunks, debug_sources = raw_retrieve(query.strip())
        
        answer, sources = explain(query.strip())

    st.markdown("###  Explanation")
    st.write(answer)

    if sources:
        st.markdown("---")
        st.caption(" From: " + ", ".join(sources))