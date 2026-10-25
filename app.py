"""
app.py
Streamlit front-end for the RAG Q&A bot.

Two ways to get documents in:
  1. Persisted: put files in data/ and run `python ingest.py` once.
  2. On-the-fly: upload files directly in the sidebar in the browser.

Both sources are combined at query time, so you can mix your pre-built
notes with a file you just dragged in.

Run with:
    streamlit run app.py
"""

import io
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_DIR = Path("index")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"  # swap here if Groq retires this one too
TOP_K = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
ALLOWED_UPLOAD_TYPES = ["txt", "md", "pdf"]


# ---------- chunking / text extraction (same logic as ingest.py) ----------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    text = text.replace("\n", " ").strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_text_from_upload(uploaded_file) -> str:
    """Read text out of an in-memory uploaded file (txt, md, or pdf)."""
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.getvalue().decode("utf-8", errors="ignore")


# ---------- cached resources ----------

@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


@st.cache_resource
def load_persisted_index():
    """Load the FAISS index built by ingest.py, if one exists."""
    if not (INDEX_DIR / "faiss.index").exists():
        return None, [], []
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["sources"]


# ---------- uploaded-file handling ----------

def process_uploads(uploaded_files, embed_model):
    """Chunk + embed uploaded files, caching by (name, size) so a Streamlit
    rerun doesn't re-embed files that haven't actually changed."""
    key = tuple((f.name, f.size) for f in uploaded_files)
    if st.session_state.get("_upload_key") == key:
        return  # nothing changed since last run

    all_chunks, all_sources, all_embeddings = [], [], []
    for f in uploaded_files:
        text = extract_text_from_upload(f)
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = embed_model.encode(chunks, convert_to_numpy=True).astype("float32")
        all_chunks.extend(chunks)
        all_sources.extend([f"\U0001F4E4 {f.name}"] * len(chunks))  # 📤 prefix marks it as uploaded
        all_embeddings.append(embeddings)

    st.session_state["_upload_key"] = key
    st.session_state["upload_chunks"] = all_chunks
    st.session_state["upload_sources"] = all_sources
    st.session_state["upload_embeddings"] = (
        np.vstack(all_embeddings) if all_embeddings else None
    )


def get_combined_pool(embed_model):
    """Combine the persisted index (from ingest.py) with anything uploaded
    this session into one in-memory FAISS index for retrieval."""
    base_index, base_chunks, base_sources = load_persisted_index()

    upload_embeddings = st.session_state.get("upload_embeddings")
    upload_chunks = st.session_state.get("upload_chunks", [])
    upload_sources = st.session_state.get("upload_sources", [])

    vectors = []
    chunks = list(base_chunks)
    sources = list(base_sources)

    if base_index is not None and base_index.ntotal > 0:
        vectors.append(base_index.reconstruct_n(0, base_index.ntotal))
    if upload_embeddings is not None:
        vectors.append(upload_embeddings)
        chunks.extend(upload_chunks)
        sources.extend(upload_sources)

    if not vectors:
        return None, [], []

    all_vectors = np.vstack(vectors).astype("float32")
    dimension = all_vectors.shape[1]
    combined_index = faiss.IndexFlatL2(dimension)
    combined_index.add(all_vectors)
    return combined_index, chunks, sources


def retrieve(question: str, embed_model, index, chunks, sources, k=TOP_K):
    q_vec = embed_model.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_vec, min(k, index.ntotal))
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append(
            {"text": chunks[idx], "source": sources[idx], "distance": float(dist)}
        )
    return results


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )
    return f"""You are a helpful assistant answering questions using ONLY the context below.
If the answer isn't in the context, say you don't know — do not make anything up.

Context:
{context}

Question: {question}

Answer:"""


def main():
    st.set_page_config(page_title="Notes & Papers Q&A Bot", page_icon="📚")
    st.title("📚 Notes & Papers Q&A Bot")
    st.caption("Ask questions grounded in your own notes and research papers — answered by a free LLM via Groq.")

    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not set. Copy .env.example to .env and add your free Groq API key.")
        return

    embed_model = load_embed_model()
    groq_client = load_groq_client()

    with st.sidebar:
        st.header("Upload documents")
        uploaded_files = st.file_uploader(
            "Add .txt, .md, or .pdf files to ask questions about",
            type=ALLOWED_UPLOAD_TYPES,
            accept_multiple_files=True,
        )
        if uploaded_files:
            with st.spinner("Reading and embedding uploaded files..."):
                process_uploads(uploaded_files, embed_model)
            st.success(f"{len(uploaded_files)} file(s) ready to query.")

        if st.session_state.get("upload_chunks"):
            if st.button("Clear uploaded files"):
                for k in ("_upload_key", "upload_chunks", "upload_sources", "upload_embeddings"):
                    st.session_state.pop(k, None)
                st.rerun()

    index, chunks, sources = get_combined_pool(embed_model)

    if index is None:
        st.info(
            "Upload a file in the sidebar to get started — or add files to "
            "`data/` and run `python ingest.py` for a permanent knowledge base."
        )
        return

    question = st.text_input("Ask a question about your notes or papers:")

    if question:
        with st.spinner("Retrieving relevant chunks..."):
            retrieved = retrieve(question, embed_model, index, chunks, sources)

        with st.spinner("Generating answer..."):
            prompt = build_prompt(question, retrieved)
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = response.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Sources used"):
            for c in retrieved:
                st.markdown(f"**{c['source']}** (distance: {c['distance']:.3f})")
                st.caption(c["text"][:300] + "...")


if __name__ == "__main__":
    main()