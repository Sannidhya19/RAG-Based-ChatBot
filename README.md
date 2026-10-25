# 📚 Notes & Papers Q&A Bot (RAG)

A Retrieval-Augmented Generation (RAG) app that answers questions grounded in
your own notes and research papers. Embeddings run locally (free), and
generation uses Groq's free-tier LLM API.

## How it works
1. `ingest.py` loads your files from `data/`, splits them into overlapping
   chunks, embeds them locally with `sentence-transformers`, and stores them
   in a FAISS vector index.
2. `app.py` (Streamlit) takes your question, retrieves the most relevant
   chunks from the index, and sends them + your question to a free Groq LLM,
   which answers using only that context.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get a free Groq API key
# -> https://console.groq.com/keys (no credit card required)

# 4. Set up your .env file
cp .env.example .env
# then edit .env and paste your GROQ_API_KEY

# 5. Add your documents
# Drop .txt, .md, or .pdf files (your notes, research papers, etc.) into data/

# 6. Build the index
python ingest.py

# 7. Run the app
streamlit run app.py
```

## Project structure
```
rag-qa-bot/
├── data/           # put your notes/papers here (.txt, .md, .pdf)
├── index/           # generated: FAISS index + chunk store
├── ingest.py         # document loading, chunking, embedding
├── app.py            # Streamlit UI + retrieval + Groq generation
├── requirements.txt
├── .env.example
└── README.md
```

## Tech stack
- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2) — free, runs locally
- **Vector store:** FAISS
- **Generation:** Groq API (Llama 3.3 70B) — free tier
- **UI:** Streamlit

## Possible improvements (good for a resume bullet or interview talking point)
- Swap character-based chunking for sentence/paragraph-aware chunking
- Add a small evaluation set of Q&A pairs to measure retrieval accuracy
- Cite exact source + page number instead of just filename
- Deploy for free on Streamlit Community Cloud or HuggingFace Spaces
