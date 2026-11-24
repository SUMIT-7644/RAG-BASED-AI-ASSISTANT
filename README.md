# 🔍 RAG Assistant

A Retrieval-Augmented Generation (RAG) application built with:

- **Streamlit** — clean, dark-themed frontend
- **PyMuPDF** — fast PDF text extraction
- **LangChain** — text splitting & vector store management
- **FAISS** — local vector database (no cloud needed)
- **sentence-transformers** — local embedding models
- **HuggingFace Inference API** — LLM (Mistral, Zephyr, Phi-3, etc.)

---

## 📁 Project Structure

```
rag_project/
├── app.py            # Streamlit frontend
├── rag_engine.py     # Core RAG logic
├── requirements.txt  # Dependencies
├── .env.example      # Environment variable template
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / enter the project folder
```bash
cd rag_project
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your HuggingFace token
```bash
cp .env.example .env
# Edit .env and paste your token:
#   HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxx
```
Get a free token at → https://huggingface.co/settings/tokens

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🧠 How It Works

```
PDF Files
   │
   ▼
PyMuPDF  ──► raw text
   │
   ▼
RecursiveCharacterTextSplitter  ──► chunks (512 tokens, 64 overlap)
   │
   ▼
HuggingFaceEmbeddings  ──► dense vectors  (local, no API)
   │
   ▼
FAISS Vector Store  ──► similarity search
   │
   ▼  (top-k chunks as context)
HuggingFace Inference API  ──► final answer
```

---

## ⚙️ Configuration (Sidebar)

| Setting | Default | Notes |
|---|---|---|
| HF Token | — | Required |
| LLM Model | Mistral-7B-Instruct-v0.3 | Swappable |
| Embedding Model | all-MiniLM-L6-v2 | Runs locally |
| Chunk Size | 512 | Tokens per chunk |
| Chunk Overlap | 64 | Overlap between chunks |
| Top-K | 3 | Chunks retrieved per query |

---

## 💡 Notes

- Embeddings run **locally** on CPU — no extra API key needed.
- The FAISS index is kept **in memory**; re-upload PDFs after restarting.
- For larger documents, increase chunk size or reduce overlap.
- Models listed require a free HuggingFace account with Inference API access.
