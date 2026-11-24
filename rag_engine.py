"""
rag_engine.py
Fully local RAG engine — no cloud API calls at runtime.

Pipeline (sequential, never concurrent):
  1. HuggingFaceEmbeddings (sentence-transformers) — embedding, local CPU
  2. ChromaDB — persistent vector store on disk
  3. Ollama — LLM inference, local (runs as a background service)

Embedding model  : downloaded from HuggingFace Hub on first use,
                   cached in ~/.cache/huggingface/
LLM              : pulled via `ollama pull <model>` once, then served
                   locally by the Ollama daemon (default http://localhost:11434)
"""

from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.config import Settings


# ── Ollama Embeddings ─────────────────────────────────────────────────────────

class OllamaEmbeddings:
    """
    Embedding client that calls Ollama's /api/embed endpoint.
    Use a dedicated embedding model for best quality, e.g.:
      • nomic-embed-text  (~274 MB, best quality)
      • all-minilm        (~46 MB,  lightweight)
    Pull with:  ollama pull nomic-embed-text
    """

    def __init__(self, model: str, ollama_url: str):
        self.model      = model
        self.ollama_url = ollama_url.rstrip("/")

    def _embed(self, text: str) -> list[float]:
        resp = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.model, "input": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.model, "input": texts},  # ← pass the whole list at once
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]
    
    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# ── Constants ────────────────────────────────────────────────────────────────

CHROMA_DIR     = "./chroma_db"
MIN_SIMILARITY = 0.30                        # drop chunks below 30% cosine similarity
OLLAMA_URL     = "http://localhost:11434"    # default Ollama daemon address


# ── RAGEngine ────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Fully local RAG engine using Ollama for LLM inference.

    Parameters
    ----------
    ollama_model : str
        Any model you have pulled with `ollama pull <model>`, e.g.
          • "llama3.2"          (~2 GB, fast, great quality)
          • "mistral"           (~4 GB, excellent for RAG)
          • "qwen2.5:3b"        (~2 GB, multilingual)
          • "phi3:mini"         (~2 GB, very fast on CPU)
          • "gemma3:4b"         (~3 GB, strong reasoning)
    embed_model : str
        Any sentence-transformers model ID from HuggingFace Hub.
    chunk_size : int
        Characters per chunk (~200–250 tokens at 1000 chars).
    chunk_overlap : int
        Overlap between consecutive chunks.
    max_new_tokens : int
        Maximum tokens the LLM may generate per answer.
    ollama_url : str
        Base URL of the Ollama daemon (change if you run it on a custom port).
    """

    def __init__(
        self,
        ollama_model:   str = "llama3.2",
        embed_model:    str = "nomic-embed-text",
        chunk_size:     int = 1000,
        chunk_overlap:  int = 150,
        max_new_tokens: int = 512,
        ollama_url:     str = OLLAMA_URL,
    ):
        self.ollama_model   = ollama_model
        self.embed_model_id = embed_model
        self.chunk_size     = chunk_size
        self.chunk_overlap  = chunk_overlap
        self.max_new_tokens = max_new_tokens
        self.ollama_url     = ollama_url.rstrip("/")

        # ── 1. Verify Ollama is running ───────────────────────────────────
        self._check_ollama()

        # ── 2. Load embedding model (Ollama, local) ───────────────────────
        print(f"[RAGEngine] Using Ollama embedding model: {embed_model}")
        self.embeddings = OllamaEmbeddings(
            model=embed_model,
            ollama_url=self.ollama_url,
        )
        print("[RAGEngine] Embedding model ready.")

        # ── 3. ChromaDB (persistent on disk) ─────────────────────────────
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[RAGEngine] ChromaDB ready. ({self.collection.count()} chunks indexed)")

        # ── 4. Text splitter ──────────────────────────────────────────────
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Ollama helpers ────────────────────────────────────────────────────────

    def _check_ollama(self):
        """Raise a clear error if the Ollama daemon is not reachable."""
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            # Warn if the chosen model isn't pulled yet
            model_base = self.ollama_model.split(":")[0]
            if models and not any(model_base in m for m in models):
                print(
                    f"[RAGEngine] WARNING: '{self.ollama_model}' not found in Ollama. "
                    f"Run:  ollama pull {self.ollama_model}"
                )
            else:
                print(f"[RAGEngine] Ollama running. Model '{self.ollama_model}' ready.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama at http://localhost:11434.\n"
                "Make sure Ollama is installed and running:\n"
                "  Windows / macOS : open the Ollama app\n"
                "  Linux           : run `ollama serve` in a terminal"
            )

    def _generate(self, context: str, question: str) -> str:
        """
        Call Ollama's /api/chat endpoint (non-streaming).
        Embeddings are computed before this is called — fully sequential.
        """
        payload = {
            "model": self.ollama_model,
            "stream": False,
            "options": {"num_predict": self.max_new_tokens, "temperature": 0.2},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a knowledgeable assistant. You are given excerpts from documents "
                        "and a question. Answer the question based on the excerpts. "
                        "Be thorough — if multiple excerpts are relevant, combine them into a complete answer. "
                        "If the excerpts contain partial information, share what you found and note what is missing. "
                        "Only say you cannot answer if the excerpts are genuinely unrelated to the question."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT EXCERPTS:\n{context}\n\n"
                        f"QUESTION: {question}\n\n"
                        "Please answer based on the excerpts above."
                    ),
                },
            ],
        }
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=120,   # large models can be slow on CPU
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Ollama request timed out (120 s). "
                "Try a smaller model like 'phi3:mini' or 'qwen2.5:3b'."
            )
        except Exception as e:
            raise RuntimeError(f"Ollama inference error: {e}")

    # ── Indexing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _file_hash(path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def already_indexed(self, pdf_path: str) -> bool:
        file_hash = self._file_hash(pdf_path)
        results   = self.collection.get(where={"file_hash": file_hash}, limit=1)
        return len(results["ids"]) > 0

    def index_pdf(self, pdf_path: str) -> dict[str, Any]:
        if self.already_indexed(pdf_path):
            fname    = Path(pdf_path).name
            existing = self.collection.get(where={"source": fname})
            return {"skipped": True, "source": fname, "chunks": len(existing["ids"])}

        file_hash   = self._file_hash(pdf_path)
        fname       = Path(pdf_path).name
        doc         = fitz.open(pdf_path)
        total_pages = len(doc)

        all_texts:  list[str]  = []
        all_metas:  list[dict] = []
        all_ids:    list[str]  = []
        chunk_index = 0

        for page_num in range(total_pages):
            page_text = doc[page_num].get_text("text").strip()
            if not page_text:
                continue
            chunks = self.splitter.split_text(page_text)
            for chunk in chunks:
                chunk_id = f"{file_hash}_p{page_num}_c{chunk_index}"
                all_ids.append(chunk_id)
                all_texts.append(chunk)
                all_metas.append({
                    "source":      fname,
                    "pdf_path":    str(Path(pdf_path).resolve()),
                    "page_number": page_num,
                    "page_label":  page_num + 1,
                    "total_pages": total_pages,
                    "chunk_index": chunk_index,
                    "file_hash":   file_hash,
                })
                chunk_index += 1

        doc.close()

        if not all_texts:
            raise ValueError(f"No text extracted from {fname}")

        # Embed in batches of 100
        for i in range(0, len(all_texts), 100):
            self.collection.add(
                ids=all_ids[i : i + 100],
                documents=all_texts[i : i + 100],
                embeddings=self._embed_texts(all_texts[i : i + 100]),
                metadatas=all_metas[i : i + 100],
            )

        return {"skipped": False, "source": fname, "pages": total_pages, "chunks": chunk_index}

    def build_index(self, pdf_paths: list[str]) -> dict[str, Any]:
        results      = []
        total_pages  = total_chunks = skipped = 0
        for path in pdf_paths:
            r = self.index_pdf(path)
            results.append(r)
            if r["skipped"]:
                skipped += 1
            else:
                total_pages  += r.get("pages", 0)
                total_chunks += r.get("chunks", 0)
        return {
            "files":         len(pdf_paths),
            "skipped":       skipped,
            "new_pages":     total_pages,
            "new_chunks":    total_chunks,
            "total_chunks":  self.collection.count(),
            "file_results":  results,
        }

    def get_indexed_files(self) -> list[str]:
        if self.collection.count() == 0:
            return []
        all_metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted(set(m["source"] for m in all_metas))

    def delete_pdf(self, filename: str):
        self.collection.delete(where={"source": filename})

    # ── Query ────────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 6) -> dict[str, Any]:
        """
        1. Embed the question (embedding model, CPU).
        2. Retrieve top_k*2 candidates from ChromaDB, filter by similarity.
        3. Build context string.
        4. Generate answer with local LLM (sequential, after embeddings).
        """
        if self.collection.count() == 0:
            raise RuntimeError("No documents indexed yet.")

        # ── Step 1: embed question ────────────────────────────────────────
        query_embedding = self.embeddings.embed_query(question)

        # ── Step 2: vector search + similarity filter ─────────────────────
        fetch_k = min(top_k * 2, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        filtered = [
            (doc, meta, dist)
            for doc, meta, dist in zip(docs, metas, distances)
            if (1 - dist) >= MIN_SIMILARITY
        ][:top_k]

        if not filtered:
            return {
                "answer": (
                    f"I could not find relevant information in the indexed documents "
                    f"to answer your question. The best match had less than "
                    f"{int(MIN_SIMILARITY * 100)}% similarity. "
                    "Try rephrasing your question or check that the right PDFs are indexed."
                ),
                "sources": [],
            }

        docs, metas, distances = zip(*filtered)

        # ── Step 3: build context ─────────────────────────────────────────
        context_parts = [
            f"[Excerpt {i+1} — {meta['source']}, Page {meta['page_label']}]\n{doc}"
            for i, (doc, meta) in enumerate(zip(docs, metas))
        ]
        context = "\n\n---\n\n".join(context_parts)

        # ── Step 4: LLM inference via Ollama (runs after embeddings are done) ──
        answer = self._generate(context, question)

        # No GPU cache to clear — Ollama manages its own memory

        sources = [
            {
                "source":      meta["source"],
                "pdf_path":    meta["pdf_path"],
                "page_number": meta["page_number"],
                "page_label":  meta["page_label"],
                "total_pages": meta["total_pages"],
                "chunk_index": meta["chunk_index"],
                "similarity":  round((1 - dist) * 100, 1),
                "text":        doc,
            }
            for doc, meta, dist in zip(docs, metas, distances)
        ]

        return {"answer": answer, "sources": sources}

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def render_page(pdf_path: str, page_number: int, dpi: int = 150) -> bytes:
        """Render a PDF page (0-indexed) to PNG bytes."""
        doc       = fitz.open(pdf_path)
        page      = doc[page_number]
        mat       = fitz.Matrix(dpi / 72, dpi / 72)
        pix       = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
