import streamlit as st
import os
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv

from rag_engine import RAGEngine

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #0d0f14;
    --surface: #161a23;
    --border:  #252b38;
    --accent:  #4f8ef7;
    --accent2: #7c5cfc;
    --text:    #e2e8f0;
    --muted:   #6b7a99;
    --success: #34d399;
    --warn:    #fbbf24;
    --error:   #f87171;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1,h2,h3 { font-family: 'Space Mono', monospace !important; }

/* Style the native sidebar collapse/expand button — never hide it */
[data-testid="collapsedControl"] {
    display: flex !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 6px 6px 0 !important;
    color: var(--accent) !important;
}
[data-testid="collapsedControl"]:hover {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}
[data-testid="collapsedControl"] svg {
    fill: var(--accent) !important;
    stroke: var(--accent) !important;
}

[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); }

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

.chat-user {
    background: linear-gradient(135deg,#4f8ef722,#7c5cfc22);
    border: 1px solid #4f8ef744;
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0 0.5rem 12%;
    font-size: 0.95rem; line-height:1.6;
}
.chat-assistant {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 12% 0.5rem 0;
    font-size: 0.95rem; line-height:1.6;
}
.chat-label {
    font-family:'Space Mono',monospace;
    font-size:0.68rem; color:var(--muted);
    margin-bottom:0.25rem; letter-spacing:0.1em;
    text-transform:uppercase;
}

.source-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.83rem;
    line-height: 1.55;
}
.source-meta { display:flex; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.4rem; }
.meta-tag {
    display:inline-block; padding:0.15rem 0.6rem; border-radius:99px;
    font-family:'Space Mono',monospace; font-size:0.72rem; font-weight:700;
}
.tag-file  { background:#4f8ef722; border:1px solid #4f8ef766; color:#4f8ef7; }
.tag-page  { background:#7c5cfc22; border:1px solid #7c5cfc66; color:#a78bfa; }
.tag-sim   { background:#34d39922; border:1px solid #34d39966; color:#34d399; }
.tag-chunk { background:#fbbf2422; border:1px solid #fbbf2466; color:#fbbf24; }
.chunk-text { color:var(--muted); font-size:0.8rem; line-height:1.5; margin-top:0.4rem; }

.pill-success { display:inline-block; background:#34d39922; border:1px solid #34d39966;
    color:#34d399; border-radius:99px; padding:0.18rem 0.7rem;
    font-size:0.75rem; font-family:'Space Mono',monospace; }
.pill-info { display:inline-block; background:#4f8ef722; border:1px solid #4f8ef766;
    color:#4f8ef7; border-radius:99px; padding:0.18rem 0.7rem;
    font-size:0.75rem; font-family:'Space Mono',monospace; }

[data-testid="stTextInput"] input {
    background: var(--surface) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
}
[data-testid="stTextInput"] input:focus { border-color: var(--accent) !important; }

[data-testid="stMetric"] {
    background: var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:0.75rem 1rem;
}
[data-testid="stMetricValue"] { color:var(--accent) !important; font-family:'Space Mono',monospace !important; }
[data-testid="stMetricLabel"] { color:var(--muted) !important; font-size:0.76rem !important; }

hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("rag", None),
    ("chat_history", []),
    ("page_viewer", None),
    ("top_k", 3),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 🦙 Ollama LLM")
    ollama_model = st.selectbox("Model (must be pulled)", [
        "llama3.2",
        "llama3.2:1b",
        "mistral",
        "qwen2.5:3b",
        "qwen2.5:1.5b",
        "phi3:mini",
        "gemma3:4b",
        "deepseek-r1:7b",
    ])
    ollama_hints = {
        "llama3.2":        "~2 GB  · great all-rounder",
        "llama3.2:1b":     "~1.3 GB · fastest on CPU",
        "mistral":         "~4 GB  · excellent for RAG",
        "qwen2.5:3b":      "~2 GB  · multilingual",
        "qwen2.5:1.5b":    "~1 GB  · very fast",
        "phi3:mini":       "~2 GB  · great on CPU",
        "gemma3:4b":       "~3 GB  · strong reasoning",
        "deepseek-r1:7b":  "~5 GB  · best reasoning, slower",
    }
    st.caption(f"💾 {ollama_hints.get(ollama_model, '—')}")
    st.caption(f"Pull command:  `ollama pull {ollama_model}`")

    st.markdown("### 🔢 Embedding Model")
    embed_model = st.selectbox("Embedding Model", [
        "nomic-embed-text",
        "all-minilm",
    ])
    embed_hints = {
        "nomic-embed-text": "~274 MB · best quality",
        "all-minilm":       "~46 MB  · lightweight",
    }
    st.caption(f"💾 {embed_hints.get(embed_model, '—')}")
    st.caption(f"Pull command:  `ollama pull {embed_model}`")

    chunk_size    = st.slider("Chunk Size (chars)",  400, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap",         0,  400,  150,  25)
    top_k         = st.slider("Top-K Retrieval",       1,   10,    6)
    st.session_state.top_k = top_k

    st.markdown("---")
    st.info("Make sure **Ollama is running** before initialising.\nEmbedding model downloads once from HuggingFace.", icon="ℹ️")

    if st.button("🔌 Initialize Engine", use_container_width=True):
        with st.spinner(f"Connecting to Ollama · loading embedding model…"):
            try:
                st.session_state.rag = RAGEngine(
                    ollama_model=ollama_model,
                    embed_model=embed_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                st.success(f"✅ Ready!  LLM: {ollama_model}  ·  Embeddings: local")
            except Exception as e:
                st.error(f"❌ {e}")

    if st.session_state.rag:
        indexed = st.session_state.rag.get_indexed_files()
        if indexed:
            st.markdown("### 📚 Indexed Files")
            st.markdown(
                f'<span class="pill-success">● {st.session_state.rag.collection.count()} chunks</span>',
                unsafe_allow_html=True,
            )
            for fname in indexed:
                col_f, col_d = st.columns([3, 1])
                col_f.markdown(f"<small>📄 {fname}</small>", unsafe_allow_html=True)
                if col_d.button("✕", key=f"del_{fname}", help=f"Remove {fname}"):
                    st.session_state.rag.delete_pdf(fname)
                    st.rerun()
        else:
            st.markdown('<span class="pill-info">○ No files indexed</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.page_viewer  = None
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:1.9rem; margin-bottom:0;'>🔍 RAG Assistant</h1>
<p style='color:#6b7a99; margin-top:0.2rem; font-size:0.92rem;'>
Persistent Vector DB · Page-level Retrieval · Inline PDF Viewer
</p>
""", unsafe_allow_html=True)
st.markdown("---")

left_col, right_col = st.columns([1, 1.7], gap="large")

# ── LEFT: Upload ──────────────────────────────────────────────────────────────
with left_col:
    st.markdown("### 📂 Upload & Index PDFs")

    if not st.session_state.rag:
        st.info("👈 Make sure Ollama is running, then click **Initialize Engine** in the sidebar.")
    else:
        uploaded_files = st.file_uploader(
            "Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            for f in uploaded_files:
                st.markdown(f"- `{f.name}` — {round(f.size/1024, 1)} KB")

        if st.button("⚡ Index PDFs", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload at least one PDF.")
            else:
                progress = st.progress(0, text="Saving files…")
                try:
                    tmp_dir = Path(tempfile.gettempdir()) / "rag_pdfs"
                    tmp_dir.mkdir(exist_ok=True)
                    pdf_paths = []
                    for f in uploaded_files:
                        p = tmp_dir / f.name
                        p.write_bytes(f.read())
                        pdf_paths.append(str(p))

                    progress.progress(20, text="Indexing chunks…")
                    meta = st.session_state.rag.build_index(pdf_paths)
                    progress.progress(100, text="Done!")

                    skipped = meta["skipped"]
                    new_f   = meta["files"] - skipped
                    if skipped and new_f == 0:
                        st.warning(f"⚠️ All {skipped} file(s) already indexed — skipped.")
                    elif skipped:
                        st.success(f"✅ Indexed {new_f} new file(s), skipped {skipped} already in DB. Total: {meta['total_chunks']} chunks.")
                    else:
                        st.success(f"✅ Indexed {meta['new_chunks']} chunks from {meta['new_pages']} pages. Total: {meta['total_chunks']} chunks.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")
                finally:
                    progress.empty()

        if st.session_state.rag:
            indexed = st.session_state.rag.get_indexed_files()
            if indexed:
                st.markdown("---")
                c1, c2 = st.columns(2)
                c1.metric("Files", len(indexed))
                c2.metric("Chunks", st.session_state.rag.collection.count())

    st.markdown("---")
    st.markdown("""
**How it works**

1. Install Ollama → run `ollama pull llama3.2`
2. Start Ollama, click **Initialize Engine**
3. Upload PDFs → **Index PDFs**
4. Ask questions → click **View Page** on any source
""")


# ── RIGHT: Chat + Page Viewer ─────────────────────────────────────────────────
with right_col:
    st.markdown("### 💬 Chat")

    chat_box = st.container(height=400)
    with chat_box:
        if not st.session_state.chat_history:
            st.markdown(
                '<div style="color:#6b7a99;text-align:center;padding:3rem 1rem;'
                'font-family:Space Mono,monospace;font-size:0.82rem;">'
                'Index some PDFs, then ask anything.</div>',
                unsafe_allow_html=True,
            )
        for turn in st.session_state.chat_history:
            st.markdown(
                f'<div class="chat-label">YOU</div>'
                f'<div class="chat-user">{turn["question"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="chat-label">ASSISTANT</div>'
                f'<div class="chat-assistant">{turn["answer"]}</div>',
                unsafe_allow_html=True,
            )

            if turn.get("sources"):
                st.markdown(
                    '<div class="chat-label" style="margin-top:0.6rem;">SOURCES</div>',
                    unsafe_allow_html=True,
                )
                for i, src in enumerate(turn["sources"]):
                    st.markdown(f"""
<div class="source-card">
  <div class="source-meta">
    <span class="meta-tag tag-file">📄 {src['source']}</span>
    <span class="meta-tag tag-page">📖 Page {src['page_label']} / {src['total_pages']}</span>
    <span class="meta-tag tag-sim">🎯 {src['similarity']}% match</span>
    <span class="meta-tag tag-chunk">🧩 Chunk #{src['chunk_index']}</span>
  </div>
  <div class="chunk-text">{src['text'][:280]}{'…' if len(src['text']) > 280 else ''}</div>
</div>""", unsafe_allow_html=True)

                    if st.button(
                        f"🔎 View Page {src['page_label']} — {src['source']}",
                        key=f"view_{turn.get('id', '')}_src{i}",
                        use_container_width=True,
                    ):
                        st.session_state.page_viewer = {
                            "pdf_path":    src["pdf_path"],
                            "page_number": src["page_number"],
                            "page_label":  src["page_label"],
                            "total_pages": src["total_pages"],
                            "source":      src["source"],
                            "chunk_text":  src["text"],
                        }
                        st.rerun()

    query   = st.text_input(
        "Ask a question…",
        placeholder="What does the document say about…?",
        label_visibility="collapsed",
    )
    ask_btn = st.button("Send →", use_container_width=True)

    if ask_btn and query:
        if not st.session_state.rag or st.session_state.rag.collection.count() == 0:
            st.warning("Index some PDFs first.")
        else:
            with st.spinner("Thinking…"):
                try:
                    result = st.session_state.rag.query(
                        query, top_k=st.session_state.top_k
                    )
                    st.session_state.chat_history.append({
                        "id":       str(time.time()),
                        "question": query,
                        "answer":   result["answer"],
                        "sources":  result["sources"],
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")

    # ── Page Viewer ───────────────────────────────────────────────────────────
    pv = st.session_state.page_viewer
    if pv:
        st.markdown("---")
        st.markdown("### 📄 Page Viewer")

        nav1, nav2, nav3, nav4 = st.columns([1, 1, 2, 1])
        with nav1:
            if st.button("◀ Prev"):
                if pv["page_number"] > 0:
                    st.session_state.page_viewer["page_number"] -= 1
                    st.session_state.page_viewer["page_label"]  -= 1
                    st.rerun()
        with nav2:
            if st.button("Next ▶"):
                if pv["page_number"] < pv["total_pages"] - 1:
                    st.session_state.page_viewer["page_number"] += 1
                    st.session_state.page_viewer["page_label"]  += 1
                    st.rerun()
        with nav3:
            st.markdown(
                f'<div style="text-align:center;font-family:Space Mono,monospace;'
                f'font-size:0.82rem;color:#6b7a99;padding-top:0.5rem;">'
                f'📄 {pv["source"]}  ·  Page {pv["page_label"]} of {pv["total_pages"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with nav4:
            if st.button("✕ Close"):
                st.session_state.page_viewer = None
                st.rerun()

        try:
            img_bytes = RAGEngine.render_page(pv["pdf_path"], pv["page_number"], dpi=160)
            st.image(
                img_bytes,
                use_container_width=True,
                caption=f"{pv['source']} — Page {pv['page_label']}",
            )
        except FileNotFoundError:
            st.error("⚠️ PDF not found on disk. Please re-upload and re-index the file.")
        except Exception as e:
            st.error(f"Could not render page: {e}")

        with st.expander("📌 Matched chunk from this page", expanded=True):
            st.markdown(
                f'<div style="background:#4f8ef711;border-left:3px solid #4f8ef7;'
                f'border-radius:6px;padding:0.8rem 1rem;font-size:0.85rem;'
                f'line-height:1.6;color:#e2e8f0;">{pv["chunk_text"]}</div>',
                unsafe_allow_html=True,
            )