"""
Streamlit UI for the Agentic Hybrid RAG pipeline.

Run (from the project root, with the venv active):
    streamlit run main.py
"""

import logging
import os
import uuid

import streamlit as st

from config.settings import OPENROUTER_API_KEY
from core.loaders import SOURCE_CODE_EXTENSIONS

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "database", "chroma_db")
DOCUMENT_TYPES = ["pdf", "docx", "txt", "ppt", "pptx", "wav"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

st.set_page_config(
    page_title="Agentic Hybrid RAG",
    page_icon="🔍",
    layout="wide",
)


# ──────────────────────────────────────────────
# Shared resources (cached across reruns)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading knowledge base and models…")
def get_knowledge_base():
    from database.chroma_pipeline import KnowledgeBasePipeline

    return KnowledgeBasePipeline(persist_directory=CHROMA_DIR)


def get_orchestrator():
    """One Orchestrator per chat session (stateless, but tied to the KB)."""
    if "orchestrator" not in st.session_state:
        from core.orchestrator import Orchestrator

        st.session_state.orchestrator = Orchestrator(
            pipeline=get_knowledge_base(),
            max_analyst_loops=2,
            source_style="numbered",
        )
    return st.session_state.orchestrator


def short_name(source: str) -> str:
    if source.startswith(("http://", "https://", "wiki:")):
        return source
    return os.path.basename(source)


# ──────────────────────────────────────────────
# Ingestion helpers
# ──────────────────────────────────────────────

def save_upload(uploaded_file) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, os.path.basename(uploaded_file.name))
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def ingest(sources: list[str]):
    kb = get_knowledge_base()
    # Re-ingesting a source replaces old chunks instead of leaving orphans.
    for src in sources:
        kb.delete_source(src)

    with st.spinner(
        f"Ingesting {len(sources)} source(s)… (audio transcription can take a while)"
    ):
        results = kb.ingest_batch(sources)

    for r in results:
        name = short_name(r["source"])
        if r["status"] == "success":
            st.success(f"{name}: {r['chunks_ingested']} chunks ingested")
        elif r["status"] == "warning":
            st.warning(f"{name}: {r['message']}")
        else:
            st.error(f"{name}: {r.get('error', 'Unknown error')}")


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def render_sidebar():
    kb = get_knowledge_base()

    st.sidebar.title("🔍 Knowledge Base")

    # ── Status indicators ──
    st.sidebar.caption(
        f"{'🟢' if OPENROUTER_API_KEY else '🔴'} OpenRouter API key"
    )
    if not OPENROUTER_API_KEY:
        st.sidebar.error("Set OPENROUTER_API_KEY in .env, then restart the app.")

    # ── Add knowledge ──
    with st.sidebar.form("add_knowledge", clear_on_submit=True):
        st.subheader("Add knowledge")
        files = st.file_uploader(
            "Files",
            type=DOCUMENT_TYPES
            + sorted(ext.lstrip(".") for ext in SOURCE_CODE_EXTENSIONS),
            accept_multiple_files=True,
            help="PDF, DOCX, TXT, PPT/PPTX, WAV audio, or source code",
        )
        urls = st.text_area(
            "Web page URLs (one per line)",
            placeholder="https://example.com/article",
        )
        wiki = st.text_input(
            "Wikipedia topic or URL",
            placeholder="Retrieval-Augmented Generation",
        )
        submitted = st.form_submit_button(
            "Ingest", type="primary", use_container_width=True
        )

    if submitted:
        sources: list[str] = [save_upload(f) for f in (files or [])]
        for url in (u.strip() for u in urls.splitlines() if u.strip()):
            if url.startswith(("http://", "https://")):
                sources.append(url)
            else:
                st.sidebar.warning(
                    f"Skipped `{url}`: URLs must start with http:// or https://"
                )
        if wiki.strip():
            topic = wiki.strip()
            sources.append(
                topic if "wikipedia.org/wiki/" in topic else f"wiki:{topic}"
            )

        if sources:
            with st.sidebar:
                ingest(sources)
        else:
            st.sidebar.warning("Add at least one file, URL, or Wikipedia topic.")

    # ── Stored sources ──
    stored = kb.list_sources()
    st.sidebar.subheader(f"Sources ({len(stored)})")
    st.sidebar.caption(f"{kb.collection.count()} chunks stored")
    for src in stored:
        col_name, col_del = st.sidebar.columns([5, 1])
        col_name.markdown(f"`{short_name(src)}`", help=src)
        if col_del.button("🗑", key=f"del_{src}", help="Remove from knowledge base"):
            kb.delete_source(src)
            if src.startswith(UPLOAD_DIR) and os.path.isfile(src):
                os.remove(src)
            st.rerun()

    if stored and st.sidebar.button(
        "Clear knowledge base", use_container_width=True
    ):
        kb.clear()
        st.rerun()

    # ── Session controls ──
    st.sidebar.divider()
    st.sidebar.caption(f"Session: `{st.session_state.session_id[:8]}`")
    if st.sidebar.button("New chat", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ──────────────────────────────────────────────
# Chat display helpers
# ──────────────────────────────────────────────

def render_result(result: dict):
    """Renders an assistant response with answer, sources, and debug details."""
    st.markdown(result["answer"])

    # Expandable details
    with st.expander("Pipeline details"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Evidence chunks", len(result.get("evidence", [])))
        col2.metric(
            "Analyst tool calls", len(result.get("analyst_tool_calls", []))
        )
        col3.metric("Time (s)", result.get("execution_time_seconds", "?"))

        if result.get("analysis"):
            st.subheader("Analyst output")
            st.markdown(result["analysis"])

        if result.get("sources"):
            st.subheader("Sources")
            st.markdown(result["sources"])


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []

    render_sidebar()

    st.title("Agentic Hybrid RAG")
    st.caption(
        "Retriever Agent → Analyst Agent → Answer Agent  ·  "
        "Ask questions about your ingested documents."
    )

    if get_knowledge_base().collection.count() == 0:
        st.info(
            "The knowledge base is empty. "
            "Add files, URLs, or Wikipedia topics from the sidebar first."
        )

    # Replay chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_result(msg["result"])

    # New user message
    question = st.chat_input("Ask a question about your knowledge base…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build chat history for the orchestrator (for query rewriting context)
    chat_history = [
        {"role": m["role"], "content": m.get("content") or m.get("result", {}).get("answer", "")}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Retrieving, analysing, and generating answer…"):
            result = get_orchestrator().run(question, chat_history=chat_history)
        render_result(result)

    st.session_state.messages.append({"role": "assistant", "result": result})


main()
