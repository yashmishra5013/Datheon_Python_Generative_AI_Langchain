"""
Task 4 - Interactive Full-Stack AI Chatbot with Streamlit.

Features
  * Real-time token streaming (st.write_stream)
  * Chat history + conversation memory (st.session_state)
  * Two modes: General Chat  |  Document Q&A (RAG, reuses the Task 2 pipeline)
  * Upload PDF / TXT / MD files from the sidebar, answers cite their sources
  * Provider / model / temperature controls and clear error messages

Run from the repository root:
    streamlit run task4_streamlit_chatbot/app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402

from common.llm_factory import (  # noqa: E402
    PROVIDERS,
    ConfigError,
    default_model,
    default_provider,
    get_chat_model,
    message_text,
)
from task2_rag_qa.rag_pipeline import RAGPipeline, source_label  # noqa: E402

CHAT_MODE = "💬 General Chat"
RAG_MODE = "📄 Document Q&A (RAG)"
MAX_HISTORY = 12  # messages sent to the model as memory
SAMPLE_DOC = ROOT / "task2_rag_qa" / "data" / "generative_ai_handbook.txt"
SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable AI assistant. Give clear, accurate answers. "
    "Use Markdown for formatting and code blocks for code. If you are unsure, say so."
)

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner=False)
def load_llm(provider: str, model: str, temperature: float):
    return get_chat_model(provider, model, temperature)


@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def load_rag(provider: str, model: str, temperature: float) -> RAGPipeline:
    return RAGPipeline.from_env(provider, model, temperature, persist_subdir="task4")


def to_lc_messages(messages: list[dict]) -> list:
    return [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in messages
    ]


def stream_chat(llm, history: list, prompt: str):
    messages = [SystemMessage(content=SYSTEM_PROMPT), *history, HumanMessage(content=prompt)]
    for chunk in llm.stream(messages):
        text = message_text(chunk.content)
        if text:
            yield text


# --------------------------------------------------------------- session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources"?}]

# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    providers = list(PROVIDERS)
    provider = st.selectbox("LLM provider", providers, index=providers.index(default_provider())
                            if default_provider() in providers else 0)
    model = st.text_input("Model", value=default_model(provider), key=f"model_{provider}")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.1)
    mode = st.radio("Mode", [CHAT_MODE, RAG_MODE])

    if mode == RAG_MODE:
        st.divider()
        st.subheader("Knowledge base")
        try:
            rag = load_rag(provider, model, temperature)
            st.metric("Indexed chunks", rag.chunk_count)

            files = st.file_uploader("Upload documents", type=["pdf", "txt", "md"],
                                     accept_multiple_files=True)
            c1, c2 = st.columns(2)
            if c1.button("Index files", disabled=not files, use_container_width=True):
                with st.spinner("Chunking and embedding..."), tempfile.TemporaryDirectory() as tmp:
                    paths = []
                    for f in files:
                        path = Path(tmp) / Path(f.name).name
                        path.write_bytes(f.getbuffer())
                        paths.append(path)
                    try:
                        n = rag.ingest(paths)
                        st.success(f"Indexed {n} chunks")
                    except Exception as exc:
                        st.error(f"Indexing failed: {exc}")
            if c2.button("Use sample", use_container_width=True, help="Index the bundled AI handbook"):
                with st.spinner("Indexing sample document..."):
                    st.success(f"Indexed {rag.ingest([SAMPLE_DOC])} chunks")
            if st.button("Delete knowledge base", use_container_width=True):
                rag.reset()
                st.rerun()
        except ConfigError as exc:
            st.error(str(exc))

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------------- main
st.title("🤖 AI Chatbot")
st.caption(f"{mode}  •  {provider} / {model}")

if not st.session_state.messages:
    st.info("Say hello! In **Document Q&A** mode, upload a file in the sidebar and ask questions about it.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                st.markdown("\n".join(f"- {s}" for s in msg["sources"]))

if prompt := st.chat_input("Type your message..."):
    history = to_lc_messages(st.session_state.messages[-MAX_HISTORY:])
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reply, sources = None, []
    with st.chat_message("assistant"):
        try:
            if mode == RAG_MODE:
                rag = load_rag(provider, model, temperature)
                if rag.chunk_count == 0:
                    reply = "The knowledge base is empty. Upload documents (or click **Use sample**) in the sidebar first."
                    st.warning(reply)
                else:
                    with st.spinner("Searching your documents..."):
                        docs = rag.retrieve(rag.rewrite_question(prompt, history))
                    reply = st.write_stream(rag.stream_answer(prompt, docs, history))
                    sources = [f"[{i}] {source_label(d)}" for i, d in enumerate(docs, 1)]
                    if sources:
                        with st.expander("📚 Sources"):
                            st.markdown("\n".join(f"- {s}" for s in sources))
            else:
                llm = load_llm(provider, model, temperature)
                reply = st.write_stream(stream_chat(llm, history, prompt))
        except ConfigError as exc:
            st.error(str(exc))
        except Exception as exc:  # network / quota / model-name errors
            st.error(f"Something went wrong: {exc}")

    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
    else:
        st.session_state.messages.pop()  # drop the unanswered user message
