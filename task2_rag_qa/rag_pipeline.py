"""
Retrieval-Augmented Generation (RAG) pipeline for question answering on custom documents.

    load (PDF / TXT / MD) -> recursive chunking -> embeddings -> ChromaDB (persistent)
    question -> [optional follow-up rewrite] -> top-k similarity search -> grounded answer + sources
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_chroma import Chroma  # noqa: E402
from langchain_community.document_loaders import PyPDFLoader, TextLoader  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.messages import BaseMessage  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from common.llm_factory import (  # noqa: E402
    embeddings_provider,
    get_chat_model,
    get_embeddings,
    message_text,
)

logger = logging.getLogger("rag")

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
NO_CONTEXT_MESSAGE = (
    "I could not find relevant information in the indexed documents. "
    "Please ingest documents first or try rephrasing your question."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful assistant that answers questions about the user's documents.\n"
            "Use ONLY the context below. If the answer is not in the context, say that you could "
            "not find it in the documents - never guess.\n"
            "Cite the context blocks you used as [1], [2], ... Keep the answer clear and concise.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation so far and a follow-up question, rewrite the follow-up as a "
            "standalone question that can be understood without the conversation. "
            "Return ONLY the rewritten question.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


@dataclass
class RAGAnswer:
    answer: str
    sources: list[str] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)


def source_label(doc: Document) -> str:
    name = Path(str(doc.metadata.get("source", "unknown"))).name
    page = doc.metadata.get("page")
    return f"{name} (page {int(page) + 1})" if page is not None else name


def format_context(docs: Sequence[Document]) -> str:
    return "\n\n".join(f"[{i}] ({source_label(d)})\n{d.page_content}" for i, d in enumerate(docs, 1))


class RAGPipeline:
    def __init__(
        self,
        llm,
        embeddings,
        persist_dir: str | Path,
        collection: str = "documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        top_k: int = 4,
    ) -> None:
        self.llm = llm
        self.embeddings = embeddings
        self.persist_dir = Path(persist_dir)
        self.collection = collection
        self.top_k = top_k
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self._parser = StrOutputParser()
        self._store = self._open_store()

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_env(cls, provider: str | None = None, model: str | None = None,
                 temperature: float = 0.0, persist_subdir: str = "task2", **kwargs) -> "RAGPipeline":
        base = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db"))
        return cls(
            llm=get_chat_model(provider, model, temperature),
            embeddings=get_embeddings(),
            persist_dir=base / persist_subdir,
            collection=f"docs_{embeddings_provider()}",  # separate index per embedding model
            **kwargs,
        )

    def _open_store(self) -> Chroma:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=self.collection,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

    # -------------------------------------------------------------- ingestion
    @staticmethod
    def load_documents(path: str | Path) -> list[Document]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if path.is_dir():
            docs: list[Document] = []
            for file in sorted(path.rglob("*")):
                if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES:
                    docs.extend(RAGPipeline.load_documents(file))
            return docs

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return PyPDFLoader(str(path)).load()
        if suffix in {".txt", ".md"}:
            return TextLoader(str(path), encoding="utf-8").load()
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}")

    def ingest(self, paths: Sequence[str | Path]) -> int:
        """Load, chunk, embed and store documents. Returns the number of chunks indexed."""
        documents: list[Document] = []
        for p in paths:
            documents.extend(self.load_documents(p))
        documents = [d for d in documents if d.page_content.strip()]
        if not documents:
            raise ValueError("No readable text found (scanned PDFs need OCR first).")

        unique: dict[str, Document] = {}
        for chunk in self.splitter.split_documents(documents):
            key = f"{chunk.metadata.get('source')}|{chunk.metadata.get('page')}|{chunk.page_content}"
            unique[hashlib.sha1(key.encode("utf-8")).hexdigest()] = chunk  # idempotent re-ingestion

        self._store.add_documents(list(unique.values()), ids=list(unique.keys()))
        logger.info("Indexed %d chunks from %d document pages", len(unique), len(documents))
        return len(unique)

    @property
    def chunk_count(self) -> int:
        return self._store._collection.count()  # noqa: SLF001 (no public count API)

    def reset(self) -> None:
        self._store.delete_collection()
        self._store = self._open_store()

    # -------------------------------------------------------------- retrieval
    def retrieve(self, question: str) -> list[Document]:
        if self.chunk_count == 0:
            return []
        return self._store.similarity_search(question, k=self.top_k)

    def rewrite_question(self, question: str, history: Sequence[BaseMessage] | None) -> str:
        """Turn a follow-up ('and its price?') into a standalone search query."""
        if not history:
            return question
        try:
            chain = REWRITE_PROMPT | self.llm | self._parser
            rewritten = chain.invoke({"history": list(history)[-6:], "question": question}).strip()
            return rewritten or question
        except Exception as exc:  # never fail the whole request because of the rewrite step
            logger.warning("Question rewrite failed (%s); using original question", exc)
            return question

    # ------------------------------------------------------------- generation
    def stream_answer(self, question: str, docs: Sequence[Document],
                      history: Sequence[BaseMessage] | None = None) -> Iterator[str]:
        if not docs:
            yield NO_CONTEXT_MESSAGE
            return
        chain = ANSWER_PROMPT | self.llm
        payload = {
            "context": format_context(docs),
            "history": list(history or [])[-6:],
            "question": question,
        }
        for chunk in chain.stream(payload):
            text = message_text(chunk.content)
            if text:
                yield text

    def answer(self, question: str, history: Sequence[BaseMessage] | None = None) -> RAGAnswer:
        standalone = self.rewrite_question(question, history)
        docs = self.retrieve(standalone)
        text = "".join(self.stream_answer(question, docs, history))
        return RAGAnswer(
            answer=text,
            sources=[f"[{i}] {source_label(d)}" for i, d in enumerate(docs, 1)],
            documents=docs,
        )
