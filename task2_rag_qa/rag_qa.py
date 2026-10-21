"""
CLI for Task 2 - Document Q&A Assistant with RAG & Vector Embeddings.

Run from the repository root:
    python task2_rag_qa/rag_qa.py ingest                 # index the bundled sample data
    python task2_rag_qa/rag_qa.py ingest my_file.pdf     # index your own PDF / TXT / MD files or folders
    python task2_rag_qa/rag_qa.py ask "What is chunk overlap?"
    python task2_rag_qa/rag_qa.py chat                   # interactive, with follow-up questions
    python task2_rag_qa/rag_qa.py reset                  # delete the vector index
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from common.llm_factory import PROVIDERS, ConfigError, default_provider  # noqa: E402
from task2_rag_qa.rag_pipeline import RAGPipeline, source_label  # noqa: E402

DEFAULT_DATA = Path(__file__).resolve().parent / "data"


def print_sources(docs) -> None:
    if docs:
        print("\nSources:")
        for i, doc in enumerate(docs, 1):
            snippet = " ".join(doc.page_content.split())[:110]
            print(f"  [{i}] {source_label(doc)}  -  \"{snippet}...\"")


def ask_once(rag: RAGPipeline, question: str, history: list) -> str:
    standalone = rag.rewrite_question(question, history)
    if standalone != question:
        print(f"(search query: {standalone})")
    docs = rag.retrieve(standalone)
    print("\nAssistant: ", end="", flush=True)
    parts = []
    for token in rag.stream_answer(question, docs, history):
        print(token, end="", flush=True)
        parts.append(token)
    print()
    print_sources(docs)
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG document Q&A")
    parser.add_argument("--provider", choices=PROVIDERS, default=default_provider())
    parser.add_argument("--model")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index documents")
    p_ingest.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_DATA])
    p_ask = sub.add_parser("ask", help="ask a single question")
    p_ask.add_argument("question")
    sub.add_parser("chat", help="interactive Q&A")
    sub.add_parser("reset", help="delete the vector index")
    sub.add_parser("stats", help="show index size")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        rag = RAGPipeline.from_env(args.provider, args.model)

        if args.command == "ingest":
            n = rag.ingest(args.paths)
            print(f"[OK] Indexed {n} chunks. Total in store: {rag.chunk_count}")
        elif args.command == "reset":
            rag.reset()
            print("[OK] Vector index deleted.")
        elif args.command == "stats":
            print(f"Chunks in store: {rag.chunk_count}")
        elif args.command == "ask":
            ask_once(rag, args.question, [])
        elif args.command == "chat":
            if rag.chunk_count == 0:
                print("The index is empty - run the 'ingest' command first.")
                return 1
            print("RAG chat - type 'exit' to quit.")
            history: list = []
            while True:
                try:
                    question = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if question.lower() in {"exit", "quit", "q"}:
                    break
                if not question:
                    continue
                answer = ask_once(rag, question, history)
                history += [HumanMessage(content=question), AIMessage(content=answer)]
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
