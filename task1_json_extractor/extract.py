"""
CLI for Task 1 - Structured LLM Prompt Pipeline & JSON Extractor.

Examples (run from the repository root):
    python task1_json_extractor/extract.py --demo
    python task1_json_extractor/extract.py --type invoice --file my_invoice.pdf
    python task1_json_extractor/extract.py --type resume --file cv.txt --provider google
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.llm_factory import PROVIDERS, ConfigError, default_provider, get_chat_model  # noqa: E402
from task1_json_extractor.extractor import ExtractionError, StructuredExtractor  # noqa: E402
from task1_json_extractor.schemas import SCHEMAS  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"


def read_document(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def run_one(extractor: StructuredExtractor, doc_type: str, file: Path, out: Path | None) -> None:
    print(f"\n=== {doc_type.upper()} : {file.name} ===")
    result = extractor.extract(read_document(file), doc_type)
    payload = json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False)
    print(payload)
    print(f"[OK] Output validated against the {type(result).__name__} schema")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"[saved] {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract validated JSON from unstructured documents")
    parser.add_argument("--type", choices=sorted(SCHEMAS), help="document type")
    parser.add_argument("--file", type=Path, help="input .txt/.md/.pdf (defaults to the bundled sample)")
    parser.add_argument("--out", type=Path, help="where to save the JSON")
    parser.add_argument("--demo", action="store_true", help="run both bundled samples")
    parser.add_argument("--provider", choices=PROVIDERS, default=default_provider())
    parser.add_argument("--model")
    parser.add_argument("--retries", type=int, default=2, help="self-repair attempts after a failed validation")
    args = parser.parse_args()

    if not args.demo and not args.type:
        parser.error("choose --type invoice|resume or use --demo")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        extractor = StructuredExtractor(get_chat_model(args.provider, args.model), args.retries)
        if args.demo:
            for doc_type in SCHEMAS:
                sample = HERE / "data" / f"sample_{doc_type}.txt"
                run_one(extractor, doc_type, sample, OUTPUT_DIR / f"{doc_type}.json")
        else:
            file = args.file or HERE / "data" / f"sample_{args.type}.txt"
            run_one(extractor, args.type, file, args.out)
    except (ConfigError, ExtractionError, FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
