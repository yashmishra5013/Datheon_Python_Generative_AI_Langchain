"""
Structured LLM extraction pipeline.

    document --> parameterised prompt (+ few-shot examples + JSON Schema)
             --> LLM --> JSON parsing --> Pydantic validation
                                 ^                 |
                                 +-- self-repair <-+   (validation errors are fed
                                                        back to the model, up to N times)
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm_factory import message_text  # noqa: E402
from task1_json_extractor.examples import FEW_SHOT  # noqa: E402
from task1_json_extractor.schemas import SCHEMAS  # noqa: E402

logger = logging.getLogger("extractor")

SYSTEM_TEMPLATE = """You are a precise information-extraction engine.
Task: extract the data of a {document_type} from the document provided by the user.

Rules:
1. Respond with ONE valid JSON object and nothing else - no markdown fences, no commentary.
2. The JSON must conform exactly to this JSON Schema (no extra keys):
{schema}
3. Use only information that is present in the document. Never invent values; use null for missing optional fields.
4. Dates must be ISO-8601 (YYYY-MM-DD). Numbers must be plain numbers: no currency symbols, no thousands separators.
5. {extra_rules}"""

EXTRA_RULES = {
    "invoice": (
        "Convert currency symbols to ISO codes (Rs/INR -> INR, $ -> USD). "
        "Each line item amount = quantity x unit_price, and subtotal - discount + tax_amount must equal total."
    ),
    "resume": (
        "skills is a flat list of short skill names. List experience from most recent to oldest. "
        "Use 'Present' for current jobs. Set total_experience_years only if stated or clearly computable, else null."
    ),
}

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder("examples"),
        ("human", "Document:\n{document}"),
    ]
)


class ExtractionError(RuntimeError):
    """The model could not produce a schema-valid JSON object."""


def strip_json(text: str) -> str:
    """Remove markdown fences / surrounding prose and return the JSON object text."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return text


def _few_shot_messages(doc_type: str) -> list:
    messages = []
    for document, expected_json in FEW_SHOT[doc_type]:
        messages.append(HumanMessage(content=f"Document:\n{document}"))
        messages.append(AIMessage(content=expected_json))
    return messages


def _error_summary(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        lines = []
        for err in exc.errors()[:8]:
            loc = ".".join(str(p) for p in err["loc"]) or "(root)"
            lines.append(f"- {loc}: {err['msg']}")
        return "\n".join(lines)
    return f"- {exc}"


class StructuredExtractor:
    """Turns unstructured text into a validated Pydantic object."""

    def __init__(self, llm, max_retries: int = 2) -> None:
        self.llm = llm
        self.max_retries = max_retries

    def build_messages(self, document: str, doc_type: str) -> list:
        schema_cls = SCHEMAS[doc_type]
        return PROMPT.format_messages(
            document_type=doc_type,
            schema=json.dumps(schema_cls.model_json_schema(), indent=2),
            extra_rules=EXTRA_RULES[doc_type],
            examples=_few_shot_messages(doc_type),
            document=document,
        )

    def extract(self, document: str, doc_type: str) -> BaseModel:
        if doc_type not in SCHEMAS:
            raise ValueError(f"Unsupported document type '{doc_type}'. Use: {', '.join(SCHEMAS)}")
        if not document.strip():
            raise ValueError("The document is empty.")

        schema_cls = SCHEMAS[doc_type]
        messages = self.build_messages(document, doc_type)
        last_error = "unknown error"

        for attempt in range(1, self.max_retries + 2):
            response = self.llm.invoke(messages)
            raw = message_text(response.content)
            try:
                data = json.loads(strip_json(raw))
                result = schema_cls.model_validate(data)
                logger.info("Valid %s extracted on attempt %d", doc_type, attempt)
                return result
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = _error_summary(exc)
                logger.warning("Attempt %d failed validation:\n%s", attempt, last_error)
                messages = messages + [
                    AIMessage(content=raw),
                    HumanMessage(
                        content=(
                            "Your JSON was rejected for these reasons:\n"
                            f"{last_error}\n"
                            "Return the corrected JSON object only."
                        )
                    ),
                ]

        raise ExtractionError(
            f"Could not obtain valid {doc_type} JSON after {self.max_retries + 1} attempts.\n{last_error}"
        )
