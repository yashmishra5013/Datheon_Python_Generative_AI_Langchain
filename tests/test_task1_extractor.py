"""Offline tests: no API key or network needed (the LLM is faked)."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from task1_json_extractor.examples import INVOICE_EXAMPLE_JSON, RESUME_EXAMPLE_JSON
from task1_json_extractor.extractor import ExtractionError, StructuredExtractor, strip_json
from task1_json_extractor.schemas import Invoice, Resume


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.replies.pop(0))


GOOD_INVOICE = json.dumps(INVOICE_EXAMPLE_JSON)


def test_few_shot_examples_are_valid():
    Invoice.model_validate(INVOICE_EXAMPLE_JSON)
    Resume.model_validate(RESUME_EXAMPLE_JSON)


def test_strip_json_handles_fences_and_prose():
    assert strip_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_json('Sure! Here it is: {"a": 1} Hope this helps') == '{"a": 1}'


def test_invoice_total_mismatch_is_rejected():
    bad = {**INVOICE_EXAMPLE_JSON, "total": 1.0}
    with pytest.raises(ValidationError):
        Invoice.model_validate(bad)


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        Resume.model_validate({**RESUME_EXAMPLE_JSON, "favourite_colour": "blue"})


def test_extract_valid_json_first_try():
    llm = FakeLLM([f"```json\n{GOOD_INVOICE}\n```"])
    result = StructuredExtractor(llm).extract("some invoice text", "invoice")
    assert result.invoice_number == "INV-2024-001"
    assert len(llm.calls) == 1


def test_self_repair_after_invalid_output():
    bad = json.dumps({**INVOICE_EXAMPLE_JSON, "total": 5.0})
    llm = FakeLLM([bad, GOOD_INVOICE])
    result = StructuredExtractor(llm, max_retries=2).extract("text", "invoice")
    assert result.total == 19116.0
    assert len(llm.calls) == 2
    assert "rejected" in llm.calls[1][-1].content  # validation error was fed back


def test_gives_up_after_max_retries():
    llm = FakeLLM(["not json"] * 3)
    with pytest.raises(ExtractionError):
        StructuredExtractor(llm, max_retries=2).extract("text", "invoice")


def test_unsupported_type_and_empty_document():
    extractor = StructuredExtractor(FakeLLM([]))
    with pytest.raises(ValueError):
        extractor.extract("text", "passport")
    with pytest.raises(ValueError):
        extractor.extract("   ", "invoice")
