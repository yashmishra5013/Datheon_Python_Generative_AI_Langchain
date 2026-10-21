"""
Pydantic schemas that every LLM answer must satisfy.

`extra="forbid"` rejects hallucinated keys, constraints reject impossible
values, and the model validators cross-check the arithmetic of an invoice.
"""

from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MONEY_TOLERANCE = 0.05


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ----------------------------------------------------------------------------
# Invoice
# ----------------------------------------------------------------------------
class Party(StrictModel):
    name: str = Field(min_length=1)
    address: str | None = None
    email: str | None = None
    tax_id: str | None = Field(default=None, description="GSTIN / VAT / EIN if present")

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str | None) -> str | None:
        if v is not None and not EMAIL_RE.match(v):
            raise ValueError(f"'{v}' is not a valid email address")
        return v


class LineItem(StrictModel):
    description: str = Field(min_length=1)
    quantity: float = Field(ge=0)
    unit_price: float = Field(ge=0)
    amount: float = Field(ge=0)

    @model_validator(mode="after")
    def _amount_matches(self) -> "LineItem":
        expected = round(self.quantity * self.unit_price, 2)
        if abs(expected - self.amount) > MONEY_TOLERANCE:
            raise ValueError(
                f"line item '{self.description}': quantity x unit_price = {expected}, "
                f"but amount = {self.amount}"
            )
        return self


class Invoice(StrictModel):
    invoice_number: str = Field(min_length=1)
    invoice_date: date
    due_date: date | None = None
    currency: str = Field(min_length=3, max_length=3, description="ISO 4217 code, e.g. INR, USD")
    vendor: Party
    customer: Party
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    discount: float = Field(default=0, ge=0)
    tax_amount: float = Field(default=0, ge=0)
    total: float = Field(ge=0)

    @field_validator("currency")
    @classmethod
    def _upper_currency(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _totals_are_consistent(self) -> "Invoice":
        items_sum = round(sum(i.amount for i in self.line_items), 2)
        if abs(items_sum - self.subtotal) > MONEY_TOLERANCE:
            raise ValueError(f"sum of line items is {items_sum}, but subtotal is {self.subtotal}")
        expected_total = round(self.subtotal - self.discount + self.tax_amount, 2)
        if abs(expected_total - self.total) > MONEY_TOLERANCE:
            raise ValueError(
                f"subtotal - discount + tax = {expected_total}, but total is {self.total}"
            )
        if self.due_date and self.due_date < self.invoice_date:
            raise ValueError("due_date cannot be earlier than invoice_date")
        return self


# ----------------------------------------------------------------------------
# Resume
# ----------------------------------------------------------------------------
DATE_PATTERN = r"^(\d{4}-\d{2}|\d{4}|Present)$"


class Experience(StrictModel):
    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    start_date: str | None = Field(default=None, pattern=DATE_PATTERN, description="YYYY-MM or YYYY")
    end_date: str | None = Field(default=None, pattern=DATE_PATTERN, description="YYYY-MM, YYYY or 'Present'")
    highlights: list[str] = Field(default_factory=list)


class Education(StrictModel):
    institution: str = Field(min_length=1)
    degree: str = Field(min_length=1)
    field_of_study: str | None = None
    graduation_year: int | None = Field(default=None, ge=1950, le=2100)


class Resume(StrictModel):
    full_name: str = Field(min_length=1)
    email: str
    phone: str | None = None
    location: str | None = None
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    total_experience_years: float | None = Field(default=None, ge=0, le=60)

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        if not EMAIL_RE.match(v):
            raise ValueError(f"'{v}' is not a valid email address")
        return v


SCHEMAS: dict[str, type[BaseModel]] = {"invoice": Invoice, "resume": Resume}
