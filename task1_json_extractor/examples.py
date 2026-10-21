"""Few-shot examples (document -> expected JSON). Each one is validated in the tests."""

from __future__ import annotations

import json

INVOICE_EXAMPLE_TEXT = """\
INVOICE #INV-2024-001
Date: 2024-01-15    Due: 2024-02-14
From: BrightTech Solutions, 12 MG Road, Pune 411001 | billing@brighttech.example | GSTIN 27ABCDE1234F1Z5
Bill To: Nova Retail Pvt Ltd, 88 Park Street, Kolkata 700016

1. Website maintenance (3 months)   qty 3   @ 5,000.00   = 15,000.00
2. Domain renewal                   qty 1   @ 1,200.00   =  1,200.00

Subtotal: 16,200.00
GST (18%): 2,916.00
Total: INR 19,116.00
"""

INVOICE_EXAMPLE_JSON = {
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "due_date": "2024-02-14",
    "currency": "INR",
    "vendor": {
        "name": "BrightTech Solutions",
        "address": "12 MG Road, Pune 411001",
        "email": "billing@brighttech.example",
        "tax_id": "27ABCDE1234F1Z5",
    },
    "customer": {
        "name": "Nova Retail Pvt Ltd",
        "address": "88 Park Street, Kolkata 700016",
        "email": None,
        "tax_id": None,
    },
    "line_items": [
        {"description": "Website maintenance (3 months)", "quantity": 3, "unit_price": 5000.0, "amount": 15000.0},
        {"description": "Domain renewal", "quantity": 1, "unit_price": 1200.0, "amount": 1200.0},
    ],
    "subtotal": 16200.0,
    "discount": 0,
    "tax_amount": 2916.0,
    "total": 19116.0,
}

RESUME_EXAMPLE_TEXT = """\
ANITA ROY
anita.roy@example.com | +91 98765 00000 | Pune, India

Summary: Backend developer who enjoys building reliable APIs.

Experience
Software Engineer, CloudNine Labs (Jun 2021 - Present)
- Built REST APIs serving 2M requests/day
- Cut deployment time by 40% with CI/CD

Education
B.Tech in Computer Science, COEP Pune, 2021

Skills: Python, FastAPI, PostgreSQL, Docker
Certifications: AWS Cloud Practitioner
"""

RESUME_EXAMPLE_JSON = {
    "full_name": "Anita Roy",
    "email": "anita.roy@example.com",
    "phone": "+91 98765 00000",
    "location": "Pune, India",
    "summary": "Backend developer who enjoys building reliable APIs.",
    "skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
    "experience": [
        {
            "company": "CloudNine Labs",
            "title": "Software Engineer",
            "start_date": "2021-06",
            "end_date": "Present",
            "highlights": ["Built REST APIs serving 2M requests/day", "Cut deployment time by 40% with CI/CD"],
        }
    ],
    "education": [
        {"institution": "COEP Pune", "degree": "B.Tech", "field_of_study": "Computer Science", "graduation_year": 2021}
    ],
    "certifications": ["AWS Cloud Practitioner"],
    "total_experience_years": None,
}

FEW_SHOT: dict[str, list[tuple[str, str]]] = {
    "invoice": [(INVOICE_EXAMPLE_TEXT, json.dumps(INVOICE_EXAMPLE_JSON, indent=2))],
    "resume": [(RESUME_EXAMPLE_TEXT, json.dumps(RESUME_EXAMPLE_JSON, indent=2))],
}
