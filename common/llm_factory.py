"""
Shared model factory used by every task.

* get_chat_model()  -> LangChain chat model (Groq / Google Gemini / OpenAI)
* get_embeddings()  -> embeddings (local FastEmbed by default, or Google / OpenAI)

Configuration comes from the .env file in the repository root.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "google": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}
KEY_VARS = {
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
}
PROVIDERS = tuple(DEFAULT_MODELS)


class ConfigError(RuntimeError):
    """Raised for missing / invalid configuration (shown nicely to the user)."""


def default_provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").strip().lower()


def default_model(provider: str) -> str:
    # LLM_MODEL only applies to the provider selected in .env
    if provider == default_provider() and os.getenv("LLM_MODEL"):
        return os.environ["LLM_MODEL"]
    return DEFAULT_MODELS[provider]


def _require_key(env_var: str) -> None:
    if not os.getenv(env_var):
        raise ConfigError(
            f"{env_var} is not set. Copy .env.example to .env and add your API key."
        )


def get_chat_model(provider: str | None = None, model: str | None = None,
                   temperature: float = 0.0):
    provider = (provider or default_provider()).lower()
    if provider not in DEFAULT_MODELS:
        raise ConfigError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    _require_key(KEY_VARS[provider])
    model = model or default_model(provider)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=temperature)


def embeddings_provider() -> str:
    return os.getenv("EMBEDDINGS_PROVIDER", "local").strip().lower()


def get_embeddings(provider: str | None = None):
    provider = (provider or embeddings_provider()).lower()
    if provider == "local":
        from langchain_community.embeddings import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    if provider == "google":
        _require_key("GOOGLE_API_KEY")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if provider == "openai":
        _require_key("OPENAI_API_KEY")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    raise ConfigError(f"Unknown EMBEDDINGS_PROVIDER '{provider}'. Use local | google | openai")


def message_text(content) -> str:
    """Normalise LangChain message content (str or list of parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)
