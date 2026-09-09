"""Shared model wiring for Parts 14-16 - Gemini OR OpenAI, switchable.

The whole point of the OpenAI Agents SDK's provider layer is that swapping the
model backend never touches your Agent/Runner code. We build the model here once
and hand it to every agent via `model=`.

Pick the provider with the AGENT_PROVIDER env var in .env:
    AGENT_PROVIDER=gemini   (default)  -> uses GEMINI_API_KEY
    AGENT_PROVIDER=openai              -> uses OPENAI_API_KEY
"""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, set_tracing_disabled

# Gemini exposes an OpenAI-compatible surface at this base URL.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.5-flash"

# A small, cheap OpenAI model - the right default for guardrail-style checks.
OPENAI_MODEL = "gpt-4o-mini"


def get_model():
    """Return a model object usable as an Agent's `model=`.

    Reads AGENT_PROVIDER (+ the matching API key) from .env. Raises a clear
    error if the key is missing so the demos fail with a message instead of a
    cryptic 401.
    """
    load_dotenv()
    provider = os.getenv("AGENT_PROVIDER", "gemini").strip().lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "AGENT_PROVIDER=openai but OPENAI_API_KEY is empty. Add it to .env."
            )
        # With OpenAI you can pass the model name as a plain string - the SDK's
        # default client already uses OPENAI_API_KEY. Tracing works natively.
        return OPENAI_MODEL

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "AGENT_PROVIDER=gemini but GEMINI_API_KEY is empty. Add it to .env."
            )
        client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
        # Gemini's key isn't an OpenAI key, so disable the SDK's default tracing
        # exporter (it would try to POST traces to OpenAI and warn). Part 15
        # re-enables an in-process trace() block to prove no model was billed.
        set_tracing_disabled(True)
        return OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=client)

    raise RuntimeError(
        f"Unknown AGENT_PROVIDER={provider!r}. Use 'gemini' or 'openai'."
    )
