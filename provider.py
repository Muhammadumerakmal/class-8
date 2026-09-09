"""Shared Gemini provider wiring for Parts 14-16.

The whole point of the OpenAI Agents SDK's provider layer is that swapping the
model backend never touches your Agent/Runner code. We build a Gemini model here
once and hand it to every agent via `model=`.
"""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel, set_tracing_disabled

# Gemini exposes an OpenAI-compatible surface at this base URL.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.5-flash"


def get_model() -> OpenAIChatCompletionsModel:
    """Return a Gemini model object usable as an Agent's `model=`.

    Reads GEMINI_API_KEY from .env. Raises a clear error if it's missing so the
    demos fail with a message instead of a cryptic 401.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is empty. Put a real key in .env before running the demos."
        )

    client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    # We use Gemini's key, not OpenAI's, so disable the SDK's default tracing
    # exporter (which would try to POST traces to OpenAI and warn about the key).
    # Part 15 re-enables an in-process trace() block to prove no model was billed.
    set_tracing_disabled(True)

    return OpenAIChatCompletionsModel(model=GEMINI_MODEL, openai_client=client)
