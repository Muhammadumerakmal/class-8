# Parts 14–16 — OpenAI Agents SDK (Gemini-backed)

Runnable demos for three parts of the *OpenAI Agents SDK Fundamentals* guide:

| Part | File | What it demonstrates |
|------|------|----------------------|
| 14   | `part14_structured_output.py` | `output_type` returns a validated Pydantic object, not prose |
| 15   | `part15_guardrails.py`        | Input/output guardrails that abort a run |
| 16   | `part16_lifecycle_hooks.py`   | `AgentHooks` callbacks firing through the agent loop |

All three share `provider.py`, which returns the model object each agent uses via
`model=`. Swapping the provider never touches the `Agent`/`Runner` code.

## Setup

```bash
uv sync
```

Then choose a provider and fill the matching key in `.env`:

```
# Gemini (default) — key from https://aistudio.google.com/apikey
AGENT_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key

# …or OpenAI — key from https://platform.openai.com/api-keys
AGENT_PROVIDER=openai
OPENAI_API_KEY=your-openai-key
```

With `openai`, `provider.py` just returns the model name (`gpt-4o-mini`) and the
SDK's default client uses `OPENAI_API_KEY` — no custom wiring needed. With
`gemini` it wires an `AsyncOpenAI` client at Gemini's OpenAI-compatible base URL
into an `OpenAIChatCompletionsModel` (`gemini-2.5-flash`).

## Run

```bash
uv run python part14_structured_output.py
uv run python part15_guardrails.py
uv run python part16_lifecycle_hooks.py
```

## What each prints

**Part 14** — Run 1 prints `type(result.final_output)` as a `CityFact` class and
does integer arithmetic on `founded_year`/`population` to prove the fields aren't
strings. Run 2 feeds a prompt that can't be validated into the model and prints
the exception it raises.

**Part 15** — An off-topic request ("hot stock tip") trips a pure-Python input
guardrail inside a `trace(...)` block and is declined politely — no model call,
nothing billed. A clean request passes through to the model, checked on the way
out by an output guardrail that trips on leaked keys.

**Part 16** — A "roll a die" question fires the hooks in order:
`on_start → on_tool_start → on_tool_end → on_end`. That ordering is the two-pass
loop: model called, tool run, model called again with the result.

> Note: `AgentHooks` uses `on_start`/`on_end`; `RunHooks` uses
> `on_agent_start`/`on_agent_end`. Override the wrong pair and your callback
> silently never runs.
