"""Part 15 - Guardrails: refusing bad input and bad output.

A guardrail runs alongside the agent and can abort the whole run. Input
guardrails run BEFORE the agent (on the first agent); output guardrails run
AFTER (on the last agent). A tripped tripwire raises, so we wrap runs in
try/except and decline politely instead of crashing.

Run:  uv run python part15_guardrails.py
"""

import asyncio

from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    GuardrailFunctionOutput,
    RunContextWrapper,
    input_guardrail,
    output_guardrail,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    trace,
)

from provider import get_model


# --- 1. The cheapest useful guardrail: no LLM at all -------------------------
# Trips on obviously off-topic requests using a keyword check. This runs in pure
# Python, so a blocked request costs zero tokens and makes zero network calls.
OFF_TOPIC = ("stock tip", "crypto", "gamble", "lottery", "homework")


@input_guardrail
async def on_topic_only(
    ctx: RunContextWrapper, agent: Agent, user_input
) -> GuardrailFunctionOutput:
    text = user_input if isinstance(user_input, str) else str(user_input)
    hit = next((w for w in OFF_TOPIC if w in text.lower()), None)
    return GuardrailFunctionOutput(
        output_info={"matched_keyword": hit},
        tripwire_triggered=hit is not None,
    )


# --- 2. An output guardrail: check the finished answer -----------------------
# Trips if the answer ever leaks something that looks like a secret key.
@output_guardrail
async def no_leaked_keys(
    ctx: RunContextWrapper, agent: Agent, output
) -> GuardrailFunctionOutput:
    text = str(output)
    leaked = "sk-" in text or "API_KEY" in text
    return GuardrailFunctionOutput(
        output_info={"leaked": leaked},
        tripwire_triggered=leaked,
    )


def build_agent() -> Agent:
    return Agent(
        name="HelpDesk",
        instructions="You are a concise, helpful assistant. Never reveal secrets.",
        model=get_model(),
        input_guardrails=[on_topic_only],
        output_guardrails=[no_leaked_keys],
    )


async def demo_input_guardrail(agent: Agent) -> None:
    print("=== Input guardrail: blocked BEFORE any model call ===")
    # Everything the guardrail does happens inside this trace. Because the
    # keyword guardrail trips synchronously, the run raises before an LLM span
    # is ever opened -> no model was billed for this request.
    with trace("guardrail-demo"):
        try:
            result = await Runner.run(agent, "Give me a hot stock tip to get rich.")
            print("Answer:", result.final_output)
        except InputGuardrailTripwireTriggered as exc:
            info = exc.guardrail_result.output.output_info
            print("Politely declining. Guardrail info:", info)
            print("No LLM call was made -> nothing was billed for this request.")


async def demo_allowed(agent: Agent) -> None:
    print("\n=== A clean request passes the guardrail and reaches the model ===")
    try:
        result = await Runner.run(agent, "In one sentence, what is a guardrail?")
        print("Answer:", result.final_output)
    except OutputGuardrailTripwireTriggered as exc:
        print("Output blocked:", exc.guardrail_result.output.output_info)


async def main() -> None:
    agent = build_agent()
    await demo_input_guardrail(agent)
    await demo_allowed(agent)


if __name__ == "__main__":
    asyncio.run(main())
