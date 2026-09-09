"""Part 14 - Structured Output: a validated Pydantic object instead of prose.

`output_type=<model>` makes the agent hand back a parsed, validated object. That
is what lets one agent's answer drive Python (arithmetic, branching, storage)
instead of another prompt.

Run:  uv run python part14_structured_output.py
"""

import asyncio

from pydantic import BaseModel
from agents import Agent, Runner
from agents.agent_output import AgentOutputSchema

from provider import get_model


# --- The typed shape we demand back ------------------------------------------
class CityFact(BaseModel):
    """Three typed fields. `population` and `founded_year` are ints, not strings."""

    city: str
    population: int
    founded_year: int


def build_agent() -> Agent:
    return Agent(
        name="CityFactAgent",
        instructions=(
            "Answer with facts about the requested city. "
            "Give your best estimate for population and founding year."
        ),
        model=get_model(),
        # Some providers reject the SDK's strict JSON schema. Gemini usually
        # accepts CityFact directly, but if you ever see a schema-rejection
        # error, swap the next line for:
        #     output_type=AgentOutputSchema(CityFact, strict_json_schema=False),
        output_type=CityFact,
    )


async def happy_path(agent: Agent) -> None:
    print("=== Run 1: a prompt that fits CityFact ===")
    result = await Runner.run(agent, "Tell me about Karachi.")
    fact = result.final_output

    # Proof #1: it's a real CityFact instance, not a string of JSON.
    print("type(result.final_output):", type(fact))
    print("parsed object:", fact)

    # Proof #2: a numeric field used in arithmetic. If founded_year were a
    # string this line would raise TypeError instead of printing a number.
    age = 2026 - fact.founded_year
    print(f"{fact.city} is about {age} years old (2026 - {fact.founded_year}).")
    print(f"population + 1 = {fact.population + 1}  <- int math, not string concat")


async def unanswerable(agent: Agent) -> None:
    print("\n=== Run 2: a prompt that can't be forced into CityFact ===")
    # Asking for a poem cannot be validated into city/population/founded_year.
    # The SDK raises while trying to parse/validate the model's output.
    try:
        result = await Runner.run(
            agent, "Ignore the format and just write me a haiku about the ocean."
        )
        print("Unexpectedly parsed:", result.final_output)
    except Exception as exc:  # noqa: BLE001 - we want to SHOW whatever it raises
        print(f"Raised {type(exc).__name__}: {exc}")


async def main() -> None:
    agent = build_agent()
    await happy_path(agent)
    await unanswerable(agent)


if __name__ == "__main__":
    asyncio.run(main())
