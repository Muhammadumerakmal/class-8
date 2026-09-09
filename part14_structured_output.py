"""Part 14 - Structured Output: a validated Pydantic object instead of prose.

`output_type=<model>` makes the agent hand back a parsed, validated object. That
is what lets one agent's answer drive Python (arithmetic, branching, storage)
instead of another prompt.

Run:  uv run python part14_structured_output.py
"""

import asyncio

from pydantic import BaseModel, field_validator
from agents import Agent, Runner
from agents.agent_output import AgentOutputSchema

from provider import get_model


# --- The typed shape we demand back ------------------------------------------
class CityFact(BaseModel):
    """Three typed fields. `population` and `founded_year` are ints, not strings."""

    city: str
    population: int
    founded_year: int


# --- A shape no real answer can satisfy --------------------------------------
class ImpossibleCityFact(BaseModel):
    """Same fields, but with a validator no real city can pass.

    Used only for the failure demo. A strict provider (OpenAI) will happily
    return a schema-valid object for almost any prompt, so a merely off-topic
    request won't raise. What DOES raise is data that violates validation: here
    we demand a founding year in the future, which the model can't truthfully
    produce. Pydantic rejects it after parsing and the SDK surfaces the error.
    """

    city: str
    population: int
    founded_year: int

    @field_validator("founded_year")
    @classmethod
    def must_be_in_the_future(cls, v: int) -> int:
        if v <= 2026:
            raise ValueError("founded_year must be in the future (impossible)")
        return v


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


async def unanswerable() -> None:
    print("\n=== Run 2: a shape the answer can't validate into ===")
    # A dedicated agent that demands an impossible field (future founding year).
    # strict_json_schema=False turns OFF the provider's forced structured output,
    # so the model's JSON is validated by Pydantic afterwards - and can fail two
    # ways: the model returns the wrong shape (parse error), or a real value
    # violates the validator (validation error). Either raises ModelBehaviorError.
    # This is the honest way to see the exception: a strict provider like OpenAI
    # coerces almost any prompt into a schema-valid object, so merely going
    # off-topic won't fail.
    strict_agent = Agent(
        name="ImpossibleCityAgent",
        instructions="Answer with facts about the requested city.",
        model=get_model(),
        output_type=AgentOutputSchema(ImpossibleCityFact, strict_json_schema=False),
    )
    try:
        result = await Runner.run(strict_agent, "Tell me about Lahore.")
        print("Unexpectedly parsed:", result.final_output)
    except Exception as exc:  # noqa: BLE001 - we want to SHOW whatever it raises
        print(f"Raised {type(exc).__name__}: {exc}")


async def main() -> None:
    agent = build_agent()
    await happy_path(agent)
    await unanswerable()


if __name__ == "__main__":
    asyncio.run(main())
