"""Part 16 - Lifecycle Hooks: watching the agent loop from inside.

Hooks give you a callback at every stage of the run - log, count, time, or
update a UI. Here we print four of them and watch the two-pass loop happen:
the model is called, a tool runs, then the model is called again with the
tool's result.

Run:  uv run python part16_lifecycle_hooks.py
"""

import asyncio
import random

from agents import Agent, Runner, AgentHooks, function_tool

from provider import get_model


class LoudHooks(AgentHooks):
    """Prints each stage as it fires. AgentHooks belongs to ONE agent."""

    def __init__(self, label: str):
        self.label = label

    async def on_start(self, context, agent):
        print(f"[{self.label}] on_start        -> {agent.name} owns the answer now")

    async def on_tool_start(self, context, agent, tool):
        print(f"[{self.label}] on_tool_start   -> calling {tool.name}")

    async def on_tool_end(self, context, agent, tool, result):
        print(f"[{self.label}] on_tool_end     -> {tool.name} returned {result!r}")

    async def on_end(self, context, agent, output):
        print(f"[{self.label}] on_end          -> finished")


@function_tool
def roll_dice(sides: int) -> int:
    """Roll a single die with the given number of sides and return the result."""
    return random.randint(1, sides)


def build_agent() -> Agent:
    return Agent(
        name="DiceAgent",
        instructions="You are a helpful assistant. Use tools when they fit.",
        model=get_model(),
        tools=[roll_dice],
        hooks=LoudHooks("dice"),
    )


async def main() -> None:
    agent = build_agent()
    print("=== One question that needs a tool ===")
    result = await Runner.run(agent, "Roll a 20-sided die for me.")
    print("\nFinal answer:", result.final_output)

    # What the order tells you about the loop:
    #   on_start        model is handed the question
    #   on_tool_start   model decided it needs roll_dice and asked for it
    #   on_tool_end     our Python ran and returned a value
    #   on_end          model was called a SECOND time with that value and
    #                   produced the final natural-language answer
    # That on_tool_start -> on_tool_end -> on_end sequence IS the two-pass loop.
    #
    # Gotcha: AgentHooks uses on_start/on_end. RunHooks (passed to
    # Runner.run(..., hooks=...)) uses on_agent_start/on_agent_end and covers
    # every agent in the run. Override the wrong pair and nothing prints.


if __name__ == "__main__":
    asyncio.run(main())
