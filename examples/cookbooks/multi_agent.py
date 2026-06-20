"""
multi_agent.py — Multi-agent routing example.

Shows how to create a parent agent that delegates to specialized subagents.

Usage:
    export AFK_LLM_API_KEY=sk-...
    python examples/multi_agent.py
"""

from afk.quickstart import Agent, Runner, ToolResult, tool


@tool(name="search_docs", description="Search internal documentation.")
def search_docs(query: str) -> ToolResult[str]:
    """Simulate a documentation search."""
    return ToolResult(output=f"Found docs for: {query}")


@tool(name="run_query", description="Run a database query.")
def run_query(sql: str) -> ToolResult[str]:
    """Simulate a database query."""
    return ToolResult(output=f"Query result for: {sql}")
  

async def main() -> None:
    # Specialized subagents
    docs_agent = Agent(
        name="docs_expert",
        model="gpt-4.1-mini",
        instructions="You help users find information in documentation.",
        tools=[search_docs],
    )

    data_agent = Agent(
        name="data_expert",
        model="gpt-4.1-mini",
        instructions="You help users query databases and analyze data.",
        tools=[run_query],
    )

    # Parent agent with subagents
    router_agent = Agent(
        name="router",
        model="gpt-4.1-mini",
        instructions=(
            "You are a helpful assistant that routes questions to specialists. "
            "Use docs_expert for documentation questions and data_expert for "
            "data/database questions."
        ),
        subagents=[docs_agent, data_agent],
    )

    runner = Runner()
    result = await runner.run(
        router_agent,
        user_message="Can you look up the API rate limits in our docs?",
    )
    print(f"Response: {result.final_text}")
    print(f"Subagents used: {[s.subagent_name for s in result.subagent_executions]}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
