"""
---
name: News Digest
description: A news digest agent that streams responses token-by-token using run_stream and AgentStreamEvent.
tags: [agent, runner, streaming, async]
---
---
This example demonstrates how to use the Runner's streaming API to receive an agent's response
incrementally — token by token — instead of waiting for the entire response to complete. This is
essential for building responsive UIs and CLI tools where the user should see output as it is
generated. The agent summarizes news articles provided as tool output, and the summary is streamed
to the console in real-time.
---
"""

import asyncio  # <- We need asyncio because streaming is an async-only API. run_stream returns an async iterator.
from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner is responsible for executing agents. run_stream() is the streaming entry point.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool  # <- @tool decorator to create tools from plain functions.


# ===========================================================================
# Simulated news data
# ===========================================================================

NEWS_ARTICLES: dict[str, dict] = {  # <- A small in-memory news database. In a real application this would be an API call to a news service — but for this example, static data keeps the focus on streaming.
    "tech": {
        "headline": "AI Startup Raises $500M to Build Autonomous Coding Agents",
        "source": "TechCrunch",
        "body": (
            "A San Francisco-based AI startup announced a $500 million Series C round today, "
            "bringing its valuation to $4 billion. The company is building autonomous coding agents "
            "that can write, test, and deploy production software with minimal human oversight. "
            "The round was led by Sequoia Capital, with participation from Andreessen Horowitz. "
            "CEO Jane Park stated that the funds will be used to scale the engineering team and "
            "expand into enterprise markets. Critics have raised concerns about code quality and "
            "security implications of fully autonomous development pipelines."
        ),
    },
    "science": {
        "headline": "James Webb Telescope Discovers New Earth-Like Exoplanet",
        "source": "Nature",
        "body": (
            "NASA's James Webb Space Telescope has identified a rocky exoplanet in the habitable "
            "zone of a nearby red dwarf star, just 40 light-years from Earth. The planet, designated "
            "JWST-2025b, shows spectroscopic signatures consistent with a nitrogen-oxygen atmosphere. "
            "Lead researcher Dr. Maria Chen called the discovery 'the most promising candidate for "
            "extraterrestrial biosignatures we have ever seen.' Follow-up observations are planned "
            "for the next observation cycle to search for water vapor and methane."
        ),
    },
    "business": {
        "headline": "Global Markets Rally as Central Banks Signal Rate Cuts",
        "source": "Financial Times",
        "body": (
            "Stock markets around the world surged today after the Federal Reserve and European "
            "Central Bank both signaled potential interest rate cuts in the coming quarter. The "
            "S&P 500 rose 2.3%, while the Euro Stoxx 50 climbed 1.8%. Bond yields fell sharply, "
            "with the US 10-year dropping to 3.6%. Analysts at Goldman Sachs noted that easing "
            "monetary policy could reignite growth in the housing and technology sectors. However, "
            "some economists warn that premature cuts could reignite inflationary pressures."
        ),
    },
    "sports": {
        "headline": "Underdog Team Wins Championship in Historic Upset",
        "source": "ESPN",
        "body": (
            "In one of the greatest upsets in sports history, the last-seeded Riverside Raptors "
            "defeated the defending champions 4-3 in a thrilling seven-game series. Rookie point "
            "guard Marcus Johnson scored 38 points in the decisive Game 7, including a buzzer-beating "
            "three-pointer. Head coach Lisa Torres dedicated the win to the city of Riverside. "
            "'Nobody believed in us except us,' Torres said during the post-game press conference. "
            "It is the franchise's first championship in its 30-year history."
        ),
    },
}


# ===========================================================================
# Tool definitions
# ===========================================================================

class FetchNewsArgs(BaseModel):  # <- Defines what the LLM must provide to call the fetch_news tool.
    category: str = Field(description="The news category to fetch. One of: tech, science, business, sports")


@tool(args_model=FetchNewsArgs, name="fetch_news", description="Fetch the latest news article for a given category")
def fetch_news(args: FetchNewsArgs) -> str:  # <- This tool returns article text for the agent to summarize. The agent will then generate a streamed summary.
    category = args.category.lower().strip()
    article = NEWS_ARTICLES.get(category)
    if article is None:
        available = ", ".join(NEWS_ARTICLES.keys())
        return f"Unknown category '{args.category}'. Available categories: {available}"
    return (
        f"Headline: {article['headline']}\n"
        f"Source: {article['source']}\n"
        f"Article:\n{article['body']}"
    )


class FetchMultipleNewsArgs(BaseModel):  # <- Allows fetching multiple categories at once for a full digest.
    categories: list[str] = Field(description="List of news categories to fetch. Options: tech, science, business, sports")


@tool(args_model=FetchMultipleNewsArgs, name="fetch_multiple_news", description="Fetch news articles for multiple categories at once to create a digest")
def fetch_multiple_news(args: FetchMultipleNewsArgs) -> str:
    results = []
    for cat in args.categories:
        article = NEWS_ARTICLES.get(cat.lower().strip())
        if article:
            results.append(
                f"[{cat.upper()}]\n"
                f"Headline: {article['headline']}\n"
                f"Source: {article['source']}\n"
                f"Article:\n{article['body']}\n"
            )
        else:
            results.append(f"[{cat.upper()}] — No article found for this category.\n")
    return "\n---\n".join(results)


# ===========================================================================
# Agent setup
# ===========================================================================

news_agent = Agent(
    name="news-digest",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use.
    instructions="""
    You are a concise news anchor. When the user asks for news, use the tools to fetch articles
    and then deliver a clear, well-structured digest.

    For a single category: fetch the article and provide a 2-3 sentence summary.
    For a full digest: fetch all categories and present each with a one-line headline summary,
    then offer to go deeper on any topic.

    Use a professional but approachable tone — like a morning news briefing.

    **NOTE**: Always attribute the source when summarizing.
    """,
    tools=[fetch_news, fetch_multiple_news],
)

runner = Runner()  # <- Create a Runner instance. We'll use its run_stream() method instead of run_sync().


# ===========================================================================
# Streaming entry point
# ===========================================================================

async def main():
    """
    Main async entry point demonstrating the streaming API.

    Instead of runner.run() (which waits for the full response), we use
    runner.run_stream() which returns an AgentStreamHandle. This handle
    is an async iterator that yields AgentStreamEvent objects as the agent
    processes the request.

    Key event types:
    - "text_delta"    : A chunk of generated text (the main content you stream to users)
    - "tool_started"  : A tool call has begun (tool_name tells you which one)
    - "tool_completed": A tool call finished (tool_success, tool_output, or tool_error)
    - "step_started"  : A new reasoning step began
    - "completed"     : The entire run is finished (event.result holds the AgentResult)
    - "error"         : An error occurred during the run
    """
    print("News Digest Agent — Streaming Demo")
    print("=" * 40)
    print("Ask for news by category (tech, science, business, sports) or request a full digest.\n")

    while True:
        user_input = input("[] > ")  # <- Take user input from the console.

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # -----------------------------------------------------------------
        # run_stream() returns an AgentStreamHandle — an async iterator
        # that yields AgentStreamEvent objects in real time.
        # -----------------------------------------------------------------
        handle = await runner.run_stream(  # <- run_stream() is the streaming counterpart to run(). It returns immediately with a handle you iterate over asynchronously.
            news_agent, user_message=user_input
        )

        print("[news-digest] > ", end="", flush=True)  # <- Print the agent name prefix, then stream text right after it.

        async for event in handle:  # <- Each iteration yields an AgentStreamEvent. The loop runs until the agent finishes.

            if event.type == "text_delta":
                # ---------------------------------------------------------
                # "text_delta" events carry incremental text chunks. Print
                # each chunk immediately without a newline to simulate
                # real-time typing. flush=True ensures it appears instantly.
                # ---------------------------------------------------------
                print(event.text_delta, end="", flush=True)  # <- This is the core of streaming: each text_delta is a small piece of the response. Printing them as they arrive gives the user instant feedback.

            elif event.type == "tool_started":
                # ---------------------------------------------------------
                # "tool_started" fires when the agent begins calling a tool.
                # Useful for showing a loading indicator or status message.
                # ---------------------------------------------------------
                print(f"\n  [fetching: {event.tool_name}...]", flush=True)  # <- Show the user which tool is being called. In a UI, you might show a spinner here.

            elif event.type == "tool_completed":
                # ---------------------------------------------------------
                # "tool_completed" fires when a tool call finishes. You can
                # check event.tool_success and event.tool_error for status.
                # ---------------------------------------------------------
                status = "done" if event.tool_success else f"failed: {event.tool_error}"
                print(f"  [{event.tool_name}: {status}]", flush=True)  # <- Confirm the tool finished. In a UI, you might dismiss the spinner here.

            elif event.type == "completed":
                # ---------------------------------------------------------
                # "completed" fires once when the entire run is done. The
                # event.result field holds the full AgentResult (same object
                # you'd get from runner.run()). You can inspect usage stats,
                # tool call records, and the final text here.
                # ---------------------------------------------------------
                print()  # <- Newline after the streamed text.
                if event.result:
                    usage = event.result.usage
                    print(
                        f"  [tokens: {usage.input_tokens} in / {usage.output_tokens} out]"
                    )  # <- Show token usage after the response. The same UsageAggregate is available on the result.

            elif event.type == "error":
                # ---------------------------------------------------------
                # "error" fires if something went wrong during execution.
                # ---------------------------------------------------------
                print(f"\n  [error: {event.error}]", flush=True)

        print()  # <- Blank line between turns for readability.


if __name__ == "__main__":
    asyncio.run(main())  # <- asyncio.run() is required because run_stream() is an async API. This is the standard Python entry point for async programs.



"""
---
Tl;dr: This example creates a news digest agent that uses the Runner's streaming API (run_stream) to deliver
responses token-by-token instead of waiting for the full completion. The async iterator yields AgentStreamEvent
objects with types like "text_delta" (incremental text), "tool_started"/"tool_completed" (tool lifecycle),
"completed" (final result with usage stats), and "error". This pattern is essential for building responsive
CLI tools and UIs where users should see output as it's generated.
---
---
What's next?
- Try adding more event types to the handler (e.g. "step_started" to track reasoning steps).
- Experiment with streaming in a web application by sending text_delta events over WebSockets or Server-Sent Events.
- Compare the user experience of run_stream() vs run() — streaming feels dramatically more responsive for long outputs.
- Explore the event.result field in the "completed" event to access tool_calls, usage, and other execution metadata.
- Check out the AgentStreamEvent dataclass to see all available fields for each event type.
---
"""
