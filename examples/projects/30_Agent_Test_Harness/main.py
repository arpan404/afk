"""
---
name: Agent Test Harness
description: An eval harness that uses EvalCase, assertions, and run_suite to systematically test agent behavior across multiple test cases.
tags: [agent, runner, evals, eval-case, assertions, testing, eval-suite]
---
---
This example demonstrates AFK's eval system for systematic agent testing. Instead of manually
sending messages and eyeballing responses, you define EvalCase objects (each with an agent, a
user_message, and tags), pair them with assertions (StateCompletedAssertion checks the agent
reached "completed" state; FinalTextContainsAssertion checks the response contains expected
text), and run them all through run_suite. The suite executes each case against a fresh Runner,
applies assertions, scores results, and returns a structured EvalSuiteResult with pass/fail
counts and per-case details. This is a non-interactive script -- it runs evals and prints
results, making it ideal for CI/CD pipelines, regression testing, and agent quality assurance.
---
"""

import asyncio  # <- Async required for arun_suite.
from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner executes agents. Each eval case gets a fresh Runner instance via runner_factory.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool  # <- The @tool decorator for creating agent-callable tools.
from afk.evals import (  # <- The eval system: suite runner, case/config models, and built-in assertions.
    run_suite,  # <- Synchronous suite runner: executes all eval cases and returns EvalSuiteResult. Use arun_suite for async.
    arun_suite,  # <- Async suite runner: same as run_suite but returns an awaitable. Use in async contexts.
    EvalCase,  # <- One test case: pairs an agent + user_message + optional tags. The suite runs the agent with this message and checks assertions against the result.
    EvalSuiteConfig,  # <- Suite-level configuration: execution_mode, max_concurrency, fail_fast, assertions, and scorers.
    FinalTextContainsAssertion,  # <- Built-in assertion: passes only when the agent's final_text contains a specific substring. Useful for checking that the agent mentions expected keywords.
    StateCompletedAssertion,  # <- Built-in assertion: passes only when the agent reaches the "completed" terminal state. Catches crashes, timeouts, and error states.
    ResultLengthScorer,  # <- Built-in scorer: returns the length of final_text as a float score. Useful for tracking response verbosity across test cases.
)


# ===========================================================================
# Step 1: Create an agent with tools to test
# ===========================================================================
# We'll create a simple Q&A agent with a couple of tools. The eval suite
# will test this agent with different inputs and check that it responds
# correctly.

class LookupArgs(BaseModel):  # <- Schema for the knowledge lookup tool.
    topic: str = Field(description="The topic to look up information about")


class CalculateArgs(BaseModel):  # <- Schema for the calculator tool.
    expression: str = Field(description="A simple math expression to evaluate, e.g., '2 + 3'")


@tool(  # <- Knowledge base tool. Returns facts about predefined topics. The eval suite will test that the agent uses this tool and includes the correct facts in its response.
    args_model=LookupArgs,
    name="lookup_knowledge",
    description="Look up factual information about a topic. Use this for geography, science, history, and general knowledge questions.",
)
def lookup_knowledge(args: LookupArgs) -> str:
    knowledge_base: dict[str, str] = {  # <- Simulated knowledge base. In a real app, this could be a database, search engine, or RAG pipeline.
        "france": "France is a country in Western Europe. Its capital is Paris. Population: ~67 million. Known for the Eiffel Tower, cuisine, and wine.",
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991. Known for readability and versatility.",
        "mars": "Mars is the fourth planet from the Sun. It's called the Red Planet due to iron oxide on its surface. It has two moons: Phobos and Deimos.",
        "photosynthesis": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen. It occurs in chloroplasts.",
    }
    topic_lower = args.topic.lower()
    for key, value in knowledge_base.items():
        if key in topic_lower:
            return value
    return f"No information found about '{args.topic}'. Available topics: france, python, mars, photosynthesis."


@tool(  # <- Calculator tool. Evaluates simple math expressions safely.
    args_model=CalculateArgs,
    name="calculate",
    description="Evaluate a simple math expression and return the result. Supports +, -, *, / operators.",
)
def calculate(args: CalculateArgs) -> str:
    try:
        # --- Safe eval for simple math only ---
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in args.expression):  # <- Only allow numeric characters and basic operators. This prevents code injection.
            return f"Error: expression contains invalid characters. Use numbers and +, -, *, / only."
        result = eval(args.expression)  # <- Safe because we validated the input above.
        return f"Result: {args.expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{args.expression}': {e}"


# --- The agent under test ---
qa_agent = Agent(
    name="qa-agent",  # <- The agent we'll test with the eval suite.
    model="ollama_chat/gpt-oss:20b",  # <- The LLM model the agent will use.
    instructions="""
    You are a helpful Q&A assistant with access to a knowledge base and a calculator.

    Rules:
    - For factual questions, use the lookup_knowledge tool to find accurate information.
    - For math questions, use the calculate tool to compute answers.
    - Always include the key facts from tool results in your response.
    - Be concise but thorough.
    - If you don't know something, say so clearly.
    """,  # <- Instructions guide the agent to use tools and include facts in responses.
    tools=[lookup_knowledge, calculate],  # <- Two tools: knowledge lookup and calculator.
)


# ===========================================================================
# Step 2: Define eval cases
# ===========================================================================
# Each EvalCase pairs the agent with a specific user_message and optional tags.
# Tags are useful for filtering and grouping results (e.g., run only "geography"
# tests, or only "math" tests).

eval_cases: list[EvalCase] = [
    EvalCase(  # <- Test case 1: Geography question. We expect the agent to look up France and mention "Paris" in its response.
        name="capital-france",
        agent=qa_agent,
        user_message="What is the capital of France?",
        tags=("geography", "factual"),  # <- Tags for filtering. You could run only "geography" cases in a targeted test.
    ),
    EvalCase(  # <- Test case 2: Math question. We expect the agent to use the calculator and include "47" in its response.
        name="calculator-add",
        agent=qa_agent,
        user_message="What is 23 + 24?",
        tags=("math", "calculator"),
    ),
    EvalCase(  # <- Test case 3: A trick question -- asking about a topic not in the knowledge base. We check that the agent completes successfully even when the lookup returns no results.
        name="unknown-topic",
        agent=qa_agent,
        user_message="Tell me about quantum computing.",
        tags=("factual", "edge-case"),
    ),
    EvalCase(  # <- Test case 4: Simple greeting. Tests that the agent handles non-tool conversations gracefully.
        name="greeting-test",
        agent=qa_agent,
        user_message="Hello! How are you today?",
        tags=("conversational",),
    ),
]


# ===========================================================================
# Step 3: Configure assertions and the eval suite
# ===========================================================================
# Assertions are applied to EVERY case in the suite. StateCompletedAssertion
# checks that the agent reached a terminal "completed" state (didn't crash or
# timeout). FinalTextContainsAssertion checks for specific substrings -- but
# note that suite-level assertions apply to ALL cases, so we use general ones
# here. For per-case assertions, you'd build custom assertion classes.

suite_config = EvalSuiteConfig(  # <- Suite-level configuration controls how cases are executed and evaluated.
    execution_mode="sequential",  # <- Run cases one at a time. Options: "sequential" (deterministic ordering), "parallel" (concurrent with max_concurrency), "adaptive" (auto-selects based on case count).
    max_concurrency=2,  # <- Max concurrent cases when running in parallel mode. Ignored in sequential mode.
    fail_fast=False,  # <- When True, stop the suite on the first failure. When False (default), run all cases regardless of failures. False is better for comprehensive test reports.
    assertions=(
        StateCompletedAssertion(),  # <- Passes only when the agent's terminal state is "completed". This catches crashes, timeouts, and error exits. Applied to every case.
    ),
    scorers=(
        ResultLengthScorer(),  # <- Scores each case by the length of final_text. Not a pass/fail check -- just a numeric score for tracking response verbosity. Useful for regression monitoring.
    ),
)


# ===========================================================================
# Step 4: Run the eval suite and print results
# ===========================================================================

async def main():
    print("Agent Test Harness")
    print("=" * 60)
    print(f"Running eval suite with {len(eval_cases)} test cases...\n")

    # --- Run the suite ---
    result = await arun_suite(  # <- arun_suite executes all eval cases. It takes a runner_factory (callable that returns a fresh Runner for each case), a list of cases, and optional config.
        runner_factory=lambda: Runner(),  # <- Each eval case gets a fresh Runner instance. This ensures test isolation -- one case's state doesn't leak into another.
        cases=eval_cases,
        config=suite_config,
    )

    # --- Print per-case results ---
    for case_result in result.results:  # <- result.results is a list of EvalCaseResult, one per case. Each has the case name, final_text, state, assertions, and scores.
        # Count assertions
        total_assertions = len(case_result.assertions)
        passed_assertions = sum(1 for a in case_result.assertions if a.passed)

        # Determine overall case status
        status = "PASS" if case_result.passed else "FAIL"
        status_marker = "+" if case_result.passed else "-"

        print(f"  [{status}] {case_result.case}")
        print(f"         State: {case_result.state}")
        print(f"         Assertions: {passed_assertions}/{total_assertions} passed")

        # --- Print individual assertion results ---
        for assertion in case_result.assertions:
            a_status = "ok" if assertion.passed else "FAIL"
            details = f" ({assertion.details})" if assertion.details else ""
            print(f"           [{a_status}] {assertion.name}{details}")

        # --- Print response preview ---
        preview = case_result.final_text[:100].replace("\n", " ")  # <- Show first 100 chars of the response for quick inspection.
        if len(case_result.final_text) > 100:
            preview += "..."
        print(f"         Response: {preview}")
        print()

    # --- Print suite summary ---
    print("-" * 60)
    print(f"  Suite Results: {result.passed}/{result.total} passed, {result.failed}/{result.total} failed")
    print(f"  Execution mode: {result.execution_mode}")
    print("-" * 60)

    # --- Print scoring summary ---
    print(f"\n  Scoring Summary (ResultLengthScorer):")
    for case_result in result.results:
        score_entries = [a for a in case_result.assertions if a.score is not None]
        if score_entries:
            for s in score_entries:
                print(f"    {case_result.case}: score={s.score:.0f} (response length in chars)")
        else:
            # ResultLengthScorer output may appear in a different field -- show final_text length
            print(f"    {case_result.case}: response_length={len(case_result.final_text)} chars")

    print(f"\nDone. Eval suite completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())  # <- asyncio.run() starts the event loop. This is a non-interactive script that runs evals and exits.



"""
---
Tl;dr: This example creates an eval test harness for systematic agent testing. It defines a Q&A
agent with two tools (lookup_knowledge for facts, calculate for math), then creates four EvalCase
objects -- each pairing the agent with a different user_message and tags. The cases are bundled
into an EvalSuiteConfig with StateCompletedAssertion (checks agent reached "completed" state),
and ResultLengthScorer (measures response verbosity). The suite is run with arun_suite, which
takes a runner_factory (for test isolation), cases, and config. Each case gets a fresh Runner
instance, runs the agent, applies assertions, and collects results. The script prints per-case
pass/fail status, assertion details, response previews, and a suite-level summary. This pattern
is ideal for CI/CD pipelines, regression testing, and agent quality assurance.
---
---
What's next?
- Add FinalTextContainsAssertion(needle="Paris") to suite_config.assertions to check that the France case mentions "Paris" -- but note that suite-level assertions apply to ALL cases.
- Build a custom per-case assertion class that checks different expectations for each case based on tags.
- Try execution_mode="parallel" with max_concurrency=4 to run cases concurrently and see the speed improvement.
- Set fail_fast=True to stop the suite immediately on the first failure -- useful for fast feedback in CI.
- Use load_eval_cases_json() to load eval cases from a JSON file instead of defining them in code.
- Add budget constraints with EvalBudget to limit token usage or execution time per case.
- Check out the System Monitor example to combine evals with telemetry for performance measurement!
---
"""
