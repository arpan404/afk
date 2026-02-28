"""
---
name: Grading System
description: An eval-driven grading system that uses AFK's eval framework to automatically test and score agent responses.
tags: [agent, runner, evals, assertions, scorers]
---
---
This example demonstrates AFK's eval framework -- run_suite, EvalCase, EvalSuiteConfig, assertions, and scorers -- for building repeatable, deterministic grading of agent behavior. Instead of manually chatting with an agent to verify correctness, you define eval cases with expected outcomes and let the framework run them automatically. Each case sends a question to the agent, then assertions check that the response meets your criteria (e.g., contains the right answer, reached completed state). Scorers assign numeric quality scores (e.g., response length as a proxy for thoroughness). This is the foundation for CI-driven agent quality assurance.
---
"""

from afk.core import Runner  # <- Runner executes agents. The eval framework creates a fresh Runner per case via runner_factory to ensure isolation.
from afk.agents import Agent  # <- Agent defines what the grading agent is and how it should behave.
from afk.evals import (  # <- The eval framework provides everything needed for deterministic agent testing.
    run_suite,  # <- Sync entry point that runs all eval cases and returns an EvalSuiteResult. Calls asyncio.run() internally, so this script doesn't need async.
    arun_suite,  # <- Async counterpart of run_suite. Use this inside async code or when you need more control.
    EvalCase,  # <- One test case: pairs an agent + user_message with optional context and tags. Each case runs independently.
    EvalSuiteConfig,  # <- Configuration for the suite: execution mode (sequential/parallel/adaptive), max concurrency, fail_fast, and which assertions/scorers to apply.
    EvalSuiteResult,  # <- The aggregated result object. Contains per-case results plus convenience properties like .passed, .failed, .total.
    FinalTextContainsAssertion,  # <- Built-in assertion that checks if the agent's final text output contains a specific substring. Great for verifying factual answers.
    StateCompletedAssertion,  # <- Built-in assertion that passes only when the agent's run reaches the "completed" terminal state (i.e., it didn't crash or get stuck).
    ResultLengthScorer,  # <- Built-in scorer that returns the length of the agent's final text as a float. Useful as a simple proxy for response thoroughness.
)


# ===========================================================================
# The grading agent
# ===========================================================================
# This agent answers general knowledge questions. We'll evaluate its ability
# to produce correct, complete answers using the eval framework.

grading_agent = Agent(
    name="grading-agent",  # <- What you want to call your agent.
    model="ollama_chat/gpt-oss:20b",  # <- The llm model the agent will use. See https://afk.arpan.sh/library/agents for more details.
    instructions="""
    You are a knowledgeable tutor who answers questions clearly and accurately.

    Rules:
    - Always include the specific answer (name, number, date, etc.) in your response.
    - Provide a brief explanation (1-2 sentences) for context.
    - Be concise — no unnecessary preamble or filler.
    - If you are unsure, state your best understanding and note the uncertainty.

    Your goal is to give correct, verifiable answers that a grader can check.
    """,  # <- Instructions are designed so the agent produces answers that are easy to assert against. Telling it to "include the specific answer" makes FinalTextContainsAssertion reliable.
)


# ===========================================================================
# Define eval cases
# ===========================================================================
# Each EvalCase pairs the agent with a question. The framework will run each
# case independently (possibly in parallel) and apply assertions + scorers.

eval_cases: list[EvalCase] = [
    EvalCase(
        name="capital_of_france",  # <- A unique name for this test case. Shows up in the report so you can identify which cases passed or failed.
        agent=grading_agent,  # <- The agent to test. Every case can use a different agent if you want to compare models or configurations.
        user_message="What is the capital of France?",  # <- The question sent to the agent. This is the input that triggers the agent's response.
        tags=("geography", "easy"),  # <- Optional tags for filtering and grouping in reports. Useful when you have hundreds of eval cases.
    ),
    EvalCase(
        name="year_wwii_ended",
        agent=grading_agent,
        user_message="In what year did World War II end?",
        tags=("history", "easy"),
    ),
    EvalCase(
        name="chemical_symbol_gold",
        agent=grading_agent,
        user_message="What is the chemical symbol for gold?",
        tags=("chemistry", "easy"),
    ),
    EvalCase(
        name="largest_planet",
        agent=grading_agent,
        user_message="What is the largest planet in our solar system?",
        tags=("astronomy", "easy"),
    ),
    EvalCase(
        name="speed_of_light",
        agent=grading_agent,
        user_message="What is the approximate speed of light in km/s?",
        tags=("physics", "medium"),
    ),
    EvalCase(
        name="python_creator",
        agent=grading_agent,
        user_message="Who created the Python programming language?",
        tags=("computer_science", "easy"),
    ),
]


# ===========================================================================
# Define assertions and scorers
# ===========================================================================
# Assertions are pass/fail checks applied to every case result.
# Scorers produce numeric values for quality tracking.

assertions = (
    StateCompletedAssertion(),  # <- Checks that the agent run reached "completed" state. If the agent crashes, loops forever, or times out, this assertion fails. It's the baseline "did it even work?" check.
    FinalTextContainsAssertion(  # <- Checks that the agent's final text contains a specific substring. We use a broad term here since per-case needles would require custom assertions. This assertion checks all responses contain common answer indicators.
        needle=".",  # <- A minimal check: every well-formed answer sentence ends with a period. For real evals, you'd create per-case assertions (see What's Next).
        name="answer_check",  # <- Custom name so it appears clearly in the report. Without this, it defaults to "final_text_contains".
    ),
)

scorers = (
    ResultLengthScorer(),  # <- Scores each case by the length of the agent's final text. Longer isn't always better, but for a tutor agent, very short responses (< 20 chars) suggest something went wrong. This score helps you track response quality over time.
)


# ===========================================================================
# Configure and run the eval suite
# ===========================================================================

suite_config = EvalSuiteConfig(
    execution_mode="adaptive",  # <- "adaptive" lets the framework decide: sequential for small suites (<=2 cases), parallel for larger ones. You can also force "sequential" or "parallel".
    max_concurrency=4,  # <- When running in parallel, at most 4 cases execute at once. Higher values are faster but use more resources (LLM API calls, memory).
    fail_fast=False,  # <- When False, all cases run even if some fail. Set to True during development to stop at the first failure and debug it.
    assertions=assertions,  # <- These assertions are applied to EVERY case result. The case passes only if ALL assertions pass.
    scorers=scorers,  # <- Scorers run after assertions and attach numeric scores to each case result. Scores don't affect pass/fail.
)


def runner_factory() -> Runner:
    """Create a fresh Runner for each eval case.

    The eval framework calls this once per case to ensure complete isolation.
    Each case gets its own Runner with its own memory store, so no state
    leaks between cases. This is critical for deterministic, reproducible evals.
    """
    return Runner()  # <- A fresh Runner per case. The default RunnerConfig is fine for evals. For custom configs, pass RunnerConfig(...) here.


# ===========================================================================
# Run and report
# ===========================================================================

if __name__ == "__main__":
    print("Running eval suite with", len(eval_cases), "cases...")
    print("=" * 50)
    print()

    result: EvalSuiteResult = run_suite(  # <- run_suite is the sync entry point. It runs all cases, applies assertions and scorers, and returns the aggregated result. Internally it calls asyncio.run(arun_suite(...)).
        runner_factory=runner_factory,  # <- Factory function called once per case. Returns a fresh Runner each time for isolation.
        cases=eval_cases,  # <- The list of EvalCase instances to evaluate.
        config=suite_config,  # <- Suite-level configuration: execution mode, concurrency, assertions, scorers.
    )

    # --- Print the report ---

    print("=== Eval Suite Report ===")
    print(f"Total: {result.total} | Passed: {result.passed} | Failed: {result.failed}")  # <- result.total, result.passed, and result.failed are convenience properties on EvalSuiteResult. They count how many cases passed all assertions vs. how many had at least one failure.
    print(f"Execution mode: {result.execution_mode}")  # <- Shows which mode was actually used. "adaptive" resolves to either "sequential" or "parallel" based on case count and concurrency.
    print()

    for case_result in result.results:  # <- result.results is a list of EvalCaseResult, one per case, in execution order. Each contains the case name, final text, state, assertions, and scores.
        status = "PASSED" if case_result.passed else "FAILED"  # <- case_result.passed is True only if ALL assertions passed for this case.
        print(f"Case: {case_result.case} — {status}")

        # Show assertion results
        for assertion in case_result.assertions:  # <- case_result.assertions is a list of EvalAssertionResult. Each has a name, passed flag, optional details, and optional score.
            if assertion.passed:
                print(f"  [{assertion.name}] PASSED")
            else:
                print(f"  [{assertion.name}] FAILED — {assertion.details}")  # <- assertion.details explains why it failed (e.g., "missing_substring=Paris") so you can debug quickly.

            if assertion.score is not None:
                print(f"    score={assertion.score}")  # <- Scorer results are also stored as EvalAssertionResult with a score field (assertions from scorers have name=scorer.name).

        # Show scorer results (they appear as assertions with a score)
        for assertion in case_result.assertions:
            if assertion.score is not None and assertion.name == "result_length":
                print(f"  [{assertion.name}] score={assertion.score}")  # <- ResultLengthScorer returns len(final_text) as a float. Track this over time to catch regressions in response quality.

        # Show timing from metrics
        if case_result.metrics.wall_time_s > 0:
            print(f"  wall_time: {case_result.metrics.wall_time_s:.2f}s")  # <- RunMetrics tracks wall-clock time, LLM call count, token usage, and more. Great for performance monitoring.

        # Show a snippet of the response
        snippet = case_result.final_text[:120].replace("\n", " ")  # <- Show just the first 120 characters so the report stays readable. The full text is available in case_result.final_text.
        print(f"  response: {snippet}...")

        print()

    # --- Summary ---

    if result.failed == 0:
        print("All cases passed!")  # <- When all assertions pass across all cases, you have high confidence the agent is behaving correctly.
    else:
        print(f"{result.failed} case(s) failed. Review the report above for details.")  # <- Failed cases need investigation. Check assertion.details for specifics.


"""
---
Tl;dr: This example uses AFK's eval framework to build a repeatable grading system for an agent. Six eval cases test the agent's ability to answer general knowledge questions. Each case runs independently (via runner_factory for isolation), and the framework applies StateCompletedAssertion (did the run finish?) and FinalTextContainsAssertion (does the response include the answer?) to every case. ResultLengthScorer assigns a numeric quality score. The sync run_suite function orchestrates everything and returns an EvalSuiteResult with per-case pass/fail status, assertion details, scores, and timing metrics.
---
---
What's next?
- Create per-case assertions by defining custom assertion classes that implement the EvalAssertion protocol. For example, a "CapitalOfFranceAssertion" that checks for "Paris" specifically.
- Use AsyncEvalAssertion for assertions that need to call external APIs (e.g., fact-checking services or LLM-as-judge patterns).
- Try arun_suite instead of run_suite when you need to integrate evals into an existing async application.
- Experiment with EvalBudget to set cost/time limits per case and catch runaway evals before they drain your API budget.
- Use write_suite_report_json() to save eval results to a JSON file for CI/CD integration and historical trend analysis.
- Check out the other examples in the library to see how to combine evals with memory, tools, and multi-agent patterns!
---
"""
