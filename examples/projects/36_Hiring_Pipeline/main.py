"""
---
name: Hiring Pipeline
description: A hiring pipeline agent that uses subagent_parallelism_mode for concurrent candidate evaluation by specialist subagents.
tags: [agent, runner, subagents, parallelism, async]
---
---
This example demonstrates how to use subagent_parallelism_mode to control how an agent's
subagents execute. When set to "parallel", all delegated subagents run concurrently, which
is ideal for independent evaluations like screening a candidate across multiple dimensions
simultaneously. The hiring pipeline has three specialist evaluators (resume, skills, culture)
that all assess the same candidate in parallel, and the coordinator combines their results
into a final hiring recommendation.

The project is split into two files:
- evaluators.py: Three specialist evaluator agents with their tools and candidate data
- main.py: Coordinator agent with parallel mode and interactive entry point
---
"""

import asyncio  # <- Async required for parallel subagent execution.
from afk.core import Runner  # <- Runner executes agents and manages subagent parallelism.
from afk.agents import Agent  # <- Agent defines the coordinator.

from evaluators import (  # <- Import specialist evaluator agents from evaluators.py.
    resume_screener,
    skills_assessor,
    culture_evaluator,
    CANDIDATES,  # <- Import candidate list for display.
)


# ===========================================================================
# Coordinator agent with parallel subagent mode
# ===========================================================================

hiring_coordinator = Agent(
    name="hiring-coordinator",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are the hiring coordinator for a senior engineering position. You manage
    three evaluation specialists:
    1. Resume Screener: checks qualifications and experience
    2. Skills Assessor: evaluates technical abilities
    3. Culture Evaluator: assesses culture fit

    When asked to evaluate a candidate, delegate to ALL three specialists.
    Then combine their assessments into a final hiring recommendation:
    - HIRE: all three evaluations are positive
    - CONDITIONAL: two positive, one needs improvement
    - PASS: two or more evaluations are negative

    Present a structured final report with each specialist's assessment and your
    overall recommendation.
    """,
    subagents=[resume_screener, skills_assessor, culture_evaluator],
    subagent_parallelism_mode="parallel",  # <- "parallel" means all subagents run concurrently when delegated. This is much faster than "single" (sequential) for independent evaluations. Options: "single", "parallel", "configurable".
    max_steps=30,  # <- Allow enough steps for the coordinator to delegate to all 3 subagents and synthesize results.
)

runner = Runner()


async def main():
    print("Hiring Pipeline Agent")
    print("=" * 40)
    print("Evaluate candidates across three dimensions in parallel.\n")
    print(f"Available candidates: {', '.join(CANDIDATES.keys())}")
    print("Try: 'Evaluate alice for the senior engineering position'\n")

    while True:
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = await runner.run(  # <- Async run because parallel subagents need async execution.
            hiring_coordinator,
            user_message=user_input,
        )

        print(f"[hiring-coordinator] > {response.final_text}")
        print(f"\n  Subagent calls: {len(response.subagent_calls)}")
        for sub in response.subagent_calls:
            status = "passed" if sub.success else "failed"
            print(f"    - {sub.target_agent}: {status}")
        print()


if __name__ == "__main__":
    asyncio.run(main())



"""
---
Tl;dr: This example creates a hiring pipeline with a coordinator agent and three specialist subagents
(resume screener, skills assessor, culture evaluator) running in parallel via
subagent_parallelism_mode="parallel". All three evaluators assess the same candidate concurrently,
and the coordinator synthesizes their independent assessments into a final hire/conditional/pass
recommendation. The project is split into evaluators.py (agents + tools + data) and main.py
(coordinator + entry point).
---
---
What's next?
- Try switching subagent_parallelism_mode to "single" and compare the execution time.
- Add more candidates to the database and batch-evaluate them.
- Implement a scoring rubric that weights each evaluation dimension differently.
- Combine with DelegationPlan for more complex pipeline logic (e.g., only run culture fit if skills pass).
- Add a "schedule_interview" tool that only activates for candidates who pass initial screening.
- Check out the Data Pipeline example for DAG-based delegation with dependencies!
---
"""
