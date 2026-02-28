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
---
"""

import asyncio  # <- Async required for parallel subagent execution.
from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner executes agents and manages subagent parallelism.
from afk.agents import Agent  # <- Agent defines each evaluator and the coordinator.
from afk.tools import tool  # <- @tool decorator for creating tools.


# ===========================================================================
# Simulated candidate data
# ===========================================================================

CANDIDATES: dict[str, dict] = {  # <- Simulated candidate database.
    "alice": {
        "name": "Alice Chen",
        "resume": {
            "education": "M.S. Computer Science, Stanford",
            "experience_years": 6,
            "previous_roles": ["Senior Engineer at Google", "Staff Engineer at Stripe"],
            "skills": ["Python", "Go", "Distributed Systems", "Kubernetes", "AWS"],
            "certifications": ["AWS Solutions Architect", "CKA Kubernetes"],
        },
        "skills_assessment": {
            "coding_score": 92,
            "system_design_score": 88,
            "algorithms_score": 85,
            "communication_score": 90,
        },
        "culture_fit": {
            "collaboration_style": "highly collaborative, enjoys pair programming",
            "values_alignment": "strong focus on code quality and mentorship",
            "growth_mindset": "regularly contributes to open source, gives tech talks",
            "references": ["Excellent team player", "Natural leader", "Great communicator"],
        },
    },
    "bob": {
        "name": "Bob Martinez",
        "resume": {
            "education": "B.S. Computer Science, State University",
            "experience_years": 2,
            "previous_roles": ["Junior Developer at Startup XYZ"],
            "skills": ["JavaScript", "React", "Node.js", "MongoDB"],
            "certifications": [],
        },
        "skills_assessment": {
            "coding_score": 65,
            "system_design_score": 40,
            "algorithms_score": 55,
            "communication_score": 72,
        },
        "culture_fit": {
            "collaboration_style": "prefers working independently",
            "values_alignment": "interested in career growth and learning",
            "growth_mindset": "taking online courses, building side projects",
            "references": ["Hard worker", "Eager to learn"],
        },
    },
}


# ===========================================================================
# Resume screener tools
# ===========================================================================

class CandidateArgs(BaseModel):
    candidate_name: str = Field(description="Name of the candidate to evaluate (e.g., 'alice' or 'bob')")


@tool(args_model=CandidateArgs, name="screen_resume", description="Screen a candidate's resume for qualifications, experience, and education")
def screen_resume(args: CandidateArgs) -> str:
    candidate = CANDIDATES.get(args.candidate_name.lower())
    if candidate is None:
        return f"Candidate '{args.candidate_name}' not found. Available: {', '.join(CANDIDATES.keys())}"
    resume = candidate["resume"]
    score = min(100, resume["experience_years"] * 10 + len(resume["skills"]) * 5 + len(resume["certifications"]) * 10)
    return (
        f"Resume Screening: {candidate['name']}\n"
        f"  Education: {resume['education']}\n"
        f"  Experience: {resume['experience_years']} years\n"
        f"  Roles: {', '.join(resume['previous_roles'])}\n"
        f"  Skills: {', '.join(resume['skills'])}\n"
        f"  Certifications: {', '.join(resume['certifications']) or 'none'}\n"
        f"  Resume Score: {score}/100"
    )


resume_screener = Agent(
    name="resume-screener",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a resume screening specialist. Use screen_resume to evaluate the
    candidate's qualifications, experience, and education. Provide a clear
    assessment of whether they meet the minimum requirements for a senior
    engineering position (4+ years experience, relevant technical skills).
    """,
    tools=[screen_resume],
)


# ===========================================================================
# Skills assessor tools
# ===========================================================================

@tool(args_model=CandidateArgs, name="assess_skills", description="Evaluate a candidate's technical skills scores across coding, design, algorithms, and communication")
def assess_skills(args: CandidateArgs) -> str:
    candidate = CANDIDATES.get(args.candidate_name.lower())
    if candidate is None:
        return f"Candidate '{args.candidate_name}' not found."
    scores = candidate["skills_assessment"]
    avg = sum(scores.values()) / len(scores)
    return (
        f"Skills Assessment: {candidate['name']}\n"
        f"  Coding: {scores['coding_score']}/100\n"
        f"  System Design: {scores['system_design_score']}/100\n"
        f"  Algorithms: {scores['algorithms_score']}/100\n"
        f"  Communication: {scores['communication_score']}/100\n"
        f"  Average: {avg:.0f}/100"
    )


skills_assessor = Agent(
    name="skills-assessor",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a technical skills assessment specialist. Use assess_skills to evaluate
    the candidate's technical abilities. For a senior role, we need:
    - Coding: 70+ (strong pass), 50-70 (conditional), <50 (fail)
    - System Design: 60+ required for senior role
    - Algorithms: 50+ required
    - Communication: 60+ required
    Give a clear pass/fail recommendation with reasoning.
    """,
    tools=[assess_skills],
)


# ===========================================================================
# Culture fit evaluator tools
# ===========================================================================

@tool(args_model=CandidateArgs, name="evaluate_culture_fit", description="Evaluate a candidate's culture fit including collaboration style, values, and growth mindset")
def evaluate_culture_fit(args: CandidateArgs) -> str:
    candidate = CANDIDATES.get(args.candidate_name.lower())
    if candidate is None:
        return f"Candidate '{args.candidate_name}' not found."
    culture = candidate["culture_fit"]
    return (
        f"Culture Fit Evaluation: {candidate['name']}\n"
        f"  Collaboration: {culture['collaboration_style']}\n"
        f"  Values: {culture['values_alignment']}\n"
        f"  Growth: {culture['growth_mindset']}\n"
        f"  References: {'; '.join(culture['references'])}"
    )


culture_evaluator = Agent(
    name="culture-evaluator",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a culture fit evaluation specialist. Use evaluate_culture_fit to assess
    the candidate's alignment with company culture. We value:
    - Strong collaboration (team-first approach)
    - Code quality and mentorship
    - Continuous learning and growth mindset
    Provide a clear recommendation on culture fit.
    """,
    tools=[evaluate_culture_fit],
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
recommendation. Parallel mode is ideal for independent evaluations where each subagent doesn't depend
on the others' results.
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
