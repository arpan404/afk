"""
---
name: Hiring Pipeline — Evaluators
description: Specialist evaluator subagents and their tools for resume screening, skills assessment, and culture fit evaluation.
tags: [agent, tools, subagents, evaluation]
---
---
Each evaluator is a standalone agent with a focused tool. They are designed to run in parallel
via subagent_parallelism_mode="parallel" on the coordinator (see main.py). The candidate
database is kept here so evaluators are self-contained. In a real system, each evaluator would
connect to an ATS, assessment platform, or HRIS.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.agents import Agent  # <- Agent defines each evaluator.
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
# Shared argument schema
# ===========================================================================

class CandidateArgs(BaseModel):
    candidate_name: str = Field(description="Name of the candidate to evaluate (e.g., 'alice' or 'bob')")


# ===========================================================================
# Evaluator 1: Resume screener
# ===========================================================================

@tool(args_model=CandidateArgs, name="screen_resume", description="Screen a candidate's resume for qualifications, experience, and education")
def screen_resume(args: CandidateArgs) -> str:  # <- Resume screening tool: computes a score based on experience, skills, and certifications.
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
# Evaluator 2: Skills assessor
# ===========================================================================

@tool(args_model=CandidateArgs, name="assess_skills", description="Evaluate a candidate's technical skills scores across coding, design, algorithms, and communication")
def assess_skills(args: CandidateArgs) -> str:  # <- Skills assessment tool: retrieves and averages technical scores.
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
# Evaluator 3: Culture fit evaluator
# ===========================================================================

@tool(args_model=CandidateArgs, name="evaluate_culture_fit", description="Evaluate a candidate's culture fit including collaboration style, values, and growth mindset")
def evaluate_culture_fit(args: CandidateArgs) -> str:  # <- Culture fit tool: returns qualitative assessment data.
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
