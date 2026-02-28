
# Hiring Pipeline

A hiring pipeline with specialist subagents (resume, skills, culture) running in parallel via subagent_parallelism_mode for concurrent candidate evaluation.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/36_Hiring_Pipeline

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/36_Hiring_Pipeline

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/36_Hiring_Pipeline

Expected interaction
User: Evaluate alice for the senior engineering position
Agent: Delegating to all three evaluators in parallel...
  Resume Screener: Strong qualifications, 6 years experience — PASS
  Skills Assessor: All scores above threshold — PASS
  Culture Evaluator: Excellent collaboration and growth — PASS
  Final Recommendation: HIRE

All three evaluators run concurrently via subagent_parallelism_mode="parallel".
