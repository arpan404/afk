
# Hiring Pipeline

A hiring pipeline with specialist subagents (resume, skills, culture) running in parallel via `subagent_parallelism_mode` for concurrent candidate evaluation.

## Project Structure

```
36_Hiring_Pipeline/
  main.py         # Entry point — coordinator agent with parallel subagent mode
  evaluators.py   # Specialist evaluator agents: resume screener, skills assessor, culture evaluator
```

## Key Concepts

- **subagent_parallelism_mode="parallel"**: All delegated subagents run concurrently instead of sequentially
- **Independent evaluators**: Each evaluator assesses a different dimension (resume, skills, culture) independently
- **Coordinator synthesis**: The coordinator combines parallel results into a final HIRE/CONDITIONAL/PASS recommendation

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/36_Hiring_Pipeline

Expected interaction
User: Evaluate alice for the senior engineering position
Agent: Delegating to all three evaluators in parallel...
  Resume Screener: Strong qualifications, 6 years experience — PASS
  Skills Assessor: All scores above threshold — PASS
  Culture Evaluator: Excellent collaboration and growth — PASS
  Final Recommendation: HIRE
