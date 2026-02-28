
# Travel Planner

A travel planner that uses DelegationPlan with different JoinPolicy options to orchestrate three specialist subagents (flights, hotels, activities) in parallel. Demonstrates allow_optional_failures and quorum join policies for resilient multi-agent execution.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/31_Travel_Planner

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/31_Travel_Planner

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/31_Travel_Planner

Expected interaction
User: Plan a trip to Tokyo for 5 days
Agent: (delegates to flights_agent, hotels_agent, and activities_agent in parallel)
Agent: Here's your Tokyo travel plan with flights, hotels, and activities...

The orchestrator combines results from whichever agents succeed, thanks to the allow_optional_failures join policy.

