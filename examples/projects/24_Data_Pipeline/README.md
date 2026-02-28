
# Data Pipeline

A data pipeline orchestrator that uses DelegationPlan for DAG-based multi-agent execution with parallel stages, dependencies, and retry policies.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/24_Data_Pipeline

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/24_Data_Pipeline

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/24_Data_Pipeline

Expected interaction
User: run
Agent: Starting pipeline...
  [extract] Extracted 8 records
  [validate] All records valid
  [transform] Department aggregates computed
  [report] Executive summary generated

The pipeline runs extract and validate in parallel, then transform, then report — defined as a DAG with DelegationPlan.
