# Fallback Reasoning Control Agent

Progressive AFK example **28** focused on **fallback_model_chain + reasoning controls override**.

## Complexity Profile
- Tier level: 7
- Stage-chain depth: 29
- Architecture: multi-module with scenario generation, quality gates, staged analytics, metrics, and report composition.

## What This Example Includes
- Feature workflow for: fallback_model_chain + reasoning controls override
- Dynamic scenario synthesis (app/scenario.py)
- Structured validation (app/quality.py)
- Progressive stage transforms (app/stages/stage_*.py)
- Derived analytics (app/metrics.py)
- Consolidated report output (app/report_builder.py)

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/28_Fallback_Reasoning_Control_Agent
- Then execute:
  cd examples/projects/28_Fallback_Reasoning_Control_Agent && python3 main.py
