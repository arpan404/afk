# LLM Builder Strategy Agent

Progressive AFK example **22** focused on **LLMBuilder + MiddlewareStack + structured output**.

## Complexity Profile
- Tier level: 1
- Stage-chain depth: 23
- Architecture: multi-module with scenario generation, quality gates, staged analytics, metrics, and report composition.

## What This Example Includes
- Feature workflow for: LLMBuilder + MiddlewareStack + structured output
- Dynamic scenario synthesis (app/scenario.py)
- Structured validation (app/quality.py)
- Progressive stage transforms (app/stages/stage_*.py)
- Derived analytics (app/metrics.py)
- Consolidated report output (app/report_builder.py)

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/22_LLM_Builder_Strategy_Agent
- Then execute:
  cd examples/projects/22_LLM_Builder_Strategy_Agent && python3 main.py
