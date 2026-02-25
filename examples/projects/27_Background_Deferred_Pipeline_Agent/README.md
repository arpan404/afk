# Background Deferred Pipeline Agent

Progressive AFK example **27** focused on **ToolDeferredHandle + background tool polling + stream events**.

## Complexity Profile
- Tier level: 6
- Stage-chain depth: 28
- Architecture: multi-module with scenario generation, quality gates, staged analytics, metrics, and report composition.

## What This Example Includes
- Feature workflow for: ToolDeferredHandle + background tool polling + stream events
- Dynamic scenario synthesis (app/scenario.py)
- Structured validation (app/quality.py)
- Progressive stage transforms (app/stages/stage_*.py)
- Derived analytics (app/metrics.py)
- Consolidated report output (app/report_builder.py)

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/27_Background_Deferred_Pipeline_Agent
- Then execute:
  cd examples/projects/27_Background_Deferred_Pipeline_Agent && python3 main.py
