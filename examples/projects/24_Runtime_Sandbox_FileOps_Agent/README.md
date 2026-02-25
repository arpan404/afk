# Runtime Sandbox FileOps Agent

Progressive AFK example **24** focused on **build_runtime_tools + sandbox policy + output limiting**.

## Complexity Profile
- Tier level: 3
- Stage-chain depth: 25
- Architecture: multi-module with scenario generation, quality gates, staged analytics, metrics, and report composition.

## What This Example Includes
- Feature workflow for: build_runtime_tools + sandbox policy + output limiting
- Dynamic scenario synthesis (app/scenario.py)
- Structured validation (app/quality.py)
- Progressive stage transforms (app/stages/stage_*.py)
- Derived analytics (app/metrics.py)
- Consolidated report output (app/report_builder.py)

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/24_Runtime_Sandbox_FileOps_Agent
- Then execute:
  cd examples/projects/24_Runtime_Sandbox_FileOps_Agent && python3 main.py
