# Eval Suite Quality Gates Agent

Progressive AFK example **29** focused on **eval suite + assertions + dataset + JSON report**.

## Complexity Profile
- Tier level: 8
- Stage-chain depth: 30
- Architecture: multi-module with scenario generation, quality gates, staged analytics, metrics, and report composition.

## What This Example Includes
- Feature workflow for: eval suite + assertions + dataset + JSON report
- Dynamic scenario synthesis (app/scenario.py)
- Structured validation (app/quality.py)
- Progressive stage transforms (app/stages/stage_*.py)
- Derived analytics (app/metrics.py)
- Consolidated report output (app/report_builder.py)

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/29_Eval_Suite_Quality_Gates_Agent
- Then execute:
  cd examples/projects/29_Eval_Suite_Quality_Gates_Agent && python3 main.py
