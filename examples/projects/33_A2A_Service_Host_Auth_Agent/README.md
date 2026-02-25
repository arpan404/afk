# A2A Service Host Auth Agent

Progressive AFK example **33** focused on **A2AServiceHost + APIKey auth provider + production mode**.

## Complexity Profile
- Tier level: 12
- Stage-chain depth: 34
- Architecture: multi-module with scenario generation, quality gates, staged analytics, metrics, and report composition.

## What This Example Includes
- Feature workflow for: A2AServiceHost + APIKey auth provider + production mode
- Dynamic scenario synthesis (app/scenario.py)
- Structured validation (app/quality.py)
- Progressive stage transforms (app/stages/stage_*.py)
- Derived analytics (app/metrics.py)
- Consolidated report output (app/report_builder.py)

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/33_A2A_Service_Host_Auth_Agent
- Then execute:
  cd examples/projects/33_A2A_Service_Host_Auth_Agent && python3 main.py
