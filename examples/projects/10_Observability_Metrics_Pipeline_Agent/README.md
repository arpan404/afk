# Observability Metrics Pipeline Agent

Combine policy and telemetry to build metrics-first operational reporting.

Progressive AFK example **10**.

## Complexity Profile
- Tier: 3
- Progressive pass depth: 11
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- policy engine + run_handle events + telemetry projection

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/10_Observability_Metrics_Pipeline_Agent
- Then execute:
  cd examples/projects/10_Observability_Metrics_Pipeline_Agent && python3 main.py
