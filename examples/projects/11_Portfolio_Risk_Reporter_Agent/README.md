# Portfolio Risk Reporter Agent

Produce portfolio risk summaries with governed analytics instrumentation.

Progressive AFK example **11**.

## Complexity Profile
- Tier: 3
- Progressive pass depth: 12
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- policy engine + run_handle events + telemetry projection

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/11_Portfolio_Risk_Reporter_Agent
- Then execute:
  cd examples/projects/11_Portfolio_Risk_Reporter_Agent && python3 main.py
