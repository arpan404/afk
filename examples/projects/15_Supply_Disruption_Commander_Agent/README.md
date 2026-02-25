# Supply Disruption Commander Agent

Analyze disruption scenarios with multi-run orchestration and memory analytics.

Progressive AFK example **15**.

## Complexity Profile
- Tier: 4
- Progressive pass depth: 16
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- sqlite memory + resume/compact + batch orchestration

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/15_Supply_Disruption_Commander_Agent
- Then execute:
  cd examples/projects/15_Supply_Disruption_Commander_Agent && python3 main.py
