# Attribution Insights Lab Agent

Evaluate attribution scenarios in batch with persistent memory and run summaries.

Progressive AFK example **16**.

## Complexity Profile
- Tier: 4
- Progressive pass depth: 17
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- sqlite memory + resume/compact + batch orchestration

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/16_Attribution_Insights_Lab_Agent
- Then execute:
  cd examples/projects/16_Attribution_Insights_Lab_Agent && python3 main.py
