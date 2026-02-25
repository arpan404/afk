# Incident Triage Desk Agent

Coordinate incident triage and mitigation recommendations with measurable runtime analytics.

Progressive AFK example **4**.

## Complexity Profile
- Tier: 1
- Progressive pass depth: 5
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- Runner sync execution + typed tools + result analytics

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/04_Incident_Triage_Desk_Agent
- Then execute:
  cd examples/projects/04_Incident_Triage_Desk_Agent && python3 main.py
