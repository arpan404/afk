# Memory CSM Assistant Agent

Maintain customer success context across turns and summarize continuity metrics.

Progressive AFK example **7**.

## Complexity Profile
- Tier: 2
- Progressive pass depth: 8
- Dynamic dataset rows scale with example number via app/dynamic_dataset.py.
- Multi-pass analytics scale with example number via app/progressive_analytics.py.

## AFK Focus
- run_stream + thread continuity + multi-turn orchestration

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/07_Memory_CSM_Assistant_Agent
- Then execute:
  cd examples/projects/07_Memory_CSM_Assistant_Agent && python3 main.py
