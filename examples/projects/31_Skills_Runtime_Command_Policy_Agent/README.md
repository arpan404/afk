# Skills Runtime Command Policy Agent

Progressive AFK example **31** focused on **build_skill_tools + SkillToolPolicy command governance**.

## Complexity Profile
- Tier level: 10
- Stage-chain depth: 32
- Architecture: multi-module with scenario generation, quality gates, staged analytics, metrics, and report composition.

## What This Example Includes
- Feature workflow for: build_skill_tools + SkillToolPolicy command governance
- Dynamic scenario synthesis (app/scenario.py)
- Structured validation (app/quality.py)
- Progressive stage transforms (app/stages/stage_*.py)
- Derived analytics (app/metrics.py)
- Consolidated report output (app/report_builder.py)

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/31_Skills_Runtime_Command_Policy_Agent
- Then execute:
  cd examples/projects/31_Skills_Runtime_Command_Policy_Agent && python3 main.py
