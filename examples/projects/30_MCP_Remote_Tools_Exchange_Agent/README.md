# MCP Remote Tools Exchange Agent

Progressive AFK example **30** focused on **MCPStore + mcp_servers runtime tool materialization**.

## Complexity Profile
- Tier level: 9
- Stage-chain depth: 31
- Architecture: multi-module with scenario generation, quality gates, staged analytics, metrics, and report composition.

## What This Example Includes
- Feature workflow for: MCPStore + mcp_servers runtime tool materialization
- Dynamic scenario synthesis (app/scenario.py)
- Structured validation (app/quality.py)
- Progressive stage transforms (app/stages/stage_*.py)
- Derived analytics (app/metrics.py)
- Consolidated report output (app/report_builder.py)

## Usage
- Run from repository root:
  ./scripts/setup_example.sh --project-dir=examples/projects/30_MCP_Remote_Tools_Exchange_Agent
- Then execute:
  cd examples/projects/30_MCP_Remote_Tools_Exchange_Agent && python3 main.py
