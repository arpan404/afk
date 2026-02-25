# AFK Feature Coverage Matrix

This matrix maps major AFK features to at least one example project.

| AFK feature | Example(s) |
|---|---|
| Agent + Runner basics | 01_Greeting_Agent, 02_Revenue_Lead_Scorer_Agent |
| Tool decorator usage | 02_Revenue_Lead_Scorer_Agent, 23_Tool_Hooks_Middleware_Agent |
| Fail-safe controls | 05_Change_Control_Assistant_Agent, 28_Fallback_Reasoning_Control_Agent |
| Streaming runs/events | 06_Streaming_NOC_Copilot_Agent, 27_Background_Deferred_Pipeline_Agent |
| Memory stores and resume/compact | 14_Sales_Forecast_Planner_Agent, 15_Supply_Disruption_Commander_Agent |
| Observability and telemetry export | 10_Observability_Metrics_Pipeline_Agent, 20_Executive_Briefing_Studio_Agent |
| Policy engine + governance gates | 14_Sales_Forecast_Planner_Agent, 21_Governance_Control_Tower_Agent |
| LLMBuilder + LLM middleware | 22_LLM_Builder_Strategy_Agent |
| Tool prehooks/posthooks/middleware | 23_Tool_Hooks_Middleware_Agent |
| Registry middleware | 23_Tool_Hooks_Middleware_Agent |
| Runtime prebuilt tools (build_runtime_tools) | 24_Runtime_Sandbox_FileOps_Agent |
| Sandbox profiles and output limiting | 24_Runtime_Sandbox_FileOps_Agent |
| Prompt loader (instruction_file, prompts_dir) | 25_Prompt_Template_Loader_Agent |
| Interactive mode + in-memory interaction provider | 26_Interactive_Approval_Workflow_Agent |
| Background/deferred tools (ToolDeferredHandle) | 27_Background_Deferred_Pipeline_Agent |
| Fallback model chain | 28_Fallback_Reasoning_Control_Agent |
| Reasoning controls | 28_Fallback_Reasoning_Control_Agent |
| Evals suite (run_suite, datasets, report) | 29_Eval_Suite_Quality_Gates_Agent |
| MCP store + mcp_servers runtime loading | 30_MCP_Remote_Tools_Exchange_Agent |
| Skill tools (build_skill_tools) | 31_Skills_Runtime_Command_Policy_Agent |
| A2A internal protocol | 32_A2A_Internal_Delegation_Protocol_Agent |
| A2A service host auth | 33_A2A_Service_Host_Auth_Agent |
| Task queues and worker execution contracts | 34_Queue_Worker_Execution_Contracts_Agent |

## Progressive Complexity Ladder (02-21)

Core examples include dynamic dataset generation and multi-pass analytics in every workflow.
Dynamic segment count and pass depth both increase with example number.

| Example | Python files | Dynamic segments | Pass depth |
|---|---:|---:|---:|
| 02_Revenue_Lead_Scorer_Agent | 10 | 3 | 3 |
| 03_Support_SLA_Triage_Agent | 10 | 4 | 4 |
| 04_Incident_Triage_Desk_Agent | 10 | 5 | 5 |
| 05_Change_Control_Assistant_Agent | 10 | 6 | 6 |
| 06_Streaming_NOC_Copilot_Agent | 11 | 7 | 7 |
| 07_Memory_CSM_Assistant_Agent | 11 | 8 | 8 |
| 08_Recovery_Orchestration_Agent | 11 | 9 | 9 |
| 09_Compliance_File_Inspector_Agent | 11 | 10 | 10 |
| 10_Observability_Metrics_Pipeline_Agent | 12 | 11 | 11 |
| 11_Portfolio_Risk_Reporter_Agent | 12 | 12 | 12 |
| 12_FinOps_Budget_Guard_Agent | 12 | 13 | 13 |
| 13_Retention_Insights_Agent | 12 | 14 | 14 |
| 14_Sales_Forecast_Planner_Agent | 13 | 15 | 15 |
| 15_Supply_Disruption_Commander_Agent | 13 | 16 | 16 |
| 16_Attribution_Insights_Lab_Agent | 13 | 17 | 17 |
| 17_Experiment_Insights_Hub_Agent | 13 | 18 | 18 |
| 18_Fraud_Investigation_Orchestrator_Agent | 15 | 19 | 19 |
| 19_SRE_Capacity_Planner_Agent | 15 | 20 | 20 |
| 20_Executive_Briefing_Studio_Agent | 15 | 21 | 21 |
| 21_Governance_Control_Tower_Agent | 15 | 22 | 22 |

## Progressive Complexity Ladder (22-34)

Advanced examples use scenario generation, quality gates, stage modules, analytics projection, and report composition.
Stage depth and Python file count both increase monotonically through the advanced series.

| Example | Python files | Stage depth |
|---|---:|---:|
| 22_LLM_Builder_Strategy_Agent | 36 | 23 |
| 23_Tool_Hooks_Middleware_Agent | 38 | 24 |
| 24_Runtime_Sandbox_FileOps_Agent | 39 | 25 |
| 25_Prompt_Template_Loader_Agent | 40 | 26 |
| 26_Interactive_Approval_Workflow_Agent | 41 | 27 |
| 27_Background_Deferred_Pipeline_Agent | 42 | 28 |
| 28_Fallback_Reasoning_Control_Agent | 43 | 29 |
| 29_Eval_Suite_Quality_Gates_Agent | 44 | 30 |
| 30_MCP_Remote_Tools_Exchange_Agent | 45 | 31 |
| 31_Skills_Runtime_Command_Policy_Agent | 46 | 32 |
| 32_A2A_Internal_Delegation_Protocol_Agent | 47 | 33 |
| 33_A2A_Service_Host_Auth_Agent | 48 | 34 |
| 34_Queue_Worker_Execution_Contracts_Agent | 49 | 35 |

## Automated Validation

Run the project validator to verify:
- index continuity (`01..34`)
- weather example removal and workspace references
- progressive complexity constraints (`02..21` and `22..34`)
- feature marker coverage for major AFK capabilities
- AFK import symbol resolution in all example Python files

```bash
python3 examples/projects/validate_examples.py
```
