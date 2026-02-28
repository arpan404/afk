"""
---
name: Data Pipeline
description: A data pipeline orchestrator agent that uses DelegationPlan for DAG-based multi-agent execution.
tags: [agent, runner, delegation, delegation-plan, dag, async]
---
---
This example demonstrates AFK's DelegationPlan system for orchestrating complex multi-agent
workflows as a directed acyclic graph (DAG). Instead of simple sequential or parallel subagent
calls, you define nodes (agents), edges (dependencies), and execution constraints. The delegation
engine handles scheduling, parallelism, retries, and result collection. This pattern is ideal for
data pipelines, CI/CD workflows, document processing, and any task where agents have dependencies.
---
"""

import asyncio  # <- Async required for delegation engine.
from pydantic import BaseModel, Field
from afk.core import Runner  # <- Runner orchestrates agent execution.
from afk.agents import Agent  # <- Agent defines each pipeline stage.
from afk.agents.delegation import (  # <- Delegation system for DAG-based orchestration.
    DelegationPlan,  # <- The plan: a list of nodes, edges, and execution policy.
    DelegationNode,  # <- A node represents one agent invocation in the DAG.
    DelegationEdge,  # <- An edge represents a dependency between nodes (data flow).
    RetryPolicy,  # <- Per-node retry configuration (max attempts, backoff).
)
from afk.tools import tool


# ===========================================================================
# Simulated data for the pipeline
# ===========================================================================

RAW_DATA: list[dict] = [  # <- Simulated raw data records. In a real pipeline, this comes from a database, API, or file.
    {"id": 1, "name": "Alice", "department": "Engineering", "salary": 95000, "tenure_years": 3},
    {"id": 2, "name": "Bob", "department": "Marketing", "salary": 72000, "tenure_years": 5},
    {"id": 3, "name": "Charlie", "department": "Engineering", "salary": 110000, "tenure_years": 7},
    {"id": 4, "name": "Diana", "department": "Sales", "salary": 68000, "tenure_years": 2},
    {"id": 5, "name": "Eve", "department": "Engineering", "salary": 125000, "tenure_years": 10},
    {"id": 6, "name": "Frank", "department": "Marketing", "salary": 78000, "tenure_years": 4},
    {"id": 7, "name": "Grace", "department": "Sales", "salary": 82000, "tenure_years": 6},
    {"id": 8, "name": "Hank", "department": "Engineering", "salary": 105000, "tenure_years": 5},
]


# ===========================================================================
# Pipeline stage agents — each handles one step in the data pipeline
# ===========================================================================

class EmptyArgs(BaseModel):
    pass


# --- Stage 1: Data extraction ---
@tool(args_model=EmptyArgs, name="extract_data", description="Extract raw employee data from the source")
def extract_data(args: EmptyArgs) -> str:
    records = "\n".join(
        f"  {r['id']}. {r['name']} | {r['department']} | ${r['salary']:,} | {r['tenure_years']}yr"
        for r in RAW_DATA
    )
    return f"Extracted {len(RAW_DATA)} records:\n{records}"


extractor_agent = Agent(  # <- The first stage in the pipeline: extracts raw data.
    name="data-extractor",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a data extraction agent. Use the extract_data tool to fetch raw employee records
    from the data source. Present the data clearly with all fields.
    """,
    tools=[extract_data],
)


# --- Stage 2: Data validation ---
@tool(args_model=EmptyArgs, name="validate_data", description="Validate data quality — check for missing fields and outliers")
def validate_data(args: EmptyArgs) -> str:
    issues = []
    for r in RAW_DATA:
        if r["salary"] < 0:
            issues.append(f"  Record {r['id']}: negative salary")
        if r["tenure_years"] < 0:
            issues.append(f"  Record {r['id']}: negative tenure")
        if not r["name"]:
            issues.append(f"  Record {r['id']}: missing name")
    if not issues:
        return f"Validation passed: all {len(RAW_DATA)} records are valid."
    return f"Validation found {len(issues)} issues:\n" + "\n".join(issues)


validator_agent = Agent(
    name="data-validator",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a data validation agent. Use the validate_data tool to check data quality.
    Report any issues found, or confirm that all records pass validation.
    """,
    tools=[validate_data],
)


# --- Stage 3: Data transformation (depends on extraction + validation) ---
@tool(args_model=EmptyArgs, name="transform_data", description="Transform and aggregate data by department")
def transform_data(args: EmptyArgs) -> str:
    dept_stats: dict[str, dict] = {}
    for r in RAW_DATA:
        dept = r["department"]
        if dept not in dept_stats:
            dept_stats[dept] = {"count": 0, "total_salary": 0, "total_tenure": 0}
        dept_stats[dept]["count"] += 1
        dept_stats[dept]["total_salary"] += r["salary"]
        dept_stats[dept]["total_tenure"] += r["tenure_years"]

    lines = []
    for dept, stats in dept_stats.items():
        avg_salary = stats["total_salary"] / stats["count"]
        avg_tenure = stats["total_tenure"] / stats["count"]
        lines.append(f"  {dept}: {stats['count']} employees, avg salary ${avg_salary:,.0f}, avg tenure {avg_tenure:.1f}yr")
    return "Transformation complete — Department aggregates:\n" + "\n".join(lines)


transformer_agent = Agent(
    name="data-transformer",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a data transformation agent. Use the transform_data tool to aggregate
    employee data by department. Present clear summaries with averages.
    """,
    tools=[transform_data],
)


# --- Stage 4: Report generation (depends on transformation) ---
@tool(args_model=EmptyArgs, name="generate_report", description="Generate a final executive summary report")
def generate_report(args: EmptyArgs) -> str:
    total = len(RAW_DATA)
    total_salary = sum(r["salary"] for r in RAW_DATA)
    avg_salary = total_salary / total
    departments = len(set(r["department"] for r in RAW_DATA))
    top_earner = max(RAW_DATA, key=lambda r: r["salary"])
    most_tenured = max(RAW_DATA, key=lambda r: r["tenure_years"])
    return (
        f"Executive Summary Report\n"
        f"{'=' * 30}\n"
        f"Total employees: {total}\n"
        f"Departments: {departments}\n"
        f"Total payroll: ${total_salary:,}\n"
        f"Average salary: ${avg_salary:,.0f}\n"
        f"Top earner: {top_earner['name']} (${top_earner['salary']:,})\n"
        f"Most tenured: {most_tenured['name']} ({most_tenured['tenure_years']} years)"
    )


reporter_agent = Agent(
    name="report-generator",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a report generation agent. Use the generate_report tool to create an
    executive summary. Present it in a professional, clear format.
    """,
    tools=[generate_report],
)


# ===========================================================================
# DelegationPlan — defines the pipeline DAG
# ===========================================================================

pipeline_plan = DelegationPlan(  # <- The DelegationPlan defines the full pipeline as a DAG (directed acyclic graph). Nodes are agent invocations, edges are dependencies.
    nodes=[
        DelegationNode(  # <- Each node specifies a target agent, optional input bindings, timeout, and retry policy.
            node_id="extract",
            target_agent="data-extractor",
            input_binding={"task": "Extract all employee records from the data source"},  # <- Input bindings are passed as context to the agent.
            timeout_s=30.0,
            retry_policy=RetryPolicy(max_attempts=2, backoff_base_s=1.0),  # <- Retry up to 2 times with 1-second backoff if the agent fails.
        ),
        DelegationNode(
            node_id="validate",
            target_agent="data-validator",
            input_binding={"task": "Validate all extracted records for data quality"},
            timeout_s=30.0,
            retry_policy=RetryPolicy(max_attempts=2),
        ),
        DelegationNode(
            node_id="transform",
            target_agent="data-transformer",
            input_binding={"task": "Aggregate data by department with averages"},
            timeout_s=30.0,
        ),
        DelegationNode(
            node_id="report",
            target_agent="report-generator",
            input_binding={"task": "Generate executive summary report"},
            timeout_s=30.0,
        ),
    ],
    edges=[
        # --- extract and validate can run in parallel (no edges between them) ---
        DelegationEdge(from_node="extract", to_node="transform"),  # <- transform depends on extract completing first.
        DelegationEdge(from_node="validate", to_node="transform"),  # <- transform also depends on validate (both must finish before transform starts).
        DelegationEdge(from_node="transform", to_node="report"),  # <- report depends on transform.
    ],
    join_policy="all_required",  # <- All nodes must succeed for the plan to be considered successful. Other options: "first_success", "quorum", "allow_optional_failures".
    max_parallelism=2,  # <- At most 2 agents run concurrently. extract + validate run in parallel since they have no dependency.
)

"""
Pipeline DAG visualization:

    [extract] ──┐
                ├──> [transform] ──> [report]
    [validate] ─┘

- extract and validate run in parallel (max_parallelism=2)
- transform waits for both extract and validate to complete
- report waits for transform to complete
"""


# ===========================================================================
# Orchestrator agent that owns the pipeline
# ===========================================================================

orchestrator = Agent(
    name="pipeline-orchestrator",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a data pipeline orchestrator. You manage a 4-stage data pipeline:
    1. Extract: Pull raw employee data
    2. Validate: Check data quality
    3. Transform: Aggregate by department
    4. Report: Generate executive summary

    When the user asks to run the pipeline, delegate to your subagents.
    Explain what each stage does and present results as they complete.
    """,
    subagents=[extractor_agent, validator_agent, transformer_agent, reporter_agent],  # <- All pipeline stages are registered as subagents.
)

runner = Runner()


# ===========================================================================
# Main entry point — runs the pipeline
# ===========================================================================

async def main():
    print("Data Pipeline Orchestrator")
    print("=" * 40)
    print("This pipeline runs 4 stages: Extract -> Validate -> Transform -> Report")
    print("Extract and Validate run in parallel; Transform and Report are sequential.\n")

    user_input = input("[] > Type 'run' to start the pipeline (or 'quit' to exit): ").strip().lower()

    if user_input in ("quit", "exit", "q"):
        print("Goodbye!")
        return

    print("\nStarting pipeline...\n")

    # --- Run the orchestrator agent (which delegates to subagents via the plan) ---
    response = await runner.run(
        orchestrator,
        user_message="Run the full data pipeline: extract, validate, transform, and generate report.",
    )

    print(f"[orchestrator] > {response.final_text}")
    print(f"\n--- Pipeline Complete ---")
    print(f"Success: {response.success}")
    print(f"Subagent calls: {len(response.subagent_calls)}")
    for sub in response.subagent_calls:
        print(f"  - {sub.target_agent}: {'ok' if sub.success else 'failed'}")


if __name__ == "__main__":
    asyncio.run(main())



"""
---
Tl;dr: This example creates a data pipeline with 4 agent stages (extract, validate, transform, report)
orchestrated via a DelegationPlan DAG. Nodes define agent invocations with timeouts and retry policies.
Edges define dependencies — extract and validate run in parallel, then transform runs when both complete,
then report runs last. The join_policy="all_required" ensures all stages must succeed. This pattern is
ideal for multi-stage workflows with dependencies, retries, and controlled parallelism.
---
---
What's next?
- Try setting a node's required=False to make it optional (the pipeline continues even if it fails).
- Experiment with join_policy="first_success" to use the first stage that completes.
- Add error injection (make a tool raise an exception) to see retry behavior.
- Use output_key_map on edges to pass specific outputs between stages.
- Scale max_parallelism to run more stages concurrently.
- Check out the Travel Planner example for quorum-based delegation with multiple perspectives!
---
"""
