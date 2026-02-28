"""
---
name: Report Generator
description: A report generator agent that uses ToolDeferredHandle for background/deferred tool execution with polling.
tags: [agent, runner, tools, deferred, background, tool-deferred-handle, async]
---
---
This example demonstrates AFK's ToolDeferredHandle for modelling long-running, background tool
executions. When a tool returns a ToolDeferredHandle instead of an immediate string result, it
signals to the runner that the work is deferred — still in progress — and can be polled for
completion later. The handle carries a ticket_id (unique identifier for tracking), tool_name
(which tool was deferred), status ("pending", "running", "completed", "failed"), poll_after_s
(how long to wait before checking again), and optional summary and resume_hint fields. This
pattern is ideal for tasks like report generation, data exports, batch processing, or any
operation that takes longer than a single LLM turn. The report generator agent starts report
generation as a background task, checks on its status, and retrieves the finished report when
ready.
---
"""

import asyncio  # <- Async required for background task simulation and deferred handle polling.
import time  # <- Used for timestamp tracking in the simulated report store.
import uuid  # <- For generating unique ticket IDs for deferred handles.
from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.core import Runner  # <- Runner orchestrates agent execution.
from afk.agents import Agent  # <- Agent defines the agent's model, instructions, and tools.
from afk.tools import tool, ToolDeferredHandle  # <- @tool decorator and ToolDeferredHandle — the deferred execution marker that tells the runner "this tool's work is still in progress."


# ===========================================================================
# Simulated report store — tracks deferred report generation tasks
# ===========================================================================

REPORT_STORE: dict[str, dict] = {}  # <- In-memory store for report tasks. Keyed by ticket_id. In production, this would be a database or task queue (Celery, Redis, etc.). Each entry tracks status, progress, and the final report content when completed.


# ===========================================================================
# Report templates — simulated report generation data
# ===========================================================================

REPORT_TEMPLATES: dict[str, dict] = {  # <- Predefined report templates. Each has sections and simulated data that gets "generated" when the deferred task completes.
    "sales": {
        "title": "Quarterly Sales Report",
        "sections": [
            "Executive Summary: Total revenue $2.4M, up 15% QoQ",
            "Product Breakdown: Widget A ($1.2M), Widget B ($800K), Widget C ($400K)",
            "Regional Performance: North America 45%, Europe 30%, Asia 25%",
            "Top Customers: Acme Corp ($500K), Global Inc ($350K), Tech Ltd ($200K)",
            "Forecast: Q2 projected at $2.8M based on pipeline analysis",
        ],
    },
    "engineering": {
        "title": "Engineering Sprint Report",
        "sections": [
            "Sprint Velocity: 45 story points completed (target: 40)",
            "Features Shipped: User dashboard redesign, API v3 endpoints, search optimization",
            "Bugs Fixed: 12 critical, 28 medium, 15 low priority",
            "Tech Debt: Reduced by 8% — migrated auth module to new framework",
            "Next Sprint: Payment system overhaul, mobile app push notifications",
        ],
    },
    "marketing": {
        "title": "Marketing Campaign Report",
        "sections": [
            "Campaign Overview: 'Summer Launch 2025' — ran for 6 weeks",
            "Reach: 2.5M impressions across social, email, and paid channels",
            "Engagement: 180K clicks (7.2% CTR), 45K sign-ups (2.5% conversion)",
            "Top Channel: Email marketing (35% of conversions) followed by paid social (28%)",
            "ROI: 340% return on ad spend — $85K budget generated $290K in new revenue",
        ],
    },
    "financial": {
        "title": "Financial Health Report",
        "sections": [
            "Revenue: $8.2M YTD (on track for $12M annual target)",
            "Expenses: $6.1M YTD — Operating margin: 25.6%",
            "Cash Position: $3.4M liquid assets, $1.2M accounts receivable",
            "Burn Rate: $510K/month — 6.7 months runway at current rate",
            "Key Risks: Currency exposure (EUR/USD), pending tax audit Q3",
        ],
    },
}


# ===========================================================================
# Simulated background task runner
# ===========================================================================

def simulate_report_progress(ticket_id: str, report_type: str) -> None:  # <- This function simulates a long-running background task. In production, this would be a Celery task, a background thread, or an async job. Here we update the REPORT_STORE directly to simulate progress over time.
    """Simulate report generation progressing through stages."""
    template = REPORT_TEMPLATES.get(report_type.lower(), REPORT_TEMPLATES["sales"])

    # --- Simulate stages of report generation ---
    REPORT_STORE[ticket_id]["status"] = "running"
    REPORT_STORE[ticket_id]["progress"] = "collecting data..."
    REPORT_STORE[ticket_id]["percent"] = 25

    # In a real system, each stage would take time. Here we pre-compute the final result.
    report_lines = [
        f"{'=' * 50}",
        f"  {template['title']}",
        f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Ticket: {ticket_id}",
        f"{'=' * 50}",
        "",
    ]
    for i, section in enumerate(template["sections"], 1):
        report_lines.append(f"  {i}. {section}")
    report_lines.append("")
    report_lines.append(f"{'=' * 50}")
    report_lines.append(f"  Report generation complete.")

    # --- Mark as completed with the final content ---
    REPORT_STORE[ticket_id]["status"] = "completed"
    REPORT_STORE[ticket_id]["progress"] = "done"
    REPORT_STORE[ticket_id]["percent"] = 100
    REPORT_STORE[ticket_id]["result"] = "\n".join(report_lines)
    REPORT_STORE[ticket_id]["completed_at"] = time.time()


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class StartReportArgs(BaseModel):  # <- Schema for starting a deferred report generation task.
    report_type: str = Field(description="Type of report: sales, engineering, marketing, or financial")
    title: str = Field(description="Custom title for the report (optional override)", default="")


class CheckStatusArgs(BaseModel):  # <- Schema for checking the status of a deferred task.
    ticket_id: str = Field(description="The ticket ID returned by start_report_generation")


class EmptyArgs(BaseModel):  # <- Empty schema for tools that take no arguments.
    pass


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(  # <- This is the KEY tool. It returns a ToolDeferredHandle instead of a direct result, signaling to the runner that the work is deferred and can be polled later.
    args_model=StartReportArgs,
    name="start_report_generation",
    description="Start generating a report in the background. Returns a deferred handle with a ticket ID for tracking. The report takes time to generate — poll with check_report_status to see progress.",
)
def start_report_generation(args: StartReportArgs) -> ToolDeferredHandle:  # <- Return type is ToolDeferredHandle, NOT str. This is what makes the tool "deferred."
    ticket_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"  # <- Generate a unique ticket ID like "RPT-A1B2C3D4".

    report_type = args.report_type.lower()
    if report_type not in REPORT_TEMPLATES:
        report_type = "sales"  # <- Default to sales if unknown type.

    title = args.title or REPORT_TEMPLATES[report_type]["title"]

    # --- Register the task in the store ---
    REPORT_STORE[ticket_id] = {  # <- Create the task entry. Status starts as "pending".
        "ticket_id": ticket_id,
        "report_type": report_type,
        "title": title,
        "status": "pending",
        "progress": "queued",
        "percent": 0,
        "result": None,
        "created_at": time.time(),
        "completed_at": None,
    }

    # --- Simulate starting background work ---
    simulate_report_progress(ticket_id, report_type)  # <- In production, this would dispatch to a task queue. Here we run it synchronously for simplicity (the result is immediately available after this call).

    return ToolDeferredHandle(  # <- Return the deferred handle. The runner receives this and knows the tool's work is "in progress."
        ticket_id=ticket_id,  # <- Unique identifier for tracking this deferred task. The agent can use this to poll for status.
        tool_name="start_report_generation",  # <- Which tool created this deferred work.
        status="pending",  # <- Initial status. In production, this would update as the background task progresses ("pending" -> "running" -> "completed" or "failed").
        poll_after_s=2.0,  # <- Hint to the runner/agent: wait at least 2 seconds before polling for status. This prevents busy-waiting.
        summary=f"Report '{title}' ({report_type}) queued for generation. Ticket: {ticket_id}",  # <- Human-readable summary the agent can relay to the user.
        resume_hint=f"Use check_report_status with ticket_id='{ticket_id}' to check progress.",  # <- Hint for the agent on how to follow up.
    )


@tool(  # <- Status checking tool. Used to poll the progress of a deferred report task.
    args_model=CheckStatusArgs,
    name="check_report_status",
    description="Check the status and progress of a deferred report generation task. Use the ticket_id from start_report_generation.",
)
def check_report_status(args: CheckStatusArgs) -> str:
    task = REPORT_STORE.get(args.ticket_id)
    if not task:
        return f"No report found with ticket ID '{args.ticket_id}'. Check the ID and try again."

    lines = [
        f"Report Status: {args.ticket_id}",
        f"{'=' * 40}",
        f"Title: {task['title']}",
        f"Type: {task['report_type']}",
        f"Status: {task['status'].upper()}",
        f"Progress: {task['progress']} ({task['percent']}%)",
    ]

    if task["status"] == "completed":
        elapsed = task["completed_at"] - task["created_at"]
        lines.append(f"Generation time: {elapsed:.1f}s")
        lines.append(f"\n--- Full Report ---")
        lines.append(task["result"])  # <- Include the full report content when completed.
    elif task["status"] == "failed":
        lines.append(f"Error: Report generation failed. Please try again.")
    else:
        lines.append(f"Estimated completion: check again in a few seconds.")

    return "\n".join(lines)


@tool(  # <- List all reports tool. Gives an overview of all deferred tasks.
    args_model=EmptyArgs,
    name="list_reports",
    description="List all report generation tasks with their current status.",
)
def list_reports(args: EmptyArgs) -> str:
    if not REPORT_STORE:
        return "No reports have been generated yet. Use start_report_generation to create one."

    lines = [f"All Reports ({len(REPORT_STORE)} total):", "=" * 40]
    for ticket_id, task in REPORT_STORE.items():
        status_icon = {
            "pending": "WAIT",
            "running": "RUN",
            "completed": "DONE",
            "failed": "FAIL",
        }.get(task["status"], "?")
        lines.append(
            f"  [{status_icon}] {ticket_id}: {task['title']} ({task['report_type']}) — {task['percent']}%"
        )

    completed = sum(1 for t in REPORT_STORE.values() if t["status"] == "completed")
    pending = sum(1 for t in REPORT_STORE.values() if t["status"] in ("pending", "running"))
    lines.append(f"\nCompleted: {completed} | In progress: {pending}")
    return "\n".join(lines)


# ===========================================================================
# Agent setup
# ===========================================================================

report_agent = Agent(
    name="report-generator",  # <- The agent's display name.
    model="ollama_chat/gpt-oss:20b",  # <- The LLM model the agent will use.
    instructions="""
    You are a report generation assistant. You help users create various types of reports:
    - Sales reports (quarterly performance, revenue, forecasts)
    - Engineering reports (sprint velocity, features shipped, bugs fixed)
    - Marketing reports (campaign performance, engagement, ROI)
    - Financial reports (revenue, expenses, cash position, runway)

    Workflow:
    1. When a user asks for a report, use start_report_generation to begin the process.
       This returns a deferred handle with a ticket_id — the report generates in the background.
    2. Tell the user their report is being generated and give them the ticket_id.
    3. Use check_report_status with the ticket_id to check progress and retrieve the
       completed report.
    4. Use list_reports to show all reports and their statuses.

    The start_report_generation tool returns a ToolDeferredHandle — this means the result
    is NOT immediately available. The handle contains a ticket_id for tracking and a
    poll_after_s hint for when to check back.

    Be helpful and proactive — after starting a report, immediately check its status to
    see if it's ready.
    """,  # <- Instructions explain the deferred workflow to the agent. The agent learns to start, poll, and retrieve.
    tools=[start_report_generation, check_report_status, list_reports],  # <- Three tools: start (deferred), check (polling), list (overview).
)

runner = Runner()  # <- A single Runner handles all executions.


# ===========================================================================
# Main entry point — interactive conversation loop
# ===========================================================================

async def main():
    print("Report Generator Agent (type 'quit' to exit)")
    print("=" * 50)
    print("Generate reports: sales, engineering, marketing, or financial.")
    print("\nTry: 'Generate a sales report', 'What reports have been generated?'\n")

    while True:  # <- Conversation loop for the report generation interaction.
        user_input = input("[] > ")

        if user_input.strip().lower() in ("quit", "exit", "q"):
            # --- Show summary before exiting ---
            if REPORT_STORE:
                completed = sum(1 for t in REPORT_STORE.values() if t["status"] == "completed")
                print(f"\nSession summary: {len(REPORT_STORE)} reports started, {completed} completed.")
            print("Goodbye!")
            break

        response = await runner.run(  # <- Async run — we use async because the deferred handle workflow benefits from async orchestration. The runner processes deferred handles and can check them automatically.
            report_agent,
            user_message=user_input,
        )

        print(f"[report-generator] > {response.final_text}\n")


if __name__ == "__main__":
    asyncio.run(main())  # <- asyncio.run() starts the event loop for our async main function.



"""
---
Tl;dr: This example creates a report generator agent that uses ToolDeferredHandle for background
tool execution. The start_report_generation tool returns a ToolDeferredHandle instead of a direct
result, signaling to the runner that the work is deferred. The handle carries a ticket_id for
tracking, a status ("pending"/"running"/"completed"/"failed"), poll_after_s (how long to wait
before checking), and human-readable summary and resume_hint fields. The agent then uses
check_report_status to poll by ticket_id and retrieve the completed report. This pattern is
ideal for long-running tasks (report generation, data exports, batch jobs) where the tool cannot
return immediately and the agent needs to check back for results.
---
---
What's next?
- Try generating multiple reports in sequence to see them tracked in the list_reports overview.
- Modify poll_after_s to different values and observe how it hints the agent on when to check back.
- Add a "cancel_report" tool that sets a task's status to "failed" to see how the agent handles cancellation.
- Implement real async background generation using asyncio.create_task so the report genuinely takes time.
- Combine deferred handles with a DelegationPlan to orchestrate multi-stage report pipelines where each stage is deferred.
- Check out the Data Pipeline example for DAG-based orchestration with dependencies and retries!
---
"""
