
# Report Generator

A report generator agent that uses ToolDeferredHandle for background/deferred tool execution. The agent starts reports as background tasks, receives deferred handles with ticket IDs, and polls for completion — ideal for long-running operations.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/27_Report_Generator

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/27_Report_Generator

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/27_Report_Generator

Expected interaction
User: Generate a sales report
Agent: Report queued! Ticket: RPT-A1B2C3D4. Let me check the status...
Agent: Report completed! [Shows full Quarterly Sales Report with revenue, breakdowns, forecasts]
User: Generate an engineering report
Agent: Report queued! Ticket: RPT-E5F6G7H8. Checking... [Shows Engineering Sprint Report]
User: List all reports
Agent: [Shows overview: 2 reports, both completed]

The start_report_generation tool returns a ToolDeferredHandle instead of an instant result. The agent polls with check_report_status to retrieve the finished report.
