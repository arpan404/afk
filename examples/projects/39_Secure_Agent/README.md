
# Secure Agent

A security-focused file reader agent that demonstrates AFK's RunnerConfig and FailSafeConfig for defense-in-depth. Shows how to set up tool output sanitization, output character limits, command allowlisting, step limits, tool timeouts, and other safety boundaries.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/39_Secure_Agent

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/39_Secure_Agent

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/39_Secure_Agent

Expected interaction
User: List all files in the documents folder
Agent: Found 5 files in /documents: report.txt, notes.md, budget.csv, ...
User: Read the report file
Agent: (Content returned within the 5000 character limit, sanitized for safety)
User: Search for files containing "password"
Agent: (Search results with output truncated to safe limits)

The agent operates within strict safety boundaries: output limits prevent data exfiltration, timeouts prevent hangs, and step limits prevent infinite loops.
