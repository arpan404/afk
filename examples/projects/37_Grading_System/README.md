
# Grading System

An eval-driven grading system that uses AFK's eval framework (run_suite, EvalCase, assertions, scorers) to automatically grade an agent's ability to answer questions correctly. Demonstrates deterministic, repeatable evaluation of agent quality.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/37_Grading_System

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/37_Grading_System

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/37_Grading_System

Expected output
Running eval suite with 6 cases...

=== Eval Suite Report ===
Total: 6 | Passed: 6 | Failed: 0
Execution mode: adaptive

Case: capital_of_france — PASSED
  [state_completed] PASSED
  [answer_check] PASSED
  [result_length] score=142.0

Case: year_wwii_ended — PASSED
  ...

The script runs all eval cases automatically and prints a detailed report with pass/fail status, assertion results, and quality scores.
