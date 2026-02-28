
# Agent Test Harness

An eval harness that uses EvalCase, assertions (FinalTextContainsAssertion, StateCompletedAssertion), and run_suite to systematically test agent behavior across multiple test cases. Demonstrates how to build a repeatable test suite for agents.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/30_Agent_Test_Harness

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/30_Agent_Test_Harness

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/30_Agent_Test_Harness

Expected output
Running eval suite with 4 test cases...
  [PASS] capital-france: 2/2 assertions passed
  [PASS] calculator-add: 2/2 assertions passed
  [FAIL] trick-question: 1/2 assertions passed
  [PASS] greeting-test: 2/2 assertions passed

Suite Results: 3/4 passed

This is a non-interactive script that runs evals and prints results.

