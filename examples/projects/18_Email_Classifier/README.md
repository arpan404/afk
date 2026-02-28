
# Email Classifier

An email triage agent that classifies emails into categories (spam, work, newsletter, social) using tools that return structured JSON data with category, confidence, and reasoning.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/18_Email_Classifier

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/18_Email_Classifier

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/18_Email_Classifier

Expected interaction
User: Show my inbox
Agent: Here are your emails: [1] Q4 Budget Review, [2] This Week in Python, ...
User: Classify email 3
Agent: Email #3 classified as SPAM (95% confidence). Reasoning: Multiple spam indicators...
User: Classify all emails
Agent: Here's the full classification summary...

The agent uses tools that return structured JSON with category, confidence, reasoning, and suggested actions.
