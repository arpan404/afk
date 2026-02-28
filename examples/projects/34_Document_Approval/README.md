
# Document Approval

A document processing agent with a draft-review-finalize workflow demonstrating RunnerConfig interaction settings for human-in-the-loop approval patterns.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/34_Document_Approval

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/34_Document_Approval

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/34_Document_Approval

Expected interaction
User: Create a memo about the team offsite
Agent: Document created: DOC-001, Title: Team Offsite Memo, Status: draft
User: Review DOC-001
Agent: Review of DOC-001 — Looks good! Word count: 45, Ready for finalization.
User: Finalize DOC-001
Agent: Document DOC-001 FINALIZED. (In production, this would require approval first.)

Demonstrates RunnerConfig with interaction_mode, approval_timeout_s, and approval_fallback settings.
