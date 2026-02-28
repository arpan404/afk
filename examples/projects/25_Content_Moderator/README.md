
# Content Moderator

A content moderation agent that uses PolicyEngine with PolicyRule to gate tool calls based on content patterns. The agent analyzes posts and publishes, flags, or rejects them, with declarative policy rules enforcing hard guardrails around the publish_post tool.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/25_Content_Moderator

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/25_Content_Moderator

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/25_Content_Moderator

Expected interaction
User: Please moderate this post by alice: "Just had the most amazing sunset hike today!"
Agent: [Analyzes content] Content is clean. [Publishes post] Post published successfully!
User: Moderate this: "This GUARANTEED miracle cure will solve all your problems!"
Agent: [Analyzes content] Flaggable keywords found. [Flags content] Content flagged for review.
User: Moderate: "Stop the violence and hate!"
Agent: [Analyzes content] Sensitive keywords found. [Rejects content] Content rejected.

The PolicyEngine gates publish_post based on declarative rules — deny for flagged/rejected content, defer for sensitive categories, allow by default.
