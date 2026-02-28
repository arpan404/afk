
# News Digest

A news agent that streams responses token-by-token using the Runner's streaming API, demonstrating real-time output with AgentStreamEvent handling.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/17_News_Digest

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/17_News_Digest

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/17_News_Digest

Expected interaction
User: Give me a full news digest
Agent: [fetching: fetch_multiple_news...]
       [fetch_multiple_news: done]
       Here's your morning briefing...  (text streams token by token)
       [tokens: 1234 in / 567 out]

The agent uses run_stream() to deliver text incrementally, with real-time tool lifecycle events visible in the output.
