
# Meeting Notes

A meeting notes agent that uses a dynamic InstructionProvider to adapt its behavior based on runtime context such as meeting type (standup, brainstorm, review, planning) and formality level.

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/19_Meeting_Notes

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/19_Meeting_Notes

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/19_Meeting_Notes

Expected interaction
Choose meeting type: 2 (brainstorm)
Formality: casual
User: idea - we could use websockets for real-time updates
Agent: Note #1 added: "Idea: Use WebSockets for real-time updates"
User: idea - or server-sent events might be simpler
Agent: Note #2 added: "Idea: Server-Sent Events as simpler alternative"
User: summarize
Agent: Brainstorm Summary: 2 ideas captured...

The agent dynamically generates instructions based on the meeting context, adapting its note-taking strategy per meeting type.
