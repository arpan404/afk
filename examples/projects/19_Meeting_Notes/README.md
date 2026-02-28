
# Meeting Notes

A meeting notes agent demonstrating two approaches to dynamic agent instructions in AFK: Jinja2 prompt templates via `instruction_file` + `prompts_dir`, and callable InstructionProvider via `instructions=`.

## Project Structure

```
19_Meeting_Notes/
  main.py                  # Entry point — two agents (template vs callable)
  tools.py                 # Tool definitions (add_note, action items, etc.)
  prompts/
    meeting_notes.md       # Jinja2 template with conditionals and filters
```

## Key Concepts

- **instruction_file**: Path to a Jinja2 Markdown template resolved relative to `prompts_dir`
- **prompts_dir**: Directory containing prompt templates (also settable via `AFK_AGENT_PROMPTS_DIR` env var)
- **Jinja2 rendering**: Templates receive runtime `context` as variables — use `{{ meeting_type }}`, `{% if %}`, `{{ attendees | join(', ') }}`
- **InstructionProvider**: Callable `(context: dict) -> str` alternative for logic-heavy instruction generation

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/19_Meeting_Notes

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/19_Meeting_Notes

Expected interaction
Choose approach: 1 (Jinja2 template)
Choose meeting type: 2 (brainstorm)
Formality: casual
User: idea - we could use websockets for real-time updates
Agent: Note #1 added: "Idea: Use WebSockets for real-time updates"
User: summarize
Agent: Brainstorm Summary: 2 ideas captured...

Both agents produce equivalent behavior — the template approach loads prompts/meeting_notes.md as a Jinja2 template, while the callable approach builds the same instructions in Python.
