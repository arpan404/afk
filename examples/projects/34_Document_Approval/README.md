
# Document Approval

A document processing agent demonstrating AgentRunHandle lifecycle controls (pause/resume/cancel/interrupt) and RunnerConfig interaction settings.

## Project Structure

```
34_Document_Approval/
  main.py       # Entry point — 3 modes: interactive, pause/resume demo, cancel demo
  tools.py      # Document management tools (draft, review, finalize, list, get)
```

## Key Concepts

- **AgentRunHandle**: `runner.run_handle()` returns a live handle with `pause()`, `resume()`, `cancel()`, `interrupt()`, `await_result()`, and `events` iterator
- **RunnerConfig**: `interaction_mode` ("headless"/"interactive"/"external"), `approval_timeout_s`, `approval_fallback`
- **Lifecycle events**: `handle.events` yields `AgentRunEvent` for real-time monitoring

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/34_Document_Approval

Modes
1. Interactive session (run_sync conversation loop)
2. AgentRunHandle lifecycle demo (pause/resume after 2 events)
3. AgentRunHandle cancel demo (cancel after 1 event)
