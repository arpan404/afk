"""
---
name: Document Approval
description: A document processing agent demonstrating AgentRunHandle lifecycle controls (pause/resume/cancel/interrupt) and RunnerConfig interaction settings.
tags: [agent, runner, run-handle, lifecycle, pause, resume, cancel, interrupt, interaction, approval]
---
---
This example demonstrates two key AFK features:

1. **AgentRunHandle** lifecycle controls: Instead of calling runner.run() (which blocks until
   completion), use runner.run_handle() to get a live handle with lifecycle methods:
   - handle.pause(): Pause cooperative execution at safe boundaries
   - handle.resume(): Resume a paused run
   - handle.cancel(): Cancel the run and terminate with no result
   - handle.interrupt(): Interrupt in-flight operations immediately
   - handle.await_result(): Await the terminal result (AgentResult or None if cancelled)
   - handle.events: AsyncIterator of AgentRunEvent for real-time lifecycle monitoring

   This is essential for production systems where you need to control long-running agent
   operations — pause for user input, cancel abandoned requests, or interrupt stuck operations.

2. **RunnerConfig** interaction settings: Configure how the Runner handles approval requests
   and user input with interaction_mode, approval_timeout_s, approval_fallback, etc.
---
"""

import asyncio  # <- Async required for run_handle() and lifecycle control.
from afk.core import Runner, RunnerConfig  # <- Runner executes agents; RunnerConfig configures interaction and approval behavior.
from afk.agents import Agent  # <- Agent defines the document processing agent.

from tools import draft_document, review_document, finalize_document, list_documents, get_document, documents  # <- Import tools and document store.


# ===========================================================================
# RunnerConfig — interaction and approval settings
# ===========================================================================

config = RunnerConfig(
    interaction_mode="headless",  # <- "headless" means auto-decide (no manual prompts). Options: "headless", "interactive", "external". In production, use "interactive" with an InteractionProvider for human-in-the-loop approval.
    approval_timeout_s=60.0,  # <- How long to wait for approval when in interactive mode.
    approval_fallback="deny",  # <- What to do if approval times out: "allow" or "deny". "deny" is safer for sensitive actions.
    input_timeout_s=60.0,  # <- How long to wait for user input.
    input_fallback="deny",  # <- Timeout fallback for input requests.
    sanitize_tool_output=True,  # <- Sanitize tool output for safety.
    tool_output_max_chars=12_000,  # <- Maximum tool output characters.
)


# ===========================================================================
# Agent definition
# ===========================================================================

doc_agent = Agent(
    name="document-processor",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a document processing assistant. You help users create, review, and finalize documents.

    Workflow:
    1. Draft: Create new documents with draft_document.
    2. Review: Check documents for issues with review_document.
    3. Finalize: Mark documents as final with finalize_document (sensitive action!).

    Always review a document before finalizing it. Warn the user that finalization is
    permanent. If there are issues in the review, suggest fixes before finalizing.
    """,
    tools=[draft_document, review_document, finalize_document, list_documents, get_document],
)

runner = Runner(config=config)


# ===========================================================================
# AgentRunHandle lifecycle demonstration
# ===========================================================================

async def demonstrate_run_handle():
    """Demonstrate AgentRunHandle lifecycle controls.

    This function shows how to use run_handle() to get a live handle with
    pause/resume/cancel controls, instead of the blocking run() method.
    """
    print("\n--- AgentRunHandle Lifecycle Demo ---")
    print("Starting a document processing run with lifecycle controls...\n")

    # --- Start the run via run_handle (non-blocking) ---
    handle = await runner.run_handle(  # <- run_handle() returns immediately with a live AgentRunHandle. The agent starts executing in the background.
        doc_agent,
        user_message="Create a memo titled 'Team Update' about the Q4 planning meeting. Then review it and finalize it.",
    )

    print("Run started. Handle lifecycle methods available:")
    print("  handle.pause()       — pause at next safe boundary")
    print("  handle.resume()      — resume paused run")
    print("  handle.cancel()      — cancel and terminate")
    print("  handle.interrupt()   — interrupt immediately")
    print("  handle.await_result() — wait for final result")
    print()

    # --- Monitor events from the handle ---
    print("[events]")
    event_count = 0
    async for event in handle.events:  # <- handle.events is an AsyncIterator[AgentRunEvent]. Each event represents a lifecycle change (step started, tool called, state change, etc.).
        event_count += 1
        # Format the event for display
        event_info = f"  [{event_count}] type={event.type}"
        if hasattr(event, "step") and event.step:
            event_info += f" step={event.step}"
        if hasattr(event, "state") and event.state:
            event_info += f" state={event.state}"
        if hasattr(event, "tool_name") and event.tool_name:
            event_info += f" tool={event.tool_name}"
        print(event_info)

        # --- Demonstrate pause/resume after 2 events ---
        if event_count == 2:
            print("\n  >> Pausing run...")
            await handle.pause()  # <- Pauses cooperative execution at the next safe boundary. The agent won't start new steps until resumed.
            print("  >> Run paused. Simulating review delay...")
            await asyncio.sleep(0.5)  # <- Simulate a review/approval delay.
            print("  >> Resuming run...")
            await handle.resume()  # <- Resume the paused run. Execution continues from where it was paused.
            print("  >> Run resumed.\n")

    # --- Await the final result ---
    result = await handle.await_result()  # <- Blocks until the run completes (or was cancelled). Returns AgentResult or None.

    if result is None:
        print("\nRun was cancelled (result is None).")
    else:
        print(f"\n[document-processor] > {result.final_text}")
        print(f"  [tokens: {result.usage.input_tokens} in / {result.usage.output_tokens} out]")

    print(f"\nTotal lifecycle events: {event_count}")


async def demonstrate_cancel():
    """Demonstrate cancelling a run via AgentRunHandle.cancel()."""
    print("\n--- Cancel Demo ---")
    print("Starting a run that will be cancelled after 1 event...\n")

    handle = await runner.run_handle(
        doc_agent,
        user_message="Create 10 different documents about various topics.",
    )

    event_count = 0
    async for event in handle.events:
        event_count += 1
        print(f"  [{event_count}] type={event.type}")
        if event_count >= 1:
            print("\n  >> Cancelling run...")
            await handle.cancel()  # <- Cancel terminates the run. await_result() will return None.
            print("  >> Run cancelled.\n")
            break

    result = await handle.await_result()
    print(f"Result after cancel: {result}")  # <- None, because the run was cancelled.
    print("Cancel demonstration complete.")


# ===========================================================================
# Interactive session with lifecycle option
# ===========================================================================

async def interactive_session():
    """Standard interactive loop with basic run_sync calls."""
    while True:
        user_input = input("[] > ")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            finalized = sum(1 for d in documents.values() if d["status"] == "finalized")
            print(f"Session: {len(documents)} docs ({finalized} finalized). Goodbye!")
            break
        response = runner.run_sync(doc_agent, user_message=user_input)
        print(f"[document-processor] > {response.final_text}\n")


async def main():
    print("Document Approval Agent")
    print("=" * 55)
    print()
    print("Modes:")
    print("  1. Interactive session (run_sync conversation loop)")
    print("  2. AgentRunHandle lifecycle demo (pause/resume)")
    print("  3. AgentRunHandle cancel demo")
    print()

    choice = input("Choose mode (1-3): ").strip()

    if choice == "2":
        await demonstrate_run_handle()
    elif choice == "3":
        await demonstrate_cancel()
    else:
        print()
        print(f"Interaction mode: {config.interaction_mode}")
        print(f"Approval fallback: {config.approval_fallback}")
        print()
        print("Try: 'Create a memo about the team offsite', 'Review DOC-001', 'Finalize DOC-001'")
        print()
        await interactive_session()


if __name__ == "__main__":
    asyncio.run(main())



"""
---
Tl;dr: This example demonstrates AgentRunHandle lifecycle controls and RunnerConfig interaction
settings. runner.run_handle() returns a live handle with methods: pause() pauses at safe boundaries,
resume() continues execution, cancel() terminates with no result, interrupt() stops immediately,
and await_result() waits for the terminal AgentResult (or None if cancelled). handle.events is an
AsyncIterator of AgentRunEvent for real-time lifecycle monitoring. The demo shows pause/resume after
2 events (simulating a review delay) and a separate cancel demonstration. RunnerConfig sets
interaction_mode ("headless"/"interactive"/"external"), approval_timeout_s, approval_fallback, and
tool output sanitization.
---
---
What's next?
- Try mode 2 to see pause/resume in action, and mode 3 to see cancel.
- Switch interaction_mode to "interactive" and implement an InteractionProvider for real approval prompts.
- Use handle.interrupt() for cases where the agent is stuck or the user abandons the request.
- Combine AgentRunHandle with the Debugger facade — use debugger.attach(handle) for event formatting.
- Add a PolicyRole that triggers "defer" for finalize_document, then resolve the deferral via handle.
- Check out the Chat History Manager example for Runner.resume() checkpoint restoration!
---
"""
