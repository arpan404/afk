"""
---
name: Production Agent — InstructionRoles
description: InstructionRole callbacks that dynamically augment the coordinator's instructions based on runtime context and task state.
tags: [instruction-role, dynamic-instructions, context-aware]
---
---
InstructionRole is a protocol that lets you append dynamic instructions to an agent at runtime.
Unlike InstructionProvider (which replaces the entire instruction), InstructionRole callbacks ADD
extra instruction text on top of the base instructions. Multiple InstructionRoles can be stacked —
each one is called in order and their outputs are concatenated after the base instructions.

Each InstructionRole receives:
  - context: dict[str, JSONValue] — the runtime context from runner.run(context={...})
  - state: AgentState — the current runtime state (e.g., "running")

And returns:
  - str | list[str] | None — additional instruction text (or None to skip)

The coordinator already uses an InstructionProvider for its base instructions. These
InstructionRoles STACK on top of it, adding cross-cutting concerns like workload awareness
and time-of-day context without modifying the base instruction logic.
---
"""

from config import memory, THREAD_ID  # <- Import shared memory and thread ID to check task state.


# ===========================================================================
# InstructionRole 1: Workload awareness
# ===========================================================================
# This role checks the current task count and adds warnings when the user
# appears overloaded. It reads task state from the shared memory store.

async def workload_awareness_role(context: dict, state: str) -> str | None:  # <- InstructionRole callback. Can be async since it reads from memory. Returns optional instruction text appended AFTER base instructions.
    try:
        all_state = await memory.list_state(thread_id=THREAD_ID, prefix="task:")  # <- Read current task state to assess workload.
    except Exception:
        return None  # <- If memory isn't available yet (e.g., setup not called), skip gracefully.

    tasks = [v for k, v in all_state.items() if k != "task_counter" and isinstance(v, dict)]
    open_tasks = [t for t in tasks if t.get("status") == "open"]
    open_count = len(open_tasks)

    if open_count == 0:
        return None  # <- No open tasks, no extra instructions needed.

    instructions = []

    if open_count > 10:
        instructions.append(  # <- This text is appended AFTER the agent's base instructions.
            "\n\n--- Workload Alert ---\n"
            f"The user currently has {open_count} open tasks. This suggests cognitive overload.\n"
            "- Proactively suggest completing or closing stale tasks before adding new ones.\n"
            "- Recommend breaking large tasks into smaller, actionable items.\n"
            "- Offer to run a quick prioritization review."
        )

    # Check for high-priority items
    critical_high = [t for t in open_tasks if t.get("priority") in ("critical", "high")]
    if len(critical_high) >= 3:
        names = ", ".join(f"#{t['id']}" for t in critical_high[:5])
        instructions.append(
            f"\n\n--- Priority Alert ---\n"
            f"There are {len(critical_high)} open critical/high-priority tasks ({names}).\n"
            "Remind the user to focus on these before taking on new medium/low priority work."
        )

    return instructions if instructions else None  # <- Return list[str] or None. Each string is a separate instruction block.


# ===========================================================================
# InstructionRole 2: Time-of-day context
# ===========================================================================
# This role adds time awareness so the agent can adapt its communication
# style and set appropriate expectations.

def time_context_role(context: dict, state: str) -> str | None:  # <- Synchronous InstructionRole — no async needed for time checks.
    import datetime
    hour = datetime.datetime.now().hour  # <- Check current local time.

    if 6 <= hour < 12:
        return (
            "\n\nTime context: Morning. The user is likely starting their day. "
            "If they ask 'what should I work on?', prioritize a quick task review "
            "and suggest tackling high-priority items while energy is fresh."
        )
    elif 12 <= hour < 14:
        return (
            "\n\nTime context: Midday. Good time for quick updates and status checks. "
            "Keep responses concise — the user may be context-switching."
        )
    elif 14 <= hour < 18:
        return (
            "\n\nTime context: Afternoon. If the user is wrapping up, suggest "
            "reviewing what was accomplished today and planning tomorrow's priorities."
        )
    else:
        return (
            "\n\nTime context: Evening/night. The user is working outside normal hours. "
            "Keep interactions efficient. Suggest capturing quick notes and deferring "
            "non-urgent planning to tomorrow."
        )
