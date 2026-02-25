"""Streaming helpers for tier-2 examples."""

from afk.core import Runner


def blank_stream_counters() -> dict[str, int]:
    """Initialize stream counters."""
    return {
        "text_delta_chunks": 0,
        "tool_started": 0,
        "tool_completed": 0,
        "errors": 0,
    }


async def stream_once(*, runner: Runner, agent, prompt: str, thread_id: str) -> tuple[dict[str, int], object]:
    """Run one streaming turn and return stream counters + final result."""
    counters = blank_stream_counters()
    handle = await runner.run_stream(agent, user_message=prompt, thread_id=thread_id)

    async for event in handle:
        if event.type == "text_delta" and event.text_delta:
            counters["text_delta_chunks"] += 1
            print(event.text_delta, end="", flush=True)
        elif event.type == "tool_started":
            counters["tool_started"] += 1
        elif event.type == "tool_completed":
            counters["tool_completed"] += 1
        elif event.type == "error":
            counters["errors"] += 1

    if handle.result is None:
        raise RuntimeError("Streaming run completed without terminal AgentResult")
    return counters, handle.result
