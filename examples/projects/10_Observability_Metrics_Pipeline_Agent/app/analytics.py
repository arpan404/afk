"""Analytics helpers for run, stream, and batch reporting."""

from afk.agents import AgentResult

from .models import RunDigest


def digest_result(result: AgentResult) -> RunDigest:
    """Build compact digest from AgentResult."""
    return RunDigest(
        state=result.state,
        total_tokens=result.usage_aggregate.total_tokens,
        tool_calls=len(result.tool_executions),
        subagent_calls=len(result.subagent_executions),
        total_cost_usd=result.total_cost_usd,
    )


def summarize_tool_latency(result: AgentResult) -> dict[str, float | int | None]:
    """Compute basic tool latency metrics."""
    latencies = [row.latency_ms for row in result.tool_executions if row.latency_ms is not None]
    if not latencies:
        return {
            "tool_call_count": len(result.tool_executions),
            "avg_latency_ms": None,
            "max_latency_ms": None,
        }
    return {
        "tool_call_count": len(result.tool_executions),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
    }


def summarize_policy_events(policy_events: list[dict]) -> dict[str, int]:
    """Summarize policy event actions."""
    counts = {"allow": 0, "deny": 0, "request_approval": 0, "request_user_input": 0, "defer": 0}
    for row in policy_events:
        action = row.get("action")
        if action in counts:
            counts[action] += 1
    return counts


def summarize_batch(digests: list[RunDigest]) -> dict[str, float | int]:
    """Aggregate a set of run digests."""
    if not digests:
        return {
            "runs": 0,
            "completed_runs": 0,
            "completion_rate_pct": 0.0,
            "total_tokens": 0,
            "avg_tokens_per_run": 0.0,
            "total_tool_calls": 0,
            "total_subagent_calls": 0,
        }

    completed = sum(1 for row in digests if row.state == "completed")
    total_tokens = sum(row.total_tokens for row in digests)
    total_tool_calls = sum(row.tool_calls for row in digests)
    total_subagent_calls = sum(row.subagent_calls for row in digests)

    return {
        "runs": len(digests),
        "completed_runs": completed,
        "completion_rate_pct": round((completed / len(digests)) * 100, 2),
        "total_tokens": total_tokens,
        "avg_tokens_per_run": round(total_tokens / len(digests), 2),
        "total_tool_calls": total_tool_calls,
        "total_subagent_calls": total_subagent_calls,
    }
