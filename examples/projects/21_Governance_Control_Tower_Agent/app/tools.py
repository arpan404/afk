"""Tool definitions for the workflow."""

from afk.tools import tool

from .models import PlanArgs, PortfolioArgs, ScenarioArgs, SignalArgs, SnapshotArgs


@tool(
    args_model=SnapshotArgs,
    name="fetch_operational_snapshot",
    description="Fetch a deterministic operational snapshot for an entity.",
)
def fetch_operational_snapshot(args: SnapshotArgs) -> dict:
    """Return deterministic snapshot values for reproducible examples."""
    seed = sum(ord(char) for char in args.entity)
    load = round(35 + (seed % 45), 2)
    error_rate = round(((seed % 11) + 1) / 100, 3)
    latency_ms = round(120 + (seed % 650), 2)
    return {
        "entity": args.entity,
        "load_pct": load,
        "error_rate": error_rate,
        "latency_ms": latency_ms,
    }


@tool(
    args_model=SignalArgs,
    name="compute_trend_signal",
    description="Compute trend and risk signal from baseline and current values.",
)
def compute_trend_signal(args: SignalArgs) -> dict:
    """Compute a simple trend/risk signal for model reasoning."""
    delta = args.current - args.baseline
    pct = 0.0 if args.baseline == 0 else (delta / args.baseline) * 100
    risk_score = max(0, min(100, int(abs(pct) * 2)))
    direction = "up" if pct > 0 else "down" if pct < 0 else "flat"
    return {
        "entity": args.entity,
        "delta": round(delta, 4),
        "percent_change": round(pct, 2),
        "direction": direction,
        "risk_score": risk_score,
    }


@tool(
    args_model=PlanArgs,
    name="build_action_plan",
    description="Build an action plan from an entity risk score.",
)
def build_action_plan(args: PlanArgs) -> dict:
    """Return action plan detail for moderate/high complexity examples."""
    priority = "critical" if args.risk_score >= 70 else "high" if args.risk_score >= 40 else "normal"
    sla_hours = 2 if args.risk_score >= 70 else 8 if args.risk_score >= 40 else 24
    return {
        "entity": args.entity,
        "priority": priority,
        "target_sla_hours": sla_hours,
        "owner_role": "incident_commander" if args.risk_score >= 70 else "operations_lead",
    }


@tool(
    args_model=ScenarioArgs,
    name="simulate_scenario",
    description="Simulate stressed scenario outcomes for an entity.",
)
def simulate_scenario(args: ScenarioArgs) -> dict:
    """Return synthetic stressed outputs for advanced examples."""
    impact = round(args.shock_factor * 12.5, 2)
    recovery_hours = round(max(1.0, 18.0 / args.shock_factor), 2)
    return {
        "entity": args.entity,
        "shock_factor": args.shock_factor,
        "impact_index": impact,
        "estimated_recovery_hours": recovery_hours,
    }


@tool(
    args_model=PortfolioArgs,
    name="aggregate_portfolio_risk",
    description="Aggregate risk signals across many entities.",
)
def aggregate_portfolio_risk(args: PortfolioArgs) -> dict:
    """Aggregate portfolio-level metrics for governance-grade examples."""
    weights = [sum(ord(ch) for ch in row) % 100 for row in args.entities]
    gross_risk = sum(weights)
    normalized = round(gross_risk / max(1, len(args.entities)), 2)
    concentration = round(max(weights) / max(1, sum(weights)), 3)
    return {
        "entity_count": len(args.entities),
        "gross_risk": gross_risk,
        "avg_risk": normalized,
        "concentration_ratio": concentration,
    }


def build_tools(tier: int) -> list:
    """Return progressively richer toolsets by tier."""
    tools = [fetch_operational_snapshot, compute_trend_signal]
    if tier >= 2:
        tools.append(build_action_plan)
    if tier >= 4:
        tools.append(simulate_scenario)
    if tier >= 5:
        tools.append(aggregate_portfolio_risk)
    return tools
