"""Workflow entry for build_skill_tools policy-governed execution."""

import asyncio
from pathlib import Path

from afk.agents import SkillRef, SkillToolPolicy
from afk.tools import ToolContext, ToolRegistry, build_skill_tools

from .complexity_chain import run_chain
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .report_builder import build_report
from .scenario import build_scenario


async def _run() -> None:
    scenario = build_scenario("skills-runtime-policy")
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    project_root = Path(__file__).resolve().parents[1]
    skill_root = project_root / "skills" / "customer_playbook"

    skill_ref = SkillRef(
        name="customer-playbook",
        description="Triage and recover enterprise customer escalations.",
        root_dir=str((project_root / "skills").resolve()),
        skill_md_path=str((skill_root / "SKILL.md").resolve()),
    )

    policy = SkillToolPolicy(
        command_allowlist=["echo"],
        deny_shell_operators=True,
        command_timeout_s=5.0,
    )

    tools = build_skill_tools(skills=[skill_ref], policy=policy)
    registry = ToolRegistry()
    registry.register_many(tools)

    listed = await registry.call("list_skills", {}, ctx=ToolContext(request_id="req-31"))
    read_md = await registry.call("read_skill_md", {"skill_name": "customer-playbook"})
    read_file = await registry.call(
        "read_skill_file",
        {
            "skill_name": "customer-playbook",
            "relative_path": "references/escalation_checklist.txt",
        },
    )
    run_cmd = await registry.call(
        "run_skill_command",
        {
            "command": "echo",
            "args": ["skill-command-ok"],
        },
    )

    feature_payload: dict[str, object] = {
        "kind": "skill_tools",
        "status": "ok"
        if listed.success and read_md.success and read_file.success and run_cmd.success
        else "error",
        "skills_count": len((listed.output or {}).get("skills", []))
        if isinstance(listed.output, dict)
        else 0,
        "command_exit": (run_cmd.output or {}).get("exit_code")
        if isinstance(run_cmd.output, dict)
        else None,
    }

    validate_feature_payload(feature_payload)
    metrics = compute_metrics(scenario, chain_state, feature_payload)
    report = build_report(
        feature_payload=feature_payload,
        chain_state=chain_state,
        metrics=metrics,
    )

    print("[skills] > report")
    print(report)


def run_example() -> None:
    asyncio.run(_run())
