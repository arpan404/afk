"""Validation utilities for example projects progression and AFK feature coverage."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_NAME_RE = re.compile(r"^(?P<index>\d{2})_(?P<slug>.+)$")

FEATURE_CHECKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "agent_runner_basics": (
        "01_Greeting_Agent",
        ("from afk.core import Runner", "from afk.agents import Agent"),
    ),
    "tool_decorator": ("02_Revenue_Lead_Scorer_Agent", ("from afk.tools import tool",)),
    "failsafe_controls": ("05_Change_Control_Assistant_Agent", ("FailSafeConfig",)),
    "streaming_runs": ("06_Streaming_NOC_Copilot_Agent", ("run_stream(",)),
    "memory_resume_compact": (
        "14_Sales_Forecast_Planner_Agent",
        ("SQLiteMemoryStore", "runner.resume(", "compact_thread("),
    ),
    "observability_telemetry": (
        "10_Observability_Metrics_Pipeline_Agent",
        ("RuntimeTelemetryCollector", "project_run_metrics_from_result"),
    ),
    "policy_governance": ("14_Sales_Forecast_Planner_Agent", ("PolicyEngine",)),
    "llm_builder": ("22_LLM_Builder_Strategy_Agent", ("LLMBuilder",)),
    "tool_hooks_middleware": (
        "23_Tool_Hooks_Middleware_Agent",
        ("@prehook(", "@middleware(", "@posthook(", "@registry_middleware("),
    ),
    "runtime_tools": ("24_Runtime_Sandbox_FileOps_Agent", ("build_runtime_tools",)),
    "sandbox_profiles": ("24_Runtime_Sandbox_FileOps_Agent", ("SandboxProfile",)),
    "prompt_loader": (
        "25_Prompt_Template_Loader_Agent",
        ("instruction_file=", "prompts_dir="),
    ),
    "interactive_mode": (
        "26_Interactive_Approval_Workflow_Agent",
        ('interaction_mode="interactive"', "InMemoryInteractiveProvider"),
    ),
    "deferred_tools": (
        "27_Background_Deferred_Pipeline_Agent",
        ("ToolDeferredHandle",),
    ),
    "fallback_chain": (
        "28_Fallback_Reasoning_Control_Agent",
        ("fallback_model_chain",),
    ),
    "reasoning_controls": (
        "28_Fallback_Reasoning_Control_Agent",
        ("reasoning_effort", "reasoning_max_tokens"),
    ),
    "eval_suite": (
        "29_Eval_Suite_Quality_Gates_Agent",
        ("run_suite(", "EvalCase(", "write_suite_report_json("),
    ),
    "mcp_loading": (
        "30_MCP_Remote_Tools_Exchange_Agent",
        ("MCPStore", "mcp_servers=["),
    ),
    "skills_runtime_tools": (
        "31_Skills_Runtime_Command_Policy_Agent",
        ("build_skill_tools",),
    ),
    "a2a_internal_protocol": (
        "32_A2A_Internal_Delegation_Protocol_Agent",
        ("InternalA2AProtocol",),
    ),
    "a2a_service_host_auth": (
        "33_A2A_Service_Host_Auth_Agent",
        ("A2AServiceHost", "APIKeyA2AAuthProvider"),
    ),
    "queues_worker_contracts": (
        "34_Queue_Worker_Execution_Contracts_Agent",
        ("TaskWorker", "ExecutionContractContext"),
    ),
}


@dataclass(frozen=True)
class ProjectInfo:
    index: int
    slug: str
    path: Path
    py_files: int

    @property
    def name(self) -> str:
        return f"{self.index:02d}_{self.slug}"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_int_assignment(path: Path, key: str) -> int | None:
    match = re.search(rf"^{re.escape(key)}\s*=\s*(\d+)\s*$", _read_text(path), re.MULTILINE)
    if not match:
        return None
    return int(match.group(1))


def discover_projects(projects_root: Path) -> list[ProjectInfo]:
    projects: list[ProjectInfo] = []
    for entry in sorted(projects_root.iterdir()):
        if not entry.is_dir():
            continue
        match = PROJECT_NAME_RE.match(entry.name)
        if not match:
            continue
        index = int(match.group("index"))
        slug = match.group("slug")
        py_files = sum(1 for _ in entry.rglob("*.py"))
        projects.append(ProjectInfo(index=index, slug=slug, path=entry, py_files=py_files))
    return sorted(projects, key=lambda item: item.index)


def _resolve_afk_module_source(repo_root: Path, module: str) -> Path | None:
    parts = module.split(".")
    rel = Path("src").joinpath(*parts)

    file_path = repo_root / f"{rel}.py"
    if file_path.exists():
        return file_path

    package_init = repo_root / rel / "__init__.py"
    if package_init.exists():
        return package_init
    return None


def _validate_afk_imports(
    project: ProjectInfo,
    repo_root: Path,
    errors: list[str],
) -> None:
    module_cache: dict[str, str] = {}
    for file_path in sorted(project.path.rglob("*.py")):
        tree = ast.parse(_read_text(file_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module or not node.module.startswith("afk"):
                continue
            module_source = _resolve_afk_module_source(repo_root, node.module)
            if module_source is None:
                errors.append(
                    f"{project.name}: cannot resolve source for `{node.module}` "
                    f"(from {file_path.relative_to(project.path)})"
                )
                continue
            module_text = module_cache.setdefault(node.module, _read_text(module_source))
            for alias in node.names:
                if alias.name == "*":
                    continue
                if re.search(rf"\b{re.escape(alias.name)}\b", module_text) is None:
                    errors.append(
                        f"{project.name}: `{alias.name}` not found in source for `{node.module}` "
                        f"(from {file_path.relative_to(project.path)})"
                    )


def _project_text(project: ProjectInfo) -> str:
    return "\n".join(_read_text(path) for path in sorted(project.path.rglob("*.py")))


def validate_projects(projects_root: Path) -> dict[str, Any]:
    repo_root = projects_root.parents[1]

    errors: list[str] = []
    projects = discover_projects(projects_root)
    by_index = {project.index: project for project in projects}
    by_name = {project.name: project for project in projects}

    expected_indices = list(range(1, 35))
    indices = [project.index for project in projects]
    if indices != expected_indices:
        errors.append(f"Project index sequence mismatch. expected={expected_indices} actual={indices}")

    if any("weather" in project.slug.lower() for project in projects):
        errors.append("Weather example still present in project folders.")
    if (projects_root / "02_Weather_Agent").exists():
        errors.append("Legacy `02_Weather_Agent` directory still exists.")

    pyproject = repo_root / "pyproject.toml"
    pyproject_text = _read_text(pyproject)
    if "examples/projects/02_Weather_Agent" in pyproject_text:
        errors.append("Root pyproject still references `examples/projects/02_Weather_Agent`.")
    if "cookbooks/01_greeting_agent" in pyproject_text:
        errors.append("Root pyproject still references deleted `cookbooks/01_greeting_agent`.")

    prev_segments = 0
    prev_pass_depth = 0
    core_progression: list[dict[str, int]] = []
    for index in range(2, 22):
        project = by_index.get(index)
        if project is None:
            errors.append(f"Missing project index {index:02d}.")
            continue
        config = project.path / "app" / "config.py"
        dataset = project.path / "app" / "dynamic_dataset.py"
        analytics = project.path / "app" / "progressive_analytics.py"
        if not config.exists() or not dataset.exists() or not analytics.exists():
            errors.append(f"{project.name}: missing progressive analytics files.")
            continue

        example_number = _extract_int_assignment(config, "EXAMPLE_NUMBER")
        if example_number != index:
            errors.append(
                f"{project.name}: EXAMPLE_NUMBER expected {index}, got {example_number}."
            )
            continue
        if "segment_count = 3 + max(0, EXAMPLE_NUMBER - 2)" not in _read_text(dataset):
            errors.append(f"{project.name}: dynamic segment formula was modified unexpectedly.")
        if "return max(3, EXAMPLE_NUMBER + 1)" not in _read_text(analytics):
            errors.append(f"{project.name}: progressive pass formula was modified unexpectedly.")

        segments = 3 + max(0, example_number - 2)
        pass_depth = max(3, example_number + 1)
        if segments <= prev_segments:
            errors.append(
                f"{project.name}: segments not strictly increasing ({segments} <= {prev_segments})."
            )
        if pass_depth <= prev_pass_depth:
            errors.append(
                f"{project.name}: pass depth not strictly increasing ({pass_depth} <= {prev_pass_depth})."
            )
        prev_segments = segments
        prev_pass_depth = pass_depth
        core_progression.append(
            {
                "index": index,
                "py_files": project.py_files,
                "segments": segments,
                "pass_depth": pass_depth,
            }
        )

    prev_stage_depth = 0
    prev_py_count = 0
    advanced_progression: list[dict[str, int]] = []
    for index in range(22, 35):
        project = by_index.get(index)
        if project is None:
            errors.append(f"Missing project index {index:02d}.")
            continue
        config = project.path / "app" / "complexity_config.py"
        if not config.exists():
            errors.append(f"{project.name}: missing complexity_config.py.")
            continue
        stage_depth = _extract_int_assignment(config, "STAGE_COUNT")
        expected_stage = index + 1
        if stage_depth != expected_stage:
            errors.append(
                f"{project.name}: STAGE_COUNT expected {expected_stage}, got {stage_depth}."
            )
            continue
        stage_files = len(list((project.path / "app" / "stages").glob("stage_*.py")))
        if stage_files != stage_depth:
            errors.append(
                f"{project.name}: stage module count mismatch ({stage_files} != {stage_depth})."
            )
        if stage_depth <= prev_stage_depth:
            errors.append(
                f"{project.name}: stage depth not strictly increasing "
                f"({stage_depth} <= {prev_stage_depth})."
            )
        if project.py_files <= prev_py_count:
            errors.append(
                f"{project.name}: python file count not strictly increasing "
                f"({project.py_files} <= {prev_py_count})."
            )
        prev_stage_depth = stage_depth
        prev_py_count = project.py_files
        advanced_progression.append(
            {
                "index": index,
                "py_files": project.py_files,
                "stage_depth": stage_depth,
            }
        )

    feature_status: dict[str, dict[str, Any]] = {}
    for feature, (project_name, patterns) in FEATURE_CHECKS.items():
        project = by_name.get(project_name)
        if project is None:
            errors.append(f"Feature `{feature}` references missing project `{project_name}`.")
            continue
        text = _project_text(project)
        missing = [pattern for pattern in patterns if pattern not in text]
        if missing:
            errors.append(
                f"{project.name}: missing feature markers for `{feature}` -> {missing}"
            )
        feature_status[feature] = {
            "project": project_name,
            "patterns": list(patterns),
            "ok": not missing,
        }

    for project in projects:
        _validate_afk_imports(project, repo_root, errors)

    return {
        "project_count": len(projects),
        "indices": indices,
        "core_progression": core_progression,
        "advanced_progression": advanced_progression,
        "feature_coverage": feature_status,
        "errors": errors,
    }
