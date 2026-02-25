"""Workspace bootstrap for runtime file examples."""

from pathlib import Path


def ensure_workspace(project_root: Path) -> Path:
    root = project_root / "workspace"
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "q1_pipeline.txt").write_text(
        (
            "Enterprise pipeline snapshot\n"
            "- Stage: legal review\n"
            "- Risk: procurement timing\n"
            "- Mitigation: exec sponsor alignment\n"
        ),
        encoding="utf-8",
    )
    return root
