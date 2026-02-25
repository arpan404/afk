from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECTS_ROOT = ROOT / "examples" / "projects"
VALIDATION_PATH = PROJECTS_ROOT / "_validation.py"


def _load_validation_module():
    spec = importlib.util.spec_from_file_location("examples_projects_validation", VALIDATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_example_projects_validator_passes() -> None:
    validation = _load_validation_module()
    report = validation.validate_projects(PROJECTS_ROOT)
    assert report["errors"] == [], "\n".join(report["errors"])


def test_example_projects_count_and_indices() -> None:
    validation = _load_validation_module()
    projects = validation.discover_projects(PROJECTS_ROOT)
    indices = [project.index for project in projects]
    assert len(projects) == 34
    assert indices == list(range(1, 35))
