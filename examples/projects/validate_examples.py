"""CLI validator for example progression and AFK feature coverage."""

from __future__ import annotations

import json
from pathlib import Path

from _validation import validate_projects


def main() -> int:
    projects_root = Path(__file__).resolve().parent
    report = validate_projects(projects_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
