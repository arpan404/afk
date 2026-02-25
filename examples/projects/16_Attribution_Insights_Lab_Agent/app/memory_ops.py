"""SQLite memory helpers for tier-4+ examples."""

from pathlib import Path

from afk.memory import SQLiteMemoryStore


MEMORY_FILE_NAME = "memory.sqlite3"


def build_sqlite_store(project_root: Path) -> SQLiteMemoryStore:
    """Build sqlite memory store under project data directory."""
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return SQLiteMemoryStore(path=str(data_dir / MEMORY_FILE_NAME))
