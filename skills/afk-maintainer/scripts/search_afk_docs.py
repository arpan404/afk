#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def score(query_terms: list[str], doc: dict) -> int:
    text = (doc.get("title", "") + " " + doc.get("description", "") + " " + doc.get("content", "")).lower()
    return sum(text.count(term) for term in query_terms)


def main() -> int:
    parser = argparse.ArgumentParser(description="Search bundled AFK docs index for this skill.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--index",
        default=str(Path(__file__).resolve().parent.parent / "references" / "afk-docs" / "docs-index.jsonl"),
        help="Path to docs-index.jsonl",
    )
    args = parser.parse_args()

    query_terms = [t.lower() for t in args.query.split() if t.strip()]
    if not query_terms:
        print("Empty query.")
        return 1

    idx_path = Path(args.index)
    if not idx_path.exists():
        print(f"Index not found: {idx_path}")
        return 1

    rows = load_records(idx_path)
    ranked = sorted(
        ((score(query_terms, r), r) for r in rows),
        key=lambda x: x[0],
        reverse=True,
    )
    shown = 0
    for s, row in ranked:
        if s <= 0:
            continue
        print(f"[{row['id']}] {row.get('title', '')}")
        print(f"  url: {row.get('url', '')}")
        print(f"  path: {row.get('path', '')}")
        desc = row.get("description", "")
        if desc:
            print(f"  desc: {desc}")
        print()
        shown += 1
        if shown >= args.top_k:
            break

    if shown == 0:
        print("No matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
