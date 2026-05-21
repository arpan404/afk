#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="${AFK_DOCS_DIR:-$ROOT_DIR/docs}"
SKILLS_DIR="${AFK_AGENT_SKILLS_DIR:-$ROOT_DIR/skills}"
OUT_DIR="${AFK_AI_INDEX_DIR:-$ROOT_DIR/ai-index}"
BUNDLE_SKILL_DOCS=true

usage() {
  cat <<'EOF'
Build AFK AI assets:
1) creates searchable index files from docs/
2) creates/refreshes skill metadata index in skills/
3) bundles docs assets into each skill folder (optional)

Usage:
  scripts/build_agentic_ai_assets.sh [--no-bundle-skill-docs]

Environment overrides:
  AFK_DOCS_DIR
  AFK_AGENT_SKILLS_DIR
  AFK_AI_INDEX_DIR
EOF
}

while (($#)); do
  case "$1" in
    --no-bundle-skill-docs)
      BUNDLE_SKILL_DOCS=false
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ ! -d "$DOCS_DIR" ]]; then
  echo "Docs directory not found: $DOCS_DIR" >&2
  exit 1
fi

if [[ ! -d "$SKILLS_DIR" ]]; then
  echo "Skills directory not found: $SKILLS_DIR" >&2
  echo "Create skill folders first under: $SKILLS_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/text" "$OUT_DIR/records"

python3 - "$DOCS_DIR" "$OUT_DIR" "$SKILLS_DIR" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

docs_dir = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
skills_dir = Path(sys.argv[3]).resolve()
snippets_dir = docs_dir / "library" / "snippets"

stopwords = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "if", "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "to", "with", "this", "these", "those", "you", "your", "we", "our",
}


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return {}, text
    raw = parts[1]
    body = parts[2]
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body


def to_doc_url(rel_path: str) -> str:
    rel = rel_path.replace("\\", "/")
    if rel == "index.mdx":
        return "/"
    if rel.endswith("/index.mdx"):
        return "/" + rel[:-10] + "/"
    if rel.endswith(".mdx"):
        return "/" + rel[:-4]
    if rel.endswith(".md"):
        return "/" + rel[:-3]
    if rel.endswith(".json"):
        return "/" + rel
    return "/" + rel


def plain_text(content: str) -> str:
    content = re.sub(r"```[\s\S]*?```", " ", content)
    content = re.sub(r"`([^`]*)`", r"\1", content)
    content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
    content = re.sub(r"<[^>]+>", " ", content)
    content = re.sub(r"^#{1,6}\s*", "", content, flags=re.M)
    content = re.sub(r"\s+", " ", content)
    return content.strip()


def tokenize(content: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,40}", content.lower())
    return [w for w in words if w not in stopwords]


def read_skill_meta(skill_md_path: Path) -> tuple[str, str]:
    text = skill_md_path.read_text(encoding="utf-8")
    meta, _ = parse_frontmatter(text)
    name = meta.get("name", skill_md_path.parent.name)
    description = meta.get("description", "")
    return name, description


def parse_snippet(path: Path) -> tuple[str, str]:
    """
    Parse snippet files formatted as:
    ````python file_name.py
    <code>
    ````
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return path.stem, ""

    first = lines[0].strip()
    file_name = path.stem
    if first.startswith("````"):
        header = first[4:].strip()
        if header:
            parts = header.split(maxsplit=1)
            if len(parts) == 2:
                file_name = parts[1].strip()

    end_idx = None
    for idx in range(len(lines) - 1, 0, -1):
        if lines[idx].strip() == "````":
            end_idx = idx
            break

    if end_idx is None:
        body = "\n".join(lines)
    else:
        body = "\n".join(lines[1:end_idx])

    return file_name, body.strip("\n")


doc_candidates = sorted(
    [
        p
        for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix in {".mdx", ".md", ".json"}
    ]
)

documents = []
inverted: dict[str, set[str]] = defaultdict(set)
path_to_id: dict[str, str] = {}
id_to_path: dict[str, str] = {}

for path in doc_candidates:
    rel = path.relative_to(docs_dir).as_posix()
    if rel.startswith(("logo", "favicon")) and path.suffix == ".svg":
        continue

    raw = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)
    title = meta.get("title") or path.stem
    description = meta.get("description", "")
    headings = re.findall(r"^#{1,6}\s+(.+)$", body, flags=re.M)
    text = plain_text(body if path.suffix != ".json" else raw)
    doc_id = "doc_" + hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    url = to_doc_url(rel)
    tokens = tokenize(text)

    for token in set(tokens):
        inverted[token].add(doc_id)

    payload = {
        "id": doc_id,
        "path": f"docs/{rel}",
        "url": url,
        "title": title,
        "description": description,
        "headings": headings,
        "content_sha256": sha,
        "content": text,
        "token_count": len(tokens),
    }
    documents.append(payload)
    path_to_id[payload["path"]] = doc_id
    id_to_path[doc_id] = payload["path"]

    (out_dir / "text" / f"{doc_id}.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / "records" / f"{doc_id}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

documents.sort(key=lambda x: x["path"])

index_payload = {
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "docs_root": str(docs_dir),
    "document_count": len(documents),
    "documents": documents,
}

(out_dir / "docs-index.json").write_text(
    json.dumps(index_payload, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)

with (out_dir / "docs-index.jsonl").open("w", encoding="utf-8") as f:
    for doc in documents:
        f.write(json.dumps(doc, ensure_ascii=True) + "\n")

inverted_payload = {
    token: sorted(ids)
    for token, ids in sorted(inverted.items(), key=lambda item: item[0])
}
(out_dir / "inverted-index.json").write_text(
    json.dumps(inverted_payload, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)

(out_dir / "path-to-id.json").write_text(
    json.dumps(path_to_id, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)

(out_dir / "id-to-path.json").write_text(
    json.dumps(id_to_path, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)

skill_entries = []
for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
    name, description = read_skill_meta(skill_md)
    rel_path = skill_md.relative_to(skills_dir).as_posix()
    skill_id = "skill_" + hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:10]
    skill_entries.append(
        {
            "id": skill_id,
            "name": name,
            "description": description,
            "path": f"skills/{rel_path}",
            "github_url": f"https://github.com/arpan404/afk/tree/main/skills/{skill_md.parent.name}",
        }
    )

(skills_dir / "index.json").write_text(
    json.dumps(
        {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "skills": skill_entries,
        },
        indent=2,
        ensure_ascii=True,
    )
    + "\n",
    encoding="utf-8",
)

manifest = {
    "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "document_count": len(documents),
    "skill_count": len(skill_entries),
    "files": [
        "docs-index.json",
        "docs-index.jsonl",
        "inverted-index.json",
        "path-to-id.json",
        "id-to-path.json",
        "examples.md",
        "records/",
        "text/",
        "../skills/index.json",
    ],
}
(out_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
    encoding="utf-8",
)

snippet_paths = sorted(snippets_dir.glob("*.mdx"))
examples_lines = [
    "# AFK Examples",
    "",
    "Merged from `docs/library/snippets/*.mdx` for agent-friendly context loading.",
    "",
    f"Generated at: {dt.datetime.now(dt.timezone.utc).isoformat()}",
    "",
]

for snippet_path in snippet_paths:
    file_name, code = parse_snippet(snippet_path)
    rel = snippet_path.relative_to(docs_dir).as_posix()
    examples_lines.append(f"## {file_name}")
    examples_lines.append("")
    examples_lines.append(f"Source: `docs/{rel}`")
    examples_lines.append("")
    examples_lines.append("```python")
    examples_lines.append(code)
    examples_lines.append("```")
    examples_lines.append("")

examples_md = "\n".join(examples_lines).strip() + "\n"
(out_dir / "examples.md").write_text(examples_md, encoding="utf-8")
(skills_dir / "examples.md").write_text(examples_md, encoding="utf-8")

print(f"Indexed {len(documents)} docs files into: {out_dir}")
print(f"Indexed {len(skill_entries)} skills into: {skills_dir / 'index.json'}")
print(f"Merged {len(snippet_paths)} snippets into: {out_dir / 'examples.md'}")
PY

to_title_case() {
  # shellcheck disable=SC2001
  echo "$1" | sed 's/-/ /g' | awk '{for (i=1;i<=NF;i++){$i=toupper(substr($i,1,1)) tolower(substr($i,2))} print}'
}

write_search_script() {
  local skill_dir="$1"
  local script_path="$skill_dir/scripts/search_afk_docs.py"
  cat > "$script_path" <<'PY'
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
PY
  chmod +x "$script_path"
}

write_openai_yaml() {
  local skill_dir="$1"
  local skill_slug="$2"
  local title
  title="$(to_title_case "$skill_slug")"
  cat > "$skill_dir/agents/openai.yaml" <<EOF
interface:
  display_name: "${title}"
  short_description: "AFK skill for ${title} workflows."
  default_prompt: "Use \$${skill_slug} to complete this AFK task with the packaged guidance."
EOF
}

bundle_docs_into_skills() {
  for skill_dir in "$SKILLS_DIR"/*; do
    [[ -d "$skill_dir" ]] || continue
    [[ -f "$skill_dir/SKILL.md" ]] || continue

    local skill_slug
    skill_slug="$(basename "$skill_dir")"
    local refs_dir="$skill_dir/references/afk-docs"
    mkdir -p "$refs_dir" "$skill_dir/scripts" "$skill_dir/agents"

    cp "$OUT_DIR/docs-index.json" "$refs_dir/docs-index.json"
    cp "$OUT_DIR/docs-index.jsonl" "$refs_dir/docs-index.jsonl"
    cp "$OUT_DIR/inverted-index.json" "$refs_dir/inverted-index.json"
    cp "$OUT_DIR/id-to-path.json" "$refs_dir/id-to-path.json"
    cp "$OUT_DIR/path-to-id.json" "$refs_dir/path-to-id.json"
    cp "$OUT_DIR/manifest.json" "$refs_dir/manifest.json"
    cp "$OUT_DIR/examples.md" "$skill_dir/references/afk-examples.md"

    cat > "$skill_dir/references/README.md" <<EOF
# Bundled AFK Docs Index

This folder is generated by:

\`\`\`bash
./scripts/build_agentic_ai_assets.sh
\`\`\`

It keeps machine-readable AFK docs inside this skill package for agent runtimes that expect docs to live with \`SKILL.md\`.

Files:
- \`afk-docs/docs-index.jsonl\`: searchable doc records
- \`afk-docs/inverted-index.json\`: token -> doc id map
- \`afk-docs/id-to-path.json\`: doc id -> docs path
- \`afk-examples.md\`: merged runnable examples from snippets

Quick search:

\`\`\`bash
python scripts/search_afk_docs.py "system prompts"
\`\`\`
EOF

    write_search_script "$skill_dir"

    if [[ ! -f "$skill_dir/agents/openai.yaml" ]]; then
      write_openai_yaml "$skill_dir" "$skill_slug"
    fi
  done
}

if $BUNDLE_SKILL_DOCS; then
  bundle_docs_into_skills
  echo "Bundled docs index into skill folders under: $SKILLS_DIR"
fi

echo "Done."
echo "Docs index: $OUT_DIR/docs-index.json"
echo "Fast search file set: $OUT_DIR/inverted-index.json and $OUT_DIR/text/*.txt"
echo "Merged examples: $OUT_DIR/examples.md"
