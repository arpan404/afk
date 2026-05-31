#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR/docs"

# Mintlify currently rejects Node 25+. Use an LTS runtime even when the
# globally active Node is newer.
exec npx -y -p node@22 -p mintlify@4.2.370 mintlify dev "$@"
