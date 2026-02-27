#!/usr/bin/env bash
# Setup an example project for local development
# - checks for `uv` and uses `uv sync` when available
# - falls back to creating a local `.venv` and installing the repo with `pip install -e` inside it
# - idempotent and has useful flags: --project-dir, --python, --force, --no-run, --install-uv

set -euo pipefail
IFS=$'\n\t'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PROJECT_RELPATH="examples/projects/01_Greeting_Agent"
PROJECT_DIR=""
PYTHON_EXEC=""
FORCE=false
NO_RUN=false
INSTALL_UV=false
QUIET=false
SKIP_INSTALL=false
SKIP_DEPS=false
VENV_PATH=""
EXTRAS=""
MAIN_ARGS=""
RECREATE_ONLY=false

log() { printf "[setup_example] %s\n" "$*"; }
err() { printf "[setup_example] ERROR: %s\n" "$*" >&2; }
usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --project-dir=PATH       Path (relative to repo or absolute) to example project (default: ${DEFAULT_PROJECT_RELPATH})
  --python=PYTHON          Python executable to use (eg. python3.13)
  --force                  Recreate environment even if it exists
  --install-uv             Try to install 'uv' via pip if missing
  --skip-install           Skip installing the repository into the example environment
  --skip-deps              Do not install the project's requirements.txt
  --venv-path=PATH         Use a custom virtualenv path (absolute or repo-relative)
  --extras=NAME[,NAME]     Install pip extras when installing the repo (e.g. a2a)
  --main-args="..."       Pass arguments to the example when it is run (quoted string)
  --recreate-only          Only create/recreate the environment, then exit
  --no-run                 Do not attempt to run the example after setup
  --quiet                  Reduce output
  -h, --help               Show this help

Examples:
  $0 --project-dir=examples/projects/01_Greeting_Agent
  $0 --project-dir=01_Greeting_Agent --python=python3.13 --force --venv-path=.venv_shared
  $0 --extras=a2a --main-args="--port 8080"

EOF
  exit 1
} 

# Parse args (simple long-option parser)
for arg in "$@"; do
  case $arg in
    --project-dir=*) PROJECT_DIR="${arg#*=}"; shift || true ;;
    --python=*) PYTHON_EXEC="${arg#*=}"; shift || true ;;
    --force) FORCE=true; shift || true ;;
    --no-run) NO_RUN=true; shift || true ;;
    --install-uv) INSTALL_UV=true; shift || true ;;
    --skip-install) SKIP_INSTALL=true; shift || true ;;
    --skip-deps) SKIP_DEPS=true; shift || true ;;
    --venv-path=*) VENV_PATH="${arg#*=}"; shift || true ;;
    --extras=*) EXTRAS="${arg#*=}"; shift || true ;;
    --main-args=*) MAIN_ARGS="${arg#*=}"; shift || true ;;
    --recreate-only) RECREATE_ONLY=true; shift || true ;;
    --quiet) QUIET=true; shift || true ;;
    -h|--help) usage ;;
    *) err "Unknown option: $arg"; usage ;;
  esac
done

[ "$QUIET" = true ] || log "repo root: $REPO_ROOT"

# Default project dir
if [ -z "$PROJECT_DIR" ]; then
  PROJECT_DIR="$REPO_ROOT/$DEFAULT_PROJECT_RELPATH"
elif [[ "$PROJECT_DIR" = /* ]]; then
  # absolute -- use as-is
  :
else
  # treat as repo-relative
  PROJECT_DIR="$REPO_ROOT/$PROJECT_DIR"
fi

if [ ! -d "$PROJECT_DIR" ]; then
  err "project directory not found: $PROJECT_DIR"
  exit 2
fi

# find python executable
if [ -n "$PYTHON_EXEC" ]; then
  if ! command -v "$PYTHON_EXEC" >/dev/null 2>&1; then
    err "specified python not found: $PYTHON_EXEC"
    exit 3
  fi
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_EXEC=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_EXEC=python
  else
    err "no python executable found in PATH"
    exit 4
  fi
fi

# check python version (warn if < required)
PY_VER=$($PYTHON_EXEC -c 'import sys; print("%d.%d" % (sys.version_info[0], sys.version_info[1]))') || true
REQUIRED_PY="3.13"
PY_OK=false
if $PYTHON_EXEC -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
  # compare using Python for reliability
  PY_OK=$($PYTHON_EXEC - <<PYCODE
from packaging.version import Version
import sys
ok = Version(sys.version.split()[0]) >= Version("$REQUIRED_PY")
print("true" if ok else "false")
PYCODE
) || true
fi

if [ "$PY_OK" != "true" ]; then
  log "warning: recommended Python >= $REQUIRED_PY (found $PY_VER). the script will still try to continue."
fi

# helper: run command and show friendly error
run_check() {
  if ! "$@"; then
    err "command failed: $*"
    return 1
  fi
}

# detect `uv`
if command -v uv >/dev/null 2>&1; then
  UV_CMD="$(command -v uv)"
  [ "$QUIET" = true ] || log "found uv: $UV_CMD"
else
  UV_CMD=""
  if [ "$INSTALL_UV" = true ]; then
    log "'uv' not found — attempting to install via pip using $PYTHON_EXEC"
    run_check "$PYTHON_EXEC" -m pip install --upgrade uv || {
      err "failed to install 'uv' via pip; please install it manually (pipx or pip) and re-run"
    }
    if command -v uv >/dev/null 2>&1; then
      UV_CMD="$(command -v uv)"
      [ "$QUIET" = true ] || log "installed uv: $UV_CMD"
    fi
  fi
fi

# Primary strategy when `uv` is available: use `uv sync` (if supported), then ensure
# repository is installed into the example env via `uv pip install`.
USE_UV=false
if [ -n "$UV_CMD" ]; then
  # check if 'sync' is supported by this uv installation
  if uv --help 2>&1 | grep -qi "\bsync\b"; then
    USE_UV=true
  elif uv --help 2>&1 | grep -qi "\brun\b"; then
    # 'run' exists — we'll use that to install the package into the project's venv
    USE_UV=true
  else
    log "found 'uv' but it doesn't advertise 'sync' or 'run' — falling back to venv method"
    USE_UV=false
  fi
fi

if [ -n "$VENV_PATH" ]; then
  if [[ "$VENV_PATH" = /* ]]; then
    VENV_DIR="$VENV_PATH"
  else
    VENV_DIR="$PROJECT_DIR/$VENV_PATH"
  fi
else
  VENV_DIR="$PROJECT_DIR/.venv"
fi

if [ "$USE_UV" = true ]; then
  log "using uv to prepare environment for project: $PROJECT_DIR"
  pushd "$PROJECT_DIR" >/dev/null
  if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$VENV_DIR" ]; then
    log "active virtualenv differs from target; unsetting VIRTUAL_ENV for uv commands"
    unset VIRTUAL_ENV
  fi
  # try uv sync if available
  if uv --help 2>&1 | grep -qi "\bsync\b"; then
    log "running: uv sync"
    if ! uv sync; then
      err "'uv sync' failed — will fall back to creating a local venv"
      USE_UV=false
    fi
  fi

  if [ "$USE_UV" = true ]; then
    # ensure the repository package is installed into the project's environment
    log "installing repository into the example project's environment using 'uv pip install -e'"
    if [ "$SKIP_INSTALL" = true ]; then
      log "skipping repo install into uv environment (--skip-install)"
    else
      REPO_INSTALL_ARG="$REPO_ROOT"
      if [ -n "$EXTRAS" ]; then REPO_INSTALL_ARG="$REPO_ROOT[$EXTRAS]"; fi
      if ! uv pip install --python "$VENV_DIR/bin/python" -e "$REPO_INSTALL_ARG"; then
        err "failed to install repository into uv environment — continuing to fallback path"
        USE_UV=false
      fi
    fi
  fi
  popd >/dev/null
fi

# If uv not usable, create/use a local .venv inside the example project
if [ "$USE_UV" != true ]; then
  log "setting up a local virtualenv at $VENV_DIR"

  if [ -d "$VENV_DIR" ] && [ "$FORCE" = true ]; then
    log "--force specified; removing existing venv"
    rm -rf "$VENV_DIR"
  fi

  if [ ! -d "$VENV_DIR" ]; then
    log "creating venv with $PYTHON_EXEC"
    run_check "$PYTHON_EXEC" -m venv "$VENV_DIR"
  else
    log "reusing existing venv: $VENV_DIR"
  fi

  PY_BIN="$VENV_DIR/bin/python"

  # uv-created venvs may exist without pip; recreate to recover a standard layout.
  if [ ! -x "$VENV_DIR/bin/pip" ]; then
    log "venv exists but pip is missing; recreating virtualenv"
    rm -rf "$VENV_DIR"
    run_check "$PYTHON_EXEC" -m venv "$VENV_DIR"
  fi

  if ! "$PY_BIN" -m pip --version >/dev/null 2>&1; then
    log "pip module missing inside venv; bootstrapping with ensurepip"
    run_check "$PY_BIN" -m ensurepip --upgrade
  fi

  # upgrade pip and tooling
  log "upgrading pip and build tools inside venv"
  run_check "$PY_BIN" -m pip install --upgrade pip setuptools wheel

  # install repo in editable mode (unless --skip-install)
  if [ "$SKIP_INSTALL" = true ]; then
    log "skipping repository install into venv (--skip-install)"
  else
    REPO_INSTALL_ARG="$REPO_ROOT"
    if [ -n "$EXTRAS" ]; then REPO_INSTALL_ARG="$REPO_ROOT[$EXTRAS]"; fi
    log "installing repository into the example venv (editable${EXTRAS:+ with extras [$EXTRAS]})"
    run_check "$PY_BIN" -m pip install -e "$REPO_INSTALL_ARG"
  fi

  # if example has requirements, install them (unless --skip-deps)
  if [ "$SKIP_DEPS" != true ] && [ -f "$PROJECT_DIR/requirements.txt" ]; then
    log "installing project requirements.txt"
    run_check "$PY_BIN" -m pip install -r "$PROJECT_DIR/requirements.txt"
  elif [ "$SKIP_DEPS" = true ]; then
    log "skipping project requirements (--skip-deps)"
  fi
fi

log "setup completed for project: $PROJECT_DIR"

if [ "$RECREATE_ONLY" = true ]; then
  [ "$QUIET" = true ] || log "--recreate-only specified; environment is ready"
  exit 0
fi

if [ "$NO_RUN" = true ]; then
  [ "$QUIET" = true ] || log "skipping example run (--no-run provided)"
  exit 0
fi

# Try to run the example to verify
if [ "$USE_UV" = true ]; then
  log "running example with uv: uv run main.py $MAIN_ARGS"
  (cd "$PROJECT_DIR" && uv run main.py $MAIN_ARGS)
else
  if [ -x "$VENV_DIR/bin/python" ]; then
    log "running example using venv python: $VENV_DIR/bin/python main.py $MAIN_ARGS"
    (cd "$PROJECT_DIR" && "$VENV_DIR/bin/python" main.py $MAIN_ARGS)
  else
    log "cannot find python in venv — printing activation instructions instead"
    echo
    echo "Activate the example venv and run the project manually:"
    echo "  source $VENV_DIR/bin/activate"
    echo "  python main.py"
    exit 0
  fi
fi

exit 0
