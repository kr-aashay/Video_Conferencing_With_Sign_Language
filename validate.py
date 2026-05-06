"""
validate.py
══════════════════════════════════════════════════════════════════════════════
Python-based project validation script

Checks that every required file and folder exists, all Python modules
import cleanly, and the JS SDK files are syntactically valid.

Usage
─────
    python validate.py          # full check
    python validate.py --quick  # file existence only (no imports)

Exit codes
──────────
    0  all checks passed
    1  one or more checks failed
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

# ── Colour helpers (no deps) ──────────────────────────────────────────────────
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"

def _ok(msg):   print(f"  {_GREEN}✓{_RESET}  {msg}")
def _warn(msg): print(f"  {_YELLOW}⚠{_RESET}  {msg}")
def _fail(msg): print(f"  {_RED}✗{_RESET}  {msg}")
def _head(msg): print(f"\n{_BOLD}{msg}{_RESET}")


# ── Required structure ────────────────────────────────────────────────────────

REQUIRED_FILES = [
    # API
    "api/__init__.py",
    "api/app.py",
    "api/config.py",
    "api/inference.py",
    "api/logger.py",
    "api/buffer_manager.py",
    "api/stream_buffer.py",
    # Model
    "cslr_model/__init__.py",
    "cslr_model/dataset.py",
    "cslr_model/decoder.py",
    "cslr_model/metrics.py",
    "cslr_model/model.py",
    "cslr_model/predict.py",
    "cslr_model/trainer.py",
    "cslr_model/export.py",
    # Frontend
    "frontend/sender.html",
    "frontend/receiver.html",
    "frontend/receiver.css",
    "frontend/sdk/holistic-bridge.js",
    "frontend/sdk/skeleton-renderer.js",
    # Orchestration
    "run.py",
    "train.py",
    "orchestrate.py",
    "debug_dashboard.py",
    "validate.py",
    # Config
    ".env.example",
    "requirements.txt",
    "pytest.ini",
]

REQUIRED_DIRS = [
    "api", "cslr_model", "frontend", "frontend/sdk", "tests",
]

PYTHON_MODULES = [
    "api.app",
    "api.config",
    "api.inference",
    "api.logger",
    "api.buffer_manager",
    "api.stream_buffer",
    "cslr_model.dataset",
    "cslr_model.decoder",
    "cslr_model.metrics",
    "cslr_model.model",
    "cslr_model.predict",
    "cslr_model.trainer",
]

JS_FILES = [
    "frontend/sdk/holistic-bridge.js",
    "frontend/sdk/skeleton-renderer.js",
]

ENV_VARS_OPTIONAL = [
    "MODEL_CKPT_PATH", "VOCAB_PATH", "DEVICE",
    "SLM_PROVIDER", "OLLAMA_MODEL", "OPENAI_API_KEY",
    "SLM_CONFIDENCE_GATE",
]


# ── Check functions ───────────────────────────────────────────────────────────

def check_structure() -> int:
    """Verify all required files and directories exist."""
    _head("1. Project Structure")
    failures = 0

    for d in REQUIRED_DIRS:
        if Path(d).is_dir():
            _ok(f"dir  {d}/")
        else:
            _fail(f"dir  {d}/  — MISSING")
            failures += 1

    for f in REQUIRED_FILES:
        if Path(f).exists():
            _ok(f"file {f}")
        else:
            _fail(f"file {f}  — MISSING")
            failures += 1

    return failures


def check_python_syntax() -> int:
    """Parse every Python file for syntax errors."""
    _head("2. Python Syntax")
    failures = 0
    py_files = list(Path(".").rglob("*.py"))
    py_files = [
        p for p in py_files
        if not any(part in p.parts for part in (".venv", "__pycache__", ".git"))
    ]

    for path in sorted(py_files):
        try:
            ast.parse(path.read_text())
            _ok(str(path))
        except SyntaxError as e:
            _fail(f"{path}  — SyntaxError: {e}")
            failures += 1

    return failures


def check_python_imports() -> int:
    """Import each module to catch missing dependencies or runtime errors."""
    _head("3. Python Imports")
    failures = 0

    for module in PYTHON_MODULES:
        try:
            importlib.import_module(module)
            _ok(f"import {module}")
        except Exception as exc:
            _fail(f"import {module}  — {type(exc).__name__}: {exc}")
            failures += 1

    return failures


def check_js_syntax() -> int:
    """Use Node.js vm.Script to syntax-check JS files (strips ES module syntax)."""
    _head("4. JavaScript Syntax")

    node = _find_node()
    if node is None:
        _warn("Node.js not found — skipping JS syntax check")
        return 0

    failures = 0
    script = (
        "const vm=require('vm'),fs=require('fs');"
        "const files=process.argv.slice(1);"
        "let ok=true;"
        "for(const f of files){"
        "  const src=fs.readFileSync(f,'utf8')"
        "    .replace(/^export\\s+(default\\s+)?/gm,'')"
        "    .replace(/^import\\s+.*?from\\s+['\"]\\S+['\"];?\\s*$/gm,'');"
        "  try{new vm.Script(src);console.log('OK  '+f);}"
        "  catch(e){if(e instanceof SyntaxError){console.error('ERR '+f+': '+e.message);ok=false;}"
        "  else{console.log('OK  '+f+' (ES module)');}}}"
        "process.exit(ok?0:1);"
    )

    result = subprocess.run(
        [node, "-e", script] + JS_FILES,
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("OK"):
            _ok(line[4:])
        else:
            _fail(line)
            failures += 1
    for line in result.stderr.splitlines():
        if line.startswith("ERR"):
            _fail(line[4:])
            failures += 1

    return failures


def check_env() -> int:
    """Check environment variables and .env file."""
    _head("5. Environment")
    failures = 0

    # Load .env if present
    if Path(".env").exists():
        _ok(".env file found")
        for line in Path(".env").read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())
    else:
        _warn(".env not found — using shell environment only")

    for var in ENV_VARS_REQUIRED:
        val = os.environ.get(var, "")
        if val and val not in ("your_app_id_here", "your_server_secret_here",
                               "your_app_sign_here"):
            _ok(f"{var} = set")
        else:
            _fail(f"{var} — NOT SET (required)")
            failures += 1

    for var in ENV_VARS_OPTIONAL:
        val = os.environ.get(var, "")
        if val:
            _ok(f"{var} = {val[:20]}{'…' if len(val) > 20 else ''}")
        else:
            _warn(f"{var} — not set (optional)")

    return failures


def check_artifacts() -> int:
    """Check for trained model artifacts (non-fatal warnings)."""
    _head("6. Model Artifacts")
    failures = 0

    ckpt = Path(os.environ.get("MODEL_CKPT_PATH", "checkpoints/best.pt"))
    if ckpt.exists():
        size_mb = ckpt.stat().st_size / (1024 ** 2)
        _ok(f"Checkpoint: {ckpt}  ({size_mb:.1f} MB)")
    else:
        _warn(f"Checkpoint not found: {ckpt}  — run: python train.py")

    vocab = Path(os.environ.get("VOCAB_PATH", "vocab.json"))
    if vocab.exists():
        _ok(f"Vocabulary: {vocab}")
    else:
        _warn(f"vocab.json not found  — run: python orchestrate.py")

    label_map = Path("label_map.json")
    if label_map.exists():
        _ok(f"Label map: {label_map}")
    else:
        _warn(f"label_map.json not found  — run: python orchestrate.py")

    return failures   # artifacts are warnings, not failures


def _find_node() -> str | None:
    for candidate in ("node", "nodejs"):
        result = subprocess.run(
            ["which", candidate], capture_output=True, text=True
        )
        if result.returncode == 0:
            return candidate
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    quick = "--quick" in sys.argv

    print(f"{_BOLD}{'═' * 56}{_RESET}")
    print(f"{_BOLD}  Aashay's Sign Lang Project Validation{_RESET}")
    print(f"{_BOLD}{'═' * 56}{_RESET}")

    total_failures = 0
    total_failures += check_structure()
    total_failures += check_python_syntax()

    if not quick:
        total_failures += check_python_imports()
        total_failures += check_js_syntax()

    total_failures += check_env()
    check_artifacts()   # warnings only — don't count toward failures

    print(f"\n{'═' * 56}")
    if total_failures == 0:
        print(f"{_GREEN}{_BOLD}  All checks passed ✓{_RESET}")
    else:
        print(f"{_RED}{_BOLD}  {total_failures} check(s) failed ✗{_RESET}")
    print(f"{'═' * 56}\n")

    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
