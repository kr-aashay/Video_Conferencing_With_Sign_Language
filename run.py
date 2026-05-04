"""
run.py
══════════════════════════════════════════════════════════════════════════════
Single-command launcher

Usage
─────
    python run.py                   # start API + frontend dev server
    python run.py --api-only        # API server only
    python run.py --frontend-only   # frontend server only
    python run.py --check           # pre-flight checks, then exit
    python run.py --port 8080       # custom API port (default 8000)
    python run.py --frontend-port 3001

What it does
────────────
  1. Pre-flight checks  — validates .env, checkpoint, vocab.json
  2. API server         — uvicorn api.app:app  (port 8000)
  3. Frontend server    — python -m http.server (port 3000, serves frontend/)
  4. Heartbeat monitor  — polls /health/detail every 5 s and logs status
  5. Graceful shutdown  — Ctrl-C stops both processes cleanly
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run.log", mode="w"),
    ],
)
log = logging.getLogger("launcher")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_API_PORT      = 8000
DEFAULT_FRONTEND_PORT = 3000
HEARTBEAT_INTERVAL    = 5      # seconds between /health/detail polls
STARTUP_TIMEOUT       = 30     # seconds to wait for API to become ready


# ══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def _check(condition: bool, msg: str, fatal: bool = True) -> bool:
    if condition:
        log.info("  ✓  %s", msg)
        return True
    if fatal:
        log.error("  ✗  %s", msg)
    else:
        log.warning("  ⚠  %s", msg)
    return False


def preflight(api_port: int) -> bool:
    """
    Validate the environment before starting any process.
    Returns True if all required checks pass.
    """
    log.info("═" * 56)
    log.info("  HexaMinds Pre-flight Checks")
    log.info("═" * 56)

    ok = True

    # ── .env file ─────────────────────────────────────────────────────────────
    env_ok = _check(Path(".env").exists(), ".env file present", fatal=False)
    if not env_ok:
        log.warning("     No .env found — credentials must be in shell environment")

    # ── Required env vars ─────────────────────────────────────────────────────
    # Load .env manually so we can check without importing pydantic-settings
    if Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    # ── Model checkpoint ──────────────────────────────────────────────────────
    ckpt = Path(os.environ.get("MODEL_CKPT_PATH", "checkpoints/epoch_099_loss1.6763_acc0.912.pt"))
    ckpt_ok = _check(ckpt.exists(), f"Checkpoint found: {ckpt}", fatal=False)
    if not ckpt_ok:
        log.warning(
            "     Checkpoint not found — run 'python train.py' first.\n"
            "     The API will start but inference will fail."
        )

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab = Path(os.environ.get("VOCAB_PATH", "vocab.json"))
    vocab_ok = _check(vocab.exists(), f"Vocabulary found: {vocab}", fatal=False)
    if not vocab_ok:
        log.warning(
            "     vocab.json not found — run 'python orchestrate.py' first."
        )

    # ── Port availability ─────────────────────────────────────────────────────
    import socket
    for port, name in [(api_port, "API"), (DEFAULT_FRONTEND_PORT, "Frontend")]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            in_use = s.connect_ex(("127.0.0.1", port)) == 0
        _check(not in_use, f"Port {port} ({name}) is free", fatal=False)

    # ── Python packages ───────────────────────────────────────────────────────
    for pkg in ("fastapi", "uvicorn", "torch", "mediapipe"):
        try:
            __import__(pkg)
            _check(True, f"Package '{pkg}' importable")
        except ImportError:
            ok &= _check(False, f"Package '{pkg}' NOT installed — run: pip install -r requirements.txt")

    log.info("═" * 56)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# FRONTEND SERVER  (python -m http.server)
# ══════════════════════════════════════════════════════════════════════════════

class _FrontendHandler(http.server.SimpleHTTPRequestHandler):
    """
    Serves files from the frontend/ directory.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="frontend", **kwargs)

    def do_GET(self):
        # Serve HTML files directly
        if self.path.rstrip("/") in ("", "/sender.html", "/receiver.html",
                                      "/sender", "/receiver"):
            filename = "sender.html" if "sender" in self.path else "receiver.html"
            if self.path.rstrip("/") in ("", "/"):
                filename = "sender.html"
            self._serve_html(filename)
        else:
            super().do_GET()

    def _serve_html(self, filename: str):
        path = Path("frontend") / filename
        if not path.exists():
            self.send_error(404, f"{filename} not found")
            return

        html = path.read_text()

        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Suppress per-request noise; only log errors
        if args and str(args[1]) not in ("200", "304"):
            log.debug("Frontend: " + fmt % args)


def start_frontend_server(port: int) -> threading.Thread:
    server = http.server.HTTPServer(("0.0.0.0", port), _FrontendHandler)

    def _run():
        log.info("Frontend server → http://localhost:%d", port)
        log.info("  Sender   → http://localhost:%d/sender.html", port)
        log.info("  Receiver → http://localhost:%d/receiver.html", port)
        server.serve_forever()

    t = threading.Thread(target=_run, daemon=True, name="frontend-server")
    t.start()
    return t


# ══════════════════════════════════════════════════════════════════════════════
# API SERVER  (uvicorn subprocess)
# ══════════════════════════════════════════════════════════════════════════════

def start_api_server(port: int) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.app:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--workers", "1",
        "--log-level", os.environ.get("LOG_LEVEL", "info").lower(),
        "--no-access-log",
    ]
    log.info("API server → http://localhost:%d", port)
    log.info("  Docs     → http://localhost:%d/docs", port)
    log.info("  Health   → http://localhost:%d/health/detail", port)
    log.info("  Metrics  → http://localhost:%d/metrics", port)

    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return proc


def wait_for_api(port: int, timeout: int = STARTUP_TIMEOUT) -> bool:
    """Poll /health until the API responds or timeout expires."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    log.info("Waiting for API to become ready (timeout %ds)…", timeout)

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    log.info("API is ready ✓")
                    return True
        except Exception:
            pass
        time.sleep(1)

    log.error("API did not become ready within %d seconds", timeout)
    return False


# ══════════════════════════════════════════════════════════════════════════════
# HEARTBEAT MONITOR
# ══════════════════════════════════════════════════════════════════════════════

def _heartbeat_loop(port: int, interval: int) -> None:
    """
    Polls /health/detail every `interval` seconds and logs the result.
    Runs as a daemon thread — stops automatically when the main process exits.
    """
    url = f"http://localhost:{port}/health/detail"
    while True:
        time.sleep(interval)
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                data = json.loads(r.read())
            stats = data.get("latency_stats", {})
            conns = data.get("active_connections", 0)
            engine = "✓" if data.get("engine_ready") else "✗"
            mean   = stats.get("mean_ms", 0)
            p95    = stats.get("p95_ms",  0)
            breach = stats.get("sla_breach_pct", 0)

            log.info(
                "♥  engine=%s conns=%d | "
                "latency mean=%.0fms p95=%.0fms breach=%.1f%%",
                engine, conns, mean, p95, breach,
            )
        except Exception as exc:
            log.warning("♥  Heartbeat failed: %s", exc)


def start_heartbeat(port: int, interval: int = HEARTBEAT_INTERVAL) -> threading.Thread:
    t = threading.Thread(
        target=_heartbeat_loop,
        args=(port, interval),
        daemon=True,
        name="heartbeat",
    )
    t.start()
    return t


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HexaMinds unified launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-only",      action="store_true", help="Start API server only")
    parser.add_argument("--frontend-only", action="store_true", help="Start frontend server only")
    parser.add_argument("--check",         action="store_true", help="Run pre-flight checks and exit")
    parser.add_argument("--port",          type=int, default=DEFAULT_API_PORT,      metavar="N")
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT, metavar="N")
    parser.add_argument("--no-heartbeat",  action="store_true", help="Disable heartbeat monitor")
    args = parser.parse_args()

    # ── Pre-flight ────────────────────────────────────────────────────────────
    checks_ok = preflight(args.port)
    if args.check:
        sys.exit(0 if checks_ok else 1)

    log.info("═" * 56)
    log.info("  HexaMinds — Starting Services")
    log.info("═" * 56)

    api_proc = None

    try:
        # ── API server ────────────────────────────────────────────────────────
        if not args.frontend_only:
            api_proc = start_api_server(args.port)
            if not wait_for_api(args.port):
                log.error("API failed to start. Check logs above.")
                sys.exit(1)

        # ── Frontend server ───────────────────────────────────────────────────
        if not args.api_only:
            start_frontend_server(args.frontend_port)

        # ── Heartbeat ─────────────────────────────────────────────────────────
        if not args.no_heartbeat and not args.frontend_only:
            start_heartbeat(args.port)

        log.info("═" * 56)
        log.info("  All services running. Press Ctrl-C to stop.")
        log.info("═" * 56)

        # ── Block until Ctrl-C ────────────────────────────────────────────────
        if api_proc:
            api_proc.wait()
        else:
            # Frontend-only: keep the main thread alive
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        log.info("\nShutting down…")
    finally:
        if api_proc and api_proc.poll() is None:
            api_proc.terminate()
            try:
                api_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_proc.kill()
        log.info("Stopped.")


if __name__ == "__main__":
    main()
