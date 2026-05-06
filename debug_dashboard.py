"""
debug_dashboard.py
══════════════════════════════════════════════════════════════════════════════
Aashay's Sign Lang — Real-time Terminal Debugging Dashboard

Displays a live, auto-refreshing overlay in the terminal showing:
  • Current predicted gloss sequence
  • Inference latency (mean / p95 / SLA breach %)
  • Active WebSocket connections
  • Frame drop rate
  • Last 8 caption events (scrolling log)

Uses only the stdlib curses module — zero extra dependencies.

Usage
─────
    python debug_dashboard.py                  # connect to localhost:8000
    python debug_dashboard.py --port 8080
    python debug_dashboard.py --interval 0.5   # refresh every 500 ms
"""

from __future__ import annotations

import argparse
import curses
import json
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_PORT     = 8000
DEFAULT_INTERVAL = 1.0   # seconds between refreshes
MAX_LOG_LINES    = 8
SLA_MS           = 300.0


# ── Data fetcher ──────────────────────────────────────────────────────────────

def fetch(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


# ── Colour pairs (curses) ─────────────────────────────────────────────────────
# Pair indices:
#   1 = header (cyan bold)
#   2 = good   (green)
#   3 = warn   (yellow)
#   4 = bad    (red)
#   5 = dim    (white dim)
#   6 = accent (magenta)

def _init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN,    -1)
    curses.init_pair(2, curses.COLOR_GREEN,   -1)
    curses.init_pair(3, curses.COLOR_YELLOW,  -1)
    curses.init_pair(4, curses.COLOR_RED,     -1)
    curses.init_pair(5, curses.COLOR_WHITE,   -1)
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _safe_addstr(win, y: int, x: int, text: str, attr: int = 0) -> None:
    """addstr that silently ignores out-of-bounds writes."""
    max_y, max_x = win.getmaxyx()
    if y < 0 or y >= max_y or x < 0:
        return
    available = max_x - x - 1
    if available <= 0:
        return
    try:
        win.addstr(y, x, text[:available], attr)
    except curses.error:
        pass


def _hline(win, y: int, char: str = "─") -> None:
    _, max_x = win.getmaxyx()
    _safe_addstr(win, y, 0, char * (max_x - 1), curses.color_pair(5) | curses.A_DIM)


def _latency_color(ms: float) -> int:
    if ms <= 0:
        return curses.color_pair(5)
    if ms < 150:
        return curses.color_pair(2)
    if ms < SLA_MS:
        return curses.color_pair(3)
    return curses.color_pair(4)


def _status_color(ok: bool) -> int:
    return curses.color_pair(2) if ok else curses.color_pair(4)


# ── Main draw loop ────────────────────────────────────────────────────────────

def _draw(
    stdscr,
    health: dict | None,
    sessions: dict | None,
    caption_log: deque,
    last_update: float,
    api_url: str,
) -> None:
    stdscr.erase()
    H, W = stdscr.getmaxyx()
    row = 0

    C1  = curses.color_pair(1) | curses.A_BOLD
    C2  = curses.color_pair(2)
    C3  = curses.color_pair(3)
    C4  = curses.color_pair(4)
    C5  = curses.color_pair(5)
    C6  = curses.color_pair(6)
    DIM = curses.A_DIM

    # ── Title bar ─────────────────────────────────────────────────────────────
    title = " Aashay's Sign Lang Debug Dashboard "
    ts    = datetime.now().strftime("%H:%M:%S")
    _safe_addstr(stdscr, row, 0, "═" * (W - 1), C1)
    _safe_addstr(stdscr, row, max(0, (W - len(title)) // 2), title, C1)
    _safe_addstr(stdscr, row, W - len(ts) - 2, ts, C5 | DIM)
    row += 1

    # ── API status ────────────────────────────────────────────────────────────
    _safe_addstr(stdscr, row, 0, f" API  {api_url}", C5 | DIM)
    row += 1

    if health is None:
        _safe_addstr(stdscr, row, 0, " ✗  API unreachable — is the server running?", C4 | curses.A_BOLD)
        stdscr.refresh()
        return

    # ── Subsystem status ──────────────────────────────────────────────────────
    _hline(stdscr, row); row += 1
    engine_ok = health.get("engine_ready", False)
    ws_ok     = health.get("ws_ready",     False)
    conns     = health.get("active_connections", 0)

    _safe_addstr(stdscr, row, 0,  " Engine  ", C5)
    _safe_addstr(stdscr, row, 9,  "✓ Ready" if engine_ok else "✗ Not ready",
                 _status_color(engine_ok) | curses.A_BOLD)
    _safe_addstr(stdscr, row, 22, " WS  ", C5)
    _safe_addstr(stdscr, row, 27, "✓ Ready" if ws_ok else "✗ Not ready",
                 _status_color(ws_ok) | curses.A_BOLD)
    _safe_addstr(stdscr, row, 40, f" Connections: {conns}", C6 | curses.A_BOLD)
    row += 1

    # ── Latency stats ─────────────────────────────────────────────────────────
    _hline(stdscr, row); row += 1
    stats = health.get("latency_stats", {})
    mean  = stats.get("mean_ms",        0)
    p50   = stats.get("p50_ms",         0)
    p95   = stats.get("p95_ms",         0)
    p99   = stats.get("p99_ms",         0)
    breach = stats.get("sla_breach_pct", 0.0)
    count  = stats.get("count",          0)

    _safe_addstr(stdscr, row, 0, " Latency  ", C5)
    _safe_addstr(stdscr, row, 10, f"mean {mean:>6.1f}ms", _latency_color(mean))
    _safe_addstr(stdscr, row, 26, f"p50 {p50:>6.1f}ms",  _latency_color(p50))
    _safe_addstr(stdscr, row, 41, f"p95 {p95:>6.1f}ms",  _latency_color(p95))
    _safe_addstr(stdscr, row, 56, f"p99 {p99:>6.1f}ms",  _latency_color(p99))
    row += 1

    breach_col = C4 if breach > 5 else (C3 if breach > 0 else C2)
    _safe_addstr(stdscr, row, 0, " SLA 300ms  ", C5)
    _safe_addstr(stdscr, row, 12,
                 f"breach {breach:.1f}%  ({count} total events)",
                 breach_col | curses.A_BOLD)
    row += 1

    # ── Active sessions ───────────────────────────────────────────────────────
    _hline(stdscr, row); row += 1
    _safe_addstr(stdscr, row, 0, " Active Sessions", C1)
    row += 1

    session_list = (sessions or {}).get("sessions", [])
    if not session_list:
        _safe_addstr(stdscr, row, 2, "No active connections", C5 | DIM)
        row += 1
    else:
        for s in session_list[:4]:   # show at most 4
            cid    = s.get("connection_id", "")[:8]
            room   = s.get("room_id", "?")
            frames = s.get("frames_received", 0)
            infer  = s.get("inferences_run",  0)
            caps   = s.get("captions_sent",   0)
            uptime = s.get("uptime_s",        0)
            line = (
                f"  {cid}  room={room:<16} "
                f"frames={frames:<6} infer={infer:<5} "
                f"captions={caps:<4} up={uptime:.0f}s"
            )
            _safe_addstr(stdscr, row, 0, line, C5)
            row += 1

    # ── Caption log ───────────────────────────────────────────────────────────
    _hline(stdscr, row); row += 1
    _safe_addstr(stdscr, row, 0, " Caption Log  (last 8)", C1)
    row += 1

    if not caption_log:
        _safe_addstr(stdscr, row, 2, "No captions yet", C5 | DIM)
        row += 1
    else:
        for entry in list(caption_log):
            if row >= H - 2:
                break
            ts_str  = entry["ts"]
            latency = entry["latency_ms"]
            glosses = " · ".join(entry["glosses"])
            caption = entry["caption"]

            _safe_addstr(stdscr, row, 0,  f"  {ts_str}", C5 | DIM)
            _safe_addstr(stdscr, row, 12, f"{latency:>6.1f}ms", _latency_color(latency))
            _safe_addstr(stdscr, row, 22, f"  [{glosses}]", C6)
            row += 1
            if row < H - 1:
                _safe_addstr(stdscr, row, 22, f"  → {caption}", C2 | curses.A_BOLD)
                row += 1

    # ── Footer ────────────────────────────────────────────────────────────────
    if H > 3:
        _hline(stdscr, H - 2)
        age = time.time() - last_update
        _safe_addstr(stdscr, H - 1, 0,
                     f" Refreshed {age:.1f}s ago  |  q = quit  |  r = reset stats",
                     C5 | DIM)

    stdscr.refresh()


# ── Caption log updater ───────────────────────────────────────────────────────

def _update_caption_log(caption_log: deque, perf_path: str) -> None:
    """Read the last few lines of perf.jsonl and append new captions."""
    try:
        with open(perf_path) as f:
            lines = f.readlines()
        for line in lines[-MAX_LOG_LINES:]:
            try:
                rec = json.loads(line)
                ts  = datetime.fromtimestamp(rec["ts"] / 1000).strftime("%H:%M:%S")
                entry = {
                    "ts":         ts,
                    "latency_ms": rec.get("latency_ms", 0),
                    "glosses":    rec.get("glosses", []),
                    "caption":    rec.get("caption", ""),
                }
                # Avoid duplicates by checking last entry
                if not caption_log or caption_log[-1] != entry:
                    caption_log.append(entry)
            except (json.JSONDecodeError, KeyError):
                pass
    except FileNotFoundError:
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

def _run(stdscr, api_url: str, interval: float, perf_path: str) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(int(interval * 1000))
    _init_colors()

    caption_log: deque = deque(maxlen=MAX_LOG_LINES)
    health      = None
    sessions    = None
    last_update = 0.0

    while True:
        # ── Key handling ──────────────────────────────────────────────────────
        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):   # q or Esc
            break
        if key in (ord("r"), ord("R")):
            caption_log.clear()

        # ── Fetch data ────────────────────────────────────────────────────────
        now = time.time()
        if now - last_update >= interval:
            health      = fetch(f"{api_url}/health/detail")
            sessions    = fetch(f"{api_url}/sessions")
            _update_caption_log(caption_log, perf_path)
            last_update = now

        # ── Draw ──────────────────────────────────────────────────────────────
        _draw(stdscr, health, sessions, caption_log, last_update, api_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aashay's Sign Lang terminal debug dashboard")
    parser.add_argument("--port",     type=int,   default=DEFAULT_PORT,     metavar="N")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, metavar="S")
    parser.add_argument("--perf-log", type=str,   default="logs/perf.jsonl")
    args = parser.parse_args()

    api_url = f"http://localhost:{args.port}"
    print(f"Connecting to {api_url} …  (press q to quit)")
    time.sleep(0.5)

    try:
        curses.wrapper(_run, api_url, args.interval, args.perf_log)
    except KeyboardInterrupt:
        pass
    print("Dashboard closed.")


if __name__ == "__main__":
    main()
