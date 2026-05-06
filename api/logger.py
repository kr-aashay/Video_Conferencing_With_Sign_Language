"""
api/logger.py
─────────────
Structured performance logger for Aashay's Sign Lang inference bridge.

Tracks every inference event with microsecond precision and writes
structured JSON lines to logs/perf.jsonl for offline analysis.

Emits a WARNING whenever bridge latency exceeds the 300 ms SLA.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Optional

_perf_log = logging.getLogger("aashay.signlang.perf")

# ── JSONL sink ────────────────────────────────────────────────────────────────
_LOG_DIR  = Path("logs")
_LOG_FILE = _LOG_DIR / "perf.jsonl"
_file_lock = Lock()

LATENCY_SLA_MS = 300.0   # warn above this threshold


def _write_jsonl(record: dict) -> None:
    """Append one JSON line to the performance log file (thread-safe)."""
    _LOG_DIR.mkdir(exist_ok=True)
    with _file_lock:
        with open(_LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")


# ── Rolling stats ─────────────────────────────────────────────────────────────

class LatencyTracker:
    """
    Rolling-window latency tracker.

    Keeps the last `window` measurements and exposes p50/p95/p99 percentiles
    plus a simple mean.  Thread-safe via a Lock.
    """

    def __init__(self, window: int = 100) -> None:
        self._samples: deque[float] = deque(maxlen=window)
        self._lock = Lock()
        self._total_events = 0

    def record(
        self,
        session_id: str,
        room_id: str,
        latency_ms: float,
        glosses: list[str],
        caption: str,
        stage: str = "bridge",          # "inference" | "bridge"
    ) -> None:
        """
        Record one latency measurement.

        Parameters
        ----------
        session_id  : WebSocket connection ID (first 8 chars for brevity)
        room_id     : room identifier
        latency_ms  : measured latency in milliseconds
        glosses     : decoded gloss list
        caption     : refined English caption
        stage       : which stage of the pipeline this measures
        """
        with self._lock:
            self._samples.append(latency_ms)
            self._total_events += 1

        # Structured log line — includes p99 for offline analysis
        record = {
            "ts":         int(time.time() * 1000),
            "session":    session_id[:8],
            "room":       room_id,
            "stage":      stage,
            "latency_ms": round(latency_ms, 2),
            "p50_ms":     self.stats().get("p50_ms", 0),
            "p95_ms":     self.stats().get("p95_ms", 0),
            "p99_ms":     self.stats().get("p99_ms", 0),
            "glosses":    glosses,
            "caption":    caption[:80],
        }
        _write_jsonl(record)

        # SLA check
        if latency_ms > LATENCY_SLA_MS:
            _perf_log.warning(
                "SLA BREACH | %s | %.1f ms > %.0f ms SLA | glosses=%s",
                stage, latency_ms, LATENCY_SLA_MS, glosses,
            )
        else:
            _perf_log.debug(
                "%s | %.1f ms | %s",
                stage, latency_ms, caption[:40],
            )

    def stats(self) -> dict:
        """Return current rolling statistics."""
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return {
                "count": 0, "mean_ms": 0,
                "p50_ms": 0, "p95_ms": 0, "p99_ms": 0,
                "sla_breach_pct": 0.0,
            }

        samples_sorted = sorted(samples)
        n = len(samples_sorted)

        def pct(p: float) -> float:
            idx = min(int(p / 100 * n), n - 1)
            return round(samples_sorted[idx], 2)

        breaches = sum(1 for s in samples if s > LATENCY_SLA_MS)

        return {
            "count":           self._total_events,
            "window":          n,
            "mean_ms":         round(sum(samples) / n, 2),
            "p50_ms":          pct(50),
            "p95_ms":          pct(95),
            "p99_ms":          pct(99),
            "sla_breach_pct":  round(breaches / n * 100, 1),
            "sla_ms":          LATENCY_SLA_MS,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
latency_tracker = LatencyTracker(window=200)
