"""
api/buffer_manager.py
─────────────────────
Frame-drop Buffer Manager for the WebSocket ingestion path.

Problem
───────
MediaPipe runs at ~30 fps on the sender.  The FastAPI event loop processes
frames sequentially.  Under network jitter, frames can arrive in bursts.
Without back-pressure, the sliding window fills with stale frames and
inference latency spikes.

Solution — FrameDropBuffer
──────────────────────────
A fixed-capacity asyncio.Queue with a DROP_OLDEST eviction policy.

When the queue is full and a new frame arrives:
  • The oldest frame is silently discarded  (it is already stale)
  • The new frame is enqueued immediately
  • A drop counter is incremented for telemetry

This keeps the queue depth bounded and ensures inference always sees
the most recent frames, not a backlog of old ones.

The buffer is drained by the WebSocket handler coroutine via
``get_nowait()`` — it never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# Maximum frames held in the drop buffer before eviction kicks in.
# At 30 fps this is ~1 second of buffering before drops start.
DEFAULT_CAPACITY = 30


@dataclass
class FrameDropBuffer:
    """
    Bounded FIFO queue with drop-oldest overflow policy.

    Parameters
    ----------
    capacity : maximum number of frames to hold before dropping
    """
    capacity: int = DEFAULT_CAPACITY
    _queue:   asyncio.Queue = field(init=False)
    _dropped: int           = field(init=False, default=0)
    _total:   int           = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._queue = asyncio.Queue(maxsize=self.capacity)

    # ── Write path (called from WebSocket receive loop) ───────────────────────

    def put_nowait(self, frame: np.ndarray) -> bool:
        """
        Enqueue a frame without blocking.

        If the queue is full, the oldest frame is dropped and the new
        frame takes its place.

        Returns True if the frame was accepted without a drop.
        """
        self._total += 1
        if self._queue.full():
            try:
                self._queue.get_nowait()   # evict oldest
                self._dropped += 1
                log.debug(
                    "Frame dropped (buffer full) | total_drops=%d", self._dropped
                )
            except asyncio.QueueEmpty:
                pass

        try:
            self._queue.put_nowait(frame)
            return True
        except asyncio.QueueFull:
            # Extremely unlikely race — just count it
            self._dropped += 1
            return False

    # ── Read path (called from inference trigger) ─────────────────────────────

    def get_nowait(self) -> np.ndarray | None:
        """Return the next frame or None if the queue is empty."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get(self) -> np.ndarray:
        """Await the next frame (blocks until one is available)."""
        return await self._queue.get()

    # ── Telemetry ─────────────────────────────────────────────────────────────

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def total(self) -> int:
        return self._total

    @property
    def drop_rate(self) -> float:
        """Fraction of frames dropped (0.0 – 1.0)."""
        return self._dropped / max(self._total, 1)

    def stats(self) -> dict:
        return {
            "capacity":   self.capacity,
            "qsize":      self.qsize,
            "total":      self._total,
            "dropped":    self._dropped,
            "drop_rate":  round(self.drop_rate, 4),
        }

    def reset(self) -> None:
        """Drain the queue and reset counters."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._dropped = 0
        self._total   = 0
