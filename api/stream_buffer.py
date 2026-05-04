"""
api/stream_buffer.py
────────────────────
Per-connection sliding window buffer and cooldown state machine.

SlidingWindowBuffer
    Maintains a deque of the last `window_size` landmark frames.
    Fires an inference trigger every `stride` new frames.
    Thread-safe via asyncio — one buffer per WebSocket connection.

CooldownGuard
    Prevents the same gloss sequence from being broadcast multiple times
    within a single continuous gesture.  Resets after `cooldown_frames`
    new frames have been ingested since the last emission.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cslr_model.dataset import FEAT_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Frame validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_frame(frame: list | np.ndarray) -> np.ndarray:
    """
    Validate and coerce an incoming landmark frame.

    Parameters
    ----------
    frame : flat list or array of length FEAT_DIM (1,629)

    Returns
    -------
    (FEAT_DIM,) float32 numpy array

    Raises
    ------
    ValueError if shape is wrong or values are non-finite.
    """
    arr = np.asarray(frame, dtype=np.float32).ravel()
    if arr.shape[0] != FEAT_DIM:
        raise ValueError(
            f"Frame must have {FEAT_DIM} features, got {arr.shape[0]}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Frame contains NaN or Inf values")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Sliding Window Buffer
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowBuffer:
    """
    Fixed-capacity deque that accumulates landmark frames and signals
    when enough frames have arrived to run inference.

    Parameters
    ----------
    window_size : number of frames to keep (e.g. 60)
    stride      : new frames between inference triggers (e.g. 15)
    """

    def __init__(self, window_size: int = 60, stride: int = 15) -> None:
        self._window_size = window_size
        self._stride      = stride
        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._frames_since_last_run     = 0

    def push(self, frame: np.ndarray) -> bool:
        """
        Add one validated frame to the buffer.

        Returns True when the buffer has enough frames AND the stride
        counter has reached the trigger threshold — caller should run
        inference now.
        """
        self._buffer.append(frame)
        self._frames_since_last_run += 1

        ready = (
            len(self._buffer) >= self._window_size
            and self._frames_since_last_run >= self._stride
        )
        if ready:
            self._frames_since_last_run = 0
        return ready

    def get_window(self) -> np.ndarray:
        """
        Return the current window as a (T, FEAT_DIM) float32 array.
        T = min(len(buffer), window_size).
        """
        return np.stack(list(self._buffer), axis=0)   # (T, FEAT_DIM)

    def reset(self) -> None:
        self._buffer.clear()
        self._frames_since_last_run = 0

    @property
    def size(self) -> int:
        return len(self._buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Cooldown Guard
# ─────────────────────────────────────────────────────────────────────────────

class CooldownGuard:
    """
    Suppresses duplicate emissions of the same gloss sequence.

    A sequence is suppressed if:
      1. It is identical to the last emitted sequence, AND
      2. Fewer than `cooldown_frames` new frames have arrived since emission.

    Parameters
    ----------
    cooldown_frames : minimum new frames before the same sequence can fire again
    """

    def __init__(self, cooldown_frames: int = 30) -> None:
        self._cooldown_frames  = cooldown_frames
        self._last_glosses:    list[str] = []
        self._frames_since_emit: int     = cooldown_frames   # start ready

    def tick(self, n: int = 1) -> None:
        """Advance the cooldown counter by n frames."""
        self._frames_since_emit = min(
            self._frames_since_emit + n,
            self._cooldown_frames,
        )

    def should_emit(self, glosses: list[str]) -> bool:
        """
        Return True if this gloss sequence should be broadcast.

        Suppresses if identical to last emission within cooldown window.
        Always allows emission if the sequence has changed.
        """
        clean = [g for g in glosses if g not in ("<blank>", "<unk>")]
        if not clean:
            return False

        same_as_last = (clean == self._last_glosses)
        cooled_down  = (self._frames_since_emit >= self._cooldown_frames)

        if same_as_last and not cooled_down:
            return False

        return True

    def record_emission(self, glosses: list[str]) -> None:
        """Call after a successful emission to reset the cooldown."""
        self._last_glosses      = [g for g in glosses if g not in ("<blank>", "<unk>")]
        self._frames_since_emit = 0


# ─────────────────────────────────────────────────────────────────────────────
# Per-connection session state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConnectionSession:
    """
    All mutable state for one active WebSocket connection.

    Created on connect, discarded on disconnect.
    """
    connection_id: str
    room_id:       str
    user_id:       str
    target_user_id: str

    buffer:   SlidingWindowBuffer = field(default_factory=SlidingWindowBuffer)
    cooldown: CooldownGuard       = field(default_factory=CooldownGuard)

    # Accumulated glosses waiting for SLM threshold
    pending_glosses: list[str] = field(default_factory=list)

    # Telemetry
    frames_received:  int   = 0
    inferences_run:   int   = 0
    captions_sent:    int   = 0
    connected_at:     float = field(default_factory=time.time)

    def summary(self) -> dict:
        return {
            "connection_id":  self.connection_id,
            "room_id":        self.room_id,
            "user_id":        self.user_id,
            "frames_received": self.frames_received,
            "inferences_run":  self.inferences_run,
            "captions_sent":   self.captions_sent,
            "uptime_s":        round(time.time() - self.connected_at, 1),
        }
