"""
tests/test_unit_api.py
═══════════════════════════════════════════════════════════════════════════════
Unit tests — api layer (no network, no model weights)

Covers:
  • validate_frame  (shape, dtype, NaN/Inf rejection)
  • SlidingWindowBuffer  (trigger logic, window shape, reset)
  • CooldownGuard  (suppress, allow after cooldown, different sequence)
  • ConnectionSession  (summary fields)
"""

import asyncio
import hashlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from api.stream_buffer import (
    validate_frame,
    SlidingWindowBuffer,
    CooldownGuard,
    ConnectionSession,
)
from cslr_model.dataset import FEAT_DIM


# ─────────────────────────────────────────────────────────────────────────────
# validate_frame
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateFrame:
    def test_valid_numpy_array(self):
        arr = np.random.rand(FEAT_DIM).astype(np.float32)
        out = validate_frame(arr)
        assert out.shape == (FEAT_DIM,)
        assert out.dtype == np.float32

    def test_valid_python_list(self):
        lst = [0.0] * FEAT_DIM
        out = validate_frame(lst)
        assert out.shape == (FEAT_DIM,)

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError, match="225"):
            validate_frame(np.zeros(100))

    def test_nan_raises(self):
        arr = np.zeros(FEAT_DIM, dtype=np.float32)
        arr[42] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            validate_frame(arr)

    def test_inf_raises(self):
        arr = np.zeros(FEAT_DIM, dtype=np.float32)
        arr[0] = float("inf")
        with pytest.raises(ValueError, match="NaN"):
            validate_frame(arr)

    def test_2d_array_flattened(self):
        """A (1, FEAT_DIM) array should be accepted after ravel."""
        arr = np.zeros((1, FEAT_DIM), dtype=np.float32)
        out = validate_frame(arr)
        assert out.shape == (FEAT_DIM,)


# ─────────────────────────────────────────────────────────────────────────────
# SlidingWindowBuffer
# ─────────────────────────────────────────────────────────────────────────────

class TestSlidingWindowBuffer:
    def _frame(self):
        return np.zeros(FEAT_DIM, dtype=np.float32)

    def test_no_trigger_before_window_full(self):
        buf = SlidingWindowBuffer(window_size=4, stride=2)
        results = [buf.push(self._frame()) for _ in range(3)]
        assert not any(results)

    def test_trigger_at_window_plus_stride(self):
        buf = SlidingWindowBuffer(window_size=4, stride=2)
        results = [buf.push(self._frame()) for _ in range(6)]
        # Triggers at frame 4 (window full, stride=2 met) and frame 6
        assert results[3] is True
        assert results[5] is True

    def test_no_double_trigger(self):
        """Stride counter resets after trigger — next trigger needs another stride."""
        buf = SlidingWindowBuffer(window_size=4, stride=2)
        results = [buf.push(self._frame()) for _ in range(7)]
        # Triggers at index 3 and 5, NOT at 4
        assert results[4] is False

    def test_get_window_shape(self):
        buf = SlidingWindowBuffer(window_size=4, stride=1)
        for _ in range(6):
            buf.push(self._frame())
        win = buf.get_window()
        assert win.shape == (4, FEAT_DIM)

    def test_window_contains_latest_frames(self):
        buf = SlidingWindowBuffer(window_size=3, stride=1)
        for i in range(5):
            frame = np.full(FEAT_DIM, float(i), dtype=np.float32)
            buf.push(frame)
        win = buf.get_window()
        # Should contain frames 2, 3, 4
        assert win[0, 0] == 2.0
        assert win[-1, 0] == 4.0

    def test_reset_clears_buffer(self):
        buf = SlidingWindowBuffer(window_size=4, stride=2)
        for _ in range(4):
            buf.push(self._frame())
        buf.reset()
        assert buf.size == 0
        # After reset, should not trigger until window fills again
        results = [buf.push(self._frame()) for _ in range(3)]
        assert not any(results)

    def test_size_property(self):
        buf = SlidingWindowBuffer(window_size=10, stride=5)
        for i in range(7):
            buf.push(self._frame())
        assert buf.size == 7


# ─────────────────────────────────────────────────────────────────────────────
# CooldownGuard
# ─────────────────────────────────────────────────────────────────────────────

class TestCooldownGuard:
    def test_first_emission_always_allowed(self):
        cd = CooldownGuard(cooldown_frames=10)
        assert cd.should_emit(["I", "HELP"]) is True

    def test_same_sequence_suppressed_in_cooldown(self):
        cd = CooldownGuard(cooldown_frames=10)
        cd.record_emission(["I", "HELP"])
        cd.tick(5)
        assert cd.should_emit(["I", "HELP"]) is False

    def test_same_sequence_allowed_after_cooldown(self):
        cd = CooldownGuard(cooldown_frames=5)
        cd.record_emission(["I", "HELP"])
        cd.tick(5)
        assert cd.should_emit(["I", "HELP"]) is True

    def test_different_sequence_always_allowed(self):
        cd = CooldownGuard(cooldown_frames=100)
        cd.record_emission(["I", "HELP"])
        cd.tick(1)
        assert cd.should_emit(["THANK", "YOU"]) is True

    def test_empty_glosses_rejected(self):
        cd = CooldownGuard(cooldown_frames=5)
        assert cd.should_emit([]) is False

    def test_only_special_tokens_rejected(self):
        cd = CooldownGuard(cooldown_frames=5)
        assert cd.should_emit(["<blank>", "<unk>"]) is False

    def test_tick_caps_at_cooldown_frames(self):
        cd = CooldownGuard(cooldown_frames=5)
        cd.record_emission(["A"])
        cd.tick(100)   # tick way past the cap
        assert cd._frames_since_emit == 5

    def test_record_emission_resets_counter(self):
        cd = CooldownGuard(cooldown_frames=10)
        cd.tick(10)
        cd.record_emission(["A"])
        assert cd._frames_since_emit == 0


# ─────────────────────────────────────────────────────────────────────────────
# ConnectionSession
# ─────────────────────────────────────────────────────────────────────────────

class TestConnectionSession:
    def _make_session(self):
        return ConnectionSession(
            connection_id="test-uuid",
            room_id="room-1",
            user_id="alice",
            target_user_id="bob",
        )

    def test_summary_keys(self):
        s = self._make_session()
        summary = s.summary()
        for key in ("connection_id", "room_id", "user_id",
                    "frames_received", "inferences_run", "captions_sent", "uptime_s"):
            assert key in summary

    def test_uptime_increases(self):
        s = self._make_session()
        time.sleep(0.05)
        assert s.summary()["uptime_s"] >= 0.0

    def test_default_counters_zero(self):
        s = self._make_session()
        assert s.frames_received == 0
        assert s.inferences_run  == 0
        assert s.captions_sent   == 0

    def test_pending_glosses_starts_empty(self):
        s = self._make_session()
        assert s.pending_glosses == []
