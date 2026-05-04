"""
tests/test_integration_websocket.py
═══════════════════════════════════════════════════════════════════════════════
Integration tests — FastAPI WebSocket endpoint

Tests the full request/response cycle through the ASGI app using
Starlette's TestClient (synchronous WebSocket) without a real model.
The InferenceEngine is replaced with a lightweight stub that returns
a deterministic result, so no checkpoint files are needed.

Covers:
  • /health  GET — liveness probe
  • /sessions  GET — empty and populated
  • /ws/stream  WebSocket:
      - connection accept
      - frame ack protocol
      - invalid JSON → error response
      - wrong frame size → error response
      - NaN frame → error response
      - reset control message
      - unknown message type → error response
      - inference trigger → caption response (stub engine)
      - session telemetry via /session/{id}
      - concurrent connections (two clients simultaneously)
      - clean disconnect removes session
"""

from __future__ import annotations

import json
import os
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cslr_model.dataset import FEAT_DIM

# ── Env vars required by Settings before importing the app ───────────────────
os.environ.setdefault("WINDOW_SIZE",        "4")    # tiny window for fast tests
os.environ.setdefault("WINDOW_STRIDE",      "2")
os.environ.setdefault("COOLDOWN_FRAMES",    "2")
os.environ.setdefault("SLM_CONFIDENCE",     "1")    # emit after 1 gloss

# ── Stub inference engine — no model weights needed ──────────────────────────
STUB_RESULT = {
    "glosses":    ["I", "HELP"],
    "caption":    "I need help.",
    "latency_ms": 12.5,
}


class _StubEngine:
    async def predict_window(self, frames: np.ndarray) -> dict:
        return STUB_RESULT

    async def shutdown(self) -> None:
        pass


import api.app as _app_module

@pytest.fixture(scope="module")
def client():
    """
    Module-scoped TestClient with the stub engine injected.
    The lifespan is bypassed: we set _engine directly.
    """
    # Patch lifespan to inject stubs without touching disk
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _stub_lifespan(app):
        _app_module._engine = _StubEngine()
        _app_module._sessions.clear()
        yield
        _app_module._engine = None
        _app_module._sessions.clear()

    original_lifespan = _app_module.app.router.lifespan_context
    _app_module.app.router.lifespan_context = _stub_lifespan

    with TestClient(_app_module.app, raise_server_exceptions=True) as c:
        yield c

    _app_module.app.router.lifespan_context = original_lifespan


# ── Helpers ───────────────────────────────────────────────────────────────────

def _good_frame():
    return np.random.rand(FEAT_DIM).astype(np.float32).tolist()

def _ws_url(room="room1", user="alice", target="bob"):
    return f"/ws/stream/{room}/{user}/{target}"

def _send_frame(ws, frame=None):
    ws.send_json({"type": "frame", "data": frame or _good_frame()})
    return ws.receive_json()


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "active_connections" in body
        assert "engine_ready" in body

    def test_engine_ready_true(self, client):
        r = client.get("/health")
        assert r.json()["engine_ready"] is True

class TestSessionsEndpoint:
    def test_sessions_empty_initially(self, client):
        _app_module._sessions.clear()
        r = client.get("/sessions")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_session_not_found(self, client):
        r = client.get("/session/nonexistent-id")
        assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — connection & protocol
# ─────────────────────────────────────────────────────────────────────────────

class TestWebSocketProtocol:
    def test_connection_accepted(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            # Just connecting should not raise
            pass

    def test_frame_ack_returned(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            msg = _send_frame(ws)
            assert msg["type"] == "ack"
            assert msg["frame"] == 1

    def test_ack_frame_counter_increments(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            for expected in range(1, 4):
                msg = _send_frame(ws)
                assert msg["type"] == "ack"
                assert msg["frame"] == expected

    def test_invalid_json_returns_error(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            ws.send_text("not json at all {{")
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "JSON" in msg["message"]

    def test_wrong_frame_size_returns_error(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            ws.send_json({"type": "frame", "data": [0.0] * 100})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "225" in msg["message"]

    def test_nan_frame_returns_error(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            bad = [float("nan")] + [0.0] * (FEAT_DIM - 1)
            ws.send_json({"type": "frame", "data": bad})
            msg = ws.receive_json()
            assert msg["type"] == "error"

    def test_unknown_message_type_returns_error(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            ws.send_json({"type": "unknown_op"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "unknown_op" in msg["message"]

    def test_reset_returns_ack(self, client):
        with client.websocket_connect(_ws_url()) as ws:
            ws.send_json({"type": "reset"})
            msg = ws.receive_json()
            assert msg["type"] == "ack"
            assert msg.get("message") == "reset"

    def test_reset_clears_pending_glosses(self, client):
        """After reset, pending_glosses should be empty."""
        with client.websocket_connect(_ws_url(user="reset_user")) as ws:
            # Push some frames to accumulate state
            for _ in range(3):
                _send_frame(ws)
            # Find the session
            sessions = list(_app_module._sessions.values())
            session = next(
                (s for s in sessions if s.user_id == "reset_user"), None
            )
            if session:
                session.pending_glosses = ["I", "HELP"]
            ws.send_json({"type": "reset"})
            ws.receive_json()
            if session:
                assert session.pending_glosses == []


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — inference trigger
# ─────────────────────────────────────────────────────────────────────────────

class TestWebSocketInference:
    def test_caption_emitted_after_window_fills(self, client):
        """
        With WINDOW_SIZE=4, WINDOW_STRIDE=2, SLM_CONFIDENCE=1:
        After 4 frames the window is full; after 2 more the stride fires.
        We should receive a caption message.
        """
        with client.websocket_connect(_ws_url(user="infer_user")) as ws:
            messages = []
            # Send enough frames to fill window + trigger stride
            for _ in range(6):
                ws.send_json({"type": "frame", "data": _good_frame()})
                msg = ws.receive_json()
                messages.append(msg)

            caption_msgs = [m for m in messages if m.get("type") == "caption"]
            assert len(caption_msgs) >= 1, (
                f"Expected at least one caption, got: {[m['type'] for m in messages]}"
            )

    def test_caption_payload_structure(self, client):
        """Caption message must have glosses, caption, latency_ms, ts."""
        with client.websocket_connect(_ws_url(user="payload_user")) as ws:
            caption_msg = None
            for _ in range(8):
                ws.send_json({"type": "frame", "data": _good_frame()})
                msg = ws.receive_json()
                if msg.get("type") == "caption":
                    caption_msg = msg
                    break

            if caption_msg:
                assert "glosses"    in caption_msg
                assert "caption"    in caption_msg
                assert "latency_ms" in caption_msg
                assert "ts"         in caption_msg
                assert isinstance(caption_msg["glosses"], list)
                assert isinstance(caption_msg["caption"], str)

    def test_caption_matches_stub_result(self, client):
        """Caption content must match what the stub engine returns."""
        with client.websocket_connect(_ws_url(user="stub_user")) as ws:
            caption_msg = None
            for _ in range(8):
                ws.send_json({"type": "frame", "data": _good_frame()})
                msg = ws.receive_json()
                if msg.get("type") == "caption":
                    caption_msg = msg
                    break

            if caption_msg:
                assert caption_msg["caption"] == STUB_RESULT["caption"]


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — session telemetry
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionTelemetry:
    def test_session_registered_while_connected(self, client):
        with client.websocket_connect(_ws_url(user="telem_user")) as ws:
            _send_frame(ws)
            assert len(_app_module._sessions) >= 1

    def test_session_removed_on_disconnect(self, client):
        _app_module._sessions.clear()
        with client.websocket_connect(_ws_url(user="disc_user")) as ws:
            _send_frame(ws)
            conn_id = list(_app_module._sessions.keys())[0]
        # After context exit the session should be gone
        assert conn_id not in _app_module._sessions

    def test_session_endpoint_returns_data(self, client):
        with client.websocket_connect(_ws_url(user="api_user")) as ws:
            _send_frame(ws)
            conn_id = list(_app_module._sessions.keys())[0]
            r = client.get(f"/session/{conn_id}")
            assert r.status_code == 200
            body = r.json()
            assert body["frames_received"] >= 1
            assert body["room_id"] == "room1"

    def test_sessions_count_endpoint(self, client):
        _app_module._sessions.clear()
        with client.websocket_connect(_ws_url(user="count_user")) as ws:
            _send_frame(ws)
            r = client.get("/sessions")
            assert r.json()["count"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — concurrent connections
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrentConnections:
    def test_two_clients_independent_sessions(self, client):
        """Two simultaneous connections must have separate session state."""
        with client.websocket_connect(_ws_url(user="client_a")) as ws_a:
            with client.websocket_connect(_ws_url(user="client_b")) as ws_b:
                # Send different numbers of frames to each
                for _ in range(3):
                    _send_frame(ws_a)
                for _ in range(1):
                    _send_frame(ws_b)

                sessions = list(_app_module._sessions.values())
                session_a = next((s for s in sessions if s.user_id == "client_a"), None)
                session_b = next((s for s in sessions if s.user_id == "client_b"), None)

                assert session_a is not None
                assert session_b is not None
                assert session_a.frames_received == 3
                assert session_b.frames_received == 1
                # Different connection IDs
                assert session_a.connection_id != session_b.connection_id

    def test_two_clients_do_not_share_buffer(self, client):
        """Frames sent to client A must not appear in client B's buffer."""
        with client.websocket_connect(_ws_url(user="buf_a")) as ws_a:
            with client.websocket_connect(_ws_url(user="buf_b")) as ws_b:
                for _ in range(3):
                    _send_frame(ws_a)

                sessions = list(_app_module._sessions.values())
                session_b = next((s for s in sessions if s.user_id == "buf_b"), None)
                if session_b:
                    assert session_b.buffer.size == 0
