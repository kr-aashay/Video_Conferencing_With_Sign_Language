"""
api/app.py
══════════════════════════════════════════════════════════════════════════════
HexaMinds — FastAPI WebSocket Inference Bridge

Endpoints
─────────
  WS  /ws/stream/{room_id}/{user_id}/{target_user_id}
        Continuous landmark ingestion + real-time caption broadcast.

  WS  /ws/heartbeat
        Lightweight ping/pong for frontend status indicator.

  GET /health          — liveness probe
  GET /health/detail   — per-subsystem status + latency stats
  GET /metrics         — rolling latency percentiles
  GET /session/{id}    — per-connection telemetry
  GET /sessions        — all active connections

Message protocol (client → server)
────────────────────────────────────
  { "type": "frame",  "data": [float × 225] }
  { "type": "reset" }

Message protocol (server → client)
────────────────────────────────────
  { "type": "caption", "glosses": [...], "caption": "...", "latency_ms": 42.1, "ts": ... }
  { "type": "ack",     "frame": 1234, "dropped": 0 }
  { "type": "error",   "message": "..." }
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config        import get_settings
from .inference     import InferenceEngine
from .logger        import latency_tracker
from .buffer_manager import FrameDropBuffer
from .stream_buffer import (
    ConnectionSession, SlidingWindowBuffer,
    CooldownGuard, validate_frame,
)

log = logging.getLogger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────
_engine:   InferenceEngine | None = None
_sessions: dict[str, ConnectionSession] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    cfg = get_settings()

    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    log.info("HexaMinds API starting up | device=%s", cfg.device)

    _engine = InferenceEngine(
        ckpt_path=cfg.model_ckpt_path,
        vocab_path=cfg.vocab_path,
        device=cfg.device,
        beam_width=cfg.beam_width,
    )

    log.info("Startup complete")
    yield

    log.info("Shutting down …")
    if _engine:
        await _engine.shutdown()
    log.info("Shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="HexaMinds Sign Language Caption Bridge",
    version="2.0.0",
    description="Real-time sign language recognition → captions",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _send_json(ws: WebSocket, payload: dict) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        pass


async def _run_inference(session: ConnectionSession) -> dict | None:
    """
    Run inference on the current sliding window.
    Returns the broadcast payload or None if suppressed by cooldown.
    """
    cfg = get_settings()

    frames = session.buffer.get_window()
    session.inferences_run += 1

    try:
        result = await _engine.predict_window(frames)
    except Exception as exc:
        log.error("Inference error for session %s: %s", session.connection_id, exc)
        return None

    glosses    = result["glosses"]
    caption    = result["caption"]
    latency_ms = result["latency_ms"]

    latency_tracker.record(
        session_id=session.connection_id,
        room_id=session.room_id,
        latency_ms=latency_ms,
        glosses=glosses,
        caption=caption,
        stage="bridge",
    )

    session.cooldown.tick(cfg.window_stride)

    if not session.cooldown.should_emit(glosses):
        return None

    session.cooldown.record_emission(glosses)

    clean = [g for g in glosses if g not in ("<blank>", "<unk>")]
    session.pending_glosses.extend(clean)

    if len(session.pending_glosses) < cfg.slm_confidence:
        return None

    final_glosses = list(session.pending_glosses)
    session.pending_glosses.clear()

    session.captions_sent += 1
    log.info(
        "Caption | session=%s → '%s'  (%.1f ms)",
        session.connection_id[:8], caption[:80], latency_ms,
    )

    return {
        "type":       "caption",
        "glosses":    final_glosses,
        "caption":    caption,
        "latency_ms": latency_ms,
        "ts":         int(time.time() * 1000),
    }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — landmark stream
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/stream/{room_id}/{user_id}/{target_user_id}")
async def ws_stream(
    websocket: WebSocket,
    room_id: str,
    user_id: str,
    target_user_id: str,
) -> None:
    cfg = get_settings()
    await websocket.accept()

    connection_id = str(uuid.uuid4())
    drop_buf = FrameDropBuffer(capacity=30)
    session = ConnectionSession(
        connection_id=connection_id,
        room_id=room_id,
        user_id=user_id,
        target_user_id=target_user_id,
        buffer=SlidingWindowBuffer(
            window_size=cfg.window_size,
            stride=cfg.window_stride,
        ),
        cooldown=CooldownGuard(cooldown_frames=cfg.cooldown_frames),
    )
    _sessions[connection_id] = session

    log.info("WS connected | id=%s room=%s user=%s",
             connection_id[:8], room_id, user_id)

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_json(websocket, {"type": "error", "message": "Invalid JSON"})
                continue

            msg_type = msg.get("type")

            if msg_type == "frame":
                try:
                    frame = validate_frame(msg["data"])
                except (KeyError, ValueError) as exc:
                    await _send_json(websocket, {"type": "error", "message": str(exc)})
                    continue

                session.frames_received += 1
                session.cooldown.tick(1)

                drop_buf.put_nowait(frame)
                buffered = drop_buf.get_nowait()
                should_infer = session.buffer.push(buffered) if buffered is not None else False

                await _send_json(websocket, {
                    "type":    "ack",
                    "frame":   session.frames_received,
                    "dropped": drop_buf.dropped,
                })

                if should_infer and _engine is not None:
                    payload = await _run_inference(session)
                    if payload:
                        await _send_json(websocket, payload)

            elif msg_type == "reset":
                session.buffer.reset()
                drop_buf.reset()
                session.pending_glosses.clear()
                await _send_json(websocket, {"type": "ack", "message": "reset"})

            else:
                await _send_json(websocket, {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type!r}",
                })

    except WebSocketDisconnect:
        log.info("WS disconnected | id=%s", connection_id[:8])
    except Exception as exc:
        log.error("WS error | id=%s: %s", connection_id[:8], exc)
    finally:
        _sessions.pop(connection_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — heartbeat
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/heartbeat")
async def ws_heartbeat(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "ping":
                await _send_json(websocket, {
                    "type":         "pong",
                    "ts":           int(time.time() * 1000),
                    "engine_ready": _engine is not None,
                    "connections":  len(_sessions),
                    "latency_stats": latency_tracker.stats(),
                })
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# REST endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status":             "ok",
        "active_connections": len(_sessions),
        "engine_ready":       _engine is not None,
    })


@app.get("/health/detail")
async def health_detail() -> JSONResponse:
    return JSONResponse({
        "status":             "ok" if _engine is not None else "degraded",
        "engine_ready":       _engine is not None,
        "ws_ready":           True,
        "active_connections": len(_sessions),
        "latency_stats":      latency_tracker.stats(),
        "ts":                 int(time.time() * 1000),
    })


@app.get("/metrics")
async def metrics() -> JSONResponse:
    return JSONResponse(latency_tracker.stats())


@app.get("/session/{connection_id}")
async def session_info(connection_id: str) -> JSONResponse:
    session = _sessions.get(connection_id)
    if session is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return JSONResponse(session.summary())


@app.get("/sessions")
async def list_sessions() -> JSONResponse:
    return JSONResponse({
        "count":    len(_sessions),
        "sessions": [s.summary() for s in _sessions.values()],
    })
