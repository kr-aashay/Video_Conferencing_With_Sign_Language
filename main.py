
from __future__ import annotations

import json
import logging
import os
import time
import threading
import urllib.request
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import heapq
from dotenv import load_dotenv
from fastapi import (
    FastAPI, Request, Form, WebSocket, WebSocketDisconnect,
    Depends, HTTPException, status,
)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt as _bcrypt

class bcrypt:
    """Thin shim so the rest of the code keeps `bcrypt.hash()` / `bcrypt.verify()` calls."""
    @staticmethod
    def hash(password: str) -> str:
        return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()

    @staticmethod
    def verify(password: str, hashed: str) -> bool:
        try:
            return _bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception:
            return False

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MONGODB_URI   = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB    = os.getenv("MONGODB_DB",  "aashay_signlang")
SECRET_KEY    = os.getenv("SECRET_KEY",  "aashay-signlang-secret-change-in-production")
ZEGO_APP_ID   = os.getenv("ZEGO_APP_ID", "")
ZEGO_SERVER_SECRET = os.getenv("ZEGO_SERVER_SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TIMEOUT = 3.0

log = logging.getLogger("aashay.signlang")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Session serializer ────────────────────────────────────────────────────────
_signer = URLSafeTimedSerializer(SECRET_KEY)

def create_session_token(user_id: str) -> str:
    return _signer.dumps(user_id, salt="session")

def verify_session_token(token: str) -> Optional[str]:
    try:
        return _signer.loads(token, salt="session", max_age=86400 * 7)
    except (BadSignature, SignatureExpired):
        return None

# ── MongoDB client (module-level, reused across requests) ─────────────────────
_mongo_client: AsyncIOMotorClient | None = None

def get_db():
    return _mongo_client[MONGODB_DB]

# ── Gemini caption generation (same logic as test_live_caption.py) ────────────

GESTURE_INTENTS = {
    "Thumb_Up":    [("yes", 3), ("agree", 2), ("good", 1)],
    "Thumb_Down":  [("no", 3), ("disagree", 2), ("bad", 1)],
    "Open_Palm":   [("hello", 3), ("stop", 2), ("wait", 1)],
    "Closed_Fist": [("stop", 3), ("no", 2), ("strong", 1)],
    "Pointing_Up": [("i", 3), ("attention", 2), ("one", 1)],
    "Victory":     [("peace", 3), ("two", 2), ("victory", 1)],
    "ILoveYou":    [("love", 3), ("care", 2), ("family", 1)],
}

GESTURE_LABELS = {
    "Closed_Fist": "STOP",
    "Open_Palm":   "HELLO",
    "Pointing_Up": "I",
    "Thumb_Down":  "NO",
    "Thumb_Up":    "YES",
    "Victory":     "PEACE",
    "ILoveYou":    "LOVE",
    "None":        "",
}

GEMINI_SYSTEM = (
    "You are a professional Sign Language interpreter. "
    "I will provide signs in Gloss format (e.g., STORE ME GO). "
    "You must translate this into natural, grammatically correct English "
    "(e.g., 'I am going to the store.'). "
    "Use previous conversation history to resolve pronouns like HE or SHE. "
    "Output ONLY the sentence — no explanation, no quotes."
)

_FALLBACK_MAP = {
    frozenset(["hello"]):              "Hello there.",
    frozenset(["hello", "i"]):         "Hello, I am here.",
    frozenset(["i", "yes"]):           "Yes, I agree.",
    frozenset(["i", "no"]):            "No, I don't agree.",
    frozenset(["i", "love"]):          "I love you.",
    frozenset(["stop", "no"]):         "Please stop.",
    frozenset(["yes", "agree"]):       "Yes, I completely agree.",
    frozenset(["no", "disagree"]):     "No, I disagree.",
    frozenset(["hello", "yes"]):       "Hello! Yes, I'm ready.",
    frozenset(["i", "stop", "no"]):    "I need to stop. This isn't right.",
    frozenset(["i", "attention"]):     "Excuse me, I have something to say.",
    frozenset(["yes", "good"]):        "Yes, that's a good idea.",
    frozenset(["no", "bad"]):          "No, that's not a good idea.",
}

def _intent_fallback(intents: list[str]) -> str:
    key = frozenset(intents[:3])
    if key in _FALLBACK_MAP:
        return _FALLBACK_MAP[key]
    best, best_score = "", 0
    for k, v in _FALLBACK_MAP.items():
        score = len(k & key)
        if score > best_score:
            best, best_score = v, score
    if best_score >= 1:
        return best
    return " ".join(intents[:3]).capitalize() + "."

def _resolve_intents(tokens: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for i, tok in enumerate(tokens):
        recency = 1.0 + (i / max(len(tokens) - 1, 1)) * 0.5
        for intent, w in GESTURE_INTENTS.get(tok, []):
            scores[intent] = scores.get(intent, 0) + w * recency
    heap = [(-s, k) for k, s in scores.items()]
    heapq.heapify(heap)
    top = []
    while heap and len(top) < 5:
        _, k = heapq.heappop(heap)
        top.append(k)
    return top

# Per-room Gemini conversation history (room_id → list of turns)
_room_histories: dict[str, list[dict]] = {}
_history_lock = threading.Lock()

def _gemini_caption(tokens: list[str], room_id: str) -> str:
    """Generate caption via Gemini with per-room conversation history."""
    intents = _resolve_intents(tokens)
    labels  = [GESTURE_LABELS.get(t, t) for t in tokens if t != "None"]
    if not labels:
        return _intent_fallback(intents) if intents else ""

    user_msg = f"Signs: {', '.join(labels)}"

    if not GEMINI_API_KEY:
        return _intent_fallback(intents)

    with _history_lock:
        history = list(_room_histories.get(room_id, []))
        history.append({"role": "user", "parts": [{"text": user_msg}]})

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = json.dumps({
        "system_instruction": {"parts": [{"text": GEMINI_SYSTEM}]},
        "contents": history,
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 60,
            "stopSequences": ["\n"],
        },
    }).encode()

    try:
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT) as r:
            data  = json.loads(r.read())
        reply = data["candidates"][0]["content"]["parts"][0]["text"]
        reply = reply.strip().split("\n")[0].strip().strip("\"'")
        if reply.lower().startswith("output:"):
            reply = reply[7:].strip()

        # Commit to history
        with _history_lock:
            hist = _room_histories.setdefault(room_id, [])
            hist.append({"role": "user",  "parts": [{"text": user_msg}]})
            hist.append({"role": "model", "parts": [{"text": reply}]})
            if len(hist) > 10:
                _room_histories[room_id] = hist[-10:]

        return reply
    except Exception as e:
        log.warning("Gemini error: %s — using fallback", e)
        return _intent_fallback(intents)

# ── Caption WebSocket room manager ────────────────────────────────────────────
# Maps room_id → set of connected receiver WebSockets
_caption_rooms: dict[str, set[WebSocket]] = {}
_caption_lock  = threading.Lock()

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mongo_client
    _mongo_client = AsyncIOMotorClient(MONGODB_URI)
    # Ensure unique indexes on users
    db = get_db()
    await db.users.create_index("email",    unique=True)
    await db.users.create_index("username", unique=True)
    # Index captions by room for fast history queries
    await db.captions.create_index([("room_id", 1), ("ts", 1)])
    log.info("MongoDB connected: %s / %s", MONGODB_URI, MONGODB_DB)
    yield
    _mongo_client.close()
    log.info("MongoDB disconnected")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Aashay's Sign Lang", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
# Serve the local gesture recognizer model so the browser doesn't need to
# fetch it from storage.googleapis.com (slow / unreliable CDN).
from fastapi.responses import FileResponse as _FileResponse
@app.get("/models/gesture_recognizer.task")
async def gesture_model():
    return _FileResponse("gesture_recognizer.task", media_type="application/octet-stream")
templates = Jinja2Templates(directory="templates")

# ── Auth helpers ──────────────────────────────────────────────────────────────

async def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get("session")
    if not token:
        return None
    user_id = verify_session_token(token)
    if not user_id:
        return None
    db   = get_db()
    user = await db.users.find_one({"_id": user_id})
    return user

async def require_user(request: Request) -> dict:
    user = await get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/login"},
        )
    return user

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home():
    return RedirectResponse(url="/login")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, msg: str = ""):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "msg": msg,
    })


@app.post("/login", response_class=HTMLResponse)
async def login_post(
    request: Request,
    email:    str = Form(...),
    password: str = Form(...),
):
    db   = get_db()
    user = await db.users.find_one({"email": email})
    if not user or not bcrypt.verify(password, user["password"]):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "msg": "Invalid email or password.",
        })
    token = create_session_token(user["_id"])
    resp  = RedirectResponse(url="/dashboard", status_code=303)
    resp.set_cookie("session", token, httponly=True, max_age=86400 * 7)
    return resp


@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie("session")
    return resp


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, msg: str = ""):
    return templates.TemplateResponse("register.html", {
        "request": request,
        "msg": msg,
    })


@app.post("/register", response_class=HTMLResponse)
async def register_post(
    request:    Request,
    email:      str = Form(...),
    first_name: str = Form(...),
    last_name:  str = Form(...),
    username:   str = Form(...),
    password:   str = Form(...),
):
    db = get_db()
    # Check duplicates
    if await db.users.find_one({"email": email}):
        return templates.TemplateResponse("register.html", {
            "request": request,
            "msg": "Email already registered.",
        })
    if await db.users.find_one({"username": username}):
        return templates.TemplateResponse("register.html", {
            "request": request,
            "msg": "Username already taken.",
        })
    user_id = str(uuid.uuid4())
    await db.users.insert_one({
        "_id":        user_id,
        "email":      email,
        "first_name": first_name,
        "last_name":  last_name,
        "username":   username,
        "password":   bcrypt.hash(password),
    })
    return templates.TemplateResponse("login.html", {
        "request": request,
        "msg": "Account created! You can now log in.",
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("dashboard.html", {
        "request":    request,
        "first_name": user["first_name"],
        "last_name":  user["last_name"],
    })


@app.get("/meeting", response_class=HTMLResponse)
async def meeting(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("meeting.html", {
        "request":      request,
        "username":     user["username"],
        "zego_app_id":  ZEGO_APP_ID,
        "zego_secret":  ZEGO_SERVER_SECRET,
    })


@app.get("/join", response_class=HTMLResponse)
async def join_page(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("join.html", {"request": request})


@app.post("/join", response_class=HTMLResponse)
async def join_post(request: Request, roomID: str = Form(...)):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url=f"/meeting?roomID={roomID}", status_code=303)


# ── Caption WebSocket — receiver side ────────────────────────────────────────
# IMPORTANT: this route must be registered BEFORE the {user_id} wildcard route
# so FastAPI matches /receiver literally instead of treating it as a user_id.

@app.websocket("/ws/caption/{room_id}/receiver")
async def ws_caption_receiver(
    websocket: WebSocket,
    room_id:   str,
):
    """
    Receiver connects here and listens for caption messages.
    No sending required — purely a push channel.
    """
    await websocket.accept()
    log.info("Receiver WS connected | room=%s", room_id)

    with _caption_lock:
        if room_id not in _caption_rooms:
            _caption_rooms[room_id] = set()
        _caption_rooms[room_id].add(websocket)

    try:
        # Keep alive — receiver just listens
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        log.info("Receiver WS disconnected | room=%s", room_id)
    except Exception:
        pass
    finally:
        with _caption_lock:
            _caption_rooms.get(room_id, set()).discard(websocket)


# ── Caption WebSocket — signer side ──────────────────────────────────────────
# The signer's browser sends confirmed gesture tokens here.
# The server generates a caption, saves it to MongoDB, and broadcasts it to
# all receivers in the room.

@app.websocket("/ws/caption/{room_id}/{user_id}")
async def ws_caption_sender(
    websocket: WebSocket,
    room_id:   str,
    user_id:   str,
):
    """
    Signer connects here and sends JSON:
      { "type": "gesture", "tokens": ["Open_Palm", "Pointing_Up"] }
      { "type": "reset" }

    Server responds with:
      { "type": "caption", "caption": "Hello, I am here.", "ts": 1234567890 }

    Caption is also:
      • Saved to MongoDB (captions collection)
      • Broadcast to all receivers in the same room
    """
    await websocket.accept()
    log.info("Caption WS connected | room=%s user=%s", room_id, user_id)

    # Register this room for receivers
    with _caption_lock:
        if room_id not in _caption_rooms:
            _caption_rooms[room_id] = set()

    # Seed in-memory Gemini history from MongoDB (last 5 turns) so context
    # survives server restarts and new connections to the same room.
    db = get_db()
    with _history_lock:
        if room_id not in _room_histories:
            recent = await db.captions.find(
                {"room_id": room_id},
                sort=[("ts", -1)],
                limit=5,
            ).to_list(length=5)
            if recent:
                history_seed = []
                for doc in reversed(recent):
                    labels = [GESTURE_LABELS.get(t, t) for t in doc.get("tokens", []) if t != "None"]
                    if labels:
                        history_seed.append({"role": "user",  "parts": [{"text": f"Signs: {', '.join(labels)}"}]})
                        history_seed.append({"role": "model", "parts": [{"text": doc["caption"]}]})
                _room_histories[room_id] = history_seed
                log.info("Seeded room history from MongoDB | room=%s turns=%d", room_id, len(history_seed))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "gesture":
                tokens = msg.get("tokens", [])
                if not tokens:
                    continue

                # Generate caption in a thread (Gemini call is blocking)
                import asyncio
                loop = asyncio.get_event_loop()
                caption = await loop.run_in_executor(
                    None, _gemini_caption, tokens, room_id
                )

                if not caption:
                    continue

                ts = int(time.time() * 1000)

                # ── Persist caption to MongoDB ────────────────────────────
                try:
                    await db.captions.insert_one({
                        "room_id": room_id,
                        "user_id": user_id,
                        "tokens":  tokens,
                        "caption": caption,
                        "ts":      ts,
                    })
                except Exception as db_err:
                    log.warning("Caption DB write failed: %s", db_err)

                payload = json.dumps({
                    "type":    "caption",
                    "caption": caption,
                    "tokens":  tokens,
                    "ts":      ts,
                })

                # Send back to signer
                try:
                    await websocket.send_text(payload)
                except Exception:
                    pass

                # Broadcast to all receivers in this room
                with _caption_lock:
                    receivers = set(_caption_rooms.get(room_id, set()))
                dead = set()
                for recv_ws in receivers:
                    try:
                        await recv_ws.send_text(payload)
                    except Exception:
                        dead.add(recv_ws)
                if dead:
                    with _caption_lock:
                        _caption_rooms.get(room_id, set()).difference_update(dead)

            elif msg.get("type") == "reset":
                with _history_lock:
                    _room_histories.pop(room_id, None)
                await websocket.send_text(json.dumps({"type": "ack", "message": "reset"}))

    except WebSocketDisconnect:
        log.info("Caption WS disconnected | room=%s user=%s", room_id, user_id)
    except Exception as exc:
        log.error("Caption WS error | room=%s: %s", room_id, exc)


# ── Caption history — REST endpoint ──────────────────────────────────────────

@app.get("/captions/{room_id}")
async def get_captions(room_id: str, limit: int = 50):
    """
    Return the last `limit` captions for a room (newest first).
    Useful for late-joiners to catch up on what was signed.
    """
    db = get_db()
    docs = await db.captions.find(
        {"room_id": room_id},
        {"_id": 0, "room_id": 0},
        sort=[("ts", -1)],
        limit=limit,
    ).to_list(length=limit)
    return JSONResponse({"room_id": room_id, "captions": docs})


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "db": MONGODB_DB})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
