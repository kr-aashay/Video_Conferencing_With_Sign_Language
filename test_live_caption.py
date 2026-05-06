"""
test_live_caption.py
════════════════════════════════════════════════════════════════════════════════
Dual-hand real-time sign language caption — production grade.
  • SignAggregator: compound sign detection (BOOK, HELP, etc.)
  • SemanticGatekeeper: only calls Gemini on Subject-Verb/Verb-Object patterns
  • GeminiChatSession: persistent conversation history (lock-free network calls)
  • AdaptiveLatencyGuard: instant fallback if LLM exceeds timeout
  • Teleprompter UI: aging captions + latency chip

Usage:  .venv/bin/python test_live_caption.py
Q = quit  |  R = reset
"""

from __future__ import annotations

import heapq
import sys
import threading
import time
import urllib.request
import json as _json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# ── Config ────────────────────────────────────────────────────────────────────
GESTURE_MODEL  = Path("gesture_recognizer.task")
GEMINI_API_KEY = ""
GEMINI_MODEL   = "gemini-2.0-flash"
OLLAMA_MODEL   = "gemma2:2b"
OLLAMA_URL     = "http://localhost:11434"
OLLAMA_TIMEOUT = 3.0
GEMINI_TIMEOUT = 3.0   # fail fast — instant fallback takes over

# Load config from .env
from pathlib import Path as _P
_env = _P(".env")
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line.startswith("GEMINI_API_KEY="):
            GEMINI_API_KEY = _line.split("=", 1)[1].strip()
        elif _line.startswith("GEMINI_MODEL="):
            GEMINI_MODEL = _line.split("=", 1)[1].strip()

GESTURE_HOLD   = 18    # frames to hold gesture (~0.6s at 30fps)
MIN_CONF       = 0.70  # >70% confidence gate
NEUTRAL_FRAMES = 45    # ~1.5s neutral pose triggers Gemini flush
PAUSE_RESET    = 90    # 3s full reset

# Only 1 LLM call in-flight at a time
_caption_sem = threading.Semaphore(1)

# Colours (BGR)
WHITE  = (255, 255, 255)
CYAN   = (255, 220, 0)
GREEN  = (0, 200, 80)
YELLOW = (0, 220, 220)
GREY   = (120, 120, 120)
BLACK  = (0, 0, 0)
ORANGE = (0, 165, 255)
RED    = (0, 60, 220)

# ── Gesture → intent mapping ──────────────────────────────────────────────────
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

# ── Gemini system instruction ─────────────────────────────────────────────────
GEMINI_SYSTEM = (
    "You are a professional Sign Language interpreter. "
    "I will provide signs in Gloss format (e.g., STORE ME GO). "
    "You must translate this into natural, grammatically correct English "
    "(e.g., 'I am going to the store.'). "
    "Use previous conversation history to resolve pronouns like HE or SHE. "
    "Output ONLY the sentence — no explanation, no quotes."
)

# ── Compound Sign Dictionary (dual-hand patterns) ─────────────────────────────
# Maps frozenset of (left_gesture, right_gesture) → compound sign label
COMPOUND_SIGNS: dict[frozenset, str] = {
    frozenset({"Open_Palm", "Closed_Fist"}): "HELP",
    frozenset({"Open_Palm", "Open_Palm"}):   "BOOK",
    frozenset({"Victory", "Victory"}):       "TOGETHER",
    frozenset({"Pointing_Up", "Open_Palm"}): "COME",
    frozenset({"ILoveYou", "ILoveYou"}):     "LOVE_YOU",
    frozenset({"Thumb_Up", "Open_Palm"}):    "THANK_YOU",
    frozenset({"Closed_Fist", "Closed_Fist"}): "STRONG",
}

# ── Semantic role tags for Gatekeeper ────────────────────────────────────────
# SUBJ = subject/pronoun, VERB = action, OBJ = object/modifier
_ROLE = {
    "Pointing_Up": "SUBJ",   # I
    "Open_Palm":   "VERB",   # HELLO / STOP
    "Closed_Fist": "VERB",   # STOP
    "Thumb_Up":    "OBJ",    # YES
    "Thumb_Down":  "OBJ",    # NO
    "Victory":     "OBJ",    # PEACE
    "ILoveYou":    "VERB",   # LOVE
    # compound signs
    "HELP":        "VERB",
    "BOOK":        "OBJ",
    "TOGETHER":    "OBJ",
    "COME":        "VERB",
    "LOVE_YOU":    "VERB",
    "THANK_YOU":   "VERB",
    "STRONG":      "OBJ",
}


# ── SignAggregator — dual-hand compound detection ─────────────────────────────

class SignAggregator:
    """
    Monitors both hand buffers simultaneously.
    If both hands are active with score >= MIN_CONF, checks the
    Compound Sign Dictionary before falling back to individual gestures.
    """

    def aggregate(
        self,
        left_gesture:  str | None,
        left_conf:     float,
        right_gesture: str | None,
        right_conf:    float,
    ) -> list[str]:
        """
        Returns a list of resolved sign tokens for this frame pair.
        May return a single compound token, two individual tokens, or one.
        """
        l_ok = left_gesture  and left_gesture  != "None" and left_conf  >= MIN_CONF
        r_ok = right_gesture and right_gesture != "None" and right_conf >= MIN_CONF

        if l_ok and r_ok:
            key = frozenset({left_gesture, right_gesture})
            if key in COMPOUND_SIGNS:
                return [COMPOUND_SIGNS[key]]
            # Both hands active but no compound match — return both
            return [right_gesture, left_gesture]

        if r_ok:
            return [right_gesture]
        if l_ok:
            return [left_gesture]
        return []


# ── SemanticGatekeeper — smart Gemini trigger ─────────────────────────────────

class SemanticGatekeeper:
    """
    Replaces the hard 90-frame reset with semantic completeness detection.

    Triggers Gemini when:
      1. A Subject-Verb pattern is detected in the token buffer
      2. A Verb-Object pattern is detected
      3. Neutral hand pose lasts > NEUTRAL_FRAMES (~1.5s) with pending tokens

    This prevents fragmented sentences like "Hello... I... am..."
    """

    # Patterns that indicate a semantically complete phrase
    _COMPLETE_PATTERNS = [
        ("SUBJ", "VERB"),   # I + HELLO, I + LOVE, I + STOP
        ("VERB", "OBJ"),    # HELLO + YES, STOP + NO
        ("SUBJ", "OBJ"),    # I + YES, I + NO
        ("VERB", "VERB"),   # HELLO + LOVE
    ]

    def is_complete(self, tokens: list[str]) -> bool:
        """Return True if the token sequence forms a complete semantic unit."""
        if len(tokens) < 2:
            return False
        roles = [_ROLE.get(t, "UNK") for t in tokens]
        for i in range(len(roles) - 1):
            pair = (roles[i], roles[i + 1])
            if pair in self._COMPLETE_PATTERNS:
                return True
        return False

    def should_flush_on_neutral(self, tokens: list[str], neutral_frames: int) -> bool:
        """Flush pending tokens after 1.5s neutral pose if we have anything."""
        return len(tokens) >= 1 and neutral_frames >= NEUTRAL_FRAMES


# ── Gemini Chat Session ───────────────────────────────────────────────────────

class GeminiChatSession:
    """
    Persistent Gemini conversation — lock-free during network calls.
    History is snapshotted before the call and committed after success.
    The lock is NEVER held during urlopen, so the camera never freezes.
    """

    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        self._key     = api_key
        self._model   = model
        self._timeout = timeout
        self._history: list[dict] = []   # [{role, parts:[{text}]}, ...]
        self._lock    = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._history.clear()

    def send(self, user_text: str) -> str:
        """Send a message and return the model reply. Thread-safe.
        
        The lock is held only for history reads/writes — NOT during the
        network call. This prevents the camera from freezing when Gemini
        is slow or rate-limited.
        """
        # Step 1: snapshot history under lock, then release
        with self._lock:
            history_snapshot = list(self._history) + [{
                "role":  "user",
                "parts": [{"text": user_text}],
            }]

        # Step 2: make network call WITHOUT holding the lock
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._key}"
        )
        payload = _json.dumps({
            "system_instruction": {
                "parts": [{"text": GEMINI_SYSTEM}]
            },
            "contents": history_snapshot,
            "generationConfig": {
                "temperature":     0.1,
                "maxOutputTokens": 60,
                "stopSequences":   ["\n"],
            },
        }).encode()

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self._timeout) as r:
            data  = _json.loads(r.read())
        reply = data["candidates"][0]["content"]["parts"][0]["text"]
        reply = reply.strip().split("\n")[0].strip().strip("\"'")
        if reply.lower().startswith("output:"):
            reply = reply[7:].strip()

        # Step 3: commit both turns to history under lock
        with self._lock:
            self._history.append({
                "role":  "user",
                "parts": [{"text": user_text}],
            })
            self._history.append({
                "role":  "model",
                "parts": [{"text": reply}],
            })
            # Keep history bounded (last 10 turns = 5 exchanges)
            if len(self._history) > 10:
                self._history = self._history[-10:]

        return reply


# ── AdaptiveLatencyGuard ──────────────────────────────────────────────────────

class AdaptiveLatencyGuard:
    """
    Tracks LLM call latency and immediately serves _intent_fallback
    if the rolling p99 exceeds the threshold.

    Also logs every call to logs/perf.jsonl for offline analysis.
    """

    def __init__(self, threshold_ms: float = 3000.0, window: int = 20) -> None:
        self._threshold = threshold_ms
        self._samples: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._log_dir = Path("logs")

    def record(self, latency_ms: float, source: str, caption: str) -> None:
        with self._lock:
            self._samples.append(latency_ms)
        # Write to perf.jsonl
        try:
            self._log_dir.mkdir(exist_ok=True)
            record = {
                "ts":         int(time.time() * 1000),
                "source":     source,
                "latency_ms": round(latency_ms, 2),
                "caption":    caption[:80],
                "p99_ms":     self.p99,
            }
            with open(self._log_dir / "perf.jsonl", "a") as f:
                f.write(_json.dumps(record) + "\n")
        except Exception:
            pass

    @property
    def p99(self) -> float:
        with self._lock:
            s = sorted(self._samples)
        if not s:
            return 0.0
        idx = min(int(0.99 * len(s)), len(s) - 1)
        return round(s[idx], 2)

    def should_skip_llm(self) -> bool:
        """Return True if recent p99 suggests LLM is too slow."""
        return self.p99 > self._threshold and len(self._samples) >= 5

    def latency_color_bgr(self) -> tuple:
        """Return BGR color for the latency chip: green/yellow/red."""
        p = self.p99
        if p == 0 or p < 500:
            return (0, 200, 80)    # green
        if p < 2000:
            return (0, 200, 220)   # yellow
        return (0, 60, 220)        # red

    def latency_label(self) -> str:
        p = self.p99
        if p == 0:
            return "LLM: --"
        return f"p99:{p:.0f}ms"


# Module-level guard instance
_latency_guard = AdaptiveLatencyGuard(threshold_ms=3000.0)


# ── Priority queue intent resolver ────────────────────────────────────────────

def resolve_intents(tokens: list[str]) -> list[str]:
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


# ── Natural fallback (when Gemini/Ollama unavailable) ────────────────────────

_FALLBACK_MAP = {
    frozenset(["hello"]):                 "Hello there.",
    frozenset(["hello", "i"]):            "Hello, I am here.",
    frozenset(["i", "yes"]):              "Yes, I agree.",
    frozenset(["i", "no"]):               "No, I don't agree.",
    frozenset(["i", "love"]):             "I love you.",
    frozenset(["stop", "no"]):            "Please stop.",
    frozenset(["yes", "agree"]):          "Yes, I completely agree.",
    frozenset(["no", "disagree"]):        "No, I disagree.",
    frozenset(["hello", "yes"]):          "Hello! Yes, I'm ready.",
    frozenset(["hello", "i", "yes"]):     "Hello, I think that's great.",
    frozenset(["i", "stop", "no"]):       "I need to stop. This isn't right.",
    frozenset(["peace", "two"]):          "Let's keep the peace.",
    frozenset(["i", "attention"]):        "Excuse me, I have something to say.",
    frozenset(["hello", "i", "attention"]):"Hello, I'd like your attention please.",
    frozenset(["yes", "good"]):           "Yes, that's a good idea.",
    frozenset(["no", "bad"]):             "No, that's not a good idea.",
    frozenset(["i", "care", "family"]):   "I care about my family.",
    frozenset(["stop", "strong"]):        "Stop. I'm serious about this.",
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


# ── Caption generation ────────────────────────────────────────────────────────

def generate_caption(
    tokens: list[str],
    session: GeminiChatSession | None,
) -> str:
    """
    Try Gemini chat session first (maintains context),
    fall back to Ollama, then local fallback.
    AdaptiveLatencyGuard skips LLM entirely if p99 is too high.
    """
    intents = resolve_intents(tokens)
    if not intents:
        return ""

    labels = [GESTURE_LABELS.get(t, t) for t in tokens if t != "None"]
    if not labels:
        return _intent_fallback(intents)

    user_msg = f"Signs: {', '.join(labels)}"

    # ── AdaptiveLatencyGuard: skip LLM if it's been too slow ─────────────────
    if _latency_guard.should_skip_llm():
        result = _intent_fallback(intents)
        print(f"  [Guard] LLM skipped (p99={_latency_guard.p99:.0f}ms) → {result}")
        return result

    # ── Gemini chat session (primary) ─────────────────────────────────────────
    if session is not None:
        t0 = time.perf_counter()
        try:
            result = session.send(user_msg)
            elapsed = (time.perf_counter() - t0) * 1000
            _latency_guard.record(elapsed, "gemini", result)
            if result:
                print(f"  [Gemini {elapsed:.0f}ms] {result}")
                return result
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            _latency_guard.record(elapsed, "gemini_fail", "")
            print(f"  [Gemini] {e} — trying Ollama")

    # ── Ollama (fallback) ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        payload = _json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": GEMINI_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 60, "stop": ["\n"]},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as r:
            raw    = _json.loads(r.read())["message"]["content"].strip()
            result = raw.split("\n")[0].strip().strip("\"'")
            if result.lower().startswith("output:"):
                result = result[7:].strip()
            if result:
                elapsed = (time.perf_counter() - t0) * 1000
                _latency_guard.record(elapsed, "ollama", result)
                return result
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        _latency_guard.record(elapsed, "ollama_fail", "")
        print(f"  [Ollama] {e}")

    return _intent_fallback(intents)


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_frame(frame, gestures_by_hand, tokens, caption_lines, status, fps, h, w, compound=None):
    """
    caption_lines: list of (text, age) where age=0 is newest.
    compound: compound sign label if detected this frame, else None.
    Teleprompter style: newest caption full brightness, older ones dim/shrink.
    """
    # Top bar background
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 85), BLACK, -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    # Show each detected hand's gesture
    y_pos = 28
    if gestures_by_hand:
        for hand_label, gesture, conf in gestures_by_hand:
            if gesture and gesture != "None" and conf >= MIN_CONF:
                lbl   = GESTURE_LABELS.get(gesture, gesture)
                color = GREEN if hand_label == "Right" else ORANGE
                cv2.putText(
                    frame,
                    f"{hand_label}: {gesture} [{lbl}] {conf*100:.0f}%",
                    (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2,
                )
                y_pos += 24
    else:
        cv2.putText(frame, "Hold a gesture for 0.6s...",
                    (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 1)

    # Compound sign badge
    if compound:
        cv2.putText(frame, f"[COMPOUND: {compound}]",
                    (20, y_pos + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)

    cv2.putText(frame, f"{status}  |  {fps:.0f} FPS",
                (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)

    # Latency chip (top-right)
    chip_color = _latency_guard.latency_color_bgr()
    chip_label = _latency_guard.latency_label()
    (cw, _), _ = cv2.getTextSize(chip_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.rectangle(frame, (w - cw - 18, 8), (w - 6, 26), (20, 20, 20), -1)
    cv2.rectangle(frame, (w - cw - 18, 8), (w - 6, 26), chip_color, 1)
    cv2.putText(frame, chip_label, (w - cw - 12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, chip_color, 1)

    # Token chips (sign buffer)
    if tokens:
        cx = 20
        for tok in tokens[-8:]:
            lbl = GESTURE_LABELS.get(tok, tok) or tok
            (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (cx - 3, 92), (cx + tw + 7, 114), (30, 30, 70), -1)
            cv2.rectangle(frame, (cx - 3, 92), (cx + tw + 7, 114), CYAN, 1)
            cv2.putText(frame, lbl, (cx, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1)
            cx += tw + 14

    # Caption box — teleprompter style
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - 130), (w, h), BLACK, -1)
    cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, h - 130), (w, h - 130), (255, 180, 0), 2)

    if caption_lines:
        # Render up to 3 lines, newest at bottom, older above and dimmer
        visible = caption_lines[-3:]  # last 3
        n = len(visible)
        y = h - 110 + (3 - n) * 34
        for idx, (text, age) in enumerate(visible):
            # age 0 = newest (full white), 1 = medium grey, 2+ = dim
            if age == 0:
                color = WHITE
                scale = 0.76
                thick = 2
            elif age == 1:
                color = (180, 180, 180)
                scale = 0.62
                thick = 1
            else:
                color = (100, 100, 100)
                scale = 0.52
                thick = 1
            # Word-wrap
            words = text.split()
            lines, line = [], []
            for word in words:
                line.append(word)
                if len(" ".join(line)) > 58:
                    lines.append(" ".join(line[:-1]))
                    line = [word]
            if line:
                lines.append(" ".join(line))
            for ln in lines[:1]:  # one line per caption entry
                cv2.putText(frame, ln, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
            y += 34
    else:
        cv2.putText(frame, "Perform a gesture to see caption...",
                    (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.62, GREY, 1)

    cv2.putText(frame, "Q=quit  R=reset",
                (w - 175, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Aashay's Sign Lang — Dual-Hand Live Sign Language Caption")
    print("=" * 60)

    if not GESTURE_MODEL.exists():
        print("  Downloading gesture model...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
        )
        urllib.request.urlretrieve(url, GESTURE_MODEL)

    # Check Ollama
    ollama_ok = False
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            models    = [m["name"] for m in _json.loads(r.read())["models"]]
            ollama_ok = len(models) > 0
        print(f"  Ollama: {'OK ' + OLLAMA_MODEL if ollama_ok else 'no models found'}")
    except Exception:
        print("  Ollama: not running")

    # Init Gemini chat session
    gemini_session: GeminiChatSession | None = None
    if GEMINI_API_KEY:
        gemini_session = GeminiChatSession(GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TIMEOUT)
        print(f"  Gemini: chat session ready ({GEMINI_MODEL})")
    else:
        print("  Gemini: no key — using Ollama/fallback")

    # Init aggregator + gatekeeper
    aggregator  = SignAggregator()
    gatekeeper  = SemanticGatekeeper()

    # MediaPipe gesture recognizer — BOTH hands
    BaseOptions = mp.tasks.BaseOptions
    GR          = mp.tasks.vision.GestureRecognizer
    GROpts      = mp.tasks.vision.GestureRecognizerOptions
    RunMode     = mp.tasks.vision.RunningMode

    gr_opts = GROpts(
        base_options=BaseOptions(model_asset_path=str(GESTURE_MODEL)),
        running_mode=RunMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Cannot open camera")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("  Camera ready. Hold a gesture for 0.6s to confirm it.\n")

    # Per-hand hold buffers
    hold_bufs: dict[str, deque] = {
        "Left":  deque(maxlen=GESTURE_HOLD),
        "Right": deque(maxlen=GESTURE_HOLD),
    }
    last_gesture: dict[str, str] = {"Left": "", "Right": ""}
    last_conf:    dict[str, float] = {"Left": 0.0, "Right": 0.0}

    # State
    tokens: list[str]                    = []
    caption_lines: list[tuple[str, int]] = []  # (text, age)
    status        = "Ready"
    no_input      = 0
    neutral_frames = 0
    compound_label: str | None = None
    fps_t0, fps_n, fps = time.perf_counter(), 0, 0.0

    def _push_caption(text: str) -> None:
        """Add a new caption line, age all existing ones."""
        nonlocal caption_lines
        # Age existing
        caption_lines = [(t, a + 1) for t, a in caption_lines]
        caption_lines.append((text, 0))
        # Keep last 3
        if len(caption_lines) > 3:
            caption_lines = caption_lines[-3:]

    def _fire_caption(snap: list[str], sess: GeminiChatSession | None) -> None:
        """Background thread: generate caption and push to display."""
        nonlocal status
        if not _caption_sem.acquire(blocking=False):
            return
        try:
            # Instant fallback shown immediately
            intents = resolve_intents(snap)
            instant = _intent_fallback(intents) if intents else ""
            if instant:
                _push_caption(instant)
                status = "Caption ready"

            # Try LLM to upgrade
            c = generate_caption(snap, sess)
            if c and c != instant:
                _push_caption(c)
                status = "Caption ready"
                print(f"  Caption: {c}")
        finally:
            _caption_sem.release()

    with GR.create_from_options(gr_opts) as recognizer:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize(mp_img)

            # ── Parse per-hand results ────────────────────────────────────────
            gestures_by_hand: list[tuple[str, str, float]] = []
            any_detected = False
            current: dict[str, tuple[str, float]] = {}

            num_hands = len(result.gestures) if result.gestures else 0
            for i in range(num_hands):
                hand_label = "Right"
                if result.handedness and i < len(result.handedness):
                    cats = result.handedness[i]
                    if cats:
                        hand_label = cats[0].display_name

                det_gesture, det_conf = "None", 0.0
                if result.gestures and i < len(result.gestures):
                    gs = result.gestures[i]
                    if gs:
                        best        = max(gs, key=lambda g: g.score)
                        det_gesture = best.category_name
                        det_conf    = best.score

                gestures_by_hand.append((hand_label, det_gesture, det_conf))
                current[hand_label] = (det_gesture, det_conf)

                if det_conf >= MIN_CONF and det_gesture != "None":
                    hold_bufs[hand_label].append(det_gesture)
                    last_conf[hand_label] = det_conf
                    any_detected = True
                else:
                    hold_bufs[hand_label].clear()

            # ── SignAggregator: check compound signs each frame ───────────────
            l_gest, l_conf = current.get("Left",  ("None", 0.0))
            r_gest, r_conf = current.get("Right", ("None", 0.0))
            compound_label = None
            if l_conf >= MIN_CONF and r_conf >= MIN_CONF:
                key = frozenset({l_gest, r_gest})
                if key in COMPOUND_SIGNS:
                    compound_label = COMPOUND_SIGNS[key]

            # ── Confirm individual gestures from hold buffers ─────────────────
            newly_confirmed: list[str] = []
            for hand_label in ("Left", "Right"):
                buf = hold_bufs[hand_label]
                if len(buf) == GESTURE_HOLD and len(set(buf)) == 1:
                    confirmed = buf[0]
                    buf.clear()
                    if confirmed != last_gesture[hand_label]:
                        last_gesture[hand_label] = confirmed
                        newly_confirmed.append(confirmed)

            # ── If compound sign confirmed by both hands ──────────────────────
            if compound_label and len(newly_confirmed) >= 1:
                tokens.append(compound_label)
                if len(tokens) > 5:
                    tokens = tokens[-5:]
                lbl = compound_label
                print(f"  [COMPOUND] {lbl}")
                status = f"Compound: {lbl}"
                newly_confirmed = []  # consumed by compound

            # ── Add individual confirmed gestures ─────────────────────────────
            for confirmed in newly_confirmed:
                tokens.append(confirmed)
                if len(tokens) > 5:
                    tokens = tokens[-5:]
                lbl = GESTURE_LABELS.get(confirmed, confirmed)
                print(f"  [{('L' if confirmed in [current.get('Left',('',0))[0]] else 'R')}] {confirmed} [{lbl}]")
                status = f"Detected: {lbl}"

            # ── SemanticGatekeeper: decide when to call Gemini ────────────────
            should_generate = False
            if newly_confirmed or compound_label:
                if gatekeeper.is_complete(tokens):
                    should_generate = True
                    print(f"  [Gate] Subject-Verb/Verb-Object pattern detected")

            if should_generate and tokens:
                snap     = list(tokens)
                sess_ref = gemini_session
                tokens   = []  # clear after firing
                threading.Thread(
                    target=_fire_caption, args=(snap, sess_ref), daemon=True
                ).start()

            # ── Track no-input / neutral frames ──────────────────────────────
            if any_detected:
                no_input      = 0
                neutral_frames = 0
            else:
                no_input       += 1
                neutral_frames += 1

            # Neutral flush: 1.5s neutral with pending tokens
            if gatekeeper.should_flush_on_neutral(tokens, neutral_frames) and not should_generate:
                snap     = list(tokens)
                sess_ref = gemini_session
                tokens   = []
                neutral_frames = 0
                print(f"  [Gate] Neutral flush after {NEUTRAL_FRAMES} frames")
                threading.Thread(
                    target=_fire_caption, args=(snap, sess_ref), daemon=True
                ).start()

            # Full reset after 3s silence
            if no_input >= PAUSE_RESET:
                tokens.clear()
                last_gesture   = {"Left": "", "Right": ""}
                no_input       = 0
                neutral_frames = 0
                status         = "Ready"
                if gemini_session:
                    gemini_session.reset()

            # Short pause — allow same gesture to fire again
            elif no_input >= 45:
                last_gesture = {"Left": "", "Right": ""}

            # ── Draw hand skeletons ───────────────────────────────────────────
            if result.hand_landmarks:
                CONN = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17),
                ]
                for i, hand in enumerate(result.hand_landmarks):
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                    hand_label = "Right"
                    if result.handedness and i < len(result.handedness):
                        cats = result.handedness[i]
                        if cats:
                            hand_label = cats[0].display_name
                    skel_color = GREEN if hand_label == "Right" else ORANGE
                    for a, b in CONN:
                        cv2.line(frame, pts[a], pts[b], skel_color, 1)
                    for pt in pts:
                        cv2.circle(frame, pt, 3, CYAN, -1)

            # FPS counter
            fps_n += 1
            if time.perf_counter() - fps_t0 >= 1.0:
                fps    = fps_n / (time.perf_counter() - fps_t0)
                fps_t0 = time.perf_counter()
                fps_n  = 0

            draw_frame(frame, gestures_by_hand, tokens,
                       caption_lines, status, fps, h, w, compound_label)

            cv2.imshow("Aashay's Sign Lang — Dual-Hand Caption  (Q=quit  R=reset)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                tokens.clear()
                caption_lines.clear()
                last_gesture   = {"Left": "", "Right": ""}
                no_input       = 0
                neutral_frames = 0
                status         = "Reset"
                if gemini_session:
                    gemini_session.reset()
                print("  [RESET]")

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Stopped.")


if __name__ == "__main__":
    main()
