"""
test_live_caption.py
════════════════════════════════════════════════════════════════════════════════
Fully hands-free sign language caption system.

NO keyboard. NO mode switching. NO pre-registration.

Dual-track detection runs simultaneously:
  Track 1 — Gesture recognition  (7 common signs: YES, NO, HELLO, etc.)
  Track 2 — ASL fingerspelling   (A-Z letters → words → sentences)

The system automatically detects which track is active.
Spelled words are committed when a natural pause is detected.
Phi-3 receives both gesture intents AND spelled words for context.

Usage:  .venv/bin/python test_live_caption.py
Quit:   Q or Esc
Reset:  R (clears current sentence)
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
GESTURE_MODEL   = Path("gesture_recognizer.task")
OLLAMA_MODEL    = "phi3"
OLLAMA_URL      = "http://localhost:11434"
OLLAMA_TIMEOUT  = 4.0

# Gesture detection
GESTURE_HOLD    = 18    # frames to hold a gesture before confirming (~0.6s)
MIN_CONF        = 0.72

# Fingerspelling
LETTER_HOLD     = 10    # frames to hold a letter shape (~0.33s)
LETTER_GAP      = 20    # frames of no letter = word boundary (~0.67s)
MIN_WORD_LEN    = 1     # minimum letters to commit a word

# Caption generation
MAX_TOKENS      = 6     # max gesture/word tokens before generating caption
PAUSE_RESET     = 90    # frames of no input before clearing (3s)

# Colours (BGR)
WHITE  = (255, 255, 255)
CYAN   = (255, 220,   0)
GREEN  = (0,   200,  80)
YELLOW = (0,   220, 220)
GREY   = (120, 120, 120)
BLACK  = (0,     0,   0)
RED    = (0,    60, 220)

# ── Gesture → intent mapping ──────────────────────────────────────────────────
GESTURE_INTENTS = {
    "Thumb_Up":    [("yes",3),("agree",2),("good",1)],
    "Thumb_Down":  [("no",3),("disagree",2),("bad",1)],
    "Open_Palm":   [("hello",3),("stop",2),("wait",1)],
    "Closed_Fist": [("stop",3),("no",2),("strong",1)],
    "Pointing_Up": [("i",3),("attention",2),("one",1)],
    "Victory":     [("peace",3),("two",2),("victory",1)],
    "ILoveYou":    [("love",3),("care",2),("family",1)],
}

GESTURE_LABELS = {
    "Closed_Fist": "STOP",
    "Open_Palm":   "HELLO",
    "Pointing_Up": "I / ATTENTION",
    "Thumb_Down":  "NO",
    "Thumb_Up":    "YES",
    "Victory":     "PEACE",
    "ILoveYou":    "LOVE",
    "None":        "",
}


# ── ASL letter classifier (from hand landmarks) ───────────────────────────────

def _ext(lms, tip, pip):
    return lms[tip][1] < lms[pip][1]

def _dist(lms, a, b):
    return float(np.linalg.norm(lms[a] - lms[b]))

def classify_letter(landmarks) -> str | None:
    lms = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    t = abs(lms[4][0] - lms[2][0]) > 0.04
    i = _ext(lms, 8,  6)
    m = _ext(lms, 12, 10)
    r = _ext(lms, 16, 14)
    p = _ext(lms, 20, 18)

    # All curled
    if not i and not m and not r and not p:
        if not t:
            return 'A'
        if _dist(lms, 4, 8) < 0.05:
            return 'E'
        if _dist(lms, 4, 6) < 0.05:
            return 'T'
        if _dist(lms, 4, 12) < 0.05:
            return 'N'
        if _dist(lms, 4, 16) < 0.05:
            return 'M'
        return 'S'

    # One finger
    if i and not m and not r and not p:
        return 'D' if not t else 'G'
    if not i and m and not r and not p:
        return 'X'
    if not i and not m and not r and p:
        return 'I' if not t else 'J'

    # Two fingers
    if i and m and not r and not p:
        if _dist(lms, 8, 12) < 0.04:
            return 'R'
        return 'U' if not t else 'V'
    if i and not m and not r and p:
        return 'H'

    # Three fingers
    if i and m and r and not p:
        return 'W'

    # Four fingers
    if i and m and r and p:
        return 'B' if not t else '4'

    # Thumb combos
    if t and i and not m and not r and not p:
        return 'L' if abs(lms[4][0] - lms[8][0]) > 0.08 else 'F'
    if t and not i and not m and not r and p:
        return 'Y'

    # C / O shapes
    if (lms[8][1] < lms[5][1] and lms[12][1] < lms[9][1] and
            lms[16][1] < lms[13][1] and lms[20][1] < lms[17][1]):
        if _dist(lms, 4, 8) > 0.08:
            return 'C'
        if _dist(lms, 4, 8) < 0.05:
            return 'O'

    if i and m and not r and p:
        return 'K'

    return None


# ── Priority queue intent resolver ────────────────────────────────────────────

def resolve_intents(tokens: list[str]) -> list[str]:
    """
    tokens: mix of gesture names and spelled words like [AASHAY]
    Returns priority-ordered list of intent words.
    Spelled words pass through directly at highest priority.
    """
    intent_scores: dict[str, float] = {}
    direct: list[str] = []

    for i, tok in enumerate(tokens):
        if tok.startswith('[') and tok.endswith(']'):
            direct.append(tok[1:-1])
            continue
        recency = 1.0 + (i / max(len(tokens) - 1, 1)) * 0.5
        for intent, w in GESTURE_INTENTS.get(tok, []):
            intent_scores[intent] = intent_scores.get(intent, 0) + w * recency

    heap = [(-s, k) for k, s in intent_scores.items()]
    heapq.heapify(heap)
    top = []
    while heap and len(top) < 5:
        _, k = heapq.heappop(heap)
        top.append(k)

    return direct + top


# ── Phi-3 caption generation ──────────────────────────────────────────────────

def generate_caption(tokens: list[str], prev: str = "") -> str:
    intents = resolve_intents(tokens)
    if not intents:
        return ""

    labels = [
        GESTURE_LABELS.get(t, t) if not t.startswith('[') else t
        for t in tokens if t != "None"
    ]

    system = (
        "You are a real-time sign language interpreter for professional settings "
        "including job interviews. Convert hand gesture intents and spelled words "
        "into ONE natural English sentence. Rules:\n"
        "- Max 15 words\n"
        "- Grammatically correct and professional\n"
        "- Spelled words in [BRACKETS] are proper nouns — use them verbatim\n"
        "- Continue naturally from the previous sentence if provided\n"
        "- Output ONLY the sentence, no explanation, no quotes"
    )

    ctx = f'Previous: "{prev}"\n' if prev else ""
    user = (
        f"{ctx}"
        f"Gestures: {', '.join(labels)}\n"
        f"Intent words (priority order): {', '.join(intents)}\n"
        "Sentence:"
    )

    payload = _json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 40},
    }).encode()

    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as r:
            return _json.loads(r.read())["message"]["content"].strip().strip('"\'')
    except Exception as e:
        print(f"  [Phi-3] {e}")
        return " ".join(intents).capitalize() + "."


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_frame(frame, gesture, conf, letter, spelling,
               tokens, caption, status, fps, h, w):

    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 85), BLACK, -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    # Gesture / letter indicator
    if letter:
        cv2.putText(frame, f"Letter: {letter}  |  Spelling: {spelling}",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)
    elif gesture and gesture != "None":
        lbl = GESTURE_LABELS.get(gesture, gesture)
        col = GREEN if conf >= MIN_CONF else GREY
        cv2.putText(frame, f"Gesture: {gesture}  [{lbl}]  {conf*100:.0f}%",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
    else:
        cv2.putText(frame, "Waiting for sign…",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREY, 1)

    cv2.putText(frame, f"{status}  |  {fps:.0f} FPS",
                (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1)

    # Token chips
    if tokens:
        cx = 20
        for tok in tokens[-6:]:
            lbl = GESTURE_LABELS.get(tok, tok) if not tok.startswith('[') else tok[1:-1]
            if not lbl: continue
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (cx-3, 90), (cx+tw+7, 112), (30,30,70), -1)
            cv2.rectangle(frame, (cx-3, 90), (cx+tw+7, 112), CYAN, 1)
            cv2.putText(frame, lbl, (cx, 107),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, CYAN, 1)
            cx += tw + 14

    # Caption box
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h-115), (w, h), BLACK, -1)
    cv2.addWeighted(ov2, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, h-115), (w, h-115), (255, 180, 0), 2)

    if caption:
        words = caption.split()
        lines, line = [], []
        for word in words:
            line.append(word)
            if len(" ".join(line)) > 58:
                lines.append(" ".join(line[:-1]))
                line = [word]
        if line: lines.append(" ".join(line))
        y = h - 92
        for ln in lines[:3]:
            cv2.putText(frame, ln, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, WHITE, 2)
            y += 34
    else:
        cv2.putText(frame, "Perform a sign or spell a word…",
                    (20, h-55), cv2.FONT_HERSHEY_SIMPLEX, 0.62, GREY, 1)

    cv2.putText(frame, "Q=quit  R=reset",
                (w-180, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("═"*60)
    print("  HexaMinds — Hands-Free Sign Language Caption")
    print("  Dual-track: Gestures + Fingerspelling")
    print("  No keyboard needed.")
    print("═"*60)

    if not GESTURE_MODEL.exists():
        print("  Downloading gesture model…")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task")
        urllib.request.urlretrieve(url, GESTURE_MODEL)

    # Ollama check
    ollama_ok = False
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            models = [m["name"] for m in _json.loads(r.read())["models"]]
            ollama_ok = any("phi3" in m for m in models)
        print(f"  Ollama: {'✓ phi3 ready' if ollama_ok else '⚠ phi3 not found'}")
    except Exception:
        print("  Ollama: not running — raw intent fallback")

    # MediaPipe gesture recognizer
    BaseOptions = mp.tasks.BaseOptions
    GR          = mp.tasks.vision.GestureRecognizer
    GROpts      = mp.tasks.vision.GestureRecognizerOptions
    RunMode     = mp.tasks.vision.RunningMode

    gr_opts = GROpts(
        base_options=BaseOptions(model_asset_path=str(GESTURE_MODEL)),
        running_mode=RunMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: Cannot open camera"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("  Camera ready. Start signing!\n")

    # ── State ─────────────────────────────────────────────────────────────────
    # Gesture track
    g_hold_buf   = deque(maxlen=GESTURE_HOLD)
    last_gesture = ""

    # Letter track
    l_hold_buf   = deque(maxlen=LETTER_HOLD)
    l_gap        = 0          # frames since last letter
    spelling     = []         # letters being accumulated
    last_letter  = ""

    # Shared
    tokens: list[str] = []   # confirmed gesture names + [WORDS]
    caption          = ""
    caption_hist: list[str] = []
    status           = "Ready"
    no_input_frames  = 0
    generating       = False

    fps_t0, fps_n, fps = time.perf_counter(), 0, 0.0

    with GR.create_from_options(gr_opts) as recognizer:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = recognizer.recognize(mp_img)

            # ── Dual-track detection ──────────────────────────────────────────
            det_gesture, det_conf = "None", 0.0
            det_letter = None

            if result.gestures and result.hand_landmarks:
                best = max(
                    (g for gs in result.gestures for g in gs),
                    key=lambda g: g.score,
                )
                det_gesture = best.category_name
                det_conf    = best.score

                # Try letter classification on same hand
                det_letter = classify_letter(result.hand_landmarks[0])

            # ── Letter track (higher priority — more specific) ────────────────
            if det_letter:
                l_hold_buf.append(det_letter)
                l_gap = 0
                no_input_frames = 0
            else:
                l_gap += 1
                # Word boundary: pause after letters
                if l_gap == LETTER_GAP and spelling:
                    word = "".join(spelling)
                    if len(word) >= MIN_WORD_LEN:
                        tokens.append(f"[{word.upper()}]")
                        print(f"  ✦ Word spelled: {word.upper()}")
                        spelling.clear()
                        last_letter = ""
                        # Trigger caption if enough tokens
                        if len(tokens) >= 2 and not generating:
                            _trigger_caption(tokens, caption_hist,
                                             ollama_ok, lambda c, s: _update(c, s))

            # Confirm letter when buffer consistent
            if (len(l_hold_buf) == LETTER_HOLD and len(set(l_hold_buf)) == 1):
                confirmed = l_hold_buf[0]
                l_hold_buf.clear()
                if confirmed != last_letter or l_gap > LETTER_GAP // 2:
                    spelling.append(confirmed)
                    last_letter = confirmed
                    status = f"Spelling: {''.join(spelling)}"

            # ── Gesture track (when no letter detected) ───────────────────────
            if (not det_letter and det_conf >= MIN_CONF
                    and det_gesture != "None"):
                g_hold_buf.append(det_gesture)
                no_input_frames = 0
            else:
                if not det_letter:
                    no_input_frames += 1

            if (len(g_hold_buf) == GESTURE_HOLD and len(set(g_hold_buf)) == 1):
                confirmed_g = g_hold_buf[0]
                g_hold_buf.clear()
                if confirmed_g != last_gesture:
                    last_gesture = confirmed_g
                    tokens.append(confirmed_g)
                    lbl = GESTURE_LABELS.get(confirmed_g, confirmed_g)
                    print(f"  ✦ Gesture: {confirmed_g}  [{lbl}]")
                    status = f"Detected: {lbl}"
                    if len(tokens) >= 1 and not generating:
                        _trigger_caption(tokens, caption_hist,
                                         ollama_ok, lambda c, s: _update(c, s))

            # ── Auto-reset on long pause ──────────────────────────────────────
            if no_input_frames >= PAUSE_RESET:
                tokens.clear(); spelling.clear()
                last_gesture = ""; last_letter = ""
                no_input_frames = 0
                status = "Ready"

            # ── Closure for thread-safe update ────────────────────────────────
            _cap_ref   = [caption]
            _stat_ref  = [status]

            def _update(c, s):
                nonlocal caption, status
                caption = c; status = s

            # ── FPS ───────────────────────────────────────────────────────────
            fps_n += 1
            if time.perf_counter() - fps_t0 >= 1.0:
                fps = fps_n / (time.perf_counter() - fps_t0)
                fps_t0 = time.perf_counter(); fps_n = 0

            # Draw hand skeleton
            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand]
                    CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                            (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
                            (15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
                    for a, b in CONN:
                        cv2.line(frame, pts[a], pts[b], GREEN, 1)
                    for pt in pts:
                        cv2.circle(frame, pt, 3, CYAN, -1)

            draw_frame(frame, det_gesture, det_conf,
                       det_letter, "".join(spelling),
                       tokens, caption, status, fps, h, w)

            cv2.imshow("HexaMinds — Hands-Free Caption  (Q=quit  R=reset)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                tokens.clear(); spelling.clear(); caption = ""
                caption_hist.clear(); last_gesture = ""; last_letter = ""
                no_input_frames = 0; status = "Reset"
                print("  [RESET]")

    cap.release()
    cv2.destroyAllWindows()
    print("\n  Stopped.")


def _trigger_caption(tokens, caption_hist, ollama_ok, update_fn):
    """Fire-and-forget caption generation in a background thread."""
    snap = list(tokens)
    prev = caption_hist[-1] if caption_hist else ""

    def _run():
        t0 = time.perf_counter()
        if ollama_ok:
            c = generate_caption(snap, prev)
        else:
            intents = resolve_intents(snap)
            c = " ".join(intents).capitalize() + "." if intents else ""
        ms = (time.perf_counter() - t0) * 1000
        if c:
            caption_hist.append(c)
            if len(caption_hist) > 5:
                caption_hist.pop(0)
        update_fn(c, f"Caption ready  ({ms:.0f}ms)")
        print(f"  ✦ Caption: {c}  ({ms:.0f}ms)")

    threading.Thread(target=_run, daemon=True).start()


if __name__ == "__main__":
    main()
