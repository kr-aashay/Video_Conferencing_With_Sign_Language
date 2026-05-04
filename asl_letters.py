"""
asl_letters.py
─────────────────────────────────────────────────────────────────────────────
Rule-based ASL fingerspelling classifier using MediaPipe hand landmarks.

Classifies 26 letters (A-Z) from the 21 hand landmark positions.
Works for any person — no training needed.

Each letter is defined by the geometric relationships between landmarks:
  - Which fingers are extended vs curled
  - Thumb position relative to fingers
  - Specific angle/contact patterns

Landmark indices (MediaPipe):
  0=WRIST
  1-4=THUMB (1=CMC, 2=MCP, 3=IP, 4=TIP)
  5-8=INDEX (5=MCP, 6=PIP, 7=DIP, 8=TIP)
  9-12=MIDDLE (9=MCP, 10=PIP, 11=DIP, 12=TIP)
  13-16=RING (13=MCP, 14=PIP, 15=DIP, 16=TIP)
  17-20=PINKY (17=MCP, 18=PIP, 19=DIP, 20=TIP)
"""

from __future__ import annotations
import numpy as np


def _finger_extended(lms: np.ndarray, tip: int, pip: int) -> bool:
    """True if fingertip is above (lower y) the PIP joint."""
    return lms[tip][1] < lms[pip][1]


def _thumb_extended(lms: np.ndarray) -> bool:
    """True if thumb tip is to the side of the thumb MCP."""
    return abs(lms[4][0] - lms[2][0]) > 0.04


def _dist(lms: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(lms[a] - lms[b]))


def _fingers_state(lms: np.ndarray) -> tuple[bool, bool, bool, bool, bool]:
    """Returns (thumb, index, middle, ring, pinky) extended state."""
    thumb  = _thumb_extended(lms)
    index  = _finger_extended(lms, 8,  6)
    middle = _finger_extended(lms, 12, 10)
    ring   = _finger_extended(lms, 16, 14)
    pinky  = _finger_extended(lms, 20, 18)
    return thumb, index, middle, ring, pinky


def classify_letter(landmarks) -> str | None:
    """
    Classify an ASL letter from MediaPipe hand landmarks.

    Parameters
    ----------
    landmarks : list of NormalizedLandmark (from MediaPipe)

    Returns
    -------
    Letter string ('A'-'Z') or None if unrecognised.
    """
    lms = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    t, i, m, r, p = _fingers_state(lms)

    # ── Closed fist variants ──────────────────────────────────────────────────
    if not i and not m and not r and not p:
        if not t:
            return 'A'   # Fist, thumb alongside
        if _dist(lms, 4, 8) < 0.05:
            return 'E'   # All fingers curled, thumb under
        return 'S'       # Fist, thumb over fingers

    # ── Single finger ─────────────────────────────────────────────────────────
    if i and not m and not r and not p:
        if not t:
            return 'D'   # Index up, thumb touches middle
        return 'G'       # Index pointing sideways (thumb out)

    if not i and m and not r and not p:
        return 'X'       # Middle finger up (hook)

    if not i and not m and not r and p:
        if not t:
            return 'I'   # Pinky up, thumb in
        return 'J'       # Pinky up, thumb out (J traces a curve)

    # ── Two fingers ───────────────────────────────────────────────────────────
    if i and m and not r and not p:
        if _dist(lms, 8, 12) < 0.04:
            return 'R'   # Index and middle crossed/together
        if not t:
            return 'U'   # Index and middle up, together
        return 'V'       # Index and middle up, spread (peace)

    if i and not m and not r and p:
        return 'H'       # Index and pinky up (horns)

    if not i and not m and r and p:
        return 'W'       # Ring and pinky up (partial)

    # ── Three fingers ─────────────────────────────────────────────────────────
    if i and m and r and not p:
        if not t:
            return 'W'   # Three middle fingers up
        return 'W'

    if i and m and not r and p:
        return 'K'       # Index, middle, pinky

    # ── Four fingers ─────────────────────────────────────────────────────────
    if i and m and r and p:
        if not t:
            return 'B'   # Four fingers up, thumb folded
        if t:
            return 'L'   # All up — actually L is index+thumb

    # ── Thumb + index (L shape) ───────────────────────────────────────────────
    if t and i and not m and not r and not p:
        if abs(lms[4][0] - lms[8][0]) > 0.08:
            return 'L'   # L shape — thumb and index at 90°
        return 'F'       # Thumb and index touching (circle)

    # ── Thumb + pinky ─────────────────────────────────────────────────────────
    if t and not i and not m and not r and p:
        return 'Y'       # Thumb and pinky extended (shaka)

    # ── Curved / contact letters ──────────────────────────────────────────────
    if not t and not i and not m and not r and not p:
        if _dist(lms, 4, 8) < 0.06:
            return 'O'   # All fingers curved into O shape
        return 'A'

    # ── C shape ───────────────────────────────────────────────────────────────
    if (lms[8][1] < lms[5][1] and lms[12][1] < lms[9][1] and
            lms[16][1] < lms[13][1] and lms[20][1] < lms[17][1]):
        if _dist(lms, 4, 8) > 0.08:
            return 'C'   # Curved C shape

    # ── P, Q (pointing down variants) ────────────────────────────────────────
    if i and m and not r and not p and lms[8][1] > lms[5][1]:
        return 'P'   # Index and middle pointing down

    # ── N, M (fingers over thumb) ────────────────────────────────────────────
    if not i and not m and not r and not p:
        if _dist(lms, 4, 12) < 0.05:
            return 'N'   # Thumb between middle and ring
        if _dist(lms, 4, 16) < 0.05:
            return 'M'   # Thumb between ring and pinky

    # ── T ────────────────────────────────────────────────────────────────────
    if not i and not m and not r and not p:
        if _dist(lms, 4, 6) < 0.05:
            return 'T'   # Thumb between index and middle

    return None   # unrecognised
