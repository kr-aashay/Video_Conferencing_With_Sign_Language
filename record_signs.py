"""
record_signs.py
═══════════════════════════════════════════════════════════════════════════════
Record your own sign language training data.

Usage
─────
    .venv/bin/python record_signs.py

How it works
────────────
1. You define a list of signs you want to recognise (SIGNS list below)
2. For each sign, the script records 15 short video clips of you performing it
3. Each clip is saved as a .npz landmark file (same format as WLASL)
4. After recording, run: .venv/bin/python retrain.py

Controls
────────
    SPACE — start/stop recording a clip
    N     — skip to next sign
    Q     — quit
"""

from __future__ import annotations

import sys
import time
import hashlib
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# ══════════════════════════════════════════════════════════════════════════════
# DEFINE YOUR SIGNS HERE
# Add/remove signs as needed. Use simple English words.
# ══════════════════════════════════════════════════════════════════════════════
SIGNS = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO",
    "HELP",
    "PLEASE",
    "SORRY",
    "GOOD",
    "BAD",
    "WATER",
]

CLIPS_PER_SIGN = 15    # how many recordings per sign
MIN_FRAMES     = 20    # minimum frames per clip
MAX_FRAMES     = 120   # maximum frames per clip (4s at 30fps)
OUTPUT_DIR     = Path("my_signs")

# Landmark indices
_LEFT_HIP  = 23
_RIGHT_HIP = 24


def extract_landmarks(results) -> dict | None:
    """Extract pose, face, lhand, rhand arrays from MediaPipe results."""
    if results.pose_landmarks is None:
        return None

    pose = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(results.pose_landmarks.landmark):
        pose[i] = [lm.x, lm.y, lm.z, lm.visibility]

    face = np.zeros((468, 3), dtype=np.float32)
    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            if i >= 468: break
            face[i] = [lm.x, lm.y, lm.z]

    lhand = np.zeros((21, 3), dtype=np.float32)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            if i >= 21: break
            lhand[i] = [lm.x, lm.y, lm.z]

    rhand = np.zeros((21, 3), dtype=np.float32)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            if i >= 21: break
            rhand[i] = [lm.x, lm.y, lm.z]

    return {"pose": pose, "face": face, "lhand": lhand, "rhand": rhand}


def save_clip(frames_data: list[dict], sign: str, clip_idx: int) -> Path:
    """Save a list of frame dicts as a .npz file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pose  = np.stack([f["pose"]  for f in frames_data])
    face  = np.stack([f["face"]  for f in frames_data])
    lhand = np.stack([f["lhand"] for f in frames_data])
    rhand = np.stack([f["rhand"] for f in frames_data])

    # Generate a unique filename
    uid  = hashlib.sha1(f"{sign}_{clip_idx}_{time.time()}".encode()).hexdigest()[:8]
    path = OUTPUT_DIR / f"{sign}__{clip_idx:02d}__{uid}.npz"
    np.savez_compressed(str(path), pose=pose, face=face, lhand=lhand, rhand=rhand)
    return path


def main():
    print("═" * 60)
    print("  Aashay's Sign Lang — Record Your Own Signs")
    print("═" * 60)
    print(f"  Signs to record : {SIGNS}")
    print(f"  Clips per sign  : {CLIPS_PER_SIGN}")
    print(f"  Output dir      : {OUTPUT_DIR}/")
    print()
    print("  Controls:")
    print("    SPACE — start/stop recording")
    print("    N     — skip to next sign")
    print("    Q     — quit")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sign_idx  = 0
    clip_idx  = 0
    recording = False
    clip_buf: list[dict] = []
    saved_clips = 0

    while sign_idx < len(SIGNS):
        sign = SIGNS[sign_idx]

        # Count existing clips for this sign
        existing = list(OUTPUT_DIR.glob(f"{sign}__*.npz"))
        clip_idx = len(existing)

        if clip_idx >= CLIPS_PER_SIGN:
            print(f"  ✓ {sign} — already has {clip_idx} clips, skipping")
            sign_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        # Draw hand skeleton
        if results.right_hand_landmarks or results.left_hand_landmarks:
            for hand_lms in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_lms:
                    for lm in hand_lms.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 100), -1)

        # Record frame if recording
        if recording:
            lm_data = extract_landmarks(results)
            if lm_data:
                clip_buf.append(lm_data)

            # Auto-stop at max frames
            if len(clip_buf) >= MAX_FRAMES:
                recording = False
                if len(clip_buf) >= MIN_FRAMES:
                    path = save_clip(clip_buf, sign, clip_idx)
                    clip_idx += 1
                    saved_clips += 1
                    print(f"  ✓ Saved: {path.name}  ({len(clip_buf)} frames)")
                clip_buf = []

        # ── UI overlay ────────────────────────────────────────────────────────
        # Dark background strip at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Sign name
        cv2.putText(frame, f"Sign: {sign}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)

        # Progress
        progress = f"Clip {clip_idx}/{CLIPS_PER_SIGN}  |  Sign {sign_idx+1}/{len(SIGNS)}"
        cv2.putText(frame, progress, (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Recording indicator
        if recording:
            cv2.circle(frame, (w - 30, 30), 12, (0, 0, 255), -1)
            cv2.putText(frame, f"REC  {len(clip_buf)}/{MAX_FRAMES}", (w - 160, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "SPACE to record", (w - 220, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

        # Instruction at bottom
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        instruction = f"Perform '{sign}' clearly, then press SPACE to start/stop"
        cv2.putText(frame, instruction, (20, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Record Signs  (Q=quit  N=next  SPACE=record)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord(' '):
            if not recording:
                # Start recording
                recording = True
                clip_buf  = []
                print(f"  ● Recording {sign} clip {clip_idx+1}/{CLIPS_PER_SIGN}…")
            else:
                # Stop recording
                recording = False
                if len(clip_buf) >= MIN_FRAMES:
                    path = save_clip(clip_buf, sign, clip_idx)
                    clip_idx += 1
                    saved_clips += 1
                    print(f"  ✓ Saved: {path.name}  ({len(clip_buf)} frames)")
                else:
                    print(f"  ✗ Too short ({len(clip_buf)} frames < {MIN_FRAMES}) — discarded")
                clip_buf = []

                # Auto-advance when sign is complete
                if clip_idx >= CLIPS_PER_SIGN:
                    print(f"  ✓ {sign} complete! Moving to next sign…")
                    sign_idx += 1

        elif key == ord('n'):
            if recording:
                recording = False
                clip_buf  = []
            print(f"  → Skipping {sign}")
            sign_idx += 1

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()

    # Summary
    all_clips = list(OUTPUT_DIR.glob("*.npz"))
    print()
    print("═" * 60)
    print(f"  Recording complete!")
    print(f"  Total clips saved : {len(all_clips)}")
    print(f"  Output directory  : {OUTPUT_DIR}/")
    print()
    print("  Next step — retrain on your signs:")
    print("    .venv/bin/python retrain.py")
    print("═" * 60)


if __name__ == "__main__":
    main()
