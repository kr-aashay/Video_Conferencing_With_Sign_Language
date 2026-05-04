"""
orchestrate.py
══════════════════════════════════════════════════════════════════════════════
ISL-CSLTR Data Orchestration Pipeline
──────────────────────────────────────────────────
Stages
  1. VOCAB SCAN     — parse dataset filenames → label_map.json + vocab.json
  2. EXTRACTION     — Stream-and-Purge MediaPipe landmark extraction
  3. SANITY SCAN    — QA every .npz (hand presence, confidence, min frames)
  4. METADATA BUILD — metadata.csv linking each .npz to its integer label
  5. REPORT         — summary of counts, quality stats, storage footprint

Run:
    python orchestrate.py

All outputs land in OUTPUT_DIR (default: ./isl_landmarks/).
Logs are written to orchestrate.log and stdout.
"""

from __future__ import annotations

import csv
import gc
import json
import logging
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
import kagglehub

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit this block only
# ══════════════════════════════════════════════════════════════════════════════

DATASET_SLUG        = "risangbaskoro/wlasl-processed"
OUTPUT_DIR          = Path("./isl_landmarks")
LABEL_MAP_PATH      = Path("./label_map.json")
VOCAB_PATH          = Path("./vocab.json")
METADATA_CSV_PATH   = Path("./metadata.csv")
REPORT_PATH         = Path("./pipeline_report.json")

# Storage
STORAGE_HARD_CAP_GB = 10.0   # absolute ceiling — pipeline halts
STORAGE_SOFT_CAP_GB = 9.0    # soft ceiling — triggers deeper purge warning
LANDMARK_TARGET_GB  = 5.0    # target for final .npz dataset

# Extraction
BATCH_SIZE          = 5      # videos per Stream-and-Purge cycle
MAX_FRAMES          = None   # None = all frames; int = cap per video
MP_MODEL_COMPLEXITY = 1      # 0=lite, 1=full, 2=heavy
VIDEO_EXTENSIONS    = {".mp4", ".avi", ".mov", ".mkv"}

# Quality thresholds
MIN_FRAMES          = 16     # minimum frames for Bi-LSTM (receptive field)
MIN_HAND_CONF       = 0.4    # minimum average hand detection confidence
# A sequence is "hand-null" if BOTH hands are all-zero for > this fraction
MAX_NULL_HAND_RATIO = 0.80   # 80 % null frames → reject

# Filename parsing
# Matches: HELP_01.mp4 → "HELP"  |  THANK_YOU_003.mp4 → "THANK_YOU"
# Strategy: strip trailing _<digits> suffix, then take the remainder as gloss
GLOSS_PATTERN       = re.compile(r"^(.+?)(?:_\d+)+$")

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("orchestrate.log", mode="a"),
    ],
)
log = logging.getLogger("orchestrator")


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QAResult:
    npz_path:        Path
    gloss:           str
    label_id:        int
    n_frames:        int
    hand_null_ratio: float   # fraction of frames where both hands are zero
    avg_hand_conf:   float   # proxy: fraction of frames with non-zero hand data
    passed:          bool
    reject_reason:   str = ""


@dataclass
class PipelineStats:
    total_videos:      int = 0
    extracted_ok:      int = 0
    extracted_fail:    int = 0
    qa_passed:         int = 0
    qa_rejected:       int = 0
    reject_reasons:    dict = field(default_factory=dict)
    unique_glosses:    int = 0
    landmark_size_gb:  float = 0.0
    elapsed_sec:       float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — VOCABULARY SCAN
# ══════════════════════════════════════════════════════════════════════════════

def parse_gloss(stem: str) -> str:
    """
    Extract the sign gloss from a video filename stem.

    Examples
    ────────
    HELP_01        → HELP
    THANK_YOU_003  → THANK_YOU
    QUESTION_1_2   → QUESTION
    NAMASTE        → NAMASTE   (no trailing digits → use as-is)
    """
    m = GLOSS_PATTERN.match(stem)
    return (m.group(1) if m else stem).upper()


def build_vocabulary(dataset_root: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    """
    Build label_map from dataset.

    Supports two dataset formats:
      1. WLASL: flat videos/ folder + WLASL_v0.3.json  (video_id → gloss)
      2. ISL-CSLTR: nested folders, gloss parsed from filename

    Returns
    -------
    label_map   : { npz_stem: [gloss_string] }
    gloss_to_id : { gloss: int }  — <blank>=0, glosses 1…N alphabetically
    """
    log.info("── Stage 1: Vocabulary Scan ──────────────────────────────────")

    # ── Detect WLASL format (has WLASL_v0.3.json) ─────────────────────────────
    wlasl_json = dataset_root / "WLASL_v0.3.json"
    if wlasl_json.exists():
        return _build_vocabulary_wlasl(dataset_root, wlasl_json)

    # ── Fallback: filename-based parsing (ISL-CSLTR) ──────────────────────────
    return _build_vocabulary_filename(dataset_root)


def _build_vocabulary_wlasl(
    dataset_root: Path, json_path: Path
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Parse WLASL: video_id (numeric filename) → gloss from JSON."""
    data = json.loads(json_path.read_text())

    # Build video_id → gloss lookup
    vid2gloss: dict[str, str] = {}
    for entry in data:
        gloss = entry["gloss"].upper()
        for inst in entry["instances"]:
            vid2gloss[inst["video_id"]] = gloss

    video_files = sorted(
        p for p in dataset_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    log.info("  Found %d video files (WLASL format)", len(video_files))

    glosses: set[str] = set()
    label_map: dict[str, list[str]] = {}

    for vp in video_files:
        gloss = vid2gloss.get(vp.stem)
        if gloss is None:
            log.debug("  No label for video_id=%s — skipping", vp.stem)
            continue
        npz_stem = _npz_stem_for(vp)
        label_map[npz_stem] = [gloss]
        glosses.add(gloss)

    log.info("  Matched %d videos to labels", len(label_map))
    return _finalise_vocab(label_map, glosses)


def _build_vocabulary_filename(
    dataset_root: Path,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Parse ISL-CSLTR: gloss extracted from video filename."""
    video_files = sorted(
        p for p in dataset_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    log.info("  Found %d video files (filename format)", len(video_files))

    glosses: set[str] = set()
    label_map: dict[str, list[str]] = {}

    for vp in video_files:
        gloss    = parse_gloss(vp.stem)
        npz_stem = _npz_stem_for(vp)
        label_map[npz_stem] = [gloss]
        glosses.add(gloss)

    return _finalise_vocab(label_map, glosses)


def _finalise_vocab(
    label_map: dict[str, list[str]],
    glosses: set[str],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    sorted_glosses = sorted(glosses)
    gloss_to_id: dict[str, int] = {"<blank>": 0}
    for i, g in enumerate(sorted_glosses, start=1):
        gloss_to_id[g] = i

    log.info("  Unique glosses: %d", len(sorted_glosses))
    log.info("  Vocabulary size (incl. blank): %d", len(gloss_to_id))

    LABEL_MAP_PATH.write_text(json.dumps(label_map, indent=2, sort_keys=True))
    VOCAB_PATH.write_text(json.dumps(gloss_to_id, indent=2, sort_keys=True))
    log.info("  label_map.json → %s", LABEL_MAP_PATH)
    log.info("  vocab.json     → %s", VOCAB_PATH)

    return label_map, gloss_to_id


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — STREAM-AND-PURGE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _npz_stem_for(video_path: Path) -> str:
    """
    Derive a short, filesystem-safe .npz stem from a video path.

    Format: {gloss_slug}__{sha8}
      gloss_slug : sanitised video stem, max 40 chars
      sha8       : first 8 hex chars of SHA-1(full_path) — collision-proof

    Examples:
      'it does not make any difference to me (2)' → 'it_does_not_make_any_difference_to__a3f2b1c4'
      'MVI_6503'                                  → 'MVI_6503__d9e1a2b3'

    Always ≤ 60 chars — well under the 255-byte macOS filename limit.
    """
    import hashlib
    # 8-char hash of the full absolute path — unique per video
    sha8 = hashlib.sha1(str(video_path).encode()).hexdigest()[:8]
    # Sanitise the stem: replace spaces/special chars with _, truncate to 40
    slug = re.sub(r"[^\w]", "_", video_path.stem)[:40].rstrip("_")
    return f"{slug}__{sha8}"


def _npz_path_for(video_path: Path) -> Path:
    return OUTPUT_DIR / f"{_npz_stem_for(video_path)}.npz"


def _disk_gb(*paths: Path) -> float:
    total = 0
    for p in paths:
        if p.exists():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def _check_storage(dataset_root: Path) -> None:
    used = _disk_gb(dataset_root, OUTPUT_DIR)
    log.debug("  Disk usage: %.2f GB", used)
    if used >= STORAGE_HARD_CAP_GB:
        raise RuntimeError(
            f"HARD storage cap reached: {used:.2f} GB ≥ {STORAGE_HARD_CAP_GB} GB. "
            "Pipeline halted to protect disk."
        )
    if used >= STORAGE_SOFT_CAP_GB:
        log.warning(
            "  ⚠ Soft cap warning: %.2f GB / %.2f GB — "
            "consider reducing batch size or purging intermediates.",
            used, STORAGE_SOFT_CAP_GB,
        )


def _extract_landmarks(video_path: Path, holistic) -> Optional[dict]:
    """Extract MediaPipe Holistic landmarks from a single video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning("    Cannot open: %s", video_path.name)
        return None

    pose_seq, face_seq, lhand_seq, rhand_seq = [], [], [], []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if MAX_FRAMES and frame_idx >= MAX_FRAMES:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # ── Pose — always (33, 4): x, y, z, visibility ───────────────────
            pose_frame = np.zeros((33, 4), dtype=np.float32)
            if results.pose_landmarks:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    if i >= 33:
                        break
                    pose_frame[i] = [lm.x, lm.y, lm.z, lm.visibility]
            pose_seq.append(pose_frame)

            # ── Face — always (468, 3): x, y, z ──────────────────────────────
            # refine_face_landmarks=True can return 478 landmarks on some frames.
            # We clamp to exactly 468 so every frame has the same shape.
            face_frame = np.zeros((468, 3), dtype=np.float32)
            if results.face_landmarks:
                for i, lm in enumerate(results.face_landmarks.landmark):
                    if i >= 468:
                        break
                    face_frame[i] = [lm.x, lm.y, lm.z]
            face_seq.append(face_frame)

            # ── Left hand — always (21, 3) ────────────────────────────────────
            lhand_frame = np.zeros((21, 3), dtype=np.float32)
            if results.left_hand_landmarks:
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    if i >= 21:
                        break
                    lhand_frame[i] = [lm.x, lm.y, lm.z]
            lhand_seq.append(lhand_frame)

            # ── Right hand — always (21, 3) ───────────────────────────────────
            rhand_frame = np.zeros((21, 3), dtype=np.float32)
            if results.right_hand_landmarks:
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    if i >= 21:
                        break
                    rhand_frame[i] = [lm.x, lm.y, lm.z]
            rhand_seq.append(rhand_frame)

            frame_idx += 1
    finally:
        cap.release()

    if frame_idx == 0:
        log.warning("    Zero frames decoded: %s", video_path.name)
        return None

    return {
        "pose":  np.stack(pose_seq,  axis=0),   # (T, 33, 4)
        "face":  np.stack(face_seq,  axis=0),   # (T, 468, 3)
        "lhand": np.stack(lhand_seq, axis=0),   # (T, 21, 3)
        "rhand": np.stack(rhand_seq, axis=0),   # (T, 21, 3)
    }


def _purge(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
        log.debug("    Purged: %s", path.name)
    except OSError as exc:
        log.warning("    Could not purge %s: %s", path.name, exc)


def run_extraction(dataset_root: Path, stats: PipelineStats) -> None:
    """Stream-and-Purge loop over all videos in dataset_root."""
    log.info("── Stage 2: Stream-and-Purge Extraction ─────────────────────")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_videos = sorted(
        p for p in dataset_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    stats.total_videos = len(all_videos)
    log.info("  Videos to process: %d", stats.total_videos)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=MP_MODEL_COMPLEXITY,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    total_batches = (len(all_videos) + BATCH_SIZE - 1) // BATCH_SIZE

    try:
        for batch_idx, batch_start in enumerate(range(0, len(all_videos), BATCH_SIZE)):
            batch = all_videos[batch_start : batch_start + BATCH_SIZE]
            log.info(
                "  Batch %d/%d  (videos %d–%d)",
                batch_idx + 1, total_batches,
                batch_start + 1, min(batch_start + BATCH_SIZE, len(all_videos)),
            )

            # Storage guard — checked before every batch
            _check_storage(dataset_root)

            for vp in batch:
                out_path = _npz_path_for(vp)
                data = None

                if out_path.exists():
                    log.info("    [SKIP] %s", vp.name)
                    stats.extracted_ok += 1
                    _purge(vp)
                    continue

                log.info("    [PROC] %s", vp.name)
                t0 = time.perf_counter()

                try:
                    data = _extract_landmarks(vp, holistic)
                    if data is None:
                        stats.extracted_fail += 1
                    else:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(str(out_path), **data)
                        elapsed = time.perf_counter() - t0
                        log.info(
                            "      → %d frames | %.1f KB | %.1fs",
                            data["pose"].shape[0],
                            out_path.stat().st_size / 1024,
                            elapsed,
                        )
                        stats.extracted_ok += 1
                except Exception:
                    log.error(
                        "      Exception on %s:\n%s",
                        vp.name, traceback.format_exc(),
                    )
                    stats.extracted_fail += 1
                finally:
                    _purge(vp)   # CRITICAL: always purge source
                    del data
                    gc.collect()

            log.info(
                "  Batch done | disk: %.2f GB",
                _disk_gb(dataset_root, OUTPUT_DIR),
            )

    except RuntimeError as exc:
        log.critical("  %s", exc)
    except KeyboardInterrupt:
        log.warning("  Extraction interrupted — partial progress saved.")
    finally:
        holistic.close()

    log.info(
        "  Extraction complete — ok: %d | failed: %d",
        stats.extracted_ok, stats.extracted_fail,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — SANITY SCAN (QA)
# ══════════════════════════════════════════════════════════════════════════════

def _is_null_array(arr: np.ndarray) -> bool:
    """True if the array is entirely zeros (landmark not detected)."""
    return not np.any(arr)


def qa_single_npz(
    npz_path: Path,
    gloss: str,
    label_id: int,
) -> QAResult:
    """
    Run quality checks on one .npz file.

    Checks
    ──────
    1. Minimum frame count  (≥ MIN_FRAMES)
    2. Hand null ratio       (fraction of frames where BOTH hands are zero)
    3. Average hand confidence proxy (fraction of frames with any hand data)
    """
    try:
        data  = np.load(str(npz_path))
        pose  = data["pose"]    # (T, 33, 4)
        lhand = data["lhand"]   # (T, 21, 3)
        rhand = data["rhand"]   # (T, 21, 3)
        T     = pose.shape[0]
    except Exception as exc:
        return QAResult(
            npz_path=npz_path, gloss=gloss, label_id=label_id,
            n_frames=0, hand_null_ratio=1.0, avg_hand_conf=0.0,
            passed=False, reject_reason=f"load_error:{exc}",
        )

    # ── Check 1: minimum frames ───────────────────────────────────────────────
    if T < MIN_FRAMES:
        return QAResult(
            npz_path=npz_path, gloss=gloss, label_id=label_id,
            n_frames=T, hand_null_ratio=1.0, avg_hand_conf=0.0,
            passed=False, reject_reason=f"too_short:{T}<{MIN_FRAMES}",
        )

    # ── Check 2 & 3: hand presence ────────────────────────────────────────────
    # A frame is "hand-null" if BOTH lhand and rhand are all zeros
    lhand_null = np.array([_is_null_array(lhand[t]) for t in range(T)])
    rhand_null = np.array([_is_null_array(rhand[t]) for t in range(T)])
    both_null  = lhand_null & rhand_null

    hand_null_ratio = float(both_null.sum()) / T
    # Confidence proxy: fraction of frames where at least one hand is detected
    avg_hand_conf   = 1.0 - hand_null_ratio

    if hand_null_ratio > MAX_NULL_HAND_RATIO:
        return QAResult(
            npz_path=npz_path, gloss=gloss, label_id=label_id,
            n_frames=T, hand_null_ratio=hand_null_ratio,
            avg_hand_conf=avg_hand_conf,
            passed=False,
            reject_reason=f"hand_null:{hand_null_ratio:.2f}>{MAX_NULL_HAND_RATIO}",
        )

    if avg_hand_conf < MIN_HAND_CONF:
        return QAResult(
            npz_path=npz_path, gloss=gloss, label_id=label_id,
            n_frames=T, hand_null_ratio=hand_null_ratio,
            avg_hand_conf=avg_hand_conf,
            passed=False,
            reject_reason=f"low_conf:{avg_hand_conf:.2f}<{MIN_HAND_CONF}",
        )

    return QAResult(
        npz_path=npz_path, gloss=gloss, label_id=label_id,
        n_frames=T, hand_null_ratio=hand_null_ratio,
        avg_hand_conf=avg_hand_conf,
        passed=True,
    )


def run_sanity_scan(
    label_map: dict[str, list[str]],
    gloss_to_id: dict[str, int],
    stats: PipelineStats,
) -> list[QAResult]:
    """
    QA every .npz in OUTPUT_DIR.

    Returns list of QAResult (both passed and rejected) for metadata building.
    """
    log.info("── Stage 3: Sanity Scan ──────────────────────────────────────")

    npz_files = sorted(OUTPUT_DIR.rglob("*.npz"))
    log.info("  Scanning %d .npz files …", len(npz_files))

    results: list[QAResult] = []
    corrupt_log: list[str]  = []

    for npz_path in npz_files:
        stem     = npz_path.stem
        glosses  = label_map.get(stem, [])
        gloss    = glosses[0] if glosses else "<unk>"
        label_id = gloss_to_id.get(gloss, -1)

        result = qa_single_npz(npz_path, gloss, label_id)
        results.append(result)

        if result.passed:
            stats.qa_passed += 1
        else:
            stats.qa_rejected += 1
            reason_key = result.reject_reason.split(":")[0]
            stats.reject_reasons[reason_key] = (
                stats.reject_reasons.get(reason_key, 0) + 1
            )
            corrupt_log.append(
                f"{npz_path.name} | gloss={gloss} | reason={result.reject_reason}"
            )
            log.warning("  [REJECT] %s — %s", npz_path.name, result.reject_reason)

    # Write corrupt sequence log
    if corrupt_log:
        corrupt_path = OUTPUT_DIR / "corrupt_sequences.log"
        corrupt_path.write_text("\n".join(corrupt_log))
        log.info("  Corrupt sequence log → %s", corrupt_path)

    log.info(
        "  QA complete — passed: %d | rejected: %d",
        stats.qa_passed, stats.qa_rejected,
    )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — METADATA CSV
# ══════════════════════════════════════════════════════════════════════════════

def build_metadata_csv(qa_results: list[QAResult], stats: PipelineStats) -> None:
    """
    Write metadata.csv with one row per QA-passed .npz file.

    Columns
    ───────
    npz_path, gloss, label_id, n_frames, hand_null_ratio, avg_hand_conf
    """
    log.info("── Stage 4: Metadata CSV ─────────────────────────────────────")

    passed = [r for r in qa_results if r.passed]

    with open(METADATA_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "npz_path", "gloss", "label_id",
            "n_frames", "hand_null_ratio", "avg_hand_conf",
        ])
        writer.writeheader()
        for r in passed:
            writer.writerow({
                "npz_path":       str(r.npz_path),
                "gloss":          r.gloss,
                "label_id":       r.label_id,
                "n_frames":       r.n_frames,
                "hand_null_ratio": f"{r.hand_null_ratio:.4f}",
                "avg_hand_conf":  f"{r.avg_hand_conf:.4f}",
            })

    log.info("  metadata.csv → %s  (%d rows)", METADATA_CSV_PATH, len(passed))

    # Count unique glosses in the clean set
    stats.unique_glosses = len({r.gloss for r in passed})


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_gb(gb: float) -> str:
    return f"{gb:.3f} GB"


def write_report(stats: PipelineStats) -> None:
    """Write pipeline_report.json and print a human-readable summary."""
    stats.landmark_size_gb = _disk_gb(OUTPUT_DIR)

    report = {
        "pipeline":          "ISL-CSLTR HexaMinds Orchestration",
        "total_videos":      stats.total_videos,
        "extracted_ok":      stats.extracted_ok,
        "extracted_fail":    stats.extracted_fail,
        "qa_passed":         stats.qa_passed,
        "qa_rejected":       stats.qa_rejected,
        "reject_breakdown":  stats.reject_reasons,
        "unique_glosses":    stats.unique_glosses,
        "landmark_size_gb":  round(stats.landmark_size_gb, 4),
        "target_size_gb":    LANDMARK_TARGET_GB,
        "within_target":     stats.landmark_size_gb <= LANDMARK_TARGET_GB,
        "elapsed_sec":       round(stats.elapsed_sec, 1),
        "outputs": {
            "label_map":  str(LABEL_MAP_PATH),
            "vocab":      str(VOCAB_PATH),
            "metadata":   str(METADATA_CSV_PATH),
            "landmarks":  str(OUTPUT_DIR),
        },
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))

    # ── Human-readable summary ────────────────────────────────────────────────
    sep = "═" * 60
    log.info(sep)
    log.info("  HexaMinds Pipeline Report")
    log.info(sep)
    log.info("  Videos found          : %d", stats.total_videos)
    log.info("  Extracted (ok/fail)   : %d / %d",
             stats.extracted_ok, stats.extracted_fail)
    log.info("  QA passed / rejected  : %d / %d",
             stats.qa_passed, stats.qa_rejected)
    if stats.reject_reasons:
        for reason, count in sorted(stats.reject_reasons.items()):
            log.info("    ↳ %-20s : %d", reason, count)
    log.info("  Unique glosses (clean): %d", stats.unique_glosses)
    log.info("  Landmark dataset size : %s  (target ≤ %s)",
             _fmt_gb(stats.landmark_size_gb), _fmt_gb(LANDMARK_TARGET_GB))
    log.info("  Storage OK            : %s",
             "✓ YES" if report["within_target"] else "✗ EXCEEDS TARGET")
    log.info("  Elapsed               : %.1f s", stats.elapsed_sec)
    log.info(sep)
    log.info("  Outputs")
    log.info("    label_map.json → %s", LABEL_MAP_PATH)
    log.info("    vocab.json     → %s", VOCAB_PATH)
    log.info("    metadata.csv   → %s", METADATA_CSV_PATH)
    log.info("    report.json    → %s", REPORT_PATH)
    log.info("    landmarks/     → %s", OUTPUT_DIR)
    log.info(sep)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.perf_counter()
    stats   = PipelineStats()

    sep = "═" * 60
    log.info(sep)
    log.info("  HexaMinds — ISL-CSLTR Orchestration Pipeline")
    log.info(sep)

    # ── Download ──────────────────────────────────────────────────────────────
    log.info("── Downloading dataset: %s", DATASET_SLUG)
    try:
        dataset_root = Path(kagglehub.dataset_download(DATASET_SLUG))
    except Exception:
        log.critical("Dataset download failed:\n%s", traceback.format_exc())
        sys.exit(1)

    log.info("  Dataset root : %s", dataset_root)
    log.info("  Dataset size : %s", _fmt_gb(_disk_gb(dataset_root)))

    # Initial storage check
    if _disk_gb(dataset_root) >= STORAGE_HARD_CAP_GB:
        log.critical(
            "Dataset alone exceeds hard cap (%.2f GB). Aborting.",
            _disk_gb(dataset_root),
        )
        sys.exit(1)

    # ── Stage 1: Vocabulary ───────────────────────────────────────────────────
    label_map, gloss_to_id = build_vocabulary(dataset_root)

    # ── Stage 2: Extraction ───────────────────────────────────────────────────
    run_extraction(dataset_root, stats)

    # ── Stage 3: QA ───────────────────────────────────────────────────────────
    qa_results = run_sanity_scan(label_map, gloss_to_id, stats)

    # ── Stage 4: Metadata ─────────────────────────────────────────────────────
    build_metadata_csv(qa_results, stats)

    # ── Stage 5: Report ───────────────────────────────────────────────────────
    stats.elapsed_sec = time.perf_counter() - t_start
    write_report(stats)


if __name__ == "__main__":
    main()
