"""
cslr_model/dataset.py
─────────────────────
PyTorch Dataset for .npz landmark files from the ISL-CSLTR pipeline.

Each .npz contains:
    pose  : (T, 33, 4)   x, y, z, visibility
    face  : (T, 468, 3)  x, y, z
    lhand : (T, 21, 3)   x, y, z
    rhand : (T, 21, 3)   x, y, z

Feature vector per frame (1,662-dim):
    pose_xyz  33×3  =   99
    lhand     21×3  =   63
    rhand     21×3  =   63
    face     468×3  = 1,404  (face last — cheapest to zero-out if absent)
    ─────────────────────
    total           = 1,629  raw
    + 3 (pose-center anchor appended for debug/viz)  → 1,632
    NOTE: the model receives FEAT_DIM = 1,629 after normalization strips
          the anchor; the anchor is only stored for interpretability.

Normalization — "pose-center relative":
    The mid-hip point (average of left-hip lm[23] and right-hip lm[24])
    is used as the spatial origin each frame.  All x,y,z coordinates are
    shifted by this center, making the representation translation-invariant
    and robust to camera position changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)

# ── Landmark indices ──────────────────────────────────────────────────────────
_LEFT_HIP  = 23   # MediaPipe Pose landmark index
_RIGHT_HIP = 24

# ── Feature dimensions ────────────────────────────────────────────────────────
POSE_DIM = 33  * 3   #   99
HAND_DIM = 21  * 3   #   63  (per hand)
FACE_DIM = 468 * 3   # 1,404  (kept for reference, not used in npz_to_feature)
FEAT_DIM = POSE_DIM + HAND_DIM * 2   # 225  (pose + hands only — 7× faster)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def _pose_center(pose_xyz: np.ndarray) -> np.ndarray:
    """
    Compute the mid-hip anchor per frame.

    Parameters
    ----------
    pose_xyz : (T, 33, 3)

    Returns
    -------
    center : (T, 1, 3)  — broadcast-ready
    """
    left_hip  = pose_xyz[:, _LEFT_HIP,  :]   # (T, 3)
    right_hip = pose_xyz[:, _RIGHT_HIP, :]   # (T, 3)
    return ((left_hip + right_hip) / 2.0)[:, np.newaxis, :]   # (T, 1, 3)


def normalize_landmarks(
    pose:  np.ndarray,   # (T, 33, 3)
    face:  np.ndarray,   # (T, 468, 3)
    lhand: np.ndarray,   # (T, 21, 3)
    rhand: np.ndarray,   # (T, 21, 3)
) -> np.ndarray:
    """
    Shift all landmark groups so the mid-hip is the spatial origin.

    Returns
    -------
    feat : (T, FEAT_DIM) float32
    """
    center = _pose_center(pose)   # (T, 1, 3)

    pose_n  = pose  - center                  # (T, 33, 3)
    face_n  = face  - center                  # (T, 468, 3)
    lhand_n = lhand - center                  # (T, 21, 3)
    rhand_n = rhand - center                  # (T, 21, 3)

    T = pose.shape[0]
    feat = np.concatenate([
        pose_n.reshape(T, -1),
        lhand_n.reshape(T, -1),
        rhand_n.reshape(T, -1),
        face_n.reshape(T, -1),
    ], axis=1).astype(np.float32)             # (T, 1629)

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def npz_to_feature(npz_path: Path) -> Tensor:
    """
    Load one .npz and return a normalized (T, FEAT_DIM) float32 tensor.

    Uses only pose + hands (225 dims) — face mesh (1404 dims) is dropped.
    Face landmarks add minimal value for sign recognition and are the
    primary cause of slow training (1629-dim Conv1D is 7× slower than 225-dim).

    Layout: pose_xyz(99) + lhand(63) + rhand(63) = 225 dims
    """
    data = np.load(str(npz_path))

    pose  = data["pose"][:, :, :3].astype(np.float32)   # (T, 33, 3)
    lhand = data["lhand"].astype(np.float32)             # (T, 21, 3)
    rhand = data["rhand"].astype(np.float32)             # (T, 21, 3)

    # Pose-centre normalisation (mid-hip anchor)
    center = _pose_center(pose)   # (T, 1, 3)
    pose_n  = pose  - center
    lhand_n = lhand - center
    rhand_n = rhand - center

    T = pose.shape[0]
    feat = np.concatenate([
        pose_n.reshape(T, -1),    #  99
        lhand_n.reshape(T, -1),   #  63
        rhand_n.reshape(T, -1),   #  63
    ], axis=1).astype(np.float32) # 225

    return torch.from_numpy(feat)


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Bidirectional gloss ↔ index mapping.

    Special tokens (always at fixed indices):
        <blank> : 0  — required by CTC
        <unk>   : 1

    Adaptive update:
        Call vocab.add("NEW_GLOSS") at inference time to extend the mapping
        without touching the spatial embedding weights.  The model's CTC head
        must then be hot-swapped via AdaptiveBiLSTM.extend_vocab().
    """
    BLANK = "<blank>"
    UNK   = "<unk>"

    def __init__(self) -> None:
        self._w2i: dict[str, int] = {self.BLANK: 0, self.UNK: 1}
        self._i2w: dict[int, str] = {0: self.BLANK, 1: self.UNK}

    # ── construction ──────────────────────────────────────────────────────────

    def add(self, word: str) -> int:
        word = word.upper()
        if word not in self._w2i:
            idx = len(self._w2i)
            self._w2i[word] = idx
            self._i2w[idx]  = word
        return self._w2i[word]

    def build_from_labels(self, label_sequences: list[list[str]]) -> None:
        for seq in label_sequences:
            for gloss in seq:
                self.add(gloss)
        log.info("Vocabulary built: %d tokens", len(self))

    # ── lookup ────────────────────────────────────────────────────────────────

    def encode(self, glosses: list[str]) -> list[int]:
        unk_idx = self._w2i.get(self.UNK, 1)   # safe fallback — never KeyError
        return [self._w2i.get(g.upper(), unk_idx) for g in glosses]

    def decode(self, indices: list[int]) -> list[str]:
        return [self._i2w.get(i, self.UNK) for i in indices]

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._w2i, f, indent=2)
        log.info("Vocabulary saved → %s  (%d tokens)", path, len(self))

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        vocab = cls()
        with open(path) as f:
            w2i = json.load(f)
        vocab._w2i = {k: int(v) for k, v in w2i.items()}
        vocab._i2w = {int(v): k for k, v in w2i.items()}

        # Guarantee <blank>=0 and <unk>=1 are always present,
        # even if the vocab file was generated without them.
        if vocab.BLANK not in vocab._w2i:
            vocab._w2i[vocab.BLANK] = 0
            vocab._i2w[0] = vocab.BLANK
        if vocab.UNK not in vocab._w2i:
            # Insert at the next available index
            unk_idx = max(vocab._i2w.keys()) + 1
            vocab._w2i[vocab.UNK] = unk_idx
            vocab._i2w[unk_idx]   = vocab.UNK

        log.info("Vocabulary loaded: %d tokens from %s", len(vocab), path)
        return vocab

    def __len__(self) -> int:
        return len(self._w2i)

    @property
    def blank_idx(self) -> int:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CSLRDataset(Dataset):
    """
    Parameters
    ----------
    npz_dir    : root directory containing .npz landmark files
    label_map  : { npz_stem: ["GLOSS1", "GLOSS2", ...] }
    vocab      : shared Vocabulary instance
    augment    : if True, apply random augmentation each __getitem__ call
    """

    def __init__(
        self,
        npz_dir: Path,
        label_map: dict[str, list[str]],
        vocab: Vocabulary,
        augment: bool = False,
    ) -> None:
        self.vocab   = vocab
        self.augment = augment
        self.samples: list[tuple[Path, list[int]]] = []

        for npz_path in sorted(npz_dir.rglob("*.npz")):
            stem = npz_path.stem
            if stem not in label_map:
                log.debug("No label for %s — skipping", stem)
                continue
            self.samples.append((npz_path, vocab.encode(label_map[stem])))

        log.info("CSLRDataset: %d samples from %s (augment=%s)",
                 len(self.samples), npz_dir, augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int, int]:
        npz_path, label_ids = self.samples[idx]
        features = npz_to_feature(npz_path)                      # (T, FEAT_DIM)

        if self.augment:
            from .augment import augment_sequence
            features = augment_sequence(features)

        labels = torch.tensor(label_ids, dtype=torch.long)
        return features, labels, features.shape[0], len(label_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Collate — dynamic temporal padding
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(
    batch: list[tuple[Tensor, Tensor, int, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Dynamic temporal padding: pads each batch to its own longest sequence,
    not a global maximum.  This minimises wasted compute on short batches.

    Returns
    -------
    features   : (B, T_max, FEAT_DIM)  zero-padded
    labels     : (sum_label_lengths,)  concatenated flat (CTC format)
    input_lens : (B,)  actual frame counts
    label_lens : (B,)  actual label lengths
    """
    features, labels, input_lens, label_lens = zip(*batch)

    padded   = pad_sequence(features, batch_first=True)   # (B, T_max, F)
    flat_lbl = torch.cat(labels)                          # (sum_L,)
    in_lens  = torch.tensor(input_lens, dtype=torch.long)
    lb_lens  = torch.tensor(label_lens, dtype=torch.long)

    return padded, flat_lbl, in_lens, lb_lens


def build_dataloader(
    npz_dir: Path,
    label_map: dict[str, list[str]],
    vocab: Vocabulary,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = CSLRDataset(npz_dir, label_map, vocab)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
