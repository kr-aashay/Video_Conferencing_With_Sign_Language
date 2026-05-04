"""
cslr_model/augment.py
─────────────────────
Real-time data augmentation for landmark sequences.

With only 1 sample per class, augmentation is the only way to prevent
the model from memorising noise rather than learning sign structure.

Augmentations applied per sample per epoch (stochastic):
  1. TemporalJitter    — randomly drop or duplicate frames (±20%)
  2. SpeedPerturbation — linearly resample to 80-120% of original length
  3. CoordinateNoise   — add Gaussian noise to all landmark coordinates
  4. HandScale         — scale hand landmarks by 0.8-1.2× (signer distance)
  5. TemporalCrop      — randomly crop to 80-100% of sequence length
"""

from __future__ import annotations

import random
import numpy as np
import torch
from torch import Tensor


def augment_sequence(feat: Tensor, p: float = 0.8) -> Tensor:
    """
    Apply a random combination of augmentations to a (T, F) feature tensor.

    Parameters
    ----------
    feat : (T, F) float32 tensor
    p    : probability of applying each augmentation

    Returns
    -------
    (T', F) augmented tensor — T' may differ from T
    """
    x = feat.numpy().copy()   # (T, F)

    if random.random() < p:
        x = _speed_perturbation(x)

    if random.random() < p:
        x = _temporal_crop(x)

    if random.random() < p:
        x = _coordinate_noise(x)

    if random.random() < p:
        x = _hand_scale(x)

    if random.random() < p:
        x = _temporal_jitter(x)

    return torch.from_numpy(x.astype(np.float32))


# ── Individual augmentations ──────────────────────────────────────────────────

def _speed_perturbation(x: np.ndarray, low: float = 0.8, high: float = 1.2) -> np.ndarray:
    """Resample sequence to simulate signing at different speeds."""
    T, F   = x.shape
    factor = random.uniform(low, high)
    new_T  = max(4, int(T * factor))
    idx    = np.linspace(0, T - 1, new_T)
    out    = np.zeros((new_T, F), dtype=x.dtype)
    for f in range(F):
        out[:, f] = np.interp(idx, np.arange(T), x[:, f])
    return out


def _temporal_crop(x: np.ndarray, min_ratio: float = 0.8) -> np.ndarray:
    """Randomly crop the start/end of the sequence."""
    T = x.shape[0]
    keep = max(4, int(T * random.uniform(min_ratio, 1.0)))
    start = random.randint(0, T - keep)
    return x[start : start + keep]


def _coordinate_noise(x: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add small Gaussian noise to all coordinates."""
    return x + np.random.normal(0, std, x.shape).astype(x.dtype)


def _hand_scale(x: np.ndarray, low: float = 0.8, high: float = 1.2) -> np.ndarray:
    """
    Scale hand landmark coordinates (dims 99-224) independently.
    Simulates the signer being at different distances from the camera.
    """
    out   = x.copy()
    scale = random.uniform(low, high)
    out[:, 99:225] *= scale   # lhand(63) + rhand(63)
    return out


def _temporal_jitter(x: np.ndarray, max_drop: float = 0.1) -> np.ndarray:
    """Randomly drop up to max_drop fraction of frames."""
    T        = x.shape[0]
    n_drop   = int(T * random.uniform(0, max_drop))
    if n_drop == 0:
        return x
    drop_idx = set(random.sample(range(T), n_drop))
    keep     = [i for i in range(T) if i not in drop_idx]
    return x[keep] if len(keep) >= 4 else x
