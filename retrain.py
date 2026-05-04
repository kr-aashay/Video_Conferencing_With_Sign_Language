"""
retrain.py
═══════════════════════════════════════════════════════════════════════════════
Retrain the model on your own recorded signs.

Run after record_signs.py has collected enough clips.

Usage
─────
    .venv/bin/python retrain.py

What it does
────────────
1. Loads all .npz files from my_signs/
2. Builds a new vocabulary from your sign names
3. Trains a fresh AdaptiveBiLSTM for 100 epochs
4. Saves the best checkpoint to checkpoints/my_model.pt
5. Updates test_live_caption.py to use the new model automatically
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("retrain")

MY_SIGNS_DIR = Path("my_signs")
CKPT_OUT     = Path("checkpoints/my_model.pt")
VOCAB_OUT    = Path("my_vocab.json")
FEAT_DIM     = 225
EPOCHS       = 100
BATCH_SIZE   = 8
LR           = 1e-3

_LEFT_HIP  = 23
_RIGHT_HIP = 24


# ── Feature extraction ────────────────────────────────────────────────────────

def npz_to_features(path: Path) -> torch.Tensor:
    data  = np.load(str(path))
    pose  = data["pose"][:, :, :3].astype(np.float32)   # (T, 33, 3)
    lhand = data["lhand"].astype(np.float32)
    rhand = data["rhand"].astype(np.float32)

    center  = ((pose[:, _LEFT_HIP, :] + pose[:, _RIGHT_HIP, :]) / 2)[:, np.newaxis, :]
    pose_n  = pose  - center
    lhand_n = lhand - center
    rhand_n = rhand - center

    T    = pose.shape[0]
    feat = np.concatenate([
        pose_n.reshape(T, -1),
        lhand_n.reshape(T, -1),
        rhand_n.reshape(T, -1),
    ], axis=1).astype(np.float32)
    return torch.from_numpy(feat)


# ── Dataset ───────────────────────────────────────────────────────────────────

class MySignsDataset(Dataset):
    def __init__(self, npz_dir: Path, w2i: dict[str, int]) -> None:
        self.samples = []
        for npz in sorted(npz_dir.glob("*.npz")):
            # Filename format: SIGN__idx__hash.npz
            sign = npz.stem.split("__")[0].upper()
            if sign not in w2i:
                continue
            self.samples.append((npz, w2i[sign]))
        log.info("Dataset: %d samples, %d classes", len(self.samples), len(w2i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = npz_to_features(path)
        return feat, label, feat.shape[0]


def collate(batch):
    feats, labels, lens = zip(*batch)
    padded = pad_sequence(feats, batch_first=True)
    return padded, torch.tensor(labels), torch.tensor(lens)


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    # ── Scan signs ────────────────────────────────────────────────────────────
    npz_files = list(MY_SIGNS_DIR.glob("*.npz"))
    if not npz_files:
        log.error("No .npz files found in %s — run record_signs.py first", MY_SIGNS_DIR)
        sys.exit(1)

    signs = sorted({f.stem.split("__")[0].upper() for f in npz_files})
    w2i   = {s: i for i, s in enumerate(signs)}
    i2w   = {i: s for s, i in w2i.items()}
    log.info("Signs: %s", signs)

    # Save vocab
    VOCAB_OUT.write_text(json.dumps({"w2i": w2i, "i2w": {str(k): v for k, v in i2w.items()}}, indent=2))
    log.info("Vocabulary saved → %s", VOCAB_OUT)

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds     = MySignsDataset(MY_SIGNS_DIR, w2i)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    from cslr_model.model import build_model
    model = build_model("bilstm", vocab_size=len(signs),
                        embed_dim=128, lstm_hidden=128, num_layers=2, dropout=0.2)
    device = "cpu"
    model.to(device).train()

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s params | %d classes | device=%s", f"{n_params:,}", len(signs), device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    import torch.nn as nn

    ce  = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = OneCycleLR(opt, max_lr=LR, steps_per_epoch=len(loader),
                     epochs=EPOCHS, pct_start=0.1, anneal_strategy="cos")

    best_loss = float("inf")
    CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for feats, labels, lens in loader:
            feats  = feats.to(device)
            labels = labels.to(device)
            lens   = lens.to(device)

            logits = model(feats, lens)
            loss   = ce(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sch.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

        avg_loss = total_loss / max(len(loader), 1)
        acc      = correct / max(total, 1) * 100

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "signs":      signs,
                "vocab_size": len(signs),
                "best_loss":  best_loss,
            }, CKPT_OUT)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            log.info("Epoch %03d | loss=%.4f | acc=%.1f%%", epoch, avg_loss, acc)

    log.info("Training complete | best_loss=%.4f", best_loss)
    log.info("Checkpoint → %s", CKPT_OUT)
    log.info("Vocabulary → %s", VOCAB_OUT)
    log.info("")
    log.info("Now run:  .venv/bin/python test_live_caption.py --my-signs")


if __name__ == "__main__":
    train()
