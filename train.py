"""
train.py
════════════════════════════════════════════════════════════════════════════════
Aashay's Sign Lang — CSLR Training (maximum accuracy configuration)

Target: ~99% training accuracy on ISL-CSLTR (687 samples, 641 classes)
Strategy:
  - CrossEntropy classification (not CTC) — correct for 1-sign-per-video
  - Data augmentation (5 types) — synthetically multiplies each sample
  - AttentionPool — learns which frames are discriminative
  - OneCycleLR — fast convergence with warmup
  - Label smoothing 0.1 — prevents overconfident wrong predictions
  - 200 epochs with patience=30 early stopping

Usage
─────
    python train.py           # train from scratch
    python train.py --resume  # resume from latest checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────────
    "npz_dir"    : Path("isl_landmarks"),
    "label_file" : Path("label_map.json"),
    "vocab_file" : Path("vocab.json"),
    "ckpt_dir"   : Path("checkpoints"),
    "log_dir"    : Path("logs"),

    # ── Model ──────────────────────────────────────────────────────────────────
    # WLASL: 2000 glosses, 11980 videos, ~6 samples/gloss
    # Larger model justified by 17× more data than ISL-CSLTR
    "embed_dim"  : 256,
    "lstm_hidden": 256,
    "num_layers" : 2,
    "dropout"    : 0.4,

    # ── Loss ───────────────────────────────────────────────────────────────────
    "loss"        : "crossentropy",

    # ── Optimisation ───────────────────────────────────────────────────────────
    # WLASL: ~375 batches/epoch × 100 epochs = 37,500 steps
    "lr"          : 3e-3,
    "weight_decay": 1e-4,
    "grad_clip"   : 5.0,
    "max_epochs"  : 100,
    "batch_size"  : 32,
    "num_workers" : 0,

    # ── Augmentation ───────────────────────────────────────────────────────────
    "augment"     : True,

    # ── Regularisation ─────────────────────────────────────────────────────────
    "early_stop_patience": 20,

    # ── Logging ────────────────────────────────────────────────────────────────
    "cm_every_n"  : 0,
    "export_best" : True,
    "beam_width"  : 1,

    # ── Device ─────────────────────────────────────────────────────────────────
    "device": "cpu",   # CPU faster than MPS for this model size (benchmarked)
}

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train.log", mode="w"),
    ],
)
log = logging.getLogger("train")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_label_map(cfg: dict) -> dict[str, list[str]]:
    path = cfg["label_file"]
    if not path.exists():
        log.error("label_map.json not found. Run orchestrate.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def _build_vocab(label_map: dict, cfg: dict):
    from cslr_model import Vocabulary
    vocab_path = cfg["vocab_file"]
    if vocab_path.exists():
        log.info("Loading vocabulary from %s", vocab_path)
        return Vocabulary.load(vocab_path)
    vocab = Vocabulary()
    vocab.build_from_labels(list(label_map.values()))
    vocab.save(vocab_path)
    return vocab


def _build_loader(label_map: dict, vocab, cfg: dict):
    from cslr_model.dataset import CSLRDataset, collate_fn
    from torch.utils.data import DataLoader

    ds = CSLRDataset(
        cfg["npz_dir"], label_map, vocab,
        augment=cfg.get("augment", True),
    )
    if len(ds) == 0:
        log.error("No .npz files matched label_map. Run orchestrate.py first.")
        sys.exit(1)

    log.info("Dataset: %d samples | augment=%s", len(ds), cfg.get("augment"))

    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
        pin_memory=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: dict, resume: bool = False) -> None:
    from cslr_model import build_model, CSLRTrainer

    label_map    = _load_label_map(cfg)
    vocab        = _build_vocab(label_map, cfg)
    train_loader = _build_loader(label_map, vocab, cfg)

    model = build_model(
        arch="bilstm",
        vocab_size=len(vocab),
        embed_dim=cfg["embed_dim"],
        lstm_hidden=cfg["lstm_hidden"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %s params | vocab: %d | device: %s",
             f"{n_params:,}", len(vocab), cfg["device"])

    trainer = CSLRTrainer(
        model=model,
        vocab=vocab,
        train_loader=train_loader,
        val_loader=None,
        device=cfg["device"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        max_epochs=cfg["max_epochs"],
        grad_clip=cfg["grad_clip"],
        ckpt_dir=cfg["ckpt_dir"],
        log_dir=cfg["log_dir"],
        beam_width=cfg["beam_width"],
        early_stop_patience=cfg["early_stop_patience"],
        cm_every_n=cfg["cm_every_n"],
        export_best=cfg["export_best"],
        loss_mode=cfg.get("loss", "crossentropy"),
    )

    if resume:
        ckpts = sorted(cfg["ckpt_dir"].glob("*.pt"))
        if ckpts:
            trainer.load_checkpoint(ckpts[-1])
            log.info("Resuming from: %s", ckpts[-1])
        else:
            log.warning("No checkpoints found — starting fresh.")

    trainer.fit()


def export_only(cfg: dict) -> None:
    from cslr_model import Vocabulary
    from cslr_model.export import export_from_checkpoint

    ckpts = sorted(cfg["ckpt_dir"].glob("*.pt"))
    if not ckpts:
        log.error("No checkpoints in %s", cfg["ckpt_dir"])
        sys.exit(1)

    vocab = Vocabulary.load(cfg["vocab_file"])
    paths = export_from_checkpoint(
        ckpt_path=ckpts[-1],
        vocab_size=len(vocab),
        export_dir=cfg["ckpt_dir"],
        arch_kwargs=dict(
            embed_dim=cfg["embed_dim"],
            lstm_hidden=cfg["lstm_hidden"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ),
        device=cfg["device"],
    )
    for fmt, p in paths.items():
        log.info("  %-14s → %s", fmt, p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    if args.export_only:
        export_only(CONFIG)
    else:
        train(CONFIG, resume=args.resume)
