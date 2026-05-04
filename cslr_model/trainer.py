"""
cslr_model/trainer.py
─────────────────────
Training loop for single-sign CrossEntropy classification.

The model now returns (B, vocab_size) logits directly — no pooling needed here.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from .dataset import Vocabulary
from .metrics import ConfusionMatrix

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 1e-5) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self._best     = float("inf")
        self._counter  = 0
        self.triggered = False

    def step(self, loss: float) -> bool:
        if loss < self._best - self.min_delta:
            self._best    = loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.triggered = True
        return self.triggered


# ─────────────────────────────────────────────────────────────────────────────
# CSV Logger
# ─────────────────────────────────────────────────────────────────────────────

class _CSVLogger:
    FIELDS = ["epoch", "train_loss", "train_acc", "lr", "elapsed_s"]

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class CSLRTrainer:
    """
    CrossEntropy classification trainer.

    The model's forward() returns (B, vocab_size) raw logits.
    CrossEntropyLoss applies log_softmax internally — never apply it in the model.
    """

    def __init__(
        self,
        model: nn.Module,
        vocab: Vocabulary,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        grad_clip: float = 5.0,
        ckpt_dir: Path = Path("checkpoints"),
        log_dir: Path = Path("logs"),
        beam_width: int = 1,
        early_stop_patience: int = 30,
        cm_every_n: int = 0,
        export_best: bool = True,
        loss_mode: str = "crossentropy",
    ) -> None:
        self.model        = model.to(device)
        self.vocab        = vocab
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.max_epochs   = max_epochs
        self.grad_clip    = grad_clip
        self.ckpt_dir     = ckpt_dir
        self.log_dir      = log_dir
        self.cm_every_n   = cm_every_n
        self.export_best  = export_best
        self._loss_mode   = loss_mode

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Label smoothing 0.1 — prevents overconfident wrong predictions
        # on a 1-sample-per-class dataset
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        # OneCycleLR: fast warmup + cosine decay — best for small datasets
        self.optimizer = AdamW(
            model.parameters(), lr=lr,
            weight_decay=weight_decay, eps=1e-8,
        )
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=max_epochs,
            pct_start=0.1,          # 10% warmup
            anneal_strategy="cos",
            div_factor=10.0,        # start at lr/10
            final_div_factor=1000,  # end at lr/1000
        )

        self.conf_mat    = ConfusionMatrix()
        self.best_loss   = float("inf")
        self.start_epoch = 0
        self._best_ckpt: Optional[Path] = None
        self._early_stop = EarlyStopping(patience=early_stop_patience)
        self._csv        = _CSVLogger(log_dir / "training_log.csv")

        # Legacy attributes
        self.best_wer      = float("inf")
        self.best_val_loss = float("inf")

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, loss: float, acc: float = 0.0) -> Path:
        path = self.ckpt_dir / f"epoch_{epoch:03d}_loss{loss:.4f}_acc{acc:.3f}.pt"
        torch.save({
            "epoch":      epoch,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
            "best_loss":  self.best_loss,
            "vocab_size": len(self.vocab),
            "best_wer":      loss,
            "best_val_loss": loss,
        }, path)
        log.info("Checkpoint saved → %s", path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.best_loss   = ckpt.get("best_loss", float("inf"))
        self.best_wer    = self.best_loss
        self.start_epoch = ckpt["epoch"] + 1
        log.info("Resumed from %s (epoch %d, loss %.4f)",
                 path, ckpt["epoch"], self.best_loss)

    # ── Training epoch ────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.perf_counter()

        for features, labels, input_lens, label_lens in self.train_loader:
            features   = features.to(self.device)
            input_lens = input_lens.to(self.device)

            # Forward → (B, vocab_size) raw logits
            logits = self.model(features, input_lens)

            # Extract class label (first gloss per sample)
            B = logits.size(0)
            class_ids = torch.zeros(B, dtype=torch.long, device=self.device)
            offset = 0
            for i, length in enumerate(label_lens.tolist()):
                class_ids[i] = labels[offset]
                offset += length

            loss = self.ce_loss(logits, class_ids)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()   # OneCycleLR steps per batch

            total_loss += loss.item()
            preds       = logits.argmax(dim=-1)
            correct    += (preds == class_ids).sum().item()
            total      += B

        avg_loss = total_loss / max(len(self.train_loader), 1)
        acc      = correct / max(total, 1)
        elapsed  = time.perf_counter() - t0

        log.info(
            "Epoch %03d | loss=%.4f | acc=%.1f%% | lr=%.2e | %.1fs",
            epoch, avg_loss, acc * 100,
            self.scheduler.get_last_lr()[0], elapsed,
        )
        return avg_loss, acc

    # ── Main loop ─────────────────────────────────────────────────────────────

    def fit(self) -> None:
        log.info(
            "Training | device=%s | epochs=%d | batches/epoch=%d | lr=%.1e",
            self.device, self.max_epochs,
            len(self.train_loader),
            self.optimizer.param_groups[0]["lr"],
        )

        epoch = self.start_epoch
        try:
            for epoch in range(self.start_epoch, self.max_epochs):
                train_loss, train_acc = self._train_epoch(epoch)

                self._csv.write({
                    "epoch":      epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc":  f"{train_acc:.4f}",
                    "lr":         f"{self.scheduler.get_last_lr()[0]:.2e}",
                    "elapsed_s":  "0",
                })

                if train_loss < self.best_loss:
                    self.best_loss     = train_loss
                    self.best_wer      = train_loss
                    self.best_val_loss = train_loss

                    # Remove previous best checkpoint to save disk space
                    if self._best_ckpt and self._best_ckpt.exists():
                        self._best_ckpt.unlink(missing_ok=True)
                        ts = self._best_ckpt.with_suffix(".torchscript.pt")
                        ts.unlink(missing_ok=True)

                    self._best_ckpt = self.save_checkpoint(
                        epoch, train_loss, train_acc
                    )
                    if self.export_best:
                        self._export_best_torchscript()

                if self._early_stop.step(train_loss):
                    log.info("Early stopping at epoch %d (loss=%.4f, acc=%.1f%%)",
                             epoch, train_loss, train_acc * 100)
                    break

        except KeyboardInterrupt:
            log.warning("Interrupted — saving checkpoint.")
            self.save_checkpoint(epoch, self.best_loss)
        finally:
            self._csv.close()

        log.info("Training complete | best_loss=%.4f", self.best_loss)

    # ── TorchScript export ────────────────────────────────────────────────────

    def _export_best_torchscript(self) -> None:
        if self._best_ckpt is None:
            return
        try:
            from .export import export_torchscript
            ts_path = self._best_ckpt.with_suffix(".torchscript.pt")
            export_torchscript(self.model, ts_path)
        except Exception as exc:
            log.warning("TorchScript export failed (non-fatal): %s", exc)

    # ── Legacy no-op ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> tuple[float, float, float]:
        return float("inf"), float("inf"), float("inf")
