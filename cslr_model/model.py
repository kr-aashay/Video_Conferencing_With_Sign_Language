"""
cslr_model/model.py
───────────────────
Sign-to-Gloss classifier for single-sign video clips.

Architecture
────────────
  (B, T, 225)  ← pose + hands only (face dropped for speed)
      │
  SpatialEmbedding
  Conv1D(225→256, k=5) + BN + GELU + Dropout
  Conv1D(256→256, k=3) + BN + GELU + Dropout
      │  (B, T, 256)
  TemporalEncoder
  2× Bi-LSTM(hidden=256 per dir → 512 total)
      │  (B, T, 512)
  AttentionPool  ← learned weighted average over time
      │  (B, 512)
  Classifier
  Linear(512→256) + GELU + Dropout
  Linear(256→vocab_size)  ← raw logits for CrossEntropyLoss
      │
  (B, vocab_size)

Design decisions
────────────────
- Raw logits (no log_softmax): CrossEntropyLoss applies it internally.
  Passing log_softmax into CE causes double-squashing → vanishing gradients.
- AttentionPool: learns which frames are most discriminative for each sign.
  Superior to last-frame or mean-pool for variable-length sequences.
- GELU activation: smoother gradients than ReLU for small datasets.
- kernel_size=5 in first conv: captures wider temporal context per frame.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .dataset import FEAT_DIM

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spatial Embedding
# ─────────────────────────────────────────────────────────────────────────────

class SpatialEmbedding(nn.Module):
    """
    Two Conv1D layers that compress per-frame landmark vectors.
    Operates over the feature dimension — no temporal leakage.

    Input  : (B, T, input_dim)
    Output : (B, T, embed_dim)
    """

    def __init__(
        self,
        input_dim:  int   = FEAT_DIM,
        embed_dim:  int   = 256,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, embed_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)   # (B, F, T)
        x = self.net(x)           # (B, embed_dim, T)
        return x.permute(0, 2, 1) # (B, T, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Temporal Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    Bidirectional LSTM with pack/pad for variable-length sequences.

    Input  : (B, T, embed_dim), lengths (B,)
    Output : (B, T, hidden_dim*2)
    """

    def __init__(
        self,
        input_dim:   int   = 256,
        lstm_hidden: int   = 256,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = lstm_hidden * 2

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        return out   # (B, T, hidden_dim*2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Attention Pooling
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPool(nn.Module):
    """
    Learned weighted average over the time dimension.

    Computes a scalar attention score per frame, masks padding,
    applies softmax, then returns the weighted sum.

    Input  : (B, T, H), lengths (B,)
    Output : (B, H)
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        # x: (B, T, H)
        scores = self.attn(x).squeeze(-1)                          # (B, T)

        # Mask padding positions with -inf before softmax
        T   = x.size(1)
        idx = torch.arange(T, device=x.device).unsqueeze(0)       # (1, T)
        mask = idx >= lengths.unsqueeze(1)                         # (B, T)
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)      # (B, T, 1)
        return (x * weights).sum(dim=1)                            # (B, H)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Classifier head
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    Two-layer MLP that maps pooled LSTM output to class logits.

    Returns RAW LOGITS — CrossEntropyLoss applies log_softmax internally.
    Never apply log_softmax here; doing so causes double-squashing and
    vanishing gradients.

    Input  : (B, hidden_dim)
    Output : (B, vocab_size)  raw logits
    """

    def __init__(self, hidden_dim: int, vocab_size: int, dropout: float = 0.3) -> None:
        super().__init__()
        mid = hidden_dim // 2
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, vocab_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)   # raw logits


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full model
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveBiLSTM(nn.Module):
    """
    End-to-end sign classifier.

    forward() returns raw logits (B, vocab_size) for CrossEntropyLoss.
    At inference, apply log_softmax externally (done in CSLRPredictor).

    Parameters
    ----------
    vocab_size   : number of output classes (including blank/unk)
    input_dim    : feature dimension per frame (default 225)
    embed_dim    : spatial embedding width
    lstm_hidden  : Bi-LSTM hidden units per direction
    num_layers   : number of Bi-LSTM layers
    dropout      : shared dropout rate
    """

    def __init__(
        self,
        vocab_size:  int,
        input_dim:   int   = FEAT_DIM,
        embed_dim:   int   = 256,
        lstm_hidden: int   = 256,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()
        self.spatial  = SpatialEmbedding(input_dim, embed_dim, dropout * 0.5)
        self.temporal = TemporalEncoder(embed_dim, lstm_hidden, num_layers, dropout)
        self.pool     = AttentionPool(self.temporal.output_dim)
        self.head     = ClassifierHead(self.temporal.output_dim, vocab_size, dropout)

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x       : (B, T, input_dim)
        lengths : (B,) actual frame counts

        Returns
        -------
        logits : (B, vocab_size)  — raw, no softmax
        """
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1),
                                 dtype=torch.long, device=x.device)

        x = self.spatial(x)              # (B, T, embed_dim)
        x = self.temporal(x, lengths)    # (B, T, lstm_hidden*2)
        x = self.pool(x, lengths)        # (B, lstm_hidden*2)
        return self.head(x)              # (B, vocab_size)

    # ── Legacy compatibility ──────────────────────────────────────────────────

    def extend_vocab(self, new_vocab_size: int) -> None:
        """Grow the output layer in-place (for adaptive vocab extension)."""
        old_fc   = self.head.net[-1]
        old_size = old_fc.out_features
        if new_vocab_size <= old_size:
            return
        new_fc = nn.Linear(old_fc.in_features, new_vocab_size)
        with torch.no_grad():
            new_fc.weight[:old_size] = old_fc.weight
            new_fc.bias[:old_size]   = old_fc.bias
            nn.init.uniform_(new_fc.weight[old_size:], -0.01, 0.01)
            nn.init.zeros_(new_fc.bias[old_size:])
        self.head.net[-1] = new_fc
        log.info("Vocab extended: %d → %d", old_size, new_vocab_size)

    def param_groups(self, lr_embed, lr_lstm, lr_head):
        return [
            {"params": self.spatial.parameters(),  "lr": lr_embed},
            {"params": self.temporal.parameters(), "lr": lr_lstm},
            {"params": list(self.pool.parameters()) +
                       list(self.head.parameters()),  "lr": lr_head},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(arch: str = "bilstm", vocab_size: int = 2, **kwargs) -> nn.Module:
    if arch == "bilstm":
        return AdaptiveBiLSTM(vocab_size=vocab_size, **kwargs)
    raise ValueError(f"Unknown arch: {arch!r}")
