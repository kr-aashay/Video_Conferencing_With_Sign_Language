"""
cslr_model/metrics.py
─────────────────────
WER, CER, and Confusion Matrix computation for CSLR evaluation.

All functions are pure (no side effects) and operate on plain Python lists
so they can be called from trainer, notebooks, or evaluation scripts equally.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Edit distance  (shared by WER and CER)
# ─────────────────────────────────────────────────────────────────────────────

def edit_distance(a: list, b: list) -> int:
    """
    Standard dynamic-programming Levenshtein distance.
    O(len(a) × len(b)) time, O(len(b)) space.
    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ─────────────────────────────────────────────────────────────────────────────
# Word Error Rate
# ─────────────────────────────────────────────────────────────────────────────

def word_error_rate(
    predictions: list[list[str]],
    references:  list[list[str]],
) -> float:
    """
    WER = total_edit_distance(words) / total_reference_words

    Parameters
    ----------
    predictions : list of predicted gloss sequences
    references  : list of ground-truth gloss sequences

    Returns
    -------
    WER in [0, ∞)  — lower is better; 0.0 = perfect
    """
    total_dist = sum(edit_distance(p, r) for p, r in zip(predictions, references))
    total_len  = sum(len(r) for r in references)
    return total_dist / max(total_len, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Character Error Rate
# ─────────────────────────────────────────────────────────────────────────────

def character_error_rate(
    predictions: list[list[str]],
    references:  list[list[str]],
) -> float:
    """
    CER = total_edit_distance(characters) / total_reference_characters

    Each gloss is treated as a sequence of characters.
    Useful for detecting partial-match confusions (e.g. BANK vs BLANK).

    Returns
    -------
    CER in [0, ∞)  — lower is better
    """
    total_dist, total_len = 0, 0
    for pred_seq, ref_seq in zip(predictions, references):
        pred_chars = list(" ".join(pred_seq))
        ref_chars  = list(" ".join(ref_seq))
        total_dist += edit_distance(pred_chars, ref_chars)
        total_len  += len(ref_chars)
    return total_dist / max(total_len, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

class ConfusionMatrix:
    """
    Accumulates per-gloss prediction/reference pairs and computes a
    confusion matrix for the top-K most frequent signs.

    Usage
    ─────
        cm = ConfusionMatrix()
        cm.update(predictions, references)   # call after each val batch
        fig = cm.plot(top_k=50)
        fig.savefig("confusion_matrix.png")
    """

    def __init__(self) -> None:
        # counts[ref][pred] = number of times ref was predicted as pred
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._ref_totals: dict[str, int] = defaultdict(int)

    def update(
        self,
        predictions: list[list[str]],
        references:  list[list[str]],
    ) -> None:
        """
        Accumulate single-gloss pairs.

        For multi-gloss sequences, pairs are aligned by position (zip).
        Unaligned tokens (length mismatch) are skipped — they contribute
        to WER but not to the per-class confusion matrix.
        """
        for pred_seq, ref_seq in zip(predictions, references):
            for pred, ref in zip(pred_seq, ref_seq):
                self._counts[ref][pred] += 1
                self._ref_totals[ref]   += 1

    def reset(self) -> None:
        self._counts.clear()
        self._ref_totals.clear()

    def top_k_labels(self, k: int) -> list[str]:
        """Return the k most frequent reference labels."""
        return [
            label for label, _ in
            sorted(self._ref_totals.items(), key=lambda x: x[1], reverse=True)[:k]
        ]

    def to_array(self, labels: list[str]) -> np.ndarray:
        """
        Build a (len(labels), len(labels)) confusion matrix.
        Rows = reference, Columns = predicted.
        Values are row-normalised (recall per class).
        """
        n   = len(labels)
        idx = {lbl: i for i, lbl in enumerate(labels)}
        mat = np.zeros((n, n), dtype=np.float32)

        for ref, pred_counts in self._counts.items():
            if ref not in idx:
                continue
            r = idx[ref]
            for pred, count in pred_counts.items():
                if pred in idx:
                    mat[r, idx[pred]] += count

        # Row-normalise (avoid div-by-zero)
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return mat / row_sums

    def plot(
        self,
        top_k: int = 50,
        title: str = "Confusion Matrix (top-K signs)",
        figsize: tuple[int, int] = (18, 16),
        cmap: str = "Blues",
    ):
        """
        Generate a matplotlib Figure of the normalised confusion matrix.

        Returns
        -------
        matplotlib.figure.Figure  — caller is responsible for saving/showing
        """
        try:
            import matplotlib
            matplotlib.use("Agg")   # non-interactive backend — safe for servers
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not installed — cannot plot confusion matrix.")
            return None

        labels = self.top_k_labels(top_k)
        if not labels:
            log.warning("No data accumulated in ConfusionMatrix.")
            return None

        mat = self.to_array(labels)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)

        # Annotate cells with value if ≥ 0.1 (avoid clutter)
        thresh = 0.5
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = mat[i, j]
                if val >= 0.05:
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=5,
                        color="white" if val > thresh else "black",
                    )

        fig.tight_layout()
        return fig
