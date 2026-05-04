"""
cslr_model/predict.py
─────────────────────
Low-latency inference interface — designed for FastAPI real-time streams.

Target: sub-100ms end-to-end latency on CPU (MPS/CUDA will be faster).

Usage (standalone)
──────────────────
    predictor = CSLRPredictor.from_checkpoint(
        ckpt_path  = Path("checkpoints/best.pt"),
        vocab_path = Path("vocab.json"),
    )
    result = predictor.predict_npz(Path("sample.npz"))
    # {"glosses": ["I", "GIVE", "BANK"], "caption": "...", "latency_ms": 18.3}

Usage (FastAPI)
───────────────
    from fastapi import FastAPI, UploadFile
    from cslr_model.predict import CSLRPredictor

    app       = FastAPI()
    predictor = CSLRPredictor.from_checkpoint(...)

    @app.post("/predict")
    async def predict(file: UploadFile):
        tmp = Path(f"/tmp/{file.filename}")
        tmp.write_bytes(await file.read())
        return predictor.predict_npz(tmp)

Adaptive vocab update (no retraining)
──────────────────────────────────────
    predictor.extend_vocab(["NEW_SIGN_1", "NEW_SIGN_2"])
    # Adds tokens to vocab + grows CTC head weights in-place.
    # Spatial embedding and LSTM weights are untouched.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor

from .dataset  import npz_to_feature, Vocabulary
from .model    import AdaptiveBiLSTM, build_model
from .decoder  import CTCPrefixBeamDecoder, refine_with_slm

log = logging.getLogger(__name__)


class CSLRPredictor:
    """
    Thread-safe, stateless inference wrapper.

    Parameters
    ----------
    model      : trained AdaptiveBiLSTM
    vocab      : Vocabulary
    device     : "cuda" | "mps" | "cpu"
    beam_width : CTC prefix beam search width
    slm_fn     : optional callable(prompt) → str for gloss → English
    """

    def __init__(
        self,
        model: torch.nn.Module,
        vocab: Vocabulary,
        device: str = "cpu",
        beam_width: int = 10,
        slm_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.model   = model.to(device).eval()
        self.vocab   = vocab
        self.device  = device
        self.decoder = CTCPrefixBeamDecoder(vocab, beam_width=beam_width)
        self.slm_fn  = slm_fn

        # Compile for faster CPU inference (PyTorch 2.x)
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            log.info("Model compiled with torch.compile (reduce-overhead)")
        except Exception:
            log.debug("torch.compile unavailable — running in eager mode")

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: Path,
        vocab_path: Path,
        device: str | None = None,
        beam_width: int = 10,
        slm_fn: Optional[Callable[[str], str]] = None,
        **model_kwargs,
    ) -> "CSLRPredictor":
        """
        Load a trained checkpoint and return a ready-to-use predictor.

        model_kwargs are forwarded to AdaptiveBiLSTM (e.g. embed_dim, lstm_hidden).
        """
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps"  if torch.backends.mps.is_available()
                else "cpu"
            )

        vocab = Vocabulary.load(vocab_path)
        # Use the architecture that was trained: embed_dim=256, lstm_hidden=256
        model = build_model(
            "bilstm",
            vocab_size=len(vocab),
            embed_dim=model_kwargs.get("embed_dim", 256),
            lstm_hidden=model_kwargs.get("lstm_hidden", 256),
            num_layers=model_kwargs.get("num_layers", 2),
            dropout=model_kwargs.get("dropout", 0.3),
        )

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        log.info(
            "Loaded checkpoint: %s  (epoch %d, WER %.4f)",
            ckpt_path, ckpt.get("epoch", -1), ckpt.get("best_wer", float("nan")),
        )

        return cls(model, vocab, device=device, beam_width=beam_width, slm_fn=slm_fn)

    # ── adaptive vocab extension ──────────────────────────────────────────────

    def extend_vocab(self, new_glosses: list[str]) -> None:
        """
        Add new gloss tokens to the vocabulary and grow the CTC head in-place.

        Spatial embedding and LSTM weights are NOT modified — only the final
        linear projection gains new rows.  Fine-tune on new-gloss data after
        calling this to activate the new classes.

        Parameters
        ----------
        new_glosses : list of new gloss strings (case-insensitive)
        """
        for g in new_glosses:
            self.vocab.add(g)

        if isinstance(self.model, AdaptiveBiLSTM):
            self.model.extend_vocab(len(self.vocab))
            log.info(
                "Vocab extended to %d tokens. Fine-tune head to activate new glosses.",
                len(self.vocab),
            )
        else:
            log.warning(
                "Model is not AdaptiveBiLSTM — vocab extended in Vocabulary only. "
                "Rebuild the model head manually."
            )

    # ── core inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_features(self, features: Tensor) -> dict:
        """
        Parameters
        ----------
        features : (T, FEAT_DIM) float32 tensor

        Returns
        -------
        {
            "glosses"    : ["I", "GIVE", "BANK"],
            "caption"    : "I have sent the details to the bank.",
            "latency_ms" : 18.3
        }
        """
        t0 = time.perf_counter()

        x       = features.unsqueeze(0).to(self.device)              # (1, T, F)
        lengths = torch.tensor([features.shape[0]], device=self.device)

        logits = self.model(x, lengths)                               # (1, vocab_size)

        # Model returns (B, vocab_size) — argmax gives the predicted class
        pred_idx = logits[0].argmax().item()
        glosses  = self.vocab.decode([pred_idx])
        # Filter special tokens
        glosses  = [g for g in glosses if g not in ("<blank>", "<unk>")]

        caption   = refine_with_slm(glosses, self.slm_fn)

        latency_ms = (time.perf_counter() - t0) * 1000
        log.debug("Inference: %s → %.1f ms", glosses, latency_ms)

        return {
            "glosses":    glosses,
            "caption":    caption,
            "latency_ms": round(latency_ms, 2),
        }

    def predict_npz(self, npz_path: Path) -> dict:
        """Load a .npz landmark file and run inference."""
        return self.predict_features(npz_to_feature(npz_path))

    def predict_frames(self, frames: np.ndarray) -> dict:
        """
        Accept a raw (T, FEAT_DIM) numpy array — for streaming landmark input
        extracted on-the-fly by the MediaPipe pipeline.
        """
        return self.predict_features(
            torch.from_numpy(frames.astype(np.float32))
        )
