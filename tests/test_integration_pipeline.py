"""
tests/test_integration_pipeline.py
═══════════════════════════════════════════════════════════════════════════════
Integration tests — end-to-end data pipeline

Tests the full data flow from raw .npz files through the model to decoded
glosses, without any real dataset or trained weights.

Covers:
  • npz_to_feature  (loads .npz, normalises, returns correct tensor)
  • CSLRDataset  (loads samples, skips unlabelled files)
  • collate_fn  (dynamic padding, CTC flat label format)
  • AdaptiveBiLSTM forward → CTCPrefixBeamDecoder  (shape contract)
  • CTC loss computation  (no NaN, finite loss)
  • Vocabulary → encode → decode roundtrip through model output
  • InferenceEngine.predict_window  (async, correct output keys)
  • Full stream-buffer → inference stub pipeline
    (SlidingWindowBuffer.push → InferenceEngine.predict_window)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from cslr_model.dataset import (
    Vocabulary, CSLRDataset, collate_fn, npz_to_feature, FEAT_DIM,
)
from cslr_model.decoder import CTCPrefixBeamDecoder
from cslr_model.model import build_model
from api.stream_buffer import SlidingWindowBuffer, validate_frame


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_npz(tmp_path):
    """Write a single synthetic .npz file and return its path."""
    T = 20
    data = {
        "pose":  np.random.rand(T, 33,  4).astype(np.float32),
        "face":  np.random.rand(T, 468, 3).astype(np.float32),
        "lhand": np.random.rand(T, 21,  3).astype(np.float32),
        "rhand": np.random.rand(T, 21,  3).astype(np.float32),
    }
    path = tmp_path / "HELP_01.npz"
    np.savez_compressed(str(path), **data)
    return path


@pytest.fixture
def vocab_4():
    v = Vocabulary()
    for g in ["HELP", "THANK", "YOU", "BANK"]:
        v.add(g)
    return v


@pytest.fixture
def small_model(vocab_4):
    return build_model(
        "bilstm", vocab_size=len(vocab_4),
        embed_dim=32, lstm_hidden=16, num_layers=2, dropout=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# npz_to_feature
# ─────────────────────────────────────────────────────────────────────────────

class TestNpzToFeature:
    def test_output_shape(self, tmp_npz):
        feat = npz_to_feature(tmp_npz)
        assert feat.shape[1] == FEAT_DIM
        assert feat.shape[0] == 20   # T frames

    def test_output_dtype(self, tmp_npz):
        feat = npz_to_feature(tmp_npz)
        assert feat.dtype == torch.float32

    def test_no_nan_in_output(self, tmp_npz):
        feat = npz_to_feature(tmp_npz)
        assert not torch.isnan(feat).any()

    def test_pose_centre_normalised(self, tmp_npz):
        """Mid-hip (lm 23+24 average) should be at origin after normalisation."""
        feat = npz_to_feature(tmp_npz)
        T = feat.shape[0]
        pose_xyz = feat[:, :99].reshape(T, 33, 3).numpy()
        mid_hip = (pose_xyz[:, 23, :] + pose_xyz[:, 24, :]) / 2
        np.testing.assert_allclose(mid_hip, 0.0, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# CSLRDataset
# ─────────────────────────────────────────────────────────────────────────────

class TestCSLRDataset:
    def _write_npz(self, path, T=15):
        data = {
            "pose":  np.random.rand(T, 33,  4).astype(np.float32),
            "face":  np.random.rand(T, 468, 3).astype(np.float32),
            "lhand": np.random.rand(T, 21,  3).astype(np.float32),
            "rhand": np.random.rand(T, 21,  3).astype(np.float32),
        }
        np.savez_compressed(str(path), **data)

    def test_loads_labelled_samples(self, tmp_path, vocab_4):
        self._write_npz(tmp_path / "HELP_01.npz")
        self._write_npz(tmp_path / "THANK_01.npz")
        label_map = {
            "HELP_01":  ["HELP"],
            "THANK_01": ["THANK"],
        }
        ds = CSLRDataset(tmp_path, label_map, vocab_4)
        assert len(ds) == 2

    def test_skips_unlabelled_files(self, tmp_path, vocab_4):
        self._write_npz(tmp_path / "HELP_01.npz")
        self._write_npz(tmp_path / "UNLABELLED.npz")
        label_map = {"HELP_01": ["HELP"]}
        ds = CSLRDataset(tmp_path, label_map, vocab_4)
        assert len(ds) == 1

    def test_getitem_returns_correct_types(self, tmp_path, vocab_4):
        self._write_npz(tmp_path / "HELP_01.npz")
        ds = CSLRDataset(tmp_path, {"HELP_01": ["HELP"]}, vocab_4)
        feat, labels, feat_len, label_len = ds[0]
        assert isinstance(feat, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert feat.shape[1] == FEAT_DIM
        assert label_len == 1

    def test_getitem_label_encodes_correctly(self, tmp_path, vocab_4):
        self._write_npz(tmp_path / "HELP_01.npz")
        ds = CSLRDataset(tmp_path, {"HELP_01": ["HELP"]}, vocab_4)
        _, labels, _, _ = ds[0]
        assert labels[0].item() == vocab_4._w2i["HELP"]


# ─────────────────────────────────────────────────────────────────────────────
# collate_fn
# ─────────────────────────────────────────────────────────────────────────────

class TestCollateFn:
    def _make_batch(self, sizes, vocab):
        batch = []
        for T in sizes:
            feat   = torch.randn(T, FEAT_DIM)
            labels = torch.tensor(vocab.encode(["HELP"]))
            batch.append((feat, labels, T, 1))
        return batch

    def test_padded_shape(self, vocab_4):
        batch = self._make_batch([10, 15, 8], vocab_4)
        feats, labels, in_lens, lb_lens = collate_fn(batch)
        assert feats.shape == (3, 15, FEAT_DIM)   # padded to max T=15

    def test_input_lens_correct(self, vocab_4):
        batch = self._make_batch([10, 15, 8], vocab_4)
        _, _, in_lens, _ = collate_fn(batch)
        assert sorted(in_lens.tolist(), reverse=True) == [15, 10, 8]

    def test_labels_concatenated_flat(self, vocab_4):
        batch = self._make_batch([10, 10], vocab_4)
        _, labels, _, lb_lens = collate_fn(batch)
        assert labels.shape[0] == lb_lens.sum().item()

    def test_padding_is_zero(self, vocab_4):
        batch = self._make_batch([5, 10], vocab_4)
        feats, _, _, _ = collate_fn(batch)
        # The shorter sequence (T=5) should have zeros in positions 5..9
        assert (feats[0, 5:, :] == 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Model → Decoder pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestModelDecoderPipeline:
    def test_forward_to_decode_no_crash(self, small_model, vocab_4):
        """Model returns (B, vocab_size) logits — argmax gives the predicted class."""
        x   = torch.randn(1, 20, FEAT_DIM)
        out = small_model(x)
        assert out.shape == (1, len(vocab_4))
        pred_idx = out[0].argmax().item()
        glosses  = vocab_4.decode([pred_idx])
        assert isinstance(glosses, list)
        assert len(glosses) == 1

    def test_ctc_loss_finite(self, small_model, vocab_4):
        """CrossEntropy loss must be finite on a valid batch."""
        import torch.nn as nn
        ce = nn.CrossEntropyLoss(label_smoothing=0.1)

        B, T = 2, 20
        x    = torch.randn(B, T, FEAT_DIM)
        lens = torch.tensor([T, T])
        out  = small_model(x, lens)   # (B, vocab_size)

        class_ids = torch.tensor([vocab_4.encode(["HELP"])[0],
                                   vocab_4.encode(["THANK"])[0]])
        loss = ce(out, class_ids)
        assert torch.isfinite(loss), f"CE loss is not finite: {loss}"

    def test_batch_decode_length_matches_batch(self, small_model, vocab_4):
        """Batch forward returns (B, vocab_size) — one prediction per sample."""
        B, T = 3, 15
        x    = torch.randn(B, T, FEAT_DIM)
        lens = torch.tensor([T, T - 3, T - 6])
        out  = small_model(x, lens)   # (B, vocab_size)
        assert out.shape == (B, len(vocab_4))
        preds = out.argmax(dim=-1).tolist()
        assert len(preds) == B

    def test_log_probs_sum_to_one(self, small_model, vocab_4):
        """Softmax of model logits should sum to 1 per sample."""
        x   = torch.randn(1, 10, FEAT_DIM)
        out = small_model(x)   # (1, vocab_size) raw logits
        probs = out[0].softmax(dim=-1)
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0), atol=1e-4, rtol=0)


# ─────────────────────────────────────────────────────────────────────────────
# InferenceEngine (async, stub predictor)
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceEngine:
    @pytest.mark.asyncio
    async def test_predict_window_returns_required_keys(self):
        """
        InferenceEngine wraps CSLRPredictor.predict_frames in a thread pool.
        We stub predict_frames to avoid loading weights.
        """
        from unittest.mock import MagicMock, patch
        from api.inference import InferenceEngine

        stub_predictor = MagicMock()
        stub_predictor.predict_frames.return_value = {
            "glosses":    ["I", "HELP"],
            "caption":    "I need help.",
            "latency_ms": 5.0,
        }

        with patch("api.inference.CSLRPredictor.from_checkpoint",
                   return_value=stub_predictor):
            engine = InferenceEngine(
                ckpt_path=Path("fake.pt"),
                vocab_path=Path("fake.json"),
                device="cpu",
            )

        frames = np.random.rand(20, FEAT_DIM).astype(np.float32)
        result = await engine.predict_window(frames)

        assert "glosses"    in result
        assert "caption"    in result
        assert "latency_ms" in result
        assert isinstance(result["glosses"], list)
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_predict_window_called_with_correct_frames(self):
        from unittest.mock import MagicMock, patch
        from api.inference import InferenceEngine

        stub_predictor = MagicMock()
        stub_predictor.predict_frames.return_value = {
            "glosses": [], "caption": "", "latency_ms": 1.0,
        }

        with patch("api.inference.CSLRPredictor.from_checkpoint",
                   return_value=stub_predictor):
            engine = InferenceEngine(
                ckpt_path=Path("fake.pt"),
                vocab_path=Path("fake.json"),
            )

        frames = np.ones((15, FEAT_DIM), dtype=np.float32)
        await engine.predict_window(frames)

        call_args = stub_predictor.predict_frames.call_args[0][0]
        np.testing.assert_array_equal(call_args, frames)
        await engine.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Stream buffer → inference stub (end-to-end data flow)
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamBufferToInference:
    @pytest.mark.asyncio
    async def test_full_pipeline_produces_result(self):
        """
        Simulate the WebSocket handler loop:
          validate_frame → SlidingWindowBuffer.push → InferenceEngine.predict_window
        """
        from unittest.mock import MagicMock, patch
        from api.inference import InferenceEngine

        stub_predictor = MagicMock()
        stub_predictor.predict_frames.return_value = {
            "glosses":    ["HELP"],
            "caption":    "Help me.",
            "latency_ms": 8.0,
        }

        with patch("api.inference.CSLRPredictor.from_checkpoint",
                   return_value=stub_predictor):
            engine = InferenceEngine(
                ckpt_path=Path("fake.pt"),
                vocab_path=Path("fake.json"),
            )

        buf = SlidingWindowBuffer(window_size=4, stride=2)
        result = None

        for _ in range(6):
            raw_frame = np.random.rand(FEAT_DIM).astype(np.float32)
            frame = validate_frame(raw_frame)
            if buf.push(frame):
                window = buf.get_window()
                result = await engine.predict_window(window)
                break

        assert result is not None
        assert result["caption"] == "Help me."
        await engine.shutdown()

    def test_buffer_window_fed_to_model_has_correct_shape(self):
        """The window extracted from the buffer must match FEAT_DIM."""
        buf = SlidingWindowBuffer(window_size=5, stride=3)
        for _ in range(5):
            frame = np.random.rand(FEAT_DIM).astype(np.float32)
            buf.push(frame)
        window = buf.get_window()
        assert window.shape == (5, FEAT_DIM)
        assert window.dtype == np.float32
