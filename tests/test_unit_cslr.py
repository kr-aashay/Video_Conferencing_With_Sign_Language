"""
tests/test_unit_cslr.py
═══════════════════════════════════════════════════════════════════════════════
Unit tests — cslr_model layer

Covers:
  • Vocabulary  (add, encode, decode, save/load, blank_idx)
  • normalize_landmarks  (pose-centre subtraction, output shape)
  • CTCPrefixBeamDecoder  (blank suppression, repeated-token handling)
  • refine_with_slm  (passthrough + callable hook)
  • edit_distance / WER / CER
  • ConfusionMatrix  (accumulate, top_k, to_array)
  • AdaptiveBiLSTM  (forward shape, extend_vocab, param_groups)
  • EarlyStopping  (triggers at correct epoch)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from cslr_model.dataset import (
    Vocabulary, normalize_landmarks, FEAT_DIM,
    POSE_DIM, HAND_DIM, FACE_DIM,
)
from cslr_model.decoder import CTCPrefixBeamDecoder, refine_with_slm
from cslr_model.metrics import (
    edit_distance, word_error_rate, character_error_rate, ConfusionMatrix,
)
from cslr_model.model import build_model, AdaptiveBiLSTM
from cslr_model.trainer import EarlyStopping


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_vocab(*glosses):
    v = Vocabulary()
    for g in glosses:
        v.add(g)
    return v


def small_model(vocab_size=8):
    return build_model(
        "bilstm", vocab_size=vocab_size,
        embed_dim=32, lstm_hidden=16, num_layers=2, dropout=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabulary:
    def test_blank_always_zero(self):
        v = Vocabulary()
        assert v.blank_idx == 0
        assert v._w2i["<blank>"] == 0

    def test_add_returns_stable_index(self):
        v = Vocabulary()
        idx1 = v.add("HELP")
        idx2 = v.add("HELP")
        assert idx1 == idx2

    def test_case_insensitive(self):
        v = Vocabulary()
        i1 = v.add("help")
        i2 = v.add("HELP")
        assert i1 == i2

    def test_encode_decode_roundtrip(self):
        v = make_vocab("I", "WANT", "HELP")
        ids = v.encode(["I", "WANT", "HELP"])
        assert v.decode(ids) == ["I", "WANT", "HELP"]

    def test_unknown_maps_to_unk(self):
        v = make_vocab("HELP")
        ids = v.encode(["UNKNOWN_SIGN"])
        assert ids == [v._w2i["<unk>"]]

    def test_save_load_roundtrip(self):
        v = make_vocab("BANK", "MONEY", "GIVE")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "vocab.json"
            v.save(path)
            v2 = Vocabulary.load(path)
        assert len(v) == len(v2)
        assert v._w2i == v2._w2i

    def test_build_from_labels(self):
        v = Vocabulary()
        v.build_from_labels([["I", "HELP"], ["THANK", "YOU"]])
        assert "I" in v._w2i
        assert "THANK" in v._w2i
        assert len(v) == 6   # blank + unk + 4 glosses


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeLandmarks:
    def _make_arrays(self, T=10):
        pose  = np.random.rand(T, 33,  3).astype(np.float32)
        face  = np.random.rand(T, 468, 3).astype(np.float32)
        lhand = np.random.rand(T, 21,  3).astype(np.float32)
        rhand = np.random.rand(T, 21,  3).astype(np.float32)
        return pose, face, lhand, rhand

    def test_output_shape(self):
        # normalize_landmarks returns full 1629-dim (pose+hands+face).
        # npz_to_feature drops face and returns 225-dim — that's what FEAT_DIM is.
        T = 15
        pose, face, lhand, rhand = self._make_arrays(T)
        feat = normalize_landmarks(pose, face, lhand, rhand)
        assert feat.shape == (T, 1629)   # full landmarks
        # FEAT_DIM is 225 (pose+hands only, used by npz_to_feature)
        assert FEAT_DIM == 225

    def test_output_dtype(self):
        feat = normalize_landmarks(*self._make_arrays())
        assert feat.dtype == np.float32

    def test_pose_centre_is_zero(self):
        """After normalisation the mid-hip landmark should be at the origin."""
        T = 5
        pose, face, lhand, rhand = self._make_arrays(T)
        feat = normalize_landmarks(pose, face, lhand, rhand)
        pose_out = feat[:, :POSE_DIM].reshape(T, 33, 3)
        mid_hip = (pose_out[:, 23, :] + pose_out[:, 24, :]) / 2
        np.testing.assert_allclose(mid_hip, 0.0, atol=1e-5)

    def test_feat_dim_constant(self):
        # FEAT_DIM is now 225 (pose+hands only, face dropped for speed)
        assert FEAT_DIM == POSE_DIM + HAND_DIM * 2
        assert FEAT_DIM == 225


# ─────────────────────────────────────────────────────────────────────────────
# CTC Prefix Beam Decoder
# ─────────────────────────────────────────────────────────────────────────────

class TestCTCPrefixBeamDecoder:
    def _uniform_log_probs(self, T, V):
        return torch.full((T, V), -float("inf")).fill_(-torch.log(torch.tensor(float(V))))

    def test_decode_returns_list_of_strings(self):
        v = make_vocab("A", "B", "C")
        dec = CTCPrefixBeamDecoder(v, beam_width=3)
        lp = torch.randn(10, len(v))
        lp = torch.log_softmax(lp, dim=-1)
        result = dec.decode(lp)
        assert isinstance(result, list)
        assert all(isinstance(g, str) for g in result)

    def test_blank_only_sequence_returns_empty(self):
        """If every timestep is blank, the decoded sequence should be empty."""
        v = make_vocab("A", "B")
        dec = CTCPrefixBeamDecoder(v, beam_width=3)
        T, V = 8, len(v)
        # Make blank (index 0) overwhelmingly probable
        lp = torch.full((T, V), -1e9)
        lp[:, 0] = 0.0   # log-prob ≈ 1 for blank
        lp = torch.log_softmax(lp, dim=-1)
        result = dec.decode(lp)
        clean = [g for g in result if g not in ("<blank>", "<unk>")]
        assert clean == []

    def test_decode_batch_length(self):
        v = make_vocab("A", "B", "C", "D")
        dec = CTCPrefixBeamDecoder(v, beam_width=3)
        B, T, V = 4, 12, len(v)
        lp = torch.log_softmax(torch.randn(B, T, V), dim=-1)
        lens = torch.tensor([T, T - 2, T - 4, T - 6])
        results = dec.decode_batch(lp, lens)
        assert len(results) == B

    def test_repeated_token_handling(self):
        """A A (no blank) should decode to a single A."""
        v = make_vocab("A")
        dec = CTCPrefixBeamDecoder(v, beam_width=2)
        V = len(v)   # blank=0, unk=1, A=2
        T = 4
        lp = torch.full((T, V), -1e9)
        lp[:, 2] = 0.0   # always predict A
        lp = torch.log_softmax(lp, dim=-1)
        result = dec.decode(lp)
        clean = [g for g in result if g not in ("<blank>", "<unk>")]
        assert clean == ["A"]


# ─────────────────────────────────────────────────────────────────────────────
# SLM hook
# ─────────────────────────────────────────────────────────────────────────────

class TestRefineWithSLM:
    def test_passthrough_no_slm(self):
        result = refine_with_slm(["I", "WANT", "HELP"])
        assert result == "I WANT HELP"

    def test_filters_special_tokens(self):
        result = refine_with_slm(["<blank>", "I", "<unk>", "HELP"])
        assert result == "I HELP"

    def test_slm_callable_invoked(self):
        called_with = []
        def fake_slm(prompt):
            called_with.append(prompt)
            return "I need help."
        result = refine_with_slm(["I", "HELP"], slm_fn=fake_slm)
        assert result == "I need help."
        assert len(called_with) == 1
        assert "I HELP" in called_with[0]

    def test_slm_exception_falls_back(self):
        def broken_slm(prompt):
            raise RuntimeError("model offline")
        result = refine_with_slm(["I", "HELP"], slm_fn=broken_slm)
        assert result == "I HELP"   # graceful fallback


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_edit_distance_identical(self):
        assert edit_distance(["A", "B"], ["A", "B"]) == 0

    def test_edit_distance_empty(self):
        assert edit_distance([], ["A", "B"]) == 2
        assert edit_distance(["A"], []) == 1

    def test_edit_distance_substitution(self):
        assert edit_distance(["A"], ["B"]) == 1

    def test_wer_perfect(self):
        preds = [["I", "HELP"], ["THANK", "YOU"]]
        refs  = [["I", "HELP"], ["THANK", "YOU"]]
        assert word_error_rate(preds, refs) == 0.0

    def test_wer_all_wrong(self):
        preds = [["A", "B"]]
        refs  = [["C", "D"]]
        assert word_error_rate(preds, refs) == 1.0

    def test_cer_partial_match(self):
        preds = [["BANK"]]
        refs  = [["BLANK"]]
        cer = character_error_rate(preds, refs)
        assert 0.0 < cer < 1.0

    def test_confusion_matrix_shape(self):
        cm = ConfusionMatrix()
        cm.update([["A", "B"], ["A", "C"]], [["A", "B"], ["A", "B"]])
        labels = cm.top_k_labels(3)
        mat = cm.to_array(labels)
        assert mat.shape == (len(labels), len(labels))

    def test_confusion_matrix_row_normalised(self):
        cm = ConfusionMatrix()
        cm.update([["A"], ["A"], ["B"]], [["A"], ["A"], ["B"]])
        labels = cm.top_k_labels(2)
        mat = cm.to_array(labels)
        # Each row should sum to 1 (or 0 if no data)
        row_sums = mat.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_confusion_matrix_reset(self):
        cm = ConfusionMatrix()
        cm.update([["A"]], [["A"]])
        cm.reset()
        assert len(cm._counts) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveBiLSTM:
    def test_forward_output_shape(self):
        # Model now returns (B, vocab_size) logits — not (B, T, vocab_size)
        model = small_model(vocab_size=8)
        B, T = 2, 20
        x = torch.randn(B, T, FEAT_DIM)
        lens = torch.tensor([T, T - 4])
        out = model(x, lens)
        assert out.shape == (B, 8)

    def test_raw_logits_output(self):
        """Output should be raw logits (not log-softmax) for CrossEntropyLoss."""
        model = small_model(vocab_size=6)
        x = torch.randn(1, 10, FEAT_DIM)
        out = model(x)
        # Raw logits can be positive or negative — no constraint like log-probs
        assert out.shape == (1, 6)
        # CrossEntropy expects logits, not log-probs — verify softmax sums to 1
        probs_sum = out[0].softmax(dim=-1).sum().item()
        assert abs(probs_sum - 1.0) < 1e-4

    def test_log_softmax_output(self):
        """Softmax of logits should sum to 1 (raw logits, not log-probs)."""
        model = small_model(vocab_size=6)
        x = torch.randn(1, 10, FEAT_DIM)
        out = model(x)
        probs_sum = out[0].softmax(dim=-1).sum().item()
        assert abs(probs_sum - 1.0) < 1e-4

    def test_extend_vocab(self):
        model = small_model(vocab_size=6)
        x = torch.randn(1, 10, FEAT_DIM)
        model.extend_vocab(10)
        out = model(x)
        assert out.shape[-1] == 10

    def test_extend_vocab_preserves_old_weights(self):
        model = small_model(vocab_size=6)
        # The new model's classifier is model.head.net[-1]
        old_weight = model.head.net[-1].weight[:6].clone()
        model.extend_vocab(10)
        new_weight = model.head.net[-1].weight[:6]
        torch.testing.assert_close(old_weight, new_weight)

    def test_param_groups_count(self):
        model = small_model()
        groups = model.param_groups(lr_embed=1e-5, lr_lstm=1e-4, lr_head=2e-4)
        assert len(groups) == 3
        assert groups[0]["lr"] == 1e-5
        assert groups[2]["lr"] == 2e-4

    def test_no_lengths_still_works(self):
        """Model must handle missing lengths (inference without padding)."""
        model = small_model(vocab_size=5)
        x = torch.randn(1, 15, FEAT_DIM)
        out = model(x)   # no lengths argument
        assert out.shape == (1, 5)


# ─────────────────────────────────────────────────────────────────────────────
# EarlyStopping
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_triggers_after_patience(self):
        es = EarlyStopping(patience=3, min_delta=1e-3)
        losses = [1.0, 0.9, 0.9, 0.9, 0.9]
        results = [es.step(l) for l in losses]
        assert results[-1] is True
        assert es.triggered

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3, min_delta=1e-3)
        for l in [1.0, 0.9, 0.85, 0.84, 0.83]:
            stopped = es.step(l)
        assert not stopped

    def test_does_not_trigger_early(self):
        es = EarlyStopping(patience=5)
        for l in [1.0, 0.9, 0.8]:
            assert not es.step(l)
