"""
cslr_model/export.py
────────────────────
Export trained AdaptiveBiLSTM to TorchScript and ONNX for low-latency
FastAPI inference.

TorchScript  — zero Python overhead, deployable without the source package.
ONNX         — portable, compatible with ONNX Runtime / TensorRT / CoreML.

Both formats are saved alongside the .pt checkpoint so the inference server
can pick whichever backend suits the deployment target.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from .dataset import FEAT_DIM

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TorchScript export
# ─────────────────────────────────────────────────────────────────────────────

def export_torchscript(
    model: nn.Module,
    out_path: Path,
    example_input: Tensor | None = None,
    seq_len: int = 64,
) -> Path:
    """
    Trace the model to TorchScript.

    TorchScript tracing is used (not scripting) because the model contains
    dynamic control flow (pack_padded_sequence) that scripting handles poorly
    across PyTorch versions.  The trace is valid for fixed-shape inference;
    variable lengths are handled by the lengths tensor at runtime.

    Parameters
    ----------
    model         : trained AdaptiveBiLSTM in eval mode
    out_path      : destination .pt file
    example_input : optional (1, T, FEAT_DIM) tensor; generated if None
    seq_len       : frame length used for the trace example

    Returns
    -------
    Path to the saved TorchScript file.
    """
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if example_input is None:
        example_input = torch.zeros(1, seq_len, FEAT_DIM)

    lengths = torch.tensor([seq_len], dtype=torch.long)

    with torch.no_grad():
        try:
            traced = torch.jit.trace(model, (example_input, lengths))
            traced.save(str(out_path))
            size_mb = out_path.stat().st_size / (1024 ** 2)
            log.info("TorchScript saved → %s  (%.1f MB)", out_path, size_mb)
        except Exception as exc:
            log.error("TorchScript export failed: %s", exc)
            raise

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(
    model: nn.Module,
    out_path: Path,
    seq_len: int = 64,
    opset: int = 17,
) -> Path:
    """
    Export the model to ONNX with dynamic axes for batch size and sequence length.

    Dynamic axes allow the ONNX Runtime to handle variable-length sequences
    without re-exporting — critical for real-time streaming inference.

    Parameters
    ----------
    model    : trained AdaptiveBiLSTM in eval mode
    out_path : destination .onnx file
    seq_len  : frame length used for the export example
    opset    : ONNX opset version (17 recommended for PyTorch 2.x)

    Returns
    -------
    Path to the saved ONNX file.
    """
    try:
        import onnx  # noqa: F401 — validate onnx is installed
    except ImportError:
        log.warning(
            "onnx package not installed. Run: pip install onnx onnxruntime\n"
            "Skipping ONNX export."
        )
        return out_path

    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input   = torch.zeros(1, seq_len, FEAT_DIM)
    dummy_lengths = torch.tensor([seq_len], dtype=torch.long)

    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                (dummy_input, dummy_lengths),
                str(out_path),
                opset_version=opset,
                input_names=["features", "lengths"],
                output_names=["log_probs"],
                dynamic_axes={
                    "features":  {0: "batch", 1: "time"},
                    "lengths":   {0: "batch"},
                    "log_probs": {0: "batch", 1: "time"},
                },
                do_constant_folding=True,
            )
            size_mb = out_path.stat().st_size / (1024 ** 2)
            log.info("ONNX saved → %s  (%.1f MB)", out_path, size_mb)
        except Exception as exc:
            log.error("ONNX export failed: %s", exc)
            raise

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: export both formats from a checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def export_from_checkpoint(
    ckpt_path: Path,
    vocab_size: int,
    export_dir: Path | None = None,
    arch_kwargs: dict | None = None,
    device: str = "cpu",
) -> dict[str, Path]:
    """
    Load a .pt checkpoint and export to both TorchScript and ONNX.

    Parameters
    ----------
    ckpt_path   : path to the .pt checkpoint file
    vocab_size  : vocabulary size (must match the checkpoint)
    export_dir  : output directory (defaults to ckpt_path.parent)
    arch_kwargs : extra kwargs forwarded to build_model (embed_dim, etc.)
    device      : device to load the model on

    Returns
    -------
    {"torchscript": Path, "onnx": Path}
    """
    from .model import build_model

    export_dir  = export_dir or ckpt_path.parent
    arch_kwargs = arch_kwargs or {}

    model = build_model("bilstm", vocab_size=vocab_size, **arch_kwargs)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stem = ckpt_path.stem
    ts_path   = export_dir / f"{stem}.torchscript.pt"
    onnx_path = export_dir / f"{stem}.onnx"

    export_torchscript(model, ts_path)
    export_onnx(model, onnx_path)

    return {"torchscript": ts_path, "onnx": onnx_path}
