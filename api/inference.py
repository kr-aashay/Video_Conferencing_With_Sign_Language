"""
api/inference.py
════════════════════════════════════════════════════════════════════════════════
HexaMinds — Inference Engine + Ollama Refinement Bridge

Pipeline
────────
  frames (T, 1629)
      │
      ▼
  CSLRPredictor          [ThreadPoolExecutor — never blocks event loop]
  Bi-LSTM + CTC beam
      │
      ├─ glosses  ──────────────────────────────────────────────────────────┐
      │                                                                     │
      └─ confidence ≥ 0.8?                                                  │
              │ YES                                                          │
              ▼                                                              │
      OllamaRefinementBridge                [asyncio — native async]        │
      phi3 / any Ollama model                                               │
      200 ms hard timeout                                                   │
              │                                                             │
              ├─ success → refined English sentence                         │
              └─ timeout / error → raw gloss string ◄────────────────────── ┘

SLM providers
─────────────
  SLM_PROVIDER=ollama   (default — local Phi-3, zero cloud cost)
  SLM_PROVIDER=openai   (any OpenAI-compatible endpoint)
  SLM_PROVIDER=none     (passthrough — raw glosses only)

Ollama quick-start
──────────────────
  1. Install:  https://ollama.com
  2. Pull:     ollama pull phi3
  3. .env:     SLM_PROVIDER=ollama

Timeout
───────
  OLLAMA_TIMEOUT_MS=200   (default — falls back to raw glosses above this)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch

from cslr_model.predict import CSLRPredictor

log = logging.getLogger(__name__)

# ── Tunable constants (all overridable via env) ───────────────────────────────
CONFIDENCE_GATE:  float = float(os.environ.get("SLM_CONFIDENCE_GATE", "0.8"))
OLLAMA_TIMEOUT_MS: float = float(os.environ.get("OLLAMA_TIMEOUT_MS",  "200"))

# ── Context-aware system prompt ───────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a linguistic expert in Indian Sign Language. "
    "Translate the following sequence of signs into a natural, professional "
    "English sentence for a banking fraud detection context. "
    "Output ONLY the sentence — no explanation, no extra punctuation."
)

_USER_TEMPLATE = "Signs: {glosses}"


# ════════════════════════════════════════════════════════════════════════════════
# Ollama Refinement Bridge
# ════════════════════════════════════════════════════════════════════════════════

class OllamaRefinementBridge:
    """
    Async refinement bridge using the official ``ollama`` Python library.

    Calls ``ollama.AsyncClient.chat()`` with a structured system + user prompt.
    A hard timeout of ``OLLAMA_TIMEOUT_MS`` milliseconds is enforced via
    ``asyncio.wait_for``; on expiry the raw gloss string is returned immediately
    so the broadcast is never delayed beyond the SLA.

    Parameters
    ----------
    model      : Ollama model tag (default "phi3")
    host       : Ollama server URL (default "http://localhost:11434")
    timeout_ms : hard timeout in milliseconds (default 200)
    """

    def __init__(
        self,
        model:      str   = "phi3",
        host:       str   = "http://localhost:11434",
        timeout_ms: float = OLLAMA_TIMEOUT_MS,
    ) -> None:
        try:
            import ollama as _ollama
            self._client     = _ollama.AsyncClient(host=host)
            self._model      = model
            self._timeout_s  = timeout_ms / 1000.0
            self._available  = True
            log.info(
                "OllamaRefinementBridge ready | model=%s host=%s timeout=%.0fms",
                model, host, timeout_ms,
            )
        except ImportError:
            log.warning(
                "ollama package not installed — bridge disabled. "
                "Run: pip install ollama"
            )
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    async def refine(self, glosses: list[str]) -> tuple[str, bool]:
        """
        Refine a gloss list into a natural English sentence.

        Parameters
        ----------
        glosses : clean gloss list, e.g. ["I", "GIVE", "BANK"]

        Returns
        -------
        (caption, slm_used)
            caption  : refined sentence or space-joined fallback
            slm_used : True if Ollama actually produced the caption
        """
        raw = " ".join(glosses)

        if not self._available or not glosses:
            return raw, False

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_TEMPLATE.format(glosses=raw)},
        ]

        t0 = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self._client.chat(
                    model=self._model,
                    messages=messages,
                    options={"temperature": 0.1, "num_predict": 64},
                ),
                timeout=self._timeout_s,
            )
            caption  = response.message.content.strip()
            elapsed  = (time.perf_counter() - t0) * 1000
            log.debug(
                "Ollama refined: %r → %r  (%.1f ms)",
                raw, caption, elapsed,
            )
            return caption, True

        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - t0) * 1000
            log.warning(
                "Ollama timeout after %.1f ms (limit %.0f ms) — "
                "returning raw glosses: %r",
                elapsed, OLLAMA_TIMEOUT_MS * 1000, raw,
            )
            return raw, False

        except Exception as exc:
            log.warning("Ollama error (%s) — returning raw glosses", exc)
            return raw, False


# ════════════════════════════════════════════════════════════════════════════════
# Legacy sync provider factories  (kept for SLM_PROVIDER=openai)
# ════════════════════════════════════════════════════════════════════════════════

def _build_openai_fn(
    api_key:  str,
    model:    str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
) -> Callable:
    """Sync callable for any OpenAI-compatible chat endpoint."""
    import urllib.request as _req
    import json as _json

    def _call(prompt: str) -> str:
        payload = _json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0.1,
        }).encode()
        req = _req.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with _req.urlopen(req, timeout=8) as r:
            data = _json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()

    return _call


# ════════════════════════════════════════════════════════════════════════════════
# Confidence gate helpers
# ════════════════════════════════════════════════════════════════════════════════

def _mean_confidence(log_probs: torch.Tensor) -> float:
    """Mean top-token probability across all timesteps — proxy for certainty."""
    probs     = log_probs.exp()
    top_probs = probs.max(dim=-1).values
    return float(top_probs.mean().item())


def refine_glosses(
    gloss_list: list[str],
    slm_fn:     Optional[Callable[[str], str]],
    confidence: float = 1.0,
) -> tuple[str, bool]:
    """
    Sync fallback refiner used when SLM_PROVIDER=openai.

    For Ollama, use ``OllamaRefinementBridge.refine()`` directly (async).
    """
    clean = [g for g in gloss_list if g not in ("<blank>", "<unk>")]
    raw   = " ".join(clean)

    if not clean or slm_fn is None:
        return raw, False

    if confidence < CONFIDENCE_GATE:
        log.debug(
            "Confidence gate: %.3f < %.3f — skipping SLM",
            confidence, CONFIDENCE_GATE,
        )
        return raw, False

    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"{_USER_TEMPLATE.format(glosses=raw)}"
    )
    try:
        caption = slm_fn(prompt)
        return caption, True
    except Exception as exc:
        log.warning("SLM call failed (%s) — returning raw glosses", exc)
        return raw, False


# ════════════════════════════════════════════════════════════════════════════════
# Inference Engine
# ════════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """
    Async inference engine: Bi-LSTM → CTC decode → Ollama refinement.

    The Bi-LSTM forward pass runs in a ``ThreadPoolExecutor`` (blocking-safe).
    The Ollama call is fully async and runs directly on the event loop with a
    hard 200 ms timeout — it never blocks frame ingestion.

    Parameters
    ----------
    ckpt_path   : path to .pt checkpoint
    vocab_path  : path to vocab.json
    device      : "cuda" | "mps" | "cpu"
    beam_width  : CTC prefix beam search width
    slm_fn      : optional sync SLM callable (used for openai provider)
    max_workers : thread pool size
    """

    def __init__(
        self,
        ckpt_path:   Path,
        vocab_path:  Path,
        device:      str   = "cpu",
        beam_width:  int   = 5,
        slm_fn:      Optional[Callable[[str], str]] = None,
        max_workers: int   = 1,
    ) -> None:
        log.info(
            "Loading inference engine | ckpt=%s vocab=%s device=%s",
            ckpt_path, vocab_path, device,
        )

        provider = os.environ.get("SLM_PROVIDER", "none").lower().strip()

        # ── Ollama bridge (async, preferred) ──────────────────────────────────
        if provider == "ollama":
            self._ollama = OllamaRefinementBridge(
                model      = os.environ.get("OLLAMA_MODEL",    "phi3"),
                host       = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                timeout_ms = float(os.environ.get("OLLAMA_TIMEOUT_MS", "200")),
            )
            resolved_slm = None   # Ollama runs async — not passed to predictor
        else:
            self._ollama = None

        # ── OpenAI-compatible sync fallback ───────────────────────────────────
        if provider == "openai" and slm_fn is None:
            api_key  = os.environ.get("OPENAI_API_KEY", "")
            model    = os.environ.get("OPENAI_MODEL",    "gpt-4o-mini")
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            if api_key:
                resolved_slm = _build_openai_fn(api_key, model, base_url)
                log.info("SLM: OpenAI-compatible | model=%s", model)
            else:
                log.warning("SLM: OPENAI_API_KEY not set — passthrough")
                resolved_slm = None
        elif provider not in ("ollama", "openai"):
            resolved_slm = slm_fn   # explicit override or None
            if provider not in ("", "none"):
                log.warning("SLM: unknown provider %r — passthrough", provider)

        # CSLRPredictor handles the sync slm_fn path (openai / explicit)
        self._predictor = CSLRPredictor.from_checkpoint(
            ckpt_path  = ckpt_path,
            vocab_path = vocab_path,
            device     = device,
            beam_width = beam_width,
            slm_fn     = resolved_slm,
        )

        self._executor   = ThreadPoolExecutor(max_workers=max_workers)
        self._slm_active = (self._ollama is not None) or (resolved_slm is not None)

        log.info(
            "Inference engine ready | provider=%s slm=%s",
            provider,
            "ollama-async" if self._ollama else
            "openai-sync"  if resolved_slm else
            "passthrough",
        )

    @property
    def slm_active(self) -> bool:
        return self._slm_active

    async def predict_window(self, frames: np.ndarray) -> dict:
        """
        Full async pipeline: Bi-LSTM → CTC → (optional) Ollama refinement.

        1. Run the Bi-LSTM forward pass in the thread pool (non-blocking).
        2. If Ollama is configured AND confidence ≥ gate:
               await OllamaRefinementBridge.refine() with 200 ms timeout.
           Otherwise use the caption already produced by CSLRPredictor
           (which may have called the sync slm_fn internally).

        Returns
        -------
        {
            "glosses":    ["I", "GIVE", "BANK"],
            "caption":    "I have provided the bank details.",
            "latency_ms": 87.4,
            "slm_used":   true,
            "confidence": 0.91
        }
        """
        loop = asyncio.get_running_loop()

        # ── Step 1: Bi-LSTM inference (blocking → thread pool) ────────────────
        t0     = time.perf_counter()
        result = await loop.run_in_executor(
            self._executor,
            self._predictor.predict_frames,
            frames,
        )
        lstm_ms = (time.perf_counter() - t0) * 1000

        glosses    = result["glosses"]
        caption    = result["caption"]   # raw glosses or sync-SLM output
        confidence = result.get("confidence", 1.0)

        # ── Step 2: Ollama async refinement (if configured) ───────────────────
        slm_used = False
        if self._ollama is not None:
            clean = [g for g in glosses if g not in ("<blank>", "<unk>")]

            if clean and confidence >= CONFIDENCE_GATE:
                caption, slm_used = await self._ollama.refine(clean)
            else:
                caption = " ".join(clean) if clean else caption
                if clean and confidence < CONFIDENCE_GATE:
                    log.debug(
                        "Confidence gate: %.3f < %.3f — Ollama skipped",
                        confidence, CONFIDENCE_GATE,
                    )

        total_ms = (time.perf_counter() - t0) * 1000

        log.debug(
            "predict_window | glosses=%s caption=%r "
            "conf=%.2f lstm=%.1fms total=%.1fms slm=%s",
            glosses, caption[:60], confidence, lstm_ms, total_ms, slm_used,
        )

        return {
            "glosses":    glosses,
            "caption":    caption,
            "latency_ms": round(total_ms, 2),
            "slm_used":   slm_used,
            "confidence": round(confidence, 4),
        }

    async def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
        log.info("Inference engine shut down")
