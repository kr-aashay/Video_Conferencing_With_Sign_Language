"""
cslr_model/decoder.py
─────────────────────
CTC Prefix Beam Search decoder + SLM agentic post-processing hook.

CTCPrefixBeamDecoder
    Implements the full CTC prefix beam search algorithm (Graves 2012).
    Tracks p_blank and p_non_blank separately per prefix to correctly
    handle repeated tokens — a correctness gap in simpler greedy decoders.

refine_with_slm
    Agentic hook: transforms raw gloss sequences into fluent English.
    Plug in any SLM (Phi-3, Gemma, LLaMA) via a single callable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Optional

import torch
from torch import Tensor

from .dataset import Vocabulary

log = logging.getLogger(__name__)

# Sentinel for "negative infinity" in log-space
NEG_INF = float("-inf")


# ─────────────────────────────────────────────────────────────────────────────
# CTC Prefix Beam Search
# ─────────────────────────────────────────────────────────────────────────────

class CTCPrefixBeamDecoder:
    """
    CTC Prefix Beam Search (Graves et al., 2012).

    Maintains a beam of the most probable output prefixes, tracking
    p_blank (probability the last emitted token was blank) and
    p_non_blank (probability the last emitted token was non-blank)
    separately.  This correctly handles:
        - Repeated tokens separated by blanks  (A blank A → "AA")
        - Repeated tokens without blanks       (A A → "A")

    Parameters
    ----------
    vocab      : Vocabulary instance
    beam_width : number of prefixes to keep per timestep
    """

    def __init__(self, vocab: Vocabulary, beam_width: int = 10) -> None:
        self.vocab      = vocab
        self.beam_width = beam_width
        self.blank      = vocab.blank_idx

    # ── single-sequence decode ────────────────────────────────────────────────

    def decode(self, log_probs: Tensor) -> list[str]:
        """
        Parameters
        ----------
        log_probs : (T, vocab_size) — log-softmax output for ONE sequence

        Returns
        -------
        Predicted gloss list, e.g. ["I", "GIVE", "BANK"]
        """
        probs = log_probs.detach().exp().cpu()   # (T, V) — work in prob space
        T, V  = probs.shape

        # beam: prefix_tuple → [p_blank, p_non_blank]
        # Initialise with empty prefix
        beam: dict[tuple, list[float]] = {(): [1.0, 0.0]}

        for t in range(T):
            p_t = probs[t]                       # (V,) probabilities at step t
            new_beam: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])

            for prefix, (p_b, p_nb) in beam.items():
                p_total = p_b + p_nb

                # ── Case 1: extend with blank ──────────────────────────────
                new_beam[prefix][0] += p_total * p_t[self.blank].item()

                # ── Case 2: extend with each non-blank token ───────────────
                for c in range(V):
                    if c == self.blank:
                        continue

                    p_c = p_t[c].item()
                    new_prefix = prefix + (c,)

                    if prefix and prefix[-1] == c:
                        # Repeated token: only p_blank can extend without merging
                        new_beam[new_prefix][1] += p_b * p_c
                        # Merging case: p_nb stays on same prefix
                        new_beam[prefix][1]     += p_nb * p_c
                    else:
                        new_beam[new_prefix][1] += p_total * p_c

            # ── Prune to beam_width ────────────────────────────────────────
            beam = dict(
                sorted(
                    new_beam.items(),
                    key=lambda kv: kv[1][0] + kv[1][1],
                    reverse=True,
                )[: self.beam_width]
            )

        # Best prefix by total probability
        best = max(beam, key=lambda p: beam[p][0] + beam[p][1])
        return self.vocab.decode(list(best))

    # ── batch decode ──────────────────────────────────────────────────────────

    def decode_batch(self, log_probs: Tensor, lengths: Tensor) -> list[list[str]]:
        """
        Parameters
        ----------
        log_probs : (B, T, vocab_size)
        lengths   : (B,) actual frame counts

        Returns
        -------
        List of gloss sequences, one per sample.
        """
        return [
            self.decode(log_probs[i, : lengths[i]])
            for i in range(log_probs.size(0))
        ]


# Keep the old name as an alias for backward compatibility
CTCBeamDecoder = CTCPrefixBeamDecoder


# ─────────────────────────────────────────────────────────────────────────────
# SLM Agentic Post-Processing Hook
# ─────────────────────────────────────────────────────────────────────────────

def refine_with_slm(
    glosses: list[str],
    slm_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Agentic hook: convert raw CTC glosses into fluent English captions.

    Parameters
    ----------
    glosses : raw decoded gloss list, e.g. ["I", "GIVE", "BANK"]
    slm_fn  : callable(prompt: str) → str
              Receives a structured prompt and returns the generated caption.
              Pass None to bypass (returns space-joined glosses — useful for
              latency benchmarks and unit tests).

    Plug-in examples
    ────────────────
    ① Phi-3-mini via HuggingFace Transformers:

        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model="microsoft/Phi-3-mini-4k-instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        def phi3(prompt):
            out = pipe(prompt, max_new_tokens=64, do_sample=False)
            return out[0]["generated_text"].split("Caption:")[-1].strip()

        caption = refine_with_slm(glosses, slm_fn=phi3)

    ② Ollama local server (zero-dependency):

        import requests
        def ollama(prompt):
            r = requests.post("http://localhost:11434/api/generate",
                              json={"model": "phi3", "prompt": prompt, "stream": False})
            return r.json()["response"].strip()

        caption = refine_with_slm(glosses, slm_fn=ollama)

    Returns
    -------
    Natural English caption string.
    """
    gloss_str = " ".join(g for g in glosses if g not in ("<blank>", "<unk>"))

    if slm_fn is None:
        log.debug("SLM not configured — returning raw glosses.")
        return gloss_str

    prompt = (
        "You are a real-time sign language interpreter for video calls.\n"
        "Convert the following ISL sign glosses into a single, natural, "
        "grammatically correct English sentence. "
        "Output only the sentence — no explanation.\n\n"
        f"Glosses: {gloss_str}\n"
        "Caption:"
    )

    try:
        result = slm_fn(prompt)
        log.debug("SLM refined: %r → %r", gloss_str, result)
        return result
    except Exception as exc:
        log.warning("SLM refinement failed (%s) — returning raw glosses.", exc)
        return gloss_str
