"""
api/config.py
─────────────
Runtime configuration loaded from environment variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Model ─────────────────────────────────────────────────────────────────
    model_ckpt_path: Path = Field(
        Path("checkpoints/epoch_099_loss1.6763_acc0.912.pt"),
        env="MODEL_CKPT_PATH",
    )
    vocab_path: Path = Field(Path("vocab.json"), env="VOCAB_PATH")
    device: str = Field(
        default_factory=lambda: (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        ),
        env="DEVICE",
    )

    # ── Sliding window ────────────────────────────────────────────────────────
    window_size:     int = Field(60,  env="WINDOW_SIZE")
    window_stride:   int = Field(15,  env="WINDOW_STRIDE")
    cooldown_frames: int = Field(30,  env="COOLDOWN_FRAMES")

    # ── Inference ─────────────────────────────────────────────────────────────
    beam_width:     int = Field(5, env="BEAM_WIDTH")
    slm_confidence: int = Field(2, env="SLM_CONFIDENCE")

    # ── Server ────────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # ── SLM ───────────────────────────────────────────────────────────────────
    slm_provider:     str   = Field("none",                       env="SLM_PROVIDER")
    ollama_model:     str   = Field("phi3",                       env="OLLAMA_MODEL")
    ollama_base_url:  str   = Field("http://localhost:11434",     env="OLLAMA_BASE_URL")
    ollama_timeout_ms: float = Field(200.0,                       env="OLLAMA_TIMEOUT_MS")
    openai_api_key:   str   = Field("",                           env="OPENAI_API_KEY")
    openai_model:     str   = Field("gpt-4o-mini",                env="OPENAI_MODEL")
    openai_base_url:  str   = Field("https://api.openai.com/v1",  env="OPENAI_BASE_URL")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
