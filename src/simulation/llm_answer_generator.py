"""
LLM-backed student answer generation  -  dual-agent only.

Every student uses a DecisionAgent + SolverAgent pair backed by either:
  - GPT-5 Nano (OpenAI)  when USE_LOCAL is unset or false
  - A local Ollama model  when USE_LOCAL=true

Set USE_LOCAL=true in .env to use Ollama instead of OpenAI.
OPENAI_API_KEY is only required when USE_LOCAL is not true.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Load .env from project root (walk up from this file to find it)
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")

logger = logging.getLogger(__name__)


@dataclass
class GPT5NanoConfig:
    model: str = "gpt-5-nano"
    reasoning_effort: str = "minimal"
    verbosity: str = "low"
    max_output_tokens: int = 1600
    timeout: float = 45.0
    max_retries: int = 3


@dataclass
class OllamaConfig:
    model: str = "qwen3.5:9b"
    base_url: str = "http://localhost:11434/v1"
    max_output_tokens: int = 6000
    timeout: float = 120.0
    max_retries: int = 3


def _use_local() -> bool:
    """Return True if USE_LOCAL env var is set to a truthy value."""
    return os.getenv("USE_LOCAL", "").strip().lower() in {"1", "true", "yes"}


def build_dual_agents_from_env(
    student_type: str = "average",
) -> "tuple[Any, Any]":
    """
    Create a (DecisionAgent, SolverAgent) pair from environment variables.

    When USE_LOCAL=true, uses a local Ollama model (no OpenAI key required).
    Otherwise falls back to OpenAI GPT-5 Nano (OPENAI_API_KEY required).

    Optional env vars (Ollama path):
    - USE_LOCAL                         -  set to true to use Ollama
    - OLLAMA_BASE_URL                   -  default http://localhost:11434/v1
    - OLLAMA_MODEL                      -  default qwen3.5:9b
    - RLMATH_STUDENT_LLM_MAX_OUTPUT_TOKENS
    - RLMATH_STUDENT_LLM_TIMEOUT
    - RLMATH_STUDENT_LLM_MAX_RETRIES

    Optional env vars (OpenAI path):
    - OPENAI_API_KEY                    -  required when USE_LOCAL is not true
    - RLMATH_STUDENT_LLM_MODEL
    - RLMATH_STUDENT_LLM_REASONING_EFFORT
    - RLMATH_STUDENT_LLM_VERBOSITY
    - RLMATH_STUDENT_LLM_MAX_OUTPUT_TOKENS
    - RLMATH_STUDENT_LLM_TIMEOUT
    - RLMATH_STUDENT_LLM_MAX_RETRIES
    """
    if _use_local():
        cfg = OllamaConfig(
            model=os.getenv("OLLAMA_MODEL", "qwen3.5:9b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            max_output_tokens=int(os.getenv("RLMATH_STUDENT_LLM_MAX_OUTPUT_TOKENS", "6000")),
            timeout=float(os.getenv("RLMATH_STUDENT_LLM_TIMEOUT", "120")),
            max_retries=max(1, int(os.getenv("RLMATH_STUDENT_LLM_MAX_RETRIES", "3"))),
        )
        logger.info(
            "[build_dual_agents_from_env] Using local Ollama model=%s  base_url=%s",
            cfg.model,
            cfg.base_url,
        )
        from .student_agents import OllamaDecisionAgent, OllamaSolverAgent

        return (
            OllamaDecisionAgent(student_type=student_type, config=cfg),
            OllamaSolverAgent(student_type=student_type, config=cfg),
        )

    # --- OpenAI path ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be set when USE_LOCAL is not true. "
            "Either set the key or add USE_LOCAL=true to your .env file."
        )

    cfg_oai = GPT5NanoConfig(
        model=os.getenv("RLMATH_STUDENT_LLM_MODEL", "gpt-5-nano"),
        reasoning_effort=os.getenv("RLMATH_STUDENT_LLM_REASONING_EFFORT", "minimal"),
        verbosity=os.getenv("RLMATH_STUDENT_LLM_VERBOSITY", "low"),
        max_output_tokens=int(os.getenv("RLMATH_STUDENT_LLM_MAX_OUTPUT_TOKENS", "1200")),
        timeout=float(os.getenv("RLMATH_STUDENT_LLM_TIMEOUT", "45")),
        max_retries=max(1, int(os.getenv("RLMATH_STUDENT_LLM_MAX_RETRIES", "3"))),
    )
    logger.info(
        "[build_dual_agents_from_env] Using OpenAI model=%s", cfg_oai.model
    )

    # Import here to avoid circular dependency at module level
    from .student_agents import DecisionAgent, SolverAgent

    return (
        DecisionAgent(api_key=api_key, student_type=student_type, config=cfg_oai),
        SolverAgent(api_key=api_key, student_type=student_type, config=cfg_oai),
    )
