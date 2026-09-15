"""
Application configuration.
Loads settings from environment variables with sensible defaults.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database (macOS Homebrew PostgreSQL uses current user without password)
    database_url: str = Field(
        default="postgresql://student@localhost:5432/rl_math",
        description="PostgreSQL connection string"
    )

    # API
    api_title: str = "RL-Math Problem Bank API"
    api_version: str = "0.1.0"
    api_prefix: str = "/api/v1"

    # Vector settings
    vector_dimension: int = Field(
        default=58,
        description="Dimension of concept vectors"
    )

    # Pagination
    default_page_size: int = 50
    max_page_size: int = 500

    # Student LLM simulation
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for student answer generation"
    )
    student_llm_enabled: bool = Field(
        default=False,
        description="Enable GPT-5 nano for student answer generation"
    )
    student_llm_model: str = Field(
        default="gpt-5-nano",
        description="Model name for student answer generation"
    )

    class Config:
        env_file = ".env"
        env_prefix = "RLMATH_"
        case_sensitive = False
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
