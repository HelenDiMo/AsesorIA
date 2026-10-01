"""Environment settings and YAML config loading."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "anthropic"
    llm_model: str = ""
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 60
    ollama_base_url: str = "http://localhost:11434"

    embedding_model: str = "intfloat/multilingual-e5-base"

    chroma_dir: str = str(PROJECT_ROOT / "chroma_db")
    chroma_collection: str = "tax_corpus"

    api_base_url: str = "http://localhost:8000"
    asesoria_mock_api: int = 0


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_config(name: str) -> dict:
    """Load config/<name>.yaml as a dict."""
    path = PROJECT_ROOT / "config" / f"{name}.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
