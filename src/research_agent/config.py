"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    use_llm: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    max_subtasks: int = 5
    max_refine_loops: int = 2
    quality_threshold: float = 0.72
    memory_capacity: int = 24
    memory_prune_ratio: float = 0.45
    ranking_recency_bias: float = 0.15
    demo_latency_ms: int = 12


@lru_cache
def get_settings() -> Settings:
    return Settings()
