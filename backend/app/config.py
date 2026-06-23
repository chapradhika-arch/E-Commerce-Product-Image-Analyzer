"""Application configuration loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # .../backend
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_*` env vars belong to us, not pydantic's protected namespace.
        protected_namespaces=(),
    )

    mock_mode: bool = False

    caption_model: str = "Salesforce/blip-image-captioning-large"
    text_model: str = "google/flan-t5-base"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    device: str = ""  # "" => auto-detect
    rag_top_k: int = 3

    cors_origins: str = "http://localhost:4200,http://127.0.0.1:4200"

    knowledge_base_path: Path = DATA_DIR / "knowledge_base.json"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def resolve_device(self) -> str:
        """Pick a torch device, falling back to cpu when torch is absent."""
        if self.device:
            return self.device
        try:
            import torch  # noqa: WPS433 (local import on purpose)

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()
