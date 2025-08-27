from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    env: str = "dev"
    allowed_origins: List[str] = []
    max_upload_mb: int = 10
    storage_dir: Path = Path("./storage")

    # новый стиль конфигурации
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings()