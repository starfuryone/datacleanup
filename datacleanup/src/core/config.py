from __future__ import annotations
import os

class Settings:
    ENV: str = os.getenv("ENV", "production")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    # API
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()
