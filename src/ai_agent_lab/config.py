from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    llm_api_key: str = Field(..., min_length=1)
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    request_timeout: float = 30.0
    max_retries: int = 3
    log_level: str = "INFO"

@lru_cache
def get_settings() -> Settings:
    return Settings()