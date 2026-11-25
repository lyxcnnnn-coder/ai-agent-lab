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

    # 总尝试次数：1 次首试 + 2 次重试 = 3 次调用
    max_attempts: int = 3

    # 重试退避：首次等 1 秒，之后翻倍，最多 4 秒
    # 测试里会覆盖成 0.001，避免测试真的等待
    retry_initial_wait: float = 1.0
    retry_max_wait: float = 4.0

    log_level: str = "INFO"

@lru_cache
def get_settings() -> Settings:
    return Settings()