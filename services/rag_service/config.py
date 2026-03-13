from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_hostname: str
    database_port: str
    database_password: str
    database_name: str
    database_username: str

    base_url: str
    api_key: str
    model_name: str

    document_service_url: str = "http://document-service:8000"
    redis_url: str = "redis://redis:6379/0"

    retrieval_top_k: int = 5
    rerank_multiplier: int = 3
    max_history_messages: int = 20
    history_ttl_seconds: int = 86400

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
