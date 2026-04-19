from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"
    app_name: str = "Agent Fabric"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    postgres_url: str = "postgresql+psycopg://agentfabric:agentfabric@localhost:5432/agentfabric"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"
    storage_root: str = "./storage"
    log_level: str = "INFO"


settings = Settings()
