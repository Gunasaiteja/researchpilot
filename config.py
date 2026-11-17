from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gcp_project: str = ""
    gcp_location: str = "us-central1"
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-005"
    embedding_dim: int = 768

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = "./qdrant_data"
    collection: str = "researchpilot"

    api_key: str = "dev-key"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    max_retries: int = 2

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
