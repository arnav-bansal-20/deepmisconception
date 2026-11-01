import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env if present
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
load_dotenv()



class Settings(BaseModel):
    """DeepMisconception application settings."""
    app_name: str = "DeepMisconception API"
    app_version: str = "1.0.0"
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1"))
    
    # Base paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    
    # Database
    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            f"sqlite:///{Path(__file__).resolve().parent.parent / 'deepmisconception.db'}"
        )
    )
    
    # LLM Settings
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "auto"))
    
    # Execution Sandbox Settings
    execution_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("EXECUTION_TIMEOUT_SECONDS", "3.0"))
    )
    execution_max_memory_mb: int = Field(
        default_factory=lambda: int(os.getenv("EXECUTION_MAX_MEMORY_MB", "128"))
    )
    safe_sandbox_enabled: bool = Field(
        default_factory=lambda: os.getenv("SAFE_SANDBOX_ENABLED", "true").lower() in ("true", "1")
    )


settings = Settings()
