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

    # Configurable Multimodal Evaluation Weights:
    # S_overall = 0.15 * S_ast + 0.20 * S_exec + 0.45 * S_viva + 0.20 * S_process
    weight_ast: float = Field(default_factory=lambda: float(os.getenv("WEIGHT_AST", "0.15")))
    weight_exec: float = Field(default_factory=lambda: float(os.getenv("WEIGHT_EXEC", "0.20")))
    weight_viva: float = Field(default_factory=lambda: float(os.getenv("WEIGHT_VIVA", "0.45")))
    weight_process: float = Field(default_factory=lambda: float(os.getenv("WEIGHT_PROCESS", "0.20")))

    # Configurable Process Feature Subweights:
    # S_process = 0.30 * S_incremental + 0.30 * S_debugging + 0.25 * S_revision + 0.15 * S_testing
    weight_proc_incremental: float = Field(default_factory=lambda: float(os.getenv("WEIGHT_PROC_INCREMENTAL", "0.30")))
    weight_proc_debugging: float = Field(default_factory=lambda: float(os.getenv("WEIGHT_PROC_DEBUGGING", "0.30")))
    weight_proc_revision: float = Field(default_factory=lambda: float(os.getenv("WEIGHT_PROC_REVISION", "0.25")))
    weight_proc_testing: float = Field(default_factory=lambda: float(os.getenv("WEIGHT_PROC_TESTING", "0.15")))


settings = Settings()
