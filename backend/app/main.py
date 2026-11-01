"""FastAPI application entrypoint for DeepMisconception."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import Base, engine, ensure_sqlite_columns
# Ensure all models are imported so Base.metadata knows about them
import app.models  # noqa
from app.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: create DB tables on startup."""
    Base.metadata.create_all(bind=engine)
    ensure_sqlite_columns()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Backend API for AI-assisted code comprehension verification and student misconception detection.",
    lifespan=lifespan
)

# Enable CORS for local Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)


@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify backend status."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "sandbox_enabled": settings.safe_sandbox_enabled,
        "llm_provider": settings.llm_provider
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
