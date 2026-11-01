"""API router registration."""
from fastapi import APIRouter
from app.api.sessions import router as sessions_router
from app.api.submissions import router as submissions_router
from app.api.telemetry import router as telemetry_router
from app.api.viva import router as viva_router
from app.api.reports import router as reports_router
from app.api.instructor import router as instructor_router
from app.api.code import router as code_router
from app.api.assignments import router as assignments_router
from app.api.classrooms import router as classrooms_router
from app.api.ml import router as ml_router

api_router = APIRouter(prefix="/api")
api_router.include_router(classrooms_router)
api_router.include_router(sessions_router)
api_router.include_router(submissions_router)
api_router.include_router(telemetry_router)
api_router.include_router(viva_router)
api_router.include_router(reports_router)
api_router.include_router(instructor_router)
api_router.include_router(code_router)
api_router.include_router(assignments_router)
api_router.include_router(ml_router)

__all__ = ["api_router"]
