"""Database connection and session factory using SQLAlchemy."""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import settings

# SQLite connection requires connect_args for check_same_thread
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=settings.debug
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency that yields a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_sqlite_columns():
    """Lightweight migration helper to ensure columns exist in SQLite."""
    if not settings.database_url.startswith("sqlite"):
        return
    with engine.connect() as conn:
        for table, col, col_type in [
            ("assignments", "classroom_id", "VARCHAR(36)"),
            ("assignments", "classroom_name", "VARCHAR(256)"),
            ("coding_sessions", "classroom_id", "VARCHAR(36)"),
            ("coding_sessions", "classroom_name", "VARCHAR(256)"),
            ("coding_sessions", "teacher_id", "VARCHAR(64)"),
            ("coding_sessions", "teacher_name", "VARCHAR(128)"),
            ("comprehension_reports", "needs_instructor_review", "BOOLEAN DEFAULT 0"),
            ("comprehension_reports", "multimodal_scores", "TEXT"),
            ("comprehension_reports", "recommended_interventions", "TEXT"),
        ]:
            try:
                res = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
                cols = [r[1] for r in res]
                if cols and col not in cols:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
                    conn.commit()
            except Exception:
                pass


def init_db():
    """Initialize database tables and run lightweight migrations."""
    # Import all models to register them on Base.metadata
    import app.models  # noqa
    Base.metadata.create_all(bind=engine)
    ensure_sqlite_columns()


# Initialize upon module load for immediate readiness in tests and dev servers
init_db()
