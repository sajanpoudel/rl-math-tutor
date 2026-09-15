"""
Database connection and session management.
Uses SQLAlchemy with pgvector extension for vector similarity search.

Best practices:
- Connection pooling with proper limits
- Pool pre-ping for stale connection detection
- Explicit transaction isolation
- Proper cleanup on errors
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

from .config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Engine configuration with production-ready settings
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_timeout=30,
    echo=False,
)


# Set session parameters on checkout
@event.listens_for(engine, "connect")
def set_pg_session_params(dbapi_connection, connection_record):
    """Configure PostgreSQL session parameters for optimal performance."""
    cursor = dbapi_connection.cursor()
    # Set search path for security
    cursor.execute("SET search_path TO public")
    # Enable parallel query execution
    cursor.execute("SET max_parallel_workers_per_gather = 2")
    cursor.close()


SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI endpoints to get database session.
    Handles commit/rollback automatically.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database session outside of FastAPI."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database: create pgvector extension and tables.
    Idempotent - safe to call multiple times.
    """
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")


def create_hnsw_index() -> None:
    """
    Create HNSW index for vector similarity search.
    Should be called after bulk data load for optimal index quality.
    """
    with engine.connect() as conn:
        # Drop existing index if any
        conn.execute(text("""
            DROP INDEX IF EXISTS idx_problems_concept_vector_hnsw
        """))

        # Create HNSW index with optimized parameters
        conn.execute(text("""
            CREATE INDEX idx_problems_concept_vector_hnsw
            ON problems
            USING hnsw (concept_vector vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
        conn.commit()
    logger.info("HNSW index created successfully")


def check_db_connection() -> bool:
    """Verify database connection is working."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def get_table_stats() -> dict:
    """Get database table statistics."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                schemaname,
                relname as table_name,
                n_live_tup as row_count,
                n_dead_tup as dead_rows,
                last_vacuum,
                last_autovacuum
            FROM pg_stat_user_tables
            WHERE relname = 'problems'
        """))
        row = result.fetchone()
        if row:
            return {
                "table_name": row[1],
                "row_count": row[2],
                "dead_rows": row[3],
                "last_vacuum": str(row[4]) if row[4] else None,
                "last_autovacuum": str(row[5]) if row[5] else None,
            }
        return {}
