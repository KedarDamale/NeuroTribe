"""SQLAlchemy engine / session plumbing.

Large scientific arrays (NIfTI, GIFTI, .npy) are NEVER stored in PostgreSQL.
The database holds paths, metadata, hashes and provenance only.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: Engine | None = None
_sessionmaker: sessionmaker[Session] | None = None


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if url:
        # SQLAlchemy 2 wants an explicit driver for postgres.
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url
    # Local fallback so the package is usable (and testable) without Docker.
    from neurotribe.config import get_settings

    root = get_settings().root
    (root / "data").mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(root / 'data' / 'neurotribe.db').as_posix()}"


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = database_url()
        if url.startswith("sqlite"):
            _engine = create_engine(
                url, future=True, poolclass=NullPool,
                connect_args={"check_same_thread": False, "timeout": 30},
            )

            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()
        else:
            _engine = create_engine(
                url, future=True, pool_pre_ping=True, pool_size=10, max_overflow=20,
                pool_recycle=1800,
            )
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False, future=True,
        )
    return _sessionmaker


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Drop cached engine/sessionmaker (used by tests switching databases)."""
    global _engine, _sessionmaker
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _sessionmaker = None


def create_all() -> None:
    """Create the schema directly (used for SQLite dev/test; prod uses Alembic)."""
    import neurotribe.database.models  # noqa: F401  (register models)

    Base.metadata.create_all(get_engine())
