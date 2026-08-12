"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Iterator

from fastapi import Depends
from sqlalchemy.orm import Session

from neurotribe.config import Settings, get_settings as _get_settings
from neurotribe.database.base import get_sessionmaker


def get_db() -> Iterator[Session]:
    """Request-scoped database session."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_settings() -> Settings:
    return _get_settings()


SettingsDep = Depends(get_settings)
DbDep = Depends(get_db)
