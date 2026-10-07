"""Database layer: reports, sources, and agent steps.

Storage is SQLAlchemy so it runs on SQLite out of the box and on real
PostgreSQL just by setting DATABASE_URL, e.g.
    postgresql+psycopg://user:pass@localhost:5432/research_agent
Phase 10's Docker Compose starts a Postgres container and uses that URL.
"""

import json
import os
import time

from sqlalchemy import Boolean, Float, ForeignKey, String, Text, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    selectinload,
    sessionmaker,
)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///research.db")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Report(Base):
    """One research run, in whatever state it is in (queued..done/error)."""

    __tablename__ = "reports"

    job_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(10))
    max_steps: Mapped[int] = mapped_column(default=8)
    max_seconds: Mapped[int] = mapped_column(default=120)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_steps: Mapped[int | None] = mapped_column(nullable=True)
    reached_limit: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[float] = mapped_column(Float)
    started_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    finished_at: Mapped[float | None] = mapped_column(Float, nullable=True)

    sources: Mapped[list["SourceRow"]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )
    steps: Mapped[list["StepRow"]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )


class SourceRow(Base):
    """A verified source read during a run; citation numbers come from `n`."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("reports.job_id"), index=True
    )
    n: Mapped[int]
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    step: Mapped[int]

    report: Mapped[Report] = relationship(back_populates="sources")


class StepRow(Base):
    """One trace event (tool call, guard stop, or the final answer)."""

    __tablename__ = "steps"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(
        String(20), ForeignKey("reports.job_id"), index=True
    )
    step: Mapped[int]
    action: Mapped[str] = mapped_column(String(30))
    args_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    duration_ms: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    report: Mapped[Report] = relationship(back_populates="steps")


def init_db() -> None:
    """Create tables if they don't exist yet."""
    Base.metadata.create_all(engine)


def mark_interrupted_runs() -> None:
    """After a restart, queued/running jobs could never finish - mark them."""
    with SessionLocal() as s:
        rows = s.query(Report).filter(Report.status.in_(("queued", "running"))).all()
        for r in rows:
            r.status = "error"
            r.error = "interrupted by server restart"
            r.finished_at = time.time()
        if rows:
            s.commit()


def dumps(value) -> str | None:
    return json.dumps(value, ensure_ascii=False) if value is not None else None


def loads(value: str | None):
    return json.loads(value) if value else None