"""FastAPI job backend for the research agent, backed by the database.

A single research run takes 10-50s and many Gemini calls, so the agent must
not run inside a request handler. A FIFO worker thread runs jobs one at a
time - bursts can't trip the free tier's rate limits - while the API answers
instantly with a job id and serves progress snapshots as the run happens.

Everything about a job (reports, sources, steps, status) lives in the
database, so the UI shows past reports even after the server restarts.
"""

import queue
import threading
import time
import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app import db
from app.agent.loop import AgentResult, StepRecord, run_agent
from app.db import Report, SourceRow, StepRow, loads
from app.llm import MODEL
from app.log import get_logger
from app.report import check_citations, render_report

logger = get_logger(__name__)

JOB_LIST_LIMIT = 50


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    db.mark_interrupted_runs()
    yield


app = FastAPI(title="Research Agent API", lifespan=lifespan)

_queue: queue.Queue[str] = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()


class ResearchRequest(BaseModel):
    question: str
    max_steps: int = Field(default=8, ge=1, le=20)
    max_seconds: int = Field(default=120, ge=5, le=600)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


def _step_event(s: StepRow) -> dict:
    return {
        "step": s.step,
        "action": s.action,
        "ok": s.ok,
        "duration_ms": s.duration_ms,
        "args": loads(s.args_json) or {},
        "error": s.error,
    }


def _source_dict(s: SourceRow) -> dict:
    return {"n": s.n, "url": s.url, "title": s.title, "step": s.step}


def _result_payload(r: Report, sources: list[SourceRow]) -> dict:
    """Final answer plus everything the UI renders: report, sources, checks."""
    result = AgentResult(
        answer=r.answer,
        sources=[_source_dict(s) for s in sources],
        steps=r.agent_steps or 0,
        reached_limit=r.reached_limit,
    )
    return {
        "answer": result.answer,
        "steps": result.steps,
        "reached_limit": result.reached_limit,
        "sources": result.sources,
        "citation_checks": check_citations(result),
        "report_md": render_report(r.question, result),
    }


def _job_payload(r: Report, sources: list[SourceRow], steps: list[StepRow]) -> dict:
    payload = {
        "job_id": r.job_id,
        "status": r.status,
        "question": r.question,
        "max_steps": r.max_steps,
        "max_seconds": r.max_seconds,
        "created_at": r.created_at,
        "started_at": r.started_at,
        "finished_at": r.finished_at,
        "steps": len(steps),
        "events": [_step_event(s) for s in steps],
        "error": r.error,
    }
    if r.status == "done" and r.answer is not None:
        payload["result"] = _result_payload(r, sources)
    return payload


def _summary(r: Report) -> dict:
    return {
        "job_id": r.job_id,
        "status": r.status,
        "question": r.question,
        "steps": len(r.steps),
        "created_at": r.created_at,
        "finished_at": r.finished_at,
        "error": r.error,
    }


def _fetch(job_id: str):
    """(report, sources, steps) rows for a job, or None."""
    with db.SessionLocal() as s:
        r = s.get(Report, job_id)
        if r is None:
            return None
        sources = [x for x in r.sources]
        steps = [x for x in r.steps]
    sources.sort(key=lambda x: x.n)
    steps.sort(key=lambda x: x.id)
    return r, sources, steps


def _ensure_worker() -> None:
    global _worker
    if _worker is not None and _worker.is_alive():
        return
    with _worker_lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_worker_loop, daemon=True)
            _worker.start()


def _worker_loop() -> None:
    """Run queued jobs forever. Errors turn into a job-level failure."""
    while True:
        job_id = _queue.get()
        with db.SessionLocal() as s:
            r = s.get(Report, job_id)
            if r is None:
                _queue.task_done()
                continue
            r.status = "running"
            r.started_at = time.time()
            s.commit()
            question, max_steps, max_seconds = r.question, r.max_steps, r.max_seconds

        def on_step(record: StepRecord) -> None:
            with db.SessionLocal() as s:
                s.add(
                    StepRow(
                        report_id=job_id,
                        step=record.step,
                        action=record.action,
                        args_json=db.dumps(record.args),
                        ok=record.ok,
                        duration_ms=record.duration_ms,
                        error=record.error,
                    )
                )
                s.commit()

        try:
            result = run_agent(
                question,
                max_steps=max_steps,
                max_seconds=max_seconds,
                on_step=on_step,
            )
            with db.SessionLocal() as s:
                r = s.get(Report, job_id)
                r.status = "done"
                r.answer = result.answer
                r.agent_steps = result.steps
                r.reached_limit = result.reached_limit
                r.finished_at = time.time()
                for src in result.sources:
                    s.add(
                        SourceRow(
                            report_id=job_id,
                            n=src["n"],
                            url=src["url"],
                            title=src.get("title"),
                            step=src.get("step", 0),
                        )
                    )
                s.commit()
        except Exception as e:  # e.g. quota exhausted mid-run
            logger.error("job %s failed: %s", job_id, e)
            with db.SessionLocal() as s:
                r = s.get(Report, job_id)
                r.status = "error"
                r.error = f"{type(e).__name__}: {e}"
                r.finished_at = time.time()
                s.commit()
        finally:
            _queue.task_done()


@app.get("/api/health")
def health() -> dict:
    with db.SessionLocal() as s:
        total = s.query(Report).count()
        running = (
            s.query(Report).filter(Report.status == "running").count()
        )
    return {
        "ok": True,
        "model": MODEL,
        "total_reports": total,
        "queue_size": _queue.qsize(),
        "running": running,
    }


@app.post("/api/research", status_code=201)
def research(req: ResearchRequest) -> dict:
    job_id = uuid.uuid4().hex[:12]
    with db.SessionLocal() as s:
        s.add(
            Report(
                job_id=job_id,
                question=req.question,
                status="queued",
                max_steps=req.max_steps,
                max_seconds=req.max_seconds,
                created_at=time.time(),
            )
        )
        s.commit()
    _queue.put(job_id)
    _ensure_worker()
    logger.info("job %s queued: %s", job_id, req.question)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> dict:
    row = _fetch(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    r, sources, steps = row
    return _job_payload(r, sources, steps)


@app.get("/api/jobs")
def job_list() -> dict:
    with db.SessionLocal() as s:
        rows = (
            s.query(Report)
            .options(db.selectinload(Report.steps))
            .order_by(Report.created_at.desc())
            .limit(JOB_LIST_LIMIT)
            .all()
        )
        items = [_summary(r) for r in rows]
    return {"jobs": items}


@app.get("/")
def root() -> dict:
    return {
        "name": "Research Agent API",
        "docs": "/docs",
        "submit": {"post": "/api/research", "body": {"question": "..."}},
    }