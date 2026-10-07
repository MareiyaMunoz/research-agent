"""FastAPI job backend for the research agent.

A single research run takes 10-50s and many Gemini calls, so the agent must
not run inside a request handler. A FIFO worker thread runs jobs one at a
time - bursts can't trip the free tier's rate limits - while the API answers
instantly with a job id and serves progress snapshots as the run happens.
"""

import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.agent.loop import AgentResult, StepRecord, run_agent
from app.llm import MODEL
from app.log import get_logger
from app.report import check_citations, render_report

logger = get_logger(__name__)

app = FastAPI(title="Research Agent API")

MAX_JOBS_KEPT = 50  # trimmed so memory stays bounded

_jobs: dict[str, "Job"] = {}
_jobs_lock = threading.Lock()
_queue: queue.Queue[str] = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()


@dataclass
class Job:
    id: str
    question: str
    created_at: float
    status: str = "queued"  # queued | running | done | error
    max_steps: int = 8
    max_seconds: int = 120
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    trace: list[StepRecord] = field(default_factory=list)
    result: AgentResult | None = None


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


def _step_event(s: StepRecord) -> dict:
    return {
        "step": s.step,
        "action": s.action,
        "ok": s.ok,
        "duration_ms": s.duration_ms,
        "args": s.args,
        "error": s.error,
    }


def _result_payload(job: "Job") -> dict:
    """Final answer plus everything the UI renders: report, sources, checks."""
    r = job.result
    return {
        "answer": r.answer,
        "steps": r.steps,
        "reached_limit": r.reached_limit,
        "sources": list(r.sources),
        "citation_checks": check_citations(r),
        "report_md": render_report(job.question, r),
    }


def _job_payload(job: "Job") -> dict:
    payload = {
        "job_id": job.id,
        "status": job.status,
        "question": job.question,
        "max_steps": job.max_steps,
        "max_seconds": job.max_seconds,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "steps": len(job.trace),
        "events": [_step_event(t) for t in job.trace],
        "error": job.error,
    }
    if job.status == "done" and job.result is not None:
        payload["result"] = _result_payload(job)
    return payload


def _summary(job: "Job") -> dict:
    return {
        "job_id": job.id,
        "status": job.status,
        "question": job.question,
        "steps": len(job.trace),
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "error": job.error,
    }


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
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            _queue.task_done()
            continue
        job.status = "running"
        job.started_at = time.time()

        def on_step(record: StepRecord) -> None:
            with _jobs_lock:
                job.trace.append(record)

        try:
            result = run_agent(
                job.question,
                max_steps=job.max_steps,
                max_seconds=job.max_seconds,
                on_step=on_step,
            )
            with _jobs_lock:
                job.result = result
                job.status = "done"
        except Exception as e:  # e.g. quota exhausted mid-run
            logger.error("job %s failed: %s", job_id, e)
            with _jobs_lock:
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
        finally:
            job.finished_at = time.time()
            _queue.task_done()


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model": MODEL,
        "jobs_kept": len(_jobs),
        "queue_size": _queue.qsize(),
    }


@app.post("/api/research", status_code=201)
def research(req: ResearchRequest) -> dict:
    job = Job(
        id=uuid.uuid4().hex[:12],
        question=req.question,
        max_steps=req.max_steps,
        max_seconds=req.max_seconds,
        created_at=time.time(),
    )
    with _jobs_lock:
        _jobs[job.id] = job
        if len(_jobs) > MAX_JOBS_KEPT:
            # drop oldest finished jobs (a queued one must never be evicted)
            finished = [j for j in _jobs.values() if j.status in ("done", "error")]
            for old in finished[: len(_jobs) - MAX_JOBS_KEPT]:
                del _jobs[old.id]
    _queue.put(job.id)
    _ensure_worker()
    logger.info("job %s queued: %s", job.id, job.question)
    return {"job_id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    with _jobs_lock:
        return _job_payload(job)


@app.get("/api/jobs")
def job_list() -> dict:
    with _jobs_lock:
        items = [_summary(j) for j in _jobs.values()]
    items.sort(key=lambda s: s["created_at"], reverse=True)
    return {"jobs": items}


@app.get("/")
def root() -> dict:
    return {
        "name": "Research Agent API",
        "docs": "/docs",
        "submit": {"post": "/api/research", "body": {"question": "..."}},
    }