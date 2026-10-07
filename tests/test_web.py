"""Offline checks for Phase 8/9: the FastAPI job backend and persistence - no API calls."""
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

_TMP = Path(tempfile.mkdtemp(prefix="research_web_"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"  # before app imports

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from fastapi.testclient import TestClient

import app.backend as backend
import app.db as db
from app.agent.loop import AgentResult, StepRecord
from app.db import Report, SourceRow, StepRow
from app.report import render_report

passed = 0
job_id_life = None


def ok(name):
    global passed
    passed += 1
    print("PASS", name)


def fail(name, err):
    print("FAIL", name, "->", err)
    raise SystemExit(1)


def wait_done(client, job_id, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in ("done", "error"):
            return payload
        time.sleep(0.05)
    raise AssertionError("timed out waiting for job " + job_id)


def reset_db():
    with db.SessionLocal() as s:
        s.query(StepRow).delete()
        s.query(SourceRow).delete()
        s.query(Report).delete()
        s.commit()


ANSWER = "Connection pooling reuses connections [1]. This speeds up apps [9]."
RESULT = AgentResult(
    answer=ANSWER,
    sources=[
        {"n": 1, "url": "https://example.com/one", "title": "Source One", "step": 2},
        {"n": 2, "url": "https://example.com/two", "title": "Source Two", "step": 2},
    ],
    steps=2,
    trace=[
        StepRecord(step=1, action="search", duration_ms=12),
        StepRecord(step=2, action="answer", result_chars=len(ANSWER)),
    ],
)


def fake_agent(question, max_steps=8, max_seconds=120, on_step=None):
    if on_step is not None:
        for t in RESULT.trace:
            on_step(t)
    return RESULT


def fifo_agent(question, max_steps=8, max_seconds=120, on_step=None):
    fifo_agent.starts.append(question[:12])
    time.sleep(0.15)
    if on_step is not None:
        on_step(StepRecord(step=1, action="answer"))
    return AgentResult(answer="done", steps=1, trace=[StepRecord(step=1, action="answer")])


fifo_agent.starts = []

# ---- first "server lifetime" ------------------------------------------------
with TestClient(backend.app) as client1:
    reset_db()

    h = client1.get("/api/health").json()
    ok("health ok") if h["ok"] and h["model"] else fail("health ok", h)
    r = client1.get("/").json()
    ok("root has docs link") if "/docs" in r["docs"] else fail("root", r)

    for bad in ("", "   "):
        resp = client1.post("/api/research", json={"question": bad})
        ok("blank question -> 422") if resp.status_code == 422 else fail(
            "blank question -> 422", resp.text
        )

    with patch.object(backend, "run_agent", side_effect=fake_agent):
        created = client1.post(
            "/api/research", json={"question": "What is pooling?", "max_steps": 5}
        )
        ok("create -> 201 + job_id") if created.status_code == 201 and created.json().get("job_id") else fail(
            "create", created.text
        )
        job_id_life = created.json()["job_id"]
        pay = wait_done(client1, job_id_life)
        ok("lifecycle -> done") if pay["status"] == "done" else fail("done", pay)
        ok("events in order search,answer") if [e["action"] for e in pay["events"]] == ["search", "answer"] else fail(
            "events", pay["events"]
        )
        ok("payload max_steps honored") if pay["max_steps"] == 5 else fail("max_steps", pay)
        res = pay["result"]
        ok("report_md == render_report") if res["report_md"] == render_report("What is pooling?", RESULT) else fail(
            "report_md", res["report_md"]
        )
        ok("citation_checks orb/unc") if res["citation_checks"]["orphans"] == [9] and res["citation_checks"]["uncited"] == [2] else fail(
            "citation_checks", res["citation_checks"]
        )
        ok("sources serialized") if len(res["sources"]) == 2 else fail("sources", res["sources"])

    with patch.object(backend, "run_agent", side_effect=fifo_agent):
        fifo_agent.starts = []
        a = client1.post("/api/research", json={"question": "AAAA first run"}).json()["job_id"]
        b = client1.post("/api/research", json={"question": "BBBB second run"}).json()["job_id"]
        wait_done(client1, a)
        wait_done(client1, b)
        ok("FIFO: A ran before B") if fifo_agent.starts == ["AAAA first r", "BBBB second "] else fail(
            "FIFO order", fifo_agent.starts
        )

    resp = client1.get("/api/jobs/nope")
    ok("unknown job -> 404") if resp.status_code == 404 else fail("404", resp.text)

    listing = client1.get("/api/jobs").json()["jobs"]
    seen = {j["job_id"] for j in listing}
    ok("job list has all runs") if {a, b, job_id_life} <= seen else fail(
        "job list", list(seen)
    )

# ---- server stopped; data must survive -------------------------------------
with db.SessionLocal() as s:
    count = s.query(Report).count()
    ok("3 reports persisted") if count == 3 else fail("reports persisted", count)
    life = s.get(Report, job_id_life)
    ok("sources persisted") if len(life.sources) == 2 else fail("sources persisted", len(life.sources))
    ok("steps persisted") if len(life.steps) == 2 else fail("steps persisted", len(life.steps))
    ok("done status persisted") if life.status == "done" else fail("status", life.status)

# a run that was live when the server died
with db.SessionLocal() as s:
    s.add(
        Report(
            job_id="crashedjob",
            question="was running when server died",
            status="running",
            max_steps=8,
            max_seconds=120,
            created_at=time.time(),
        )
    )
    s.commit()

# ---- second "server lifetime" = restart -------------------------------------
with TestClient(backend.app) as client2:
    listing = client2.get("/api/jobs").json()["jobs"]
    ok("old reports appear after restart") if any(j["job_id"] == job_id_life for j in listing) else fail(
        "reports appear after restart", [j["job_id"] for j in listing]
    )
    again = client2.get(f"/api/jobs/{job_id_life}").json()
    ok("restart payload intact") if again["status"] == "done" and again["result"]["report_md"] == render_report("What is pooling?", RESULT) and ["search", "answer"] == [e["action"] for e in again["events"]] else fail(
        "restart payload", again["status"]
    )
    crashed = client2.get("/api/jobs/crashedjob").json()
    ok("interrupted run marked error") if crashed["status"] == "error" and "restart" in (crashed["error"] or "") else fail(
        "mark interrupted", crashed
    )
    h2 = client2.get("/api/health").json()
    ok("health after restart counts 4") if h2["ok"] and h2["total_reports"] == 4 else fail("health counts", h2)

print(f"{passed} checks passed")