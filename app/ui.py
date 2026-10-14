"""Research agent UI. Run with: streamlit run app/ui.py

Needs the API running too: python -m app.web  (http://127.0.0.1:8000).
"""

import os
import time

import httpx
import streamlit as st

API_BASE = os.getenv("STREAMLIT_API_BASE", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Research Agent",
    page_icon=":material/travel_explore:",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    .stApp { font-family: 'DM Sans', system-ui, sans-serif; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', 'DM Sans', sans-serif; letter-spacing: -0.01em; }
    a:focus-visible, button:focus-visible { outline: 2px solid #7C3AED; outline-offset: 2px; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

TIER_COLOR = {"high": "green", "medium": "orange", "low": "red"}


def format_event(t: dict) -> str:
    """One trace event -> one human-readable log line."""
    ms = t.get("duration_ms") or 0
    detail = ""
    if t["action"] == "read_page":
        detail = (t.get("args") or {}).get("url", "")
    elif t["action"] == "search":
        detail = ((t.get("args") or {}).get("query") or "")[:60]
    elif t["action"] == "limit":
        return f"step {t['step']} · stopped: {(t.get('error') or '').strip()}"
    if t["action"] == "answer":
        return f"step {t['step']} · final answer"
    if t.get("ok"):
        return f"step {t['step']} · {t['action']} {detail} — {ms}ms"
    return f"step {t['step']} · {t['action']} {detail} — failed ({t.get('error')})"


def render_result(result: dict) -> None:
    """The final answer: report, warnings, source list."""
    checks = result.get("citation_checks") or {}
    claims = result.get("claim_checks") or {}
    counts = claims.get("counts") or {}
    quality = result.get("quality_checks") or {}
    qcounts = quality.get("counts") or {}
    if checks.get("orphans"):
        st.warning(
            "Unverified citation(s) "
            + ", ".join(f"[{n}]" for n in checks["orphans"])
            + " are not in the source list."
        )
    if checks.get("uncited"):
        st.info(
            "Source(s) "
            + ", ".join(f"[{n}]" for n in checks["uncited"])
            + " were read but never cited."
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Verified claims", counts.get("supported", 0))
    c2.metric("Loosely matched", counts.get("weak", 0))
    c3.metric("Unsupported", counts.get("unsupported", 0))

    if qcounts:
        st.caption(
            f"Source quality — avg {quality.get('avg_score', 0)}/100 "
            f"(high {qcounts.get('high', 0)}, medium {qcounts.get('medium', 0)}, "
            f"low {qcounts.get('low', 0)})"
        )
    if result.get("reached_limit"):
        st.warning("The agent hit the step limit — this report may be incomplete.")

    st.markdown(result["report_md"])

    st.download_button(
        "Download report (.md)",
        result["report_md"],
        file_name="research-report.md",
        mime="text/markdown",
    )

    sources = result.get("sources") or []
    with st.expander(f"Sources ({len(sources)})"):
        if sources:
            for s in sources:
                q = s.get("quality") or {}
                tier = q.get("tier")
                color = TIER_COLOR.get(tier, "gray")
                badge = f" · :{color}[{tier} {q.get('score')}/100]" if q else ""
                st.markdown(f"**[{s['n']}]** [{s['title']}]({s['url']}){badge}")
        else:
            st.write("No sources were read.")


def show_job(payload: dict) -> None:
    st.divider()
    status = payload["status"]
    heading = {
        "done": f"Research complete · {payload['steps']} step(s)",
        "error": "Research failed",
    }.get(status, f"Still working · {payload['steps']} step(s)")
    st.subheader(heading)
    st.caption(f"job {payload['job_id']} · {status}")

    if payload["events"]:
        with st.expander("Trace", expanded=status != "done"):
            for e in payload["events"]:
                st.caption(format_event(e))

    if status == "done":
        render_result(payload["result"])
    elif status == "error":
        st.error(f"Job failed: {payload.get('error')}")
    else:
        st.info("Still working — refresh the page to see the latest progress.")


def submit(question: str, max_steps: int, max_seconds: int) -> dict:
    """Create the job and poll until it finishes, streaming live steps."""
    try:
        created = httpx.post(
            f"{API_BASE}/api/research",
            json={
                "question": question,
                "max_steps": max_steps,
                "max_seconds": max_seconds,
            },
            timeout=10,
        )
        created.raise_for_status()
    except httpx.ConnectError:
        st.error(f"Can't reach the API at {API_BASE}. Start it with: python -m app.web")
        return {}
    job_id = created.json()["job_id"]

    with st.status("Researching…", expanded=True) as status:
        live = st.empty()
        payload = None
        while True:
            resp = httpx.get(f"{API_BASE}/api/jobs/{job_id}", timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            lines = [format_event(e) for e in payload["events"]]
            if payload["status"] == "queued":
                lines.append("Waiting for a free run slot…")
            live.caption("\n".join(lines) or "Starting…")
            if payload["status"] in ("done", "error"):
                break
            time.sleep(1.0)
        if payload["status"] == "done":
            status.update(label=f"Done · {payload['steps']} step(s)", state="complete")
        else:
            status.update(label="Failed", state="error")
    return payload


st.title("Research Agent")
st.caption("Ask a question — an agent searches, reads pages, and writes a cited report.")


def sidebar() -> str | None:
    """Recent runs. Returns a job id to open, or None."""
    st.sidebar.header("Recent runs")
    try:
        resp = httpx.get(f"{API_BASE}/api/jobs", timeout=10)
        resp.raise_for_status()
    except httpx.ConnectError:
        st.sidebar.error(f"API offline at {API_BASE}")
        return None
    jobs = resp.json()["jobs"]
    if not jobs:
        st.sidebar.caption("No runs yet. Ask a question to start one.")
        return None
    st.sidebar.caption(f"{len(jobs)} run(s)")
    for s in jobs[:20]:
        label = f"{s['status']} · {s['question'][:38]}{'…' if len(s['question']) > 38 else ''}"
        if st.sidebar.button(label, key=s["job_id"], use_container_width=True):
            return s["job_id"]
    return None


def main() -> None:
    question = st.text_input(
        "Question", key="question", placeholder="e.g. How does connection pooling work?"
    )
    with st.expander("Advanced"):
        max_steps = st.slider("Max steps", 3, 12, 8)
        max_seconds = st.slider("Max seconds", 10, 600, 120, step=10)

    if st.button("Research", type="primary"):
        if not (question or "").strip():
            st.error("Write a question first.")
        else:
            payload = submit(question.strip(), int(max_steps), int(max_seconds))
            if payload:
                st.session_state["open_job"] = payload["job_id"]

    picked = sidebar()
    if picked:
        st.session_state["open_job"] = picked

    job_id = st.session_state.get("open_job")
    if job_id:
        try:
            resp = httpx.get(f"{API_BASE}/api/jobs/{job_id}", timeout=10)
            resp.raise_for_status()
            show_job(resp.json())
        except httpx.ConnectError:
            st.error(f"Can't reach the API at {API_BASE}.")


main()
