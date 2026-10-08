# Research Agent

An autonomous web research agent that thinks, searches, reads pages, and produces well-cited markdown reports. Built as a learning project following a phased plan from setup through deployment.

## Features

- **Autonomous agent loop**: Uses an LLM with tool calling to repeat think → use tool → see results until it has enough information (max steps/time guardrails)
- **Real web tools**: `search` (Tavily API) and `read_page` (httpx + trafilatura) with caching and error handling
- **Citations that are trustworthy**: A source ledger tracks only pages actually read in order; reports auto-generate references and `check_citations()` flags orphans/uncited claims
- **Guardrails & observability**: Step/time limits, duplicate-call prevention, quota pacing/retries, structured logging, and full step traces
- **API + UI**: FastAPI job backend (FIFO worker, survives restarts) and Streamlit interface
- **Persistence**: SQLAlchemy models for reports, sources, and steps (SQLite by default, Postgres-ready via `DATABASE_URL`)
- **Evaluation harness**: Judge-based scoring against a curated dataset with baseline and iterated results
- **Containerized & CI**: Docker/Docker Compose for local runs and GitHub Actions for tests

## Architecture

The system is split into modular pieces:

```
app/
  agent/
    loop.py      # Core agent loop (tool calling, guardrails, trace)
    tools.py     # search/read_page + tool registry/declarations
    sources.py   # Source ledger (verified citations)
    cache.py     # Simple in-memory cache
  backend.py     # FastAPI app + FIFO job worker + persistence
  db.py          # SQLAlchemy models and helpers
  llm.py         # Gemini client with pacing/retries/usage logging
  log.py         # Structured logging (console + logs/agent.log)
  report.py      # Report rendering + citation checking
  web.py         # uvicorn entrypoint
  ui.py          # Streamlit UI

evaluation/
  dataset.json   # Test questions + reference points
  judge.py       # LLM judge for scoring answers
  run.py         # Batch runner with resilience (quota handling)
  score.py       # Aggregation + JSON reports
  results/       # Baseline and iterated scores/answers

tests/
  test_guardrails.py   # Offline guardrail checks
  test_resilience.py  # Quota resilience + batching
  test_web.py         # API + persistence lifecycle

experiments/     # Prototyping demos (tool calling, report, etc.)
```

Key data flow:
1. UI/API receives a question → creates job in DB (queued)
2. FIFO worker runs `run_agent()` with `on_step()` persisting each `StepRecord`
3. `run_agent()` calls tools (`search`, `read_page`), records verified reads in `SourceLedger`
4. On completion, answer + sources + steps are saved; `render_report()` produces markdown with references and warnings
5. UI polls job detail until done/error and renders report + live trace

## Demo

> A short demo GIF/video is recommended. If you have one, place it at `docs/demo.gif` and link it here.

_(Demo GIF to be added — run the UI locally to see it in action.)_

## Getting started

### Prerequisites
- Python 3.12+
- API keys: [Google Gemini](https://ai.google.dev/) and [Tavily](https://tavily.com/) (for web search)

### Local setup (venv)

```bash
# Clone repo and enter
cd research-agent

# Create/activate venv (Windows example)
python -m venv .venv
.venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Configure env
cp .env.example .env
# Fill GEMINI_API_KEY and TAVILY_API_KEY
```

### Run API + UI

```bash
# Terminal 1: API
python -m app.web
# → http://127.0.0.1:8000 (docs at /docs)

# Terminal 2: UI
streamlit run app/ui.py
# → http://127.0.0.1:8501
```

Submit a question in the UI. The agent will search/read pages and return a cited report with live step logs.

### Run tests

```bash
# Run all test suites
python tests/test_guardrails.py
python tests/test_resilience.py
python tests/test_web.py
```

### Run evaluation

```bash
# Run baseline/iterated eval (uses dataset.json)
python -m evaluation.run --tag baseline --limit 16
python -m evaluation.score --tag baseline

# Judge-based scoring
python -m evaluation.run --tag v2 --limit 3
python -m evaluation.score --tag v2
```

## Docker & Compose

Spin up API, UI, and Postgres in one command:

```bash
docker compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
# DB:  postgres://research:research@localhost:5432/research_agent (if exposed)
```

Compose uses `DATABASE_URL=postgresql+psycopg://research:research@db:5432/research_agent` for the API service and runs migrations on startup via `db.init_db()`. SQLite remains the default when running locally without Compose.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs:
- Offline test suites (`tests/test_*.py`)
- `docker compose config -q` and `docker compose build`

## Evaluation results

### Baseline (16 questions)
- **Tag**: `baseline`
- **Questions scored**: 16
- **Accuracy**: 90.9%
- **Pass rate**: 100.0%
- **Citation validity**: 100.0% (no orphan citations; ledger enforced)
- **Avg steps**: 3.4, **Avg sources**: 2.3
- **Hit step limit**: 0, **Judge errors**: 0
- **Notes**: High citation discipline. Some answers missed specific named tools/libraries or explicit trade-offs on a few questions (see per-question `missing_points` in `evaluation/results/scores_baseline.json`).

### v2 iteration (3 questions)
- **Tag**: `v2`
- **Questions scored**: 3 (q02, q03, q08)
- **Accuracy**: 91%
- **Pass rate**: 100.0%
- **Citation validity**: 100.0%
- **Avg steps**: 3.7, **Avg sources**: 3
- **Improvements**: q02/q08 near-perfect; q03 improved structure but still missed naming a couple of common tools and explicit trade-offs. Prompt/system instructions were tuned to request concrete names, numbers, trade-offs, and "choose X when… choose Y when…" conclusions.

Full JSON results: see `evaluation/results/scores_baseline.json` and `evaluation/results/scores_v2.json`. Raw answers: `answers_baseline.json`, `answers_v2.json`.

## Lessons learned

- **Ledger > model claims**: Treating only actually-read pages as citable (SourceLedger) eliminates citation hallucinations and makes reports auditable.
- **Guardrails early**: Step/time limits, duplicate-detection, and blocked/empty response handling prevent runaway loops and noisy failures.
- **Quota-aware LLM calls**: Pacing between calls (`MIN_INTERVAL`), retrying only transient errors, detecting daily-quota messages, and aborting on sustained failures made batch eval resilient without hammering the API.
- **Batching judge calls**: Chunking dataset into batches (size 16) cut judge round-trips and preserved input order; retrying on malformed JSON avoided flaky parse crashes.
- **Persistence matters**: DB-backed jobs + marking interrupted runs on restart makes the system usable beyond a single process lifetime (critical for demos/UI).
- **Small prompts + clear instructions**: Explicitly requiring concrete names/numbers, trade-offs, different sources, multiple reads, and citation format improved answer quality without overengineering.
- **Offline tests are essential**: Mocking LLM/tool calls let us verify guardrails, resilience, and API lifecycle deterministically in CI and locally.

## Tech stack

- **Language**: Python 3.12
- **LLM**: Google Gemini (`google-genai`)
- **Search**: Tavily Python
- **Web extraction**: httpx + trafilatura
- **API**: FastAPI + uvicorn
- **UI**: Streamlit
- **DB**: SQLAlchemy (SQLite/Postgres) + psycopg
- **Logging**: stdlib logging (DEBUG file + INFO console)
- **Eval**: dataset + LLM judge + scoring
- **Container/CI**: Docker, Docker Compose, GitHub Actions

## Deployment

This project is designed to be deployable (e.g., Render/Railway for API/UI, Neon/Supabase for Postgres). The current implementation satisfies the plan’s Phase 11 requirements (docs + containerization). Add a live link here once deployed.

## License

This is a learning project. Feel free to fork and adapt.