"""Offline checks for Phase 7 quota-resilience changes - no API calls."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluation.judge as jmod
import evaluation.run as rmod
from app.agent.loop import StepRecord

PASS = []


def check(name, cond, extra=""):
    PASS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))


class FakeResp:
    def __init__(self, payload):
        self._payload = payload

    @property
    def text(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def make_items(n):
    return [
        {"id": f"q{i:02d}", "question": f"Q{i}?", "reference": ["point a", "point b"], "answer": f"A{i}"}
        for i in range(1, n + 1)
    ]


# 1. judge batches 16 questions into ceil(16/BATCH_SIZE) calls, order preserved
items = make_items(16)
calls = []


def batch_ok(model, contents, config):
    calls.append(contents)
    start = (len(calls) - 1) * jmod.BATCH_SIZE + 1
    ids = [f"q{i:02d}" for i in range(start, start + jmod.BATCH_SIZE)]
    return FakeResp(json.dumps([{"id": i, "score": 70, "missing_points": [], "errors": []} for i in ids]))


with patch.object(jmod, "generate", side_effect=batch_ok):
    verdicts = jmod.judge(items)

expected_calls = -(-16 // jmod.BATCH_SIZE)  # ceil
check(f"16 items -> {expected_calls} judge calls", len(calls) == expected_calls, f"calls={len(calls)}")
check("verdict count", len(verdicts) == 16)
check("input order preserved", [v["id"] for v in verdicts] == [i["id"] for i in items])
check("scores parsed", all(v["score"] == 70 for v in verdicts))

# 2. invalid batch reply retries, then succeeds
replies = [
    FakeResp('{"not": "an array"'),
    FakeResp(json.dumps([{"id": "q01", "score": 81, "missing_points": [], "errors": []}])),
]
with patch.object(jmod, "generate", side_effect=replies):
    v = jmod.judge(items[:1])
check("retry after bad JSON", v[0]["score"] == 81, str(v))

# 3. never-valid reply -> judge_error, no crash
with patch.object(jmod, "generate", side_effect=[FakeResp("garbage"), FakeResp("garbage")]):
    v = jmod.judge(items[:1])
check("persistent bad reply -> judge_error", v[0]["score"] is None and "judge_error" in v[0], str(v))

# 4. run.py survives a failed question and keeps going
class FakeResult:
    answer = "text"
    sources = []
    steps = 3
    reached_limit = False
    trace = [StepRecord(step=1, action="answer")]


calls_made = {"n": 0}


def flaky_agent(question, max_steps=8, max_seconds=120):
    calls_made["n"] += 1
    if calls_made["n"] == 1:
        raise RuntimeError("quota gone")
    return FakeResult()


tag = "unittest_tmp"
path = rmod.answers_path(tag)
if path.exists():
    path.unlink()
try:
    with patch.object(rmod, "run_agent", side_effect=flaky_agent), patch.object(
        sys, "argv", ["run", "--tag", tag, "--limit", "3"]
    ):
        rmod.main()
    saved = json.loads(path.read_text(encoding="utf-8"))
    check("failed question skipped, rest saved", sorted(saved) == ["q02", "q03"], str(sorted(saved)))
finally:
    if path.exists():
        path.unlink()

# 5. two consecutive failures abort the run early
calls_made["n"] = 0


def always_fail(question, max_steps=8, max_seconds=120):
    calls_made["n"] += 1
    raise RuntimeError("quota gone")


if path.exists():
    path.unlink()
try:
    with patch.object(rmod, "run_agent", side_effect=always_fail), patch.object(
        sys, "argv", ["run", "--tag", tag, "--limit", "5"]
    ):
        rmod.main()
    check("aborts after 2 consecutive failures", calls_made["n"] == 2, f"calls={calls_made['n']}")
finally:
    if path.exists():
        path.unlink()

# 6. daily-quota futility parser
from app.llm import _daily_quota_hours

check("parses hours", _daily_quota_hours(Exception("...retry in 15h17m4s...")) == 15)
check("ignores seconds-scale", _daily_quota_hours(Exception("Please retry in 30s")) == 0)
check("ignores unrelated", _daily_quota_hours(Exception("connection reset")) == 0)

failed = PASS.count(False)
print(f"\n{len(PASS) - failed}/{len(PASS)} passed")
sys.exit(1 if failed else 0)