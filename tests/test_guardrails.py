"""Offline sanity checks for Phase 6 guardrails - no API calls."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.genai import types

import app.agent.loop as loop
from app.agent.loop import run_agent

PASS = []


def check(name, cond, extra=""):
    PASS.append((name, bool(cond), extra))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))


class FakeResponse:
    def __init__(self, candidates=None, function_calls=None, text=None):
        self._cands = candidates or []
        self.function_calls = function_calls or []
        self._text = text

    @property
    def candidates(self):
        return self._cands

    @property
    def text(self):
        if self._text is None:
            raise ValueError("No text found in response")
        return self._text


class FakeCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


def model_content():
    return types.Content(role="model", parts=[types.Part(text="thinking...")])


def candidate():
    """candidates[0] must expose .content, like google-genai's Candidate."""
    from types import SimpleNamespace

    return SimpleNamespace(content=model_content())


# 1. deadline guard: max_seconds=0 trips before the first LLM call
with patch.object(loop, "generate") as g:
    r = run_agent("q", max_steps=5, max_seconds=0)
    check("deadline stops run", r.reached_limit and g.call_count == 0)
    check("deadline trace record", r.trace and r.trace[-1].action == "limit")
    check("deadline mentions time", "ran out of time" in r.answer)

# 2. blocked response: empty candidates
with patch.object(loop, "generate", return_value=FakeResponse(candidates=[])):
    r = run_agent("q")
    check("blocked -> clean result", "no usable response" in r.answer and not r.reached_limit)
    check("blocked trace", r.trace and r.trace[-1].ok is False)

# 3. response with neither tools nor text (text property raises)
with patch.object(loop, "generate", return_value=FakeResponse(candidates=[candidate()])):
    r = run_agent("q")
    check("no-text -> clean result", "neither a tool call nor any text" in r.answer)

# 4. duplicate-call short-circuit across consecutive steps
calls = [FakeCall("search", {"query": "ai"})]
responses = [
    FakeResponse(candidates=[candidate()], function_calls=calls),
    FakeResponse(candidates=[candidate()], function_calls=calls),
    FakeResponse(candidates=[candidate()], text="Final answer [1]"),
]
tool_runs = []


def counting_tool(name, args):
    tool_runs.append((name, str(args)))
    return {"results": []}


with patch.object(loop, "generate", side_effect=responses), patch.object(
    loop, "run_tool", side_effect=counting_tool
):
    r = run_agent("q", max_steps=6)

check("duplicate tool NOT re-run", len(tool_runs) == 1, f"runs={tool_runs}")
dupes = [t for t in r.trace if t.error == "duplicate of previous step"]
check("duplicate trace record", len(dupes) == 1)
check("final answer reached", "Final answer" in r.answer and r.steps == 3)
check("trace has answer record", r.trace[-1].action == "answer")

# 5. trace records tool details
t = r.trace[0]
check(
    "StepRecord fields",
    t.action == "search" and t.args == {"query": "ai"} and t.ok and t.result_chars > 0,
    str(t),
)

# 6. search() catches Tavily failures
import app.agent.tools as tools


class BoomTavily:
    def search(self, **kw):
        raise RuntimeError("quota exceeded")


with patch.object(tools, "tavily", BoomTavily()):
    out = tools.search("python releases")
check("search error -> dict", isinstance(out, dict) and out.get("error", "").startswith("Search failed: RuntimeError"), str(out))

# 7. step limit path still records 'limit'
with patch.object(
    loop,
    "generate",
    return_value=FakeResponse(
        candidates=[candidate()],
        function_calls=[FakeCall("search", {"query": "a"})],
    ),
), patch.object(loop, "run_tool", side_effect=lambda n, a: {"results": []}):
    r = run_agent("q", max_steps=1)
check("max_steps -> reached_limit + trace", r.reached_limit and r.trace[-1].action == "limit")

failed = [p for p in PASS if not p[1]]
print(f"\n{len(PASS) - len(failed)}/{len(PASS)} passed")
sys.exit(1 if failed else 0)