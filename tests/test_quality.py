"""Offline checks for deterministic source-quality scoring - no API calls."""
import os
import sys
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "test-key")  # app.llm builds a client at import

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.loop import AgentResult
from app.agent.quality import ensure_qualities, score_url, summarize
from app.agent.sources import SourceLedger
from app.report import render_report

PASS = []


def check(name, cond, extra=""):
    PASS.append((name, bool(cond), extra))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))


# 1. TLD / allowlist / platform heuristics
cases = [
    ("https://www.nasa.gov/missions", "high"),
    ("https://cs.stanford.edu/research", "high"),
    ("https://arxiv.org/abs/2401.00001", "high"),
    ("https://docs.python.org/3/library/sqlite3.html", "high"),
    ("https://developer.mozilla.org/en-US/docs/Web", "high"),
    ("https://example.readthedocs.io/en/latest/", "high"),
    ("https://www.ibm.com/docs/en/cognos-analytics/12.0.x", "high"),
    ("https://www.reuters.com/technology/", "medium"),
    ("https://medium.com/@someone/a-post", "low"),
    ("https://dev.to/someone/tutorial", "low"),
    ("https://stackoverflow.com/questions/123", "low"),
    ("http://192.168.0.1/page", "low"),
    ("https://some-random-blog.example.com/x", "medium"),
]
for url, tier in cases:
    got = score_url(url)["tier"]
    check(f"{url} -> {tier}", got == tier, f"got {got}")


# 2. content cues nudge the score
base = score_url("https://example.com/x")["score"]
up = score_url("https://example.com/x", text="Official documentation for the API reference.")["score"]
down = score_url("https://example.com/x", text="A tutorial blog post about things.")["score"]
check("docs cue raises score", up > base, f"{base} -> {up}")
check("tutorial cue lowers score", down < base, f"{base} -> {down}")


# 3. non-URL input is low
check("non-url -> low", score_url("not a url")["tier"] == "low")


# 4. reasons explain the decision
check("reasons present", bool(score_url("https://docs.python.org/3/")["reasons"]))


# 5. ensure_qualities fills missing quality without mutating input
raw = [
    {"n": 1, "url": "https://docs.python.org/3/", "title": "Python", "step": 1},
    {"n": 2, "url": "https://medium.com/@a/b", "title": "Blog", "step": 1},
]
filled = ensure_qualities(raw)
check("ensure fills quality", all(s.get("quality") for s in filled))
check("ensure does not mutate input", "quality" not in raw[0])


# 6. summarize counts tiers, average, and low-quality numbers
summary = summarize(filled)
check("summary counts high", summary["counts"]["high"] == 1, str(summary))
check("summary counts low", summary["counts"]["low"] == 1, str(summary))
check("summary low_quality list", summary["low_quality"] == [2], str(summary))
check("summary avg_score", 0 < summary["avg_score"] <= 100, str(summary))


# 7. the ledger stores quality
led = SourceLedger()
led.record("https://docs.python.org/3/", step=1, title="Python", quality=score_url("https://docs.python.org/3/"))
check("ledger stores quality", led.as_list()[0]["quality"]["tier"] == "high")


# 8. report annotates references and notes low-quality sources
result = AgentResult(
    answer="Python is documented [1] and a blog says so too [2].",
    sources=[
        {"n": 1, "url": "https://docs.python.org/3/", "title": "Python Docs", "step": 1},
        {"n": 2, "url": "https://medium.com/@a/b", "title": "Some Blog", "step": 1},
    ],
    steps=2,
)
report = render_report("q", result)
check("report annotates tier", "high (" in report and "low (" in report, report)
check("report notes low quality", "low-quality" in report, report)


# 9. no low-quality note when all sources are good
good = AgentResult(
    answer="See the docs [1].",
    sources=[{"n": 1, "url": "https://docs.python.org/3/", "title": "Docs", "step": 1}],
    steps=1,
)
check("no false low-quality note", "low-quality" not in render_report("q", good))


failed = [p for p in PASS if not p[1]]
print(f"\n{len(PASS) - len(failed)}/{len(PASS)} passed")
sys.exit(1 if failed else 0)
