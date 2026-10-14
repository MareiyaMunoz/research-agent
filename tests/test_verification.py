"""Offline checks for claim verification (Phase 12) - no API calls.

Verification is deterministic: it compares each cited claim's wording against
the stored text of the source it cites, so it can be tested without Gemini.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "test-key")  # app.llm builds a client at import

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.loop import AgentResult
from app.agent.sources import SourceLedger
from app.report import check_citations, render_report, verify_claims

PASS = []


def check(name, cond, extra=""):
    PASS.append((name, bool(cond), extra))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))


POOL_TEXT = (
    "Connection pooling reuses an existing open database connection instead of "
    "opening and closing a new one for every request. Reusing connections avoids "
    "the TCP handshake and authentication cost and caps concurrent connections."
)
PLANT_TEXT = (
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose and "
    "oxygen inside plant chloroplasts."
)

POOL_CLAIM = "Connection pooling reuses open connections instead of opening a new one per request [1]."
PLANT_MISMATCH = "The TLS handshake encrypts all traffic before any data is sent [2]."
ORPHAN_CLAIM = "A completely invented fact with no backing source [9]."


def make_result(answer, sources):
    return AgentResult(answer=answer, sources=sources, steps=2)


# 1. the ledger keeps the page text
led = SourceLedger()
led.record("https://example.com/pool", step=1, title="Pooling", text=POOL_TEXT)
check("ledger stores text", led.as_list()[0]["text"] == POOL_TEXT)


def sources():
    return [
        {"n": 1, "url": "https://example.com/pool", "title": "Pooling", "step": 1, "text": POOL_TEXT},
        {"n": 2, "url": "https://example.com/plants", "title": "Plants", "step": 2, "text": PLANT_TEXT},
    ]


# 2. a claim that matches its source is supported
res = make_result(POOL_CLAIM, sources())
v = verify_claims(res)
check("supported claim", v["claims"][0]["status"] == "supported" and v["counts"]["supported"] == 1, str(v))
check("supported score high", v["claims"][0]["score"] >= 0.5, str(v["claims"][0]))

# 3. a claim citing an unrelated source is unsupported
res = make_result(PLANT_MISMATCH, sources())
v = verify_claims(res)
check("unsupported claim", v["claims"][0]["status"] == "unsupported" and v["counts"]["unsupported"] == 1, str(v))
check("unsupported names source", v["claims"][0]["best_source"] == 2, str(v["claims"][0]))

# 4. a citation not in the ledger is an orphan claim
res = make_result(ORPHAN_CLAIM, sources())
v = verify_claims(res)
check("orphan claim", v["claims"][0]["status"] == "orphan" and v["counts"]["orphan"] == 1, str(v))

# 5. no stored text -> unverified (older runs are not called unsupported)
res = make_result(POOL_CLAIM, [{"n": 1, "url": "u", "title": "t", "step": 1, "text": None}])
v = verify_claims(res)
check("missing text -> unverified", v["claims"][0]["status"] == "unverified", str(v))

# 6. multi-citation claims use the best-matching valid source
res = make_result(POOL_CLAIM.replace("[1]", "[1], [2]"), sources())
v = verify_claims(res)
check("multi-citation best source", v["claims"][0]["status"] == "supported" and v["claims"][0]["best_source"] == 1, str(v["claims"][0]))

# 7. sentence splitting finds each cited claim
two = f"{POOL_CLAIM} {PLANT_MISMATCH}"
res = make_result(two, sources())
v = verify_claims(res)
check("two claims found", v["total"] == 2, str(v))

# 8. report warns on unsupported claims and stays quiet when all supported
warn = render_report("q", make_result(PLANT_MISMATCH, sources()))
check("report warns unsupported", "may not be supported" in warn, warn)
clean = render_report("q", make_result(POOL_CLAIM, sources()))
check("report quiet when supported", "may not be supported" not in clean, clean)

# 9. existing citation checker is unaffected by the extra text key
checks = check_citations(make_result(ORPHAN_CLAIM, sources()))
check("check_citations still flags orphan", checks["orphans"] == [9], str(checks))

failed = [p for p in PASS if not p[1]]
print(f"\n{len(PASS) - len(failed)}/{len(PASS)} passed")
sys.exit(1 if failed else 0)
