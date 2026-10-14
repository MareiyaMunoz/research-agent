"""Deterministic source-quality scoring.

Python - not the LLM - decides how trustworthy a source looks, from the URL
(domain, TLD), the title, and any page text. This is intentionally heuristic
and explainable: every adjustment is recorded in `reasons`. It costs no API
calls, so it works within the free tier and is easy to test.
"""

from urllib.parse import urlparse

# Tier cutoffs on the 0-100 score.
HIGH_MIN = 80
MEDIUM_MIN = 50

# Curated allowlist: official documentation and primary/reference material.
DOC_DOMAINS = {
    "docs.python.org",
    "developer.mozilla.org",
    "kubernetes.io",
    "postgresql.org",
    "redis.io",
    "cloud.google.com",
    "aws.amazon.com",
    "learn.microsoft.com",
    "docker.com",
    "github.io",
}
DOC_HOST_PREFIXES = ("docs.", "developer.", "learn.")
DOC_SUFFIXES = (".readthedocs.io", ".readthedocs.org")
# Path segments that also signal official documentation on any site.
DOC_PATH_SEGMENTS = ("/docs/", "/documentation/", "/reference/", "/manual/")

# Academic / research.
ACADEMIC_DOMAINS = {
    "arxiv.org",
    "acm.org",
    "ieee.org",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "springer.com",
    "jstor.org",
}
ACADEMIC_TLDS = {".edu", ".gov", ".mil"}

# Major publications (reputable, not primary docs).
PUBLICATION_DOMAINS = {
    "reuters.com",
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "apnews.com",
    "arstechnica.com",
    "wired.com",
    "theguardian.com",
    "wsj.com",
    "economist.com",
}

# Platforms that host unvetted user content.
LOW_QUALITY_DOMAINS = {
    "medium.com",
    "dev.to",
    "substack.com",
    "wordpress.com",
    "blogspot.com",
    "hashnode.dev",
    "reddit.com",
    "quora.com",
    "stackexchange.com",
    "stackoverflow.com",
    "medium.com",
}
LOW_QUALITY_SUFFIXES = (".blogspot.com", ".wordpress.com")

KNOWN_CODE_HOSTS = {"github.com", "gitlab.com", "bitbucket.org"}

HIGH_CUES = ("official documentation", "official docs", "reference manual", "api reference")
LOW_CUES = ("tutorial", "blog post", "guest post", "forum", "opinion")


def _host_and_suffix(url: str) -> tuple[str, str]:
    host = (urlparse(url).netloc or "").lower()
    host = host.split("@")[-1].split(":")[0].removeprefix("www.")
    suffix = ""
    if "." in host:
        suffix = "." + host.rsplit(".", 1)[1]
    return host, suffix


def _is_ip(host: str) -> bool:
    parts = host.split(".")
    return len(parts) == 4 and all(p.isdigit() for p in parts)


def _matches(host: str, domains: set[str], prefixes=(), suffixes=()) -> bool:
    if host in domains:
        return True
    if any(host.startswith(p) for p in prefixes):
        return True
    return any(host.endswith(s) for s in suffixes)


def score_url(url: str, title: str | None = None, text: str | None = None) -> dict:
    """Score one source 0-100 and classify it high/medium/low.

    `reasons` lists every signal that moved the score, so the result is
    auditable rather than a black box.
    """
    if not url.startswith(("http://", "https://")):
        return {"score": 10, "tier": "low", "reasons": ["not a web URL"]}

    host, suffix = _host_and_suffix(url)
    path = (urlparse(url).path or "").lower()
    score = 55  # neutral default for an unknown .com/.net/.org
    reasons = ["unknown domain (default)"]

    if _is_ip(host):
        score, reasons = 20, ["bare IP address"]
    elif _matches(host, DOC_DOMAINS, DOC_HOST_PREFIXES, DOC_SUFFIXES) or any(
        seg in path for seg in DOC_PATH_SEGMENTS
    ):
        score, reasons = 92, ["official documentation/reference"]
    elif _matches(host, ACADEMIC_DOMAINS) or suffix in ACADEMIC_TLDS:
        score, reasons = 90, ["academic/government domain"]
    elif _matches(host, PUBLICATION_DOMAINS):
        score, reasons = 75, ["major publication"]
    elif _matches(host, LOW_QUALITY_DOMAINS, suffixes=LOW_QUALITY_SUFFIXES):
        score, reasons = 30, ["user-content platform"]
    elif host in KNOWN_CODE_HOSTS:
        score, reasons = 60, ["code hosting"]

    blob = f"{title or ''} {text or ''}".lower()
    if score >= 50 and any(cue in blob for cue in HIGH_CUES):
        score += 10
        reasons.append("content signals official documentation")
    if any(cue in blob for cue in LOW_CUES):
        score -= 10
        reasons.append("content signals tutorial/blog/forum")

    score = max(0, min(100, score))
    if score >= HIGH_MIN:
        tier = "high"
    elif score >= MEDIUM_MIN:
        tier = "medium"
    else:
        tier = "low"
    return {"score": score, "tier": tier, "reasons": reasons}


def ensure_qualities(sources: list[dict]) -> list[dict]:
    """Return sources with a `quality` dict, computing it when missing.

    Old rows (stored before quality scoring existed) have no `quality`, so it
    is derived from url/title/text on the fly. Pages read with no stored text
    are still scored from the domain.
    """
    out = []
    for s in sources:
        if s.get("quality"):
            out.append(s)
            continue
        s = dict(s)
        s["quality"] = score_url(s.get("url", ""), s.get("title"), s.get("text"))
        out.append(s)
    return out


def summarize(sources: list[dict]) -> dict:
    """Tier counts, average score, and the numbers of low-quality sources."""
    scored = ensure_qualities(sources)
    counts = {"high": 0, "medium": 0, "low": 0}
    scores = []
    low = []
    for s in scored:
        q = s["quality"]
        counts[q["tier"]] = counts.get(q["tier"], 0) + 1
        scores.append(q["score"])
        if q["tier"] == "low":
            low.append(s["n"])
    return {
        "counts": counts,
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0.0,
        "low_quality": low,
    }
