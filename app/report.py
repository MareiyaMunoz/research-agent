import re

from app.agent.loop import AgentResult

# Matches [1], [2, 3], [1,2,4] - the model groups citations in various ways
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

GENERIC_TOPICS = {"what", "why", "how", "explain", "describe"}

# Claim verification is deterministic on purpose: no LLM calls, so it is free
# and cannot be fooled by the same model that wrote the answer.
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#._-]*")
SUPPORTED_THRESHOLD = 0.5   # share of a claim's content words found in its source
WEAK_THRESHOLD = 0.25

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "so", "of",
    "to", "in", "on", "for", "with", "as", "by", "at", "from", "into", "about",
    "that", "this", "these", "those", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "they", "them", "their", "you", "your", "we", "our",
    "he", "she", "his", "her", "not", "no", "can", "could", "will", "would",
    "should", "may", "might", "must", "do", "does", "did", "done", "has",
    "have", "had", "also", "such", "when", "which", "who", "whom", "whose",
    "what", "why", "how", "where", "more", "most", "less", "many", "much",
    "some", "any", "each", "both", "all", "one", "two", "use", "used", "using",
    "via", "per", "while", "because", "between", "over", "under", "after",
    "before", "other", "another", "same", "only", "just", "very", "well",
}


def extract_citations(answer: str) -> list[int]:
    """All citation numbers used in the answer, in order of appearance."""
    numbers = []
    for group in CITATION_RE.findall(answer):
        numbers.extend(int(n) for n in group.split(","))
    return numbers


def check_citations(result: AgentResult) -> dict:
    """Compare citations in the answer against the verified source ledger.

    Returns:
        valid:      numbers that point to an actually-read source
        orphans:    numbers cited but not in the ledger (hallucinated)
        uncited:    ledger entries never cited by the model
    """
    cited = extract_citations(result.answer)
    valid, orphans = [], []
    for n in cited:
        if n in valid or n in orphans:
            continue
        if any(s["n"] == n for s in result.sources):
            valid.append(n)
        else:
            orphans.append(n)
    cited_set = set(valid) | set(orphans)
    uncited = [s["n"] for s in result.sources if s["n"] not in cited_set]
    return {"valid": valid, "orphans": orphans, "uncited": uncited}


def _iter_claim_sentences(answer: str):
    """Yield each cited sentence/bullet of the answer, one at a time."""
    for line in answer.splitlines():
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            sentence = sentence.strip()
            if sentence and CITATION_RE.search(sentence):
                yield sentence


def _content_words(text: str) -> set[str]:
    """Significant words of a text, ignoring citations and stopwords."""
    text = CITATION_RE.sub(" ", text)
    return {
        w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 2
    }


def _overlap(claim: str, source_text: str) -> float:
    """Share of the claim's content words that appear in the source text."""
    claim_words = _content_words(claim)
    source_words = _content_words(source_text or "")
    if not claim_words or not source_words:
        return 0.0
    return len(claim_words & source_words) / len(claim_words)


def verify_claims(result: AgentResult) -> dict:
    """Check each cited claim against the text of the source it cites.

    This is a lexical-overlap approximation, not proof: it flags claims whose
    wording barely appears in the cited page (a sign the model may have cited a
    source it did not really use). Sources without stored text are reported as
    `unverified` rather than `unsupported`, so older runs are not misjudged.

    Returns the per-claim detail plus a status tally.
    """
    by_n = {s["n"]: s for s in result.sources}
    claims = []
    for sentence in _iter_claim_sentences(result.answer):
        nums = extract_citations(sentence)
        valid = [n for n in nums if n in by_n]
        orphans = [n for n in nums if n not in by_n]

        scored = [
            (n, _overlap(sentence, by_n[n].get("text")))
            for n in valid
            if by_n[n].get("text")
        ]

        if not valid:
            status, best_source, best_score = "orphan", None, 0.0
        elif not scored:
            status, best_source, best_score = "unverified", None, 0.0
        else:
            best_source, best_score = max(scored, key=lambda pair: pair[1])
            if best_score >= SUPPORTED_THRESHOLD:
                status = "supported"
            elif best_score >= WEAK_THRESHOLD:
                status = "weak"
            else:
                status = "unsupported"

        claims.append(
            {
                "claim": sentence,
                "citations": nums,
                "valid_citations": valid,
                "orphan_citations": orphans,
                "best_source": best_source,
                "score": round(best_score, 2),
                "status": status,
            }
        )

    statuses = ("supported", "weak", "unsupported", "unverified", "orphan")
    counts = {k: sum(c["status"] == k for c in claims) for k in statuses}
    return {"claims": claims, "counts": counts, "total": len(claims)}


def _short(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _topic_from_question(question: str) -> str:
    """Turn a question into a report heading (first ~80 chars, Title Case)."""
    topic = question.strip().rstrip("?.!")
    if len(topic) > 80:
        topic = topic[:80].rsplit(" ", 1)[0] + "..."
    words = topic.split()
    if len(words) > 3 and words[0].lower() in GENERIC_TOPICS:
        topic = " ".join(words[1:])  # "Explain X" -> "X"
    return topic[0].upper() + topic[1:] if topic else "Research Report"


def render_report(question: str, result: AgentResult) -> str:
    """Build a Markdown report: heading, summary, warnings, references."""
    checks = check_citations(result)
    claims = verify_claims(result)
    lines = [
        f"# {_topic_from_question(question)}",
        "",
        f"*Research report — {len(result.sources)} sources, "
        f"{result.steps} steps*",
        "",
        "## Summary",
        "",
        result.answer,
        "",
    ]

    if checks["orphans"]:
        lines += [
            "> **Warning:** unverified citation(s) "
            + ", ".join(f"[{n}]" for n in checks["orphans"])
            + " — these numbers are not in the source list above.",
            "",
        ]
    if checks["uncited"]:
        lines += [
            "> **Note:** source(s) "
            + ", ".join(f"[{n}]" for n in checks["uncited"])
            + " were read but never cited.",
            "",
        ]
    unsupported = [c for c in claims["claims"] if c["status"] == "unsupported"]
    if unsupported:
        lines += [
            "> **Warning:** "
            + f"{len(unsupported)} claim(s) may not be supported by the cited "
            "source (low wording overlap):",
            "",
        ]
        for c in unsupported[:5]:
            cite = ", ".join(f"[{n}]" for n in c["citations"])
            lines.append(
                f"> - ({cite}, match {int(c['score'] * 100)}%) {_short(c['claim'])}"
            )
        lines.append("")
    if claims["counts"]["weak"]:
        lines += [
            "> **Note:** "
            + f"{claims['counts']['weak']} claim(s) only loosely match their "
            "cited source.",
            "",
        ]
    if result.reached_limit:
        lines += [
            "> **Note:** the agent hit the step limit — this report may be "
            "incomplete.",
            "",
        ]

    lines += ["## References", ""]
    if result.sources:
        for s in result.sources:
            lines.append(f"[{s['n']}] [{s['title']}]({s['url']})")
    else:
        lines.append("*No sources were read.*")

    return "\n".join(lines)
