import re

from app.agent.loop import AgentResult

# Matches [1], [2, 3], [1,2,4] - the model groups citations in various ways
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

GENERIC_TOPICS = {"what", "why", "how", "explain", "describe"}


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
