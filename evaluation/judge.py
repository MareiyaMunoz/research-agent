"""Grading: an LLM judge scores answers 0-100, citation checks run for free.

The judge grades in batches of BATCH_SIZE so scoring 16 questions costs
1 API call instead of 16 — important while the free tier allows 20/day.
"""

import json

from google.genai import types

from app.agent.loop import AgentResult
from app.llm import MODEL, generate
from app.log import get_logger
from app.report import check_citations

logger = get_logger(__name__)

BATCH_SIZE = 16

BATCH_PROMPT = """You are grading research agent answers for an evaluation. Be strict and consistent.

Grade every item below against its reference key points, 0-100.

## Grading rules
- 90-100: nearly all points covered and correct. 70-89: most points, minor gaps. 50-69: about half covered or notable errors. 30-49: shallow. 0-29: empty, irrelevant, or wrong.
- If an answer contradicts a reference point, that is a factual error — subtract for it.
- Grade meaning, not wording: answers may phrase points differently.
- Do NOT judge style, tone, length, or citation formatting. Citations are graded separately.

## Items
{items}

Reply with ONLY a JSON array, one object per item, same ids, nothing else:
[{{"id": "q01", "score": 0, "missing_points": ["reference point not covered"], "errors": ["factual error in the answer"]}}]"""

JUDGE_CONFIG = types.GenerateContentConfig(
    temperature=0,
    response_mime_type="application/json",
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def _format_items(items: list[dict]) -> str:
    blocks = []
    for item in items:
        points = "\n".join(f"  {i}. {p}" for i, p in enumerate(item["reference"], 1))
        blocks.append(
            f"### {item['id']}\n"
            f"Question: {item['question']}\n"
            f"Reference points:\n{points}\n"
            f"Answer:\n{item['answer']}"
        )
    return "\n\n".join(blocks)


def _verdict(item_id: str, data: dict) -> dict:
    score = int(data["score"])
    if not 0 <= score <= 100:
        raise ValueError(f"score out of range for {item_id}: {score}")
    return {
        "id": item_id,
        "score": score,
        "missing_points": [str(x) for x in data.get("missing_points", [])],
        "errors": [str(x) for x in data.get("errors", [])],
    }


def _judge_batch(items: list[dict]) -> list[dict]:
    """Grade one batch. Returns one verdict dict per item, input order."""
    prompt = BATCH_PROMPT.format(items=_format_items(items))
    last_error = None

    for attempt in range(2):
        try:
            response = generate(model=MODEL, contents=prompt, config=JUDGE_CONFIG)
            data = json.loads(response.text)
            by_id = {str(d["id"]): d for d in data if "id" in d}
            verdicts = {}
            for item in items:
                entry = by_id.get(item["id"])
                if entry is None or "score" not in entry:
                    raise ValueError(f"batch reply is missing item {item['id']}")
                verdicts[item["id"]] = _verdict(item["id"], entry)
            return [verdicts[item["id"]] for item in items]
        except Exception as e:
            last_error = e
            logger.warning("judge batch attempt %s failed: %s", attempt + 1, e)
            prompt += (
                f"\n\nYour previous reply could not be used ({e}). "
                "Reply again with ONLY the JSON array."
            )

    return [
        {"id": item["id"], "score": None, "judge_error": str(last_error)}
        for item in items
    ]


def judge(items: list[dict]) -> list[dict]:
    """Grade a list of items: [{id, question, reference, answer}].

    Returns one verdict per item in input order:
    {id, score 0-100, missing_points, errors} or {id, score: None, judge_error}.
    """
    verdicts = []
    for start in range(0, len(items), BATCH_SIZE):
        verdicts.extend(_judge_batch(items[start : start + BATCH_SIZE]))
    return verdicts


def citation_metrics(answer: str, sources: list[dict]) -> dict:
    """Deterministic citation checks from Phase 5's checker (free)."""
    return check_citations(AgentResult(answer=answer, sources=sources))
