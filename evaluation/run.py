"""Run the agent over the evaluation dataset and save its answers.

Usage:
    python -m evaluation.run                 # resume/append answers_baseline.json
    python -m evaluation.run --tag v2        # save to answers_v2.json instead
    python -m evaluation.run --limit 3       # run at most 3 unanswered questions
    python -m evaluation.run --fresh         # ignore saved answers, start over
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from app.agent.loop import run_agent
from app.log import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path("evaluation/results")
DATASET_PATH = Path("evaluation/dataset.json")


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))["questions"]


def answers_path(tag: str) -> Path:
    return RESULTS_DIR / f"answers_{tag}.json"


def load_answers(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent over the dataset")
    parser.add_argument("--tag", default="baseline", help="output file suffix")
    parser.add_argument("--limit", type=int, default=None, help="max unanswered questions")
    parser.add_argument("--only", default=None, help="comma-separated ids to run, e.g. q01,q05")
    parser.add_argument("--fresh", action="store_true", help="discard saved answers")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    path = answers_path(args.tag)
    answers = {} if args.fresh else load_answers(path)

    pending = [q for q in load_dataset() if q["id"] not in answers]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        known = {q["id"] for q in load_dataset()}
        unknown = wanted - known
        if unknown:
            raise SystemExit(f"Unknown question ids: {', '.join(sorted(unknown))}")
        pending = [q for q in pending if q["id"] in wanted]
    if args.limit is not None:
        pending = pending[: args.limit]
    if not pending:
        print(f"Nothing to run — {len(answers)} answers already saved in {path}")
        return

    print(f"Running {len(pending)} question(s) -> {path} ({len(answers)} already saved)")
    failed: list[str] = []
    consecutive = 0
    for n, q in enumerate(pending, 1):
        logger.info("[%s/%s] %s: %s", n, len(pending), q["id"], q["question"])
        t0 = time.monotonic()
        try:
            result = run_agent(q["question"], max_steps=8, max_seconds=120)
        except Exception as e:
            failed.append(q["id"])
            consecutive += 1
            logger.error("%s failed: %s", q["id"], e)
            if consecutive >= 2:
                print(
                    f"\nAborting after {consecutive} consecutive failures "
                    f"(API quota?). Resume later with: "
                    f"python -m evaluation.run --tag {args.tag}"
                )
                break
            continue
        consecutive = 0
        answers[q["id"]] = {
            "id": q["id"],
            "question": q["question"],
            "answer": result.answer,
            "sources": result.sources,
            "steps": result.steps,
            "reached_limit": result.reached_limit,
            "duration_s": round(time.monotonic() - t0, 1),
            "trace": [asdict(record) for record in result.trace],
        }
        path.write_text(
            json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "%s done in %.1fs | steps=%s limit=%s answer=%s chars",
            q["id"],
            time.monotonic() - t0,
            result.steps,
            result.reached_limit,
            len(result.answer),
        )

    if failed:
        print(f"Failed questions (will retry on next run): {', '.join(failed)}")
    print(f"Saved {len(answers)} answers to {path}")


if __name__ == "__main__":
    main()
