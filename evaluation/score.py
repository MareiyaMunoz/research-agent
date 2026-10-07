"""Score saved answers: LLM judge 0-100 + free citation checks.

Usage:
    python -m evaluation.score --tag baseline
"""

import argparse
import json
import statistics
from pathlib import Path

from evaluation.judge import citation_metrics, judge
from evaluation.run import answers_path, load_dataset

PASS_THRESHOLD = 60


def main() -> None:
    parser = argparse.ArgumentParser(description="Score saved answers")
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()

    dataset = {q["id"]: q for q in load_dataset()}
    path = answers_path(args.tag)
    if not path.exists():
        raise SystemExit(f"No answers at {path} — run: python -m evaluation.run --tag {args.tag}")
    answers = json.loads(path.read_text(encoding="utf-8"))

    items = [
        {
            "id": qid,
            "question": saved["question"],
            "reference": dataset[qid]["reference"],
            "answer": saved["answer"],
        }
        for qid, saved in answers.items()
    ]
    verdicts = {v["id"]: v for v in judge(items)}

    rows = []
    for qid, saved in answers.items():
        verdict = verdicts[qid]
        cites = citation_metrics(saved["answer"], saved["sources"])
        row = {
            "id": qid,
            "question": saved["question"],
            "score": verdict["score"],
            "missing_points": verdict.get("missing_points", []),
            "errors": verdict.get("errors", []),
            "judge_error": verdict.get("judge_error"),
            "orphans": cites["orphans"],
            "uncited": cites["uncited"],
            "steps": saved["steps"],
            "reached_limit": saved["reached_limit"],
            "sources": len(saved["sources"]),
            "duration_s": saved["duration_s"],
        }
        rows.append(row)
        print(
            f"{qid}  score={row['score']}  steps={row['steps']}  "
            f"orphans={len(cites['orphans'])}  {saved['question'][:55]}"
        )

    scored = [r for r in rows if r["score"] is not None]
    accuracy = statistics.mean(r["score"] for r in scored) if scored else 0.0
    summary = {
        "tag": args.tag,
        "questions": len(rows),
        "scored": len(scored),
        "accuracy": round(accuracy, 1),
        "pass_rate_pct": round(100 * sum(r["score"] >= PASS_THRESHOLD for r in scored) / len(scored), 1)
        if scored
        else 0.0,
        "citation_valid_pct": round(100 * sum(not r["orphans"] for r in rows) / len(rows), 1)
        if rows
        else 0.0,
        "judge_errors": sum(r["judge_error"] is not None for r in rows),
        "hit_step_limit": sum(r["reached_limit"] for r in rows),
        "avg_steps": round(statistics.mean(r["steps"] for r in rows), 1) if rows else 0.0,
        "avg_sources": round(statistics.mean(r["sources"] for r in rows), 1) if rows else 0.0,
    }

    out = Path(f"evaluation/results/scores_{args.tag}.json")
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    worst = sorted(scored, key=lambda r: r["score"])[:3]
    if worst:
        print("\n=== lowest scores ===")
        for r in worst:
            print(f"{r['id']} {r['score']}: {r['question'][:70]}")
            for m in r["missing_points"][:3]:
                print(f"    missing: {m}")
            for e in r["errors"][:2]:
                print(f"    error: {e}")

    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
