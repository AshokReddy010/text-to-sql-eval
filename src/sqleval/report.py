"""Aggregate every configuration's results into one report document for the leaderboard site.

Accuracy is reported with a 95% Wilson interval: with 50 questions, a 4-point gap between two models is
usually inside the noise, and the report should say so rather than crown a winner.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from .runner import Question, load_results, qhash

FAILURE_ORDER = [
    "wrong_values",
    "wrong_shape",
    "wrong_order",
    "extra_columns",
    "empty_result",
    "unknown_identifier",
    "syntax",
    "runtime",
    "timeout",
    "unsafe",
    "no_sql",
    "too_many_rows",
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def build_report(results_dir: Path, questions: list[Question]) -> dict[str, Any]:
    qmap = {q.id: q for q in questions}
    configs = []
    per_question: dict[str, dict[str, Any]] = defaultdict(dict)
    for path in sorted(results_dir.glob("*.jsonl")):
        res = load_results(path)
        if not res:
            continue
        config = next(iter(res.values()))["config"]
        provider, _, variant = config.rpartition("@")
        if provider == "gold":
            continue  # the harness check is not a contestant
        # only results for the current version of each question count
        cur = {
            qid: r
            for qid, r in res.items()
            if qid in qmap and r.get("hash") == qhash(qmap[qid], variant) and r.get("category") != "provider_error"
        }
        n = len(cur)
        if n == 0:
            continue
        k = sum(1 for r in cur.values() if r["correct"])
        kl = sum(1 for r in cur.values() if r["lenient"])
        by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for qid, r in cur.items():
            for t in qmap[qid].tags:
                by_tag[t][0] += int(r["correct"])
                by_tag[t][1] += 1
        lat = [r["latency_s"] for r in cur.values() if r.get("latency_s")]
        tin = [r["tokens_in"] for r in cur.values() if r.get("tokens_in")]
        tout = [r["tokens_out"] for r in cur.values() if r.get("tokens_out")]
        fails = Counter(r["category"] for r in cur.values() if not r["correct"])
        configs.append(
            {
                "config": config,
                "provider": provider,
                "model": provider.split("/", 1)[-1],
                "variant": variant,
                "answered": n,
                "total": len(questions),
                "complete": n == len(questions),
                "correct": k,
                "accuracy": round(k / n, 4),
                "ci95": wilson(k, n),
                "lenient_accuracy": round(kl / n, 4),
                "by_tag": {
                    t: {"correct": c, "n": m, "accuracy": round(c / m, 4)} for t, (c, m) in sorted(by_tag.items())
                },
                "failures": {c: fails[c] for c in FAILURE_ORDER if fails.get(c)},
                "median_latency_s": round(median(lat), 2) if lat else None,
                "median_tokens_in": int(median(tin)) if tin else None,
                "median_tokens_out": int(median(tout)) if tout else None,
                "last_run": max(r["at"] for r in cur.values()),
            }
        )
        for qid, r in cur.items():
            per_question[qid][config] = {
                "correct": r["correct"],
                "category": r["category"],
                "sql": r.get("sql", ""),
                "detail": r.get("detail", ""),
            }
    configs.sort(key=lambda c: (-c["accuracy"], c["config"]))
    qs = []
    for q in questions:
        got = per_question.get(q.id, {})
        solved = sum(1 for v in got.values() if v["correct"])
        qs.append(
            {
                "id": q.id,
                "question": q.question,
                "tags": list(q.tags),
                "ordered": q.ordered,
                "gold_sql": q.sql,
                "solved_by": solved,
                "attempted_by": len(got),
                "results": got,
            }
        )
    # does context help? pair the two variants of the same model
    ablation = []
    by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for c in configs:
        by_model[c["provider"]][c["variant"]] = c
    for prov, vs in by_model.items():
        if "schema" in vs and "context" in vs:
            ablation.append(
                {
                    "provider": prov,
                    "schema": vs["schema"]["accuracy"],
                    "context": vs["context"]["accuracy"],
                    "delta": round(vs["context"]["accuracy"] - vs["schema"]["accuracy"], 4),
                }
            )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "questions_total": len(questions),
        "configs": configs,
        "questions": qs,
        "ablation": ablation,
        "tags": sorted({t for q in questions for t in q.tags}),
    }
