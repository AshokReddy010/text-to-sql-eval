"""Run the benchmark for one configuration (provider + prompt variant), resumably.

Results go to results/runs/<config>.jsonl, one line per question. A question is skipped when it already has a
scored result for the *same* question text, gold query and prompt variant (a content hash), so an interrupted
or rate-limited run continues where it stopped, and editing a question re-runs only that question.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import yaml

from .prompts import ROOT, build_prompt
from .providers import Provider, ProviderError
from .scoring import score

log = logging.getLogger("sqleval")
QUESTIONS = ROOT / "benchmark" / "questions.yaml"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    sql: str
    tags: tuple[str, ...]
    ordered: bool


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = [
        Question(q["id"], q["question"].strip(), q["sql"].strip(), tuple(q.get("tags", [])), bool(q.get("ordered")))
        for q in raw["questions"]
    ]
    ids = [q.id for q in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids")
    return out


def config_name(provider: str, variant: str) -> str:
    return f"{provider}@{variant}"


def slug(config: str) -> str:
    return re.sub(r"[^A-Za-z0-9._@-]+", "_", config)


def qhash(q: Question, variant: str) -> str:
    return hashlib.sha1(f"{q.question}\n{q.sql}\n{q.ordered}\n{variant}".encode()).hexdigest()[:12]


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def save_results(path: Path, results: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(json.dumps(results[k], sort_keys=True, default=str) + "\n" for k in sorted(results)), encoding="utf-8"
    )
    tmp.replace(path)


def run_config(
    provider: Provider,
    variant: str,
    con: duckdb.DuckDBPyConnection,
    questions: list[Question],
    out_dir: Path,
    max_calls: int | None = None,
) -> dict[str, int]:
    config = config_name(provider.name, variant)
    path = out_dir / f"{slug(config)}.jsonl"
    results = load_results(path)
    stats = {"scored": 0, "skipped": 0, "provider_errors": 0}
    calls = 0
    for q in questions:
        h = qhash(q, variant)
        prev = results.get(q.id)
        if prev and prev.get("hash") == h and prev.get("category") != "provider_error":
            stats["skipped"] += 1
            continue
        if max_calls is not None and calls >= max_calls:
            break
        system, user = build_prompt(q.question, variant, con)
        calls += 1
        rec: dict[str, Any] = {"id": q.id, "hash": h, "config": config, "at": datetime.now(timezone.utc).isoformat()}
        try:
            reply = provider.complete(system, user, gold=q.sql)
        except ProviderError as error:
            rec.update({"category": "provider_error", "correct": False, "lenient": False, "detail": str(error)[:300]})
            results[q.id] = rec
            stats["provider_errors"] += 1
            log.warning("%s %s: provider error: %s", config, q.id, error)
            if "rate limited" in str(error) or error.fatal:
                break  # a quota is exhausted or the model is not available; resume on the next run
            continue
        sql, execution, verdict = score(con, q.sql, reply.text, q.ordered)
        rec.update(
            {
                "reply": reply.text[:4000],
                "sql": sql,
                "correct": verdict.correct,
                "lenient": verdict.lenient,
                "category": verdict.category,
                "detail": verdict.detail[:300],
                "latency_s": reply.latency_s,
                "tokens_in": reply.tokens_in,
                "tokens_out": reply.tokens_out,
                "rows": len(execution.rows) if execution.rows is not None else None,
            }
        )
        results[q.id] = rec
        stats["scored"] += 1
        save_results(path, results)  # save as we go: a cancelled job keeps its progress
        log.info("%s %s: %s", config, q.id, verdict.category)
    save_results(path, results)
    return stats
