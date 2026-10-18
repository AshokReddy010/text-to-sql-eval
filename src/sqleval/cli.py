"""sqleval: build the warehouse, run models on the benchmark, and publish the report.

sqleval build                                    create build/warehouse.duckdb
sqleval run --model github/openai/gpt-4.1-mini --variant context
sqleval run --model auto --variant context       every preferred GitHub Models model that is available
sqleval run --model gold                         harness self-check (must score 100%)
sqleval report                                   results/runs/*.jsonl -> results/report.json
sqleval ask "How many orders were cancelled?" --model github/openai/gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .prompts import VARIANTS, build_prompt
from .providers import auto_models, get_provider
from .report import build_report
from .runner import load_questions, run_config
from .scoring import connect, execute, extract_sql
from .warehouse import build

DB = Path("build/warehouse.duckdb")
RUNS = Path("results/runs")


def _ensure_db(path: Path) -> None:
    if not path.exists():
        print(f"building {path} ...", file=sys.stderr)
        build(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sqleval", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--db", type=Path, default=DB)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--model", action="append", required=True, help="provider spec, 'auto' or 'gold'; repeatable")
    r.add_argument("--variant", action="append", choices=VARIANTS, help="prompt variant(s); default context")
    r.add_argument("--only", nargs="*", help="question ids to run")
    r.add_argument("--max-calls", type=int, help="stop after this many model calls per configuration")
    r.add_argument("--out", type=Path, default=RUNS)
    rep = sub.add_parser("report")
    rep.add_argument("--runs", type=Path, default=RUNS)
    rep.add_argument("-o", "--output", type=Path, default=Path("results/report.json"))
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--model", required=True)
    a.add_argument("--variant", choices=VARIANTS, default="context")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "build":
        print(build(args.db))
        return 0
    _ensure_db(args.db)
    con = connect(str(args.db))
    if args.cmd == "run":
        qs = load_questions()
        if args.only:
            qs = [q for q in qs if q.id in set(args.only)]
        models: list[str] = []
        for m in args.model:
            models += auto_models() if m == "auto" else [m]
        for m in models:
            for v in args.variant or ["context"]:
                stats = run_config(get_provider(m), v, con, qs, args.out, args.max_calls)
                print(f"{m}@{v}: {stats}")
        return 0
    if args.cmd == "report":
        rep_doc = build_report(args.runs, load_questions())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rep_doc, indent=1, default=str), encoding="utf-8")
        for c in rep_doc["configs"]:
            print(f"{c['config']:<60} {c['accuracy']:.0%}  ({c['correct']}/{c['answered']})")
        return 0
    if args.cmd == "ask":
        system, user = build_prompt(args.question, args.variant, con)
        reply = get_provider(args.model).complete(system, user)
        sql = extract_sql(reply.text)
        print(sql)
        ex = execute(con, sql)
        print(ex.rows[:20] if ex.ok and ex.rows is not None else f"error: {ex.error}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
