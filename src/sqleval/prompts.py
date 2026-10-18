"""The prompts. Two variants, so the benchmark also measures how much business context helps:

schema      the DDL only: what a model sees when pointed at a raw warehouse
context     the DDL, three sample rows per table, and the team's business glossary
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from .warehouse import SCHEMA


def _root() -> Path:
    """The repository root: the benchmark files live there, not inside the installed package."""
    for base in (Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]):
        if (base / "benchmark" / "questions.yaml").exists():
            return base
    raise FileNotFoundError("benchmark/questions.yaml not found; run sqleval from the repository")


ROOT = _root()
GLOSSARY = ROOT / "benchmark" / "glossary.md"
VARIANTS = ("schema", "context")

SYSTEM = (
    "You are a senior analytics engineer. Translate the question into one DuckDB SQL query over the warehouse "
    "described below. Reply with the query in a single ```sql code block and nothing else."
)


def samples(con: duckdb.DuckDBPyConnection, n: int = 3) -> str:
    out = []
    for (table,) in con.execute("SELECT table_name FROM information_schema.tables ORDER BY table_name").fetchall():
        rel = con.execute(f"SELECT * FROM {table} ORDER BY 1 LIMIT {n}")
        cols = [d[0] for d in rel.description]
        rows = rel.fetchall()
        out.append(f"-- {table}\n" + " | ".join(cols) + "\n" + "\n".join(" | ".join(str(v) for v in r) for r in rows))
    return "\n\n".join(out)


def build_prompt(question: str, variant: str, con: duckdb.DuckDBPyConnection | None = None) -> tuple[str, str]:
    """(system, user) messages for one question."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant}")
    parts = ["## Schema (DuckDB)", SCHEMA.strip()]
    if variant == "context":
        if con is None:
            raise ValueError("the context variant needs a warehouse connection for sample rows")
        parts += ["## Sample rows", samples(con), "## Business glossary", GLOSSARY.read_text(encoding="utf-8").strip()]
    parts += ["## Question", question.strip()]
    return SYSTEM, "\n\n".join(parts)
