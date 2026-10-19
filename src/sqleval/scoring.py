"""Score a generated query: is it safe, does it run, and does it return the same answer as the gold query?

Execution accuracy compares *results*, not SQL text, because many different queries are correct. Two results
match when they have the same rows, allowing for:

- column names and column order (a model may alias or reorder columns),
- row order, unless the question asks for an order,
- numeric precision: each predicted number is rounded to the precision the gold answer was rounded to, so
  "12.3" matches 12.3456 when the question asked for one decimal; integers must match exactly,
- numeric types (Decimal, float and int are compared as numbers) and dates vs ISO date strings.

A "lenient" match also accepts extra predicted columns as long as the gold columns are all present: useful for
spotting answers that are right but over-delivered.
"""

from __future__ import annotations

import datetime as dt
import decimal
import re
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

MAX_ROWS = 10_000
TIMEOUT_S = 20.0


@dataclass
class Execution:
    ok: bool
    rows: list[tuple[Any, ...]] | None = None
    columns: list[str] | None = None
    error: str | None = None
    error_kind: str | None = None  # unsafe | syntax | unknown_identifier | runtime | timeout | too_many_rows


@dataclass
class Verdict:
    correct: bool
    lenient: bool
    category: str  # correct | unsafe | syntax | unknown_identifier | runtime | timeout |
    # empty_result | wrong_shape | wrong_order | wrong_values |
    # extra_columns | no_sql
    detail: str = ""


# ------------------------------------------------------------------ extraction and safety
FENCE = re.compile(r"```(?:sql|duckdb|postgresql)?\s*(.*?)```", re.S | re.I)


def extract_sql(text: str) -> str:
    """The SQL in a model reply: the last fenced block if any, else the reply from the first SELECT/WITH."""
    if not text:
        return ""
    blocks = FENCE.findall(text)
    if blocks:
        return blocks[-1].strip().rstrip(";").strip()
    m = re.search(r"\b(WITH|SELECT)\b", text, re.I)
    if m:
        return text[m.start() :].strip().rstrip(";").strip()
    first = text.strip().split(None, 1)[0].upper() if text.strip() else ""
    if first in {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "ATTACH",
        "COPY",
        "PRAGMA",
        "INSTALL",
        "LOAD",
    }:
        return text.strip().rstrip(";")  # returned so the safety check can refuse it explicitly
    return ""


def check_safe(sql: str) -> str | None:
    """None if the query is a single read-only statement, else the reason it is refused."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as error:
        return f"syntax: {str(error).splitlines()[0][:200]}"
    except (RecursionError, sqlglot.errors.SqlglotError, ValueError, TypeError, IndexError) as error:
        # garbage from small models can drive the parser into deep recursion or tokenizer errors
        return f"syntax: could not parse ({type(error).__name__})"
    if len(statements) != 1:
        return f"unsafe: expected one statement, got {len(statements)}"
    st = statements[0]
    if not isinstance(st, exp.Query):
        return f"unsafe: {type(st).__name__} statements are not allowed"
    banned = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Command, exp.Merge)
    if any(st.find(b) for b in banned):
        return "unsafe: data-changing statement inside the query"
    for fn in st.find_all(exp.Func):
        name = (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()
        if name.startswith(("read_", "readcsv", "readparquet")) or name in {"glob", "getenv", "sniff_csv"}:
            return f"unsafe: function {name} reads outside the warehouse"
    return None


# ------------------------------------------------------------------ execution
def connect(path: str) -> duckdb.DuckDBPyConnection:
    """The warehouse opened read-only with file and network access disabled: the second line of defence."""
    return duckdb.connect(path, read_only=True, config={"enable_external_access": False})


def execute(con: duckdb.DuckDBPyConnection, sql: str, timeout: float = TIMEOUT_S) -> Execution:
    reason = check_safe(sql)
    if reason:
        kind = reason.split(":", 1)[0]
        return Execution(False, error=reason, error_kind=kind)
    result: dict[str, Any] = {}
    cur = con.cursor()

    def run() -> None:
        try:
            rel = cur.execute(sql)
            cols = [d[0] for d in (rel.description or [])]
            rows = rel.fetchmany(MAX_ROWS + 1)
            result["out"] = (cols, rows)
        except Exception as error:  # DuckDB raises many exception types
            result["err"] = error

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        cur.interrupt()
        t.join(10)
        return Execution(False, error=f"query took longer than {timeout:.0f}s", error_kind="timeout")
    if "err" in result:
        msg = str(result["err"]).splitlines()[0][:300]
        kind = (
            "unknown_identifier"
            if re.search(r"Binder Error|Catalog Error|not found|does not exist", msg)
            else "syntax"
            if "Parser Error" in msg
            else "runtime"
        )
        return Execution(False, error=msg, error_kind=kind)
    cols, rows = result["out"]
    if len(rows) > MAX_ROWS:
        return Execution(False, error=f"more than {MAX_ROWS:,} rows", error_kind="too_many_rows")
    return Execution(True, rows=[tuple(r) for r in rows], columns=cols)


# ------------------------------------------------------------------ comparison
def _decimals(v: Any) -> int | None:
    """How many decimals the gold value was rounded to (None for integers)."""
    if isinstance(v, (bool, int)):
        return None
    if isinstance(v, decimal.Decimal):
        if v == v.to_integral_value() and v.as_tuple().exponent >= 0:  # type: ignore[operator]
            return None
        s = format(v.normalize(), "f")
    elif isinstance(v, float):
        s = repr(v)
        if "e" in s:
            return 6
    else:
        return None
    return min(len(s.split(".")[1]) if "." in s else 0, 6)


def _norm(v: Any, places: int | None) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, decimal.Decimal)):
        f = float(v)
        if places is None:
            return round(f, 6) if not float(f).is_integer() else int(f)
        return round(f + 0.0, places)
    if isinstance(v, dt.datetime):
        return v.isoformat(sep=" ")
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return _norm(decimal.Decimal(s), places)
        return s
    return str(v)


def _places(gold: list[tuple[Any, ...]], ncols: int) -> list[int | None]:
    """Per gold column, the largest number of decimals any gold value used (integers stay exact)."""
    out: list[int | None] = []
    for j in range(ncols):
        ds = [_decimals(r[j]) for r in gold if r[j] is not None]
        nums = [d for d in ds if d is not None]
        out.append(max(nums) if nums else None)
    return out


def _canon(rows: list[tuple[Any, ...]], places: list[int | None]) -> list[tuple[Any, ...]]:
    return [tuple(_norm(v, places[j] if j < len(places) else None) for j, v in enumerate(r)) for r in rows]


def compare(gold: Execution, pred: Execution, ordered: bool) -> Verdict:
    if not gold.ok or gold.rows is None:
        raise ValueError(f"the gold query failed: {gold.error}")
    if not pred.ok or pred.rows is None:
        return Verdict(False, False, pred.error_kind or "runtime", pred.error or "")
    g_rows, p_rows = gold.rows, pred.rows
    if g_rows and not p_rows:
        return Verdict(False, False, "empty_result", "the query returned no rows")
    ncols = len(gold.columns or [])
    p_width = len(p_rows[0]) if p_rows else ncols
    if len(p_rows) != len(g_rows):
        return Verdict(False, False, "wrong_shape", f"{len(p_rows):,} rows, expected {len(g_rows):,}")
    places = _places(g_rows, ncols)
    g = _canon(g_rows, places)
    mapping = _match_with_precision(g, p_rows, places)
    if mapping is None:
        if p_width < ncols:
            return Verdict(False, False, "wrong_shape", f"{p_width} columns, expected {ncols}")
        return Verdict(False, False, "wrong_values", "same shape, different values")
    p = [tuple(row[k] for k in mapping) for row in _canon_cols(p_rows, mapping, places)]
    if Counter(p) != Counter(g):
        return Verdict(False, False, "wrong_values", "each column matches, but the rows pair up differently")
    if ordered and p != g:
        return Verdict(False, False, "wrong_order", "right rows in the wrong order")
    if p_width != ncols:
        return Verdict(False, True, "extra_columns", f"{p_width} columns, expected {ncols}")
    return Verdict(True, True, "correct")


def _canon_cols(p_rows: list[tuple[Any, ...]], mapping: list[int], places: list[int | None]) -> list[tuple[Any, ...]]:
    """Predicted rows with each matched column rounded to its gold column's precision."""
    width = len(p_rows[0]) if p_rows else 0
    col_places: list[int | None] = [None] * width
    for gj, pk in enumerate(mapping):
        col_places[pk] = places[gj]
    return [tuple(_norm(v, col_places[k]) for k, v in enumerate(r)) for r in p_rows]


def _match_with_precision(
    g: list[tuple[Any, ...]], p_rows: list[tuple[Any, ...]], places: list[int | None]
) -> list[int] | None:
    if not p_rows:
        return []
    width = len(p_rows[0])
    g_cols = list(zip(*g, strict=True)) if g else []
    used: set[int] = set()
    mapping = []
    for gj, gc in enumerate(g_cols):
        target = Counter(gc)
        found = None
        for k in range(width):
            if k in used:
                continue
            col = [_norm(r[k], places[gj]) for r in p_rows]
            if Counter(col) == target:
                found = k
                break
        if found is None:
            return None
        used.add(found)
        mapping.append(found)
    return mapping


def score(con: duckdb.DuckDBPyConnection, gold_sql: str, reply: str, ordered: bool) -> tuple[str, Execution, Verdict]:
    sql = extract_sql(reply)
    if not sql:
        return (
            "",
            Execution(False, error="no SQL in the reply", error_kind="no_sql"),
            Verdict(False, False, "no_sql", "no SQL in the reply"),
        )
    gold = execute(con, gold_sql)
    pred = execute(con, sql)
    return sql, pred, compare(gold, pred, ordered)
