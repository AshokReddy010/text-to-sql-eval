import datetime as dt
from decimal import Decimal

import pytest

from sqleval.scoring import Execution, check_safe, compare, execute, extract_sql, score


def ex(rows, cols=None):
    return Execution(True, rows=rows, columns=cols or [f"c{i}" for i in range(len(rows[0]) if rows else 1)])


@pytest.mark.parametrize(
    "reply, sql",
    [
        ("```sql\nSELECT 1;\n```", "SELECT 1"),
        ("Here you go:\n```SQL\nSELECT 2\n```\nand a better one\n```sql\nSELECT 3\n```", "SELECT 3"),
        ("The answer is SELECT a FROM t;", "SELECT a FROM t"),
        ("WITH x AS (SELECT 1) SELECT * FROM x", "WITH x AS (SELECT 1) SELECT * FROM x"),
        ("I cannot answer that.", ""),
        ("", ""),
        ("DROP TABLE orders;", "DROP TABLE orders"),
    ],
)
def test_extract_sql(reply, sql):
    assert extract_sql(reply) == sql


@pytest.mark.parametrize(
    "sql, fragment",
    [
        ("DELETE FROM orders", "unsafe"),
        ("SELECT 1; SELECT 2", "one statement"),
        ("CREATE TABLE x AS SELECT 1", "unsafe"),
        ("SELECT * FROM read_parquet('x.parquet')", "read_parquet"),
        ("SELEC 1 FROM", "syntax"),
        ("UPDATE orders SET status = 'x'", "unsafe"),
    ],
)
def test_unsafe_or_invalid_sql_is_refused(sql, fragment):
    assert fragment in (check_safe(sql) or "")


def test_safe_sql_passes():
    assert check_safe("WITH a AS (SELECT 1 AS x) SELECT x FROM a UNION ALL SELECT 2") is None


def test_execute_classifies_errors(con):
    assert execute(con, "SELECT nope FROM orders").error_kind == "unknown_identifier"
    assert execute(con, "SELECT * FROM nowhere").error_kind == "unknown_identifier"
    assert execute(con, "SELECT 1/0::INTEGER + 'a'::INTEGER").error_kind == "runtime"
    assert execute(con, "SELECT * FROM order_items").error_kind == "too_many_rows"
    assert execute(con, "SELECT * FROM '/etc/hostname'").ok is False


def test_execute_times_out(con):
    slow = "SELECT count(*) FROM range(100000000) a, range(100000) b"
    r = execute(con, slow, timeout=0.5)
    assert r.error_kind == "timeout"
    assert execute(con, "SELECT 1").ok  # connection still usable


def test_compare_tolerates_names_order_types_and_rounding():
    gold = ex([("West", Decimal("12.3")), ("South", Decimal("7.5"))], ["region", "pct"])
    assert compare(gold, ex([(12.3456, "West"), (7.4999, "South")]), ordered=False).correct
    assert compare(gold, ex([("South", 7.5), ("West", 12.3)]), ordered=False).correct
    v = compare(gold, ex([("South", 7.5), ("West", 12.3)]), ordered=True)
    assert not v.correct and v.category == "wrong_order"


def test_compare_integers_are_exact_and_strings_of_numbers_match():
    assert not compare(ex([(1045,)]), ex([(1046,)]), False).correct
    assert compare(ex([(1045,)]), ex([("1045",)]), False).correct
    assert compare(ex([(Decimal("7822844.2495"),)]), ex([(7822844.2495,)]), False).correct
    assert not compare(ex([(Decimal("12.3"),)]), ex([(0.123,)]), False).correct  # fraction vs percent


def test_compare_dates_and_rows_pairing():
    assert compare(ex([(dt.date(2026, 1, 1), 3)]), ex([("2026-01-01", 3)]), False).correct
    gold = ex([("a", 1), ("b", 2)])
    v = compare(gold, ex([("a", 2), ("b", 1)]), False)  # columns match as sets, rows do not
    assert v.category == "wrong_values"


def test_compare_shapes():
    gold = ex([("a", 1), ("b", 2)])
    assert compare(gold, ex([("a", 1)]), False).category == "wrong_shape"
    assert compare(gold, ex([("a",), ("b",)]), False).category == "wrong_shape"
    v = compare(gold, ex([("a", 1, "x"), ("b", 2, "y")]), False)
    assert v.category == "extra_columns" and v.lenient and not v.correct
    assert compare(gold, Execution(True, rows=[], columns=["a"]), False).category == "empty_result"
    assert compare(ex([]), Execution(True, rows=[], columns=["a"]), False).correct
    with pytest.raises(ValueError):
        compare(Execution(False, error="boom"), gold, False)


def test_score_end_to_end(con):
    gold = "SELECT status, count(*) FROM orders GROUP BY status"
    _, _, v = score(con, gold, "```sql\nSELECT count(*) AS n, status FROM orders GROUP BY 2\n```", False)
    assert v.correct
    _, _, v = score(
        con, gold, "```sql\nSELECT status, count(*) FROM orders WHERE status <> 'cancelled' GROUP BY 1\n```", False
    )
    assert v.category == "wrong_shape"
    assert score(con, gold, "no idea", False)[2].category == "no_sql"


def test_pathological_sql_is_a_syntax_miss_not_a_crash():
    from sqleval.scoring import check_safe

    reason = check_safe("SELECT " + "(" * 5000 + "1" + ")" * 5000)
    assert reason is not None and reason.startswith("syntax")
