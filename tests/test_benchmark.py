"""The benchmark itself: deterministic data, valid questions, gold queries that run and answer non-trivially."""

from sqleval.prompts import build_prompt
from sqleval.runner import load_questions
from sqleval.scoring import execute, score

EXPECTED = {
    "customers": 3000,
    "products": 114,
    "orders": 32079,
    "order_items": 62586,
    "payments": 31311,
    "returns": 2048,
}


def test_warehouse_is_deterministic(con):
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in EXPECTED}
    assert counts == EXPECTED


def test_warehouse_has_the_traps_questions_rely_on(con):
    q = lambda s: con.execute(s).fetchone()[0]  # noqa: E731
    assert q("SELECT count(*) FROM orders WHERE status = 'cancelled'") > 1000
    assert (
        q(
            "SELECT count(*) FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id)"
        )
        > 100
    )
    assert q("SELECT count(*) FROM orders WHERE status = 'cancelled' AND ship_date IS NOT NULL") == 0
    assert q("SELECT count(*) FROM (SELECT order_id FROM payments GROUP BY 1 HAVING count(*) > 1)") > 100
    assert q("SELECT max(order_date) FROM orders").isoformat() == "2026-06-30"


def test_fifty_unique_questions_with_tags():
    qs = load_questions()
    assert len(qs) == 50 and len({q.id for q in qs}) == 50
    assert all(q.tags for q in qs)
    assert sum(q.ordered for q in qs) >= 10


def test_every_gold_query_runs_and_matches_itself(con):
    for q in load_questions():
        gold = execute(con, q.sql)
        assert gold.ok, (q.id, gold.error)
        assert gold.rows, q.id
        assert score(con, q.sql, q.sql, q.ordered)[2].correct, q.id


def test_ordered_questions_say_so():
    for q in load_questions():
        if q.ordered:
            assert any(w in q.question.lower() for w in ("order", "first", "sorted", "top")), q.id


def test_prompt_variants(con):
    s, schema_only = build_prompt("How many orders?", "schema")
    _, context = build_prompt("How many orders?", "context", con)
    assert "```sql" in s and "CREATE TABLE orders" in schema_only and "glossary" not in schema_only.lower()
    assert "Business glossary" in context and "-- orders" in context and context.endswith("How many orders?")
