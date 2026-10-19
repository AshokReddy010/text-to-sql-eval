# SQL Bench: text-to-SQL evaluated by answers, not by looks

[![ci](https://github.com/AshokReddy010/text-to-sql-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/AshokReddy010/text-to-sql-eval/actions/workflows/ci.yml)
[![eval](https://github.com/AshokReddy010/text-to-sql-eval/actions/workflows/eval.yml/badge.svg)](https://github.com/AshokReddy010/text-to-sql-eval/actions/workflows/eval.yml)

**▶ [Live leaderboard](https://ashokreddy010.github.io/text-to-sql-eval/)**: refreshed daily from the results committed in [`results/`](results/).

"Ask your data in plain English" demos look great on `SELECT count(*)`. Real analyst questions have definitions hidden in them: revenue excludes cancelled orders, an active customer means something specific, a growth rate needs last month's number. This benchmark asks models 50 such questions over a realistic warehouse and counts an answer as correct only if the query **returns the right result**.

## What is in it

| | |
|---|---|
| **Warehouse** | A retail business in DuckDB: `customers`, `products`, `orders`, `order_items`, `payments`, `returns` (~32k orders, 63k lines). Generated from a fixed seed, so it is reproducible and free to share. It has the traps real warehouses have: 6% cancelled orders, partial refunds, split-tender payments, customers who never ordered, nullable referral and promo columns, weekend and holiday seasonality. |
| **Questions** | [50 business questions](benchmark/questions.yaml), each with a hand-written gold query and tags: basic, aggregation, joins, missing rows (anti-joins), window functions, dates, NULLs, top-N, and business rules that need the [glossary](benchmark/glossary.md). Thirteen require a specific row order. |
| **Prompts** | `schema` (table definitions only) and `context` (plus three sample rows per table and the business glossary), to measure what context is worth. |
| **Models** | Any OpenAI-compatible endpoint, Anthropic, Groq and GitHub Models. The daily workflow uses Groq's free tier through a `GROQ_API_KEY` repository secret. `groq/auto:5` asks Groq which chat models it serves that day and picks the five best families, so retired models drop out on their own. Without the secret it falls back to GitHub Models where that is enabled. Keys only ever come from the environment. |

## How answers are scored

`sqleval.scoring` runs both queries and compares results, allowing what does not change the answer and nothing else:

- column names and column order may differ; row order may differ unless the question asks for one;
- each predicted number is rounded to the precision the gold answer used ("rounded to 1 decimal" accepts 12.3456 for 12.3), integers must match exactly, and Decimal/float/int/strings of digits compare as numbers;
- dates match ISO strings.

A *lenient* score also accepts correct answers with extra columns. Every miss is classified by running the query: wrong values, wrong shape, wrong order, empty result, made-up table or column, invalid SQL, runtime error, timeout, refused as unsafe, or no SQL at all.

**Safety.** Generated SQL is untrusted. It must parse (sqlglot) as exactly one read-only query; data-changing statements and file-reading functions are refused before execution; the warehouse is opened read-only with DuckDB's external access disabled; every query has a 20-second timeout and a 10,000-row cap.

**Honest numbers.** Accuracy is shown with a 95% Wilson interval. With 50 questions an interval is roughly ±12 points, so the leaderboard says when two models are not distinguishable instead of ranking noise.

**Harness check.** Every run first feeds the gold queries through the full pipeline (prompt → extraction → safety → execution → comparison); anything below 50/50 fails the job.

## Running it

```bash
pip install -e ".[dev]"
sqleval build                                            # build/warehouse.duckdb in ~2 s
sqleval run --model gold                                 # harness self-check: 50/50
export GROQ_API_KEY=...                                  # keys only ever come from the environment
sqleval run --model groq/llama-3.3-70b-versatile --variant context --variant schema
sqleval report                                           # results/report.json
sqleval ask "Which region had the most revenue last year?" --model groq/llama-3.3-70b-versatile
```

Runs are resumable: each answer is saved as it is scored and keyed by a hash of the question, gold query and prompt variant. A run cut short by a rate limit continues where it stopped, and editing one question re-runs only that question. This is what makes free-tier daily quotas workable: the workflow just picks up the rest the next day.

## Layout

```
benchmark/questions.yaml   50 questions, gold SQL, tags, order flags
benchmark/glossary.md      the business definitions the context prompt includes
src/sqleval/
  warehouse.py   deterministic warehouse generator
  prompts.py     schema and context prompt variants
  providers.py   GitHub Models, OpenAI, Groq, Anthropic, gold; retries and pacing
  scoring.py     extraction, safety, sandboxed execution, result comparison, error classes
  runner.py      resumable runs -> results/runs/<model@variant>.jsonl
  report.py      accuracy, Wilson intervals, per-tag accuracy, failure mix, ablation
.github/workflows/eval.yml   daily run (Groq free tier or GitHub Models), commits results
site/index.html              the leaderboard
tests/                       scorer edge cases, gold self-check, resumability, provider retries
```

## Limits

Fifty questions over one schema is a focused benchmark, not a general ranking of models: it measures business-analytics SQL in DuckDB's dialect with the context given here. The warehouse is synthetic. Free-tier models have daily token limits: a model that hits its quota shows as partially answered until a later run finishes it.

## License

MIT
