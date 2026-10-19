import io
import json
import urllib.error

import pytest

from sqleval import providers
from sqleval.cli import main
from sqleval.providers import Gold, Provider, ProviderError, Reply, get_provider
from sqleval.report import build_report, wilson
from sqleval.runner import load_questions, load_results, run_config


class Scripted(Provider):
    """Answers with the gold query, except for scripted failures."""

    def __init__(self, name, wrong=(), errors=(), rate_limit_at=None):
        super().__init__(name)
        self.wrong, self.errors, self.rate_limit_at, self.calls = set(wrong), set(errors), rate_limit_at, 0

    def complete(self, system, user, gold=""):
        self.calls += 1
        if self.rate_limit_at and self.calls >= self.rate_limit_at:
            raise ProviderError("rate limited for 86400s: daily quota")
        qid = next((q.id for q in load_questions() if q.question in user), "")
        if qid in self.errors:
            raise ProviderError("HTTP 500: boom")
        if qid in self.wrong:
            return Reply("```sql\nSELECT 42\n```", 0.5, 1000, 20)
        return Reply(f"```sql\n{gold}\n```", 0.4, 1200, 60)


def test_run_scores_and_resumes(con, tmp_path):
    qs = load_questions()[:6]
    p = Scripted("fake/m1", wrong={"q02"}, errors={"q03"})
    stats = run_config(p, "context", con, qs, tmp_path)
    assert stats == {"scored": 5, "skipped": 0, "provider_errors": 1}
    res = load_results(tmp_path / "fake_m1@context.jsonl")
    assert res["q01"]["correct"] and not res["q02"]["correct"] and res["q03"]["category"] == "provider_error"
    p2 = Scripted("fake/m1")
    stats = run_config(p2, "context", con, qs, tmp_path)  # only the provider error is retried
    assert stats == {"scored": 1, "skipped": 5, "provider_errors": 0} and p2.calls == 1


def test_rate_limit_stops_the_run(con, tmp_path):
    p = Scripted("fake/m2", rate_limit_at=3)
    stats = run_config(p, "schema", con, load_questions()[:10], tmp_path)
    assert stats["scored"] == 2 and stats["provider_errors"] == 1 and p.calls == 3


def test_max_calls(con, tmp_path):
    assert run_config(Scripted("fake/m3"), "schema", con, load_questions()[:10], tmp_path, max_calls=4)["scored"] == 4


def test_report(con, tmp_path):
    qs = load_questions()[:8]
    run_config(Scripted("fake/a", wrong={"q01", "q02"}), "context", con, qs, tmp_path)
    run_config(Scripted("fake/a", wrong={"q01", "q02", "q03", "q04"}), "schema", con, qs, tmp_path)
    run_config(Gold("gold"), "context", con, qs, tmp_path)
    rep = build_report(tmp_path, qs)
    assert [c["config"] for c in rep["configs"]] == ["fake/a@context", "fake/a@schema"]  # gold excluded
    top = rep["configs"][0]
    assert top["accuracy"] == 0.75 and top["failures"] == {"wrong_values": 2} and top["median_latency_s"] == 0.4
    assert rep["ablation"] == [{"provider": "fake/a", "schema": 0.5, "context": 0.75, "delta": 0.25}]
    q1 = next(q for q in rep["questions"] if q["id"] == "q01")
    assert q1["solved_by"] == 0 and q1["attempted_by"] == 2


def test_wilson():
    lo, hi = wilson(40, 50)
    assert 0.66 < lo < 0.7 and 0.88 < hi < 0.9
    assert wilson(0, 0) == (0.0, 0.0)


def test_provider_specs(monkeypatch):
    assert get_provider("github/openai/gpt-4.1-mini").min_interval_s == 4.5
    assert get_provider("anthropic/claude-x").name == "anthropic/claude-x"
    with pytest.raises(ValueError):
        get_provider("nope/model")
    with pytest.raises(ValueError):
        get_provider("github")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        get_provider("groq/llama").complete("s", "u")


def test_openai_compatible_call_and_retry(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    calls = []

    def urlopen(req, timeout):
        calls.append(json.loads(req.data))
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "slow down", {"Retry-After": "1"}, io.BytesIO(b"{}"))
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [{"message": {"content": "```sql\nSELECT 1\n```"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ).encode()
        )

    monkeypatch.setattr(providers._OPENER, "open", urlopen)
    r = get_provider("github/openai/gpt-4.1-mini").complete("sys", "user")
    assert r.text.startswith("```sql") and r.tokens_in == 10 and len(calls) == 2
    assert calls[0]["temperature"] == 0 and calls[0]["model"] == "openai/gpt-4.1-mini"


def test_daily_quota_is_not_waited_out(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setattr(
        providers._OPENER,
        "open",
        lambda req, timeout: (_ for _ in ()).throw(
            urllib.error.HTTPError(req.full_url, 429, "quota", {"Retry-After": "86400"}, io.BytesIO(b"daily"))
        ),
    )
    with pytest.raises(ProviderError, match="rate limited"):
        get_provider("github/x/y").complete("s", "u")


def test_auto_models_falls_back_when_catalog_fails(monkeypatch):
    monkeypatch.setattr(providers, "github_catalog", lambda: (_ for _ in ()).throw(ValueError("empty body")))
    assert providers.auto_models(limit=2) == [f"github/{m}" for m in providers.GITHUB_PREFERRED[:2]]


def test_unknown_model_stops_the_config(con, tmp_path):
    class Unknown(Provider):
        calls = 0

        def complete(self, system, user, gold=""):
            Unknown.calls += 1
            raise ProviderError("HTTP 404: unknown model", fatal=True)

    stats = run_config(Unknown("github/x/nope"), "context", con, load_questions()[:5], tmp_path)
    assert stats["provider_errors"] == 1 and Unknown.calls == 1


def test_auto_models(monkeypatch):
    monkeypatch.setattr(providers, "github_catalog", lambda: ["openai/gpt-4.1-mini", "microsoft/Phi-4", "other/x"])
    assert providers.auto_models() == ["github/openai/gpt-4.1-mini", "github/microsoft/Phi-4"]


def test_cli_gold_run_and_report(db_path, tmp_path, capsys):
    out = tmp_path / "runs"
    assert main(["--db", str(db_path), "run", "--model", "gold", "--only", "q01", "q02", "--out", str(out)]) == 0
    assert main(["--db", str(db_path), "report", "--runs", str(out), "-o", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["questions_total"] == 50


def test_empty_response_falls_back_to_a_configured_endpoint(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    seen = []

    def open_(req, timeout):
        seen.append((req.full_url, json.loads(req.data)["model"]))
        if "models.github.ai" in req.full_url:
            return io.BytesIO(b"")
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "SELECT 1"}}]}).encode())

    class R(io.BytesIO):
        status, headers = 200, {"Content-Type": "text/plain"}

    monkeypatch.setattr(providers._OPENER, "open", lambda req, timeout: R(open_(req, timeout).getvalue()))
    p = get_provider("github/openai/gpt-4.1-mini")
    p.fallback = (providers.GITHUB_LEGACY_URL, "gpt-4.1-mini")
    assert p.complete("s", "u").text == "SELECT 1"
    assert seen == [(providers.GITHUB_URL, "openai/gpt-4.1-mini"), (providers.GITHUB_LEGACY_URL, "gpt-4.1-mini")]
    p.complete("s", "u")
    assert seen[-1][0] == providers.GITHUB_LEGACY_URL  # stays on the endpoint that works


def test_requests_carry_a_user_agent(monkeypatch):
    # Groq sits behind Cloudflare, which rejects urllib's default User-Agent with "error code: 1010"
    monkeypatch.setenv("GROQ_API_KEY", "k")
    seen = []

    class R(io.BytesIO):
        status = 200
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def open_(req, timeout):
        seen.append(req.get_header("User-agent"))
        return R(json.dumps({"choices": [{"message": {"content": "SELECT 1"}}]}).encode())

    monkeypatch.setattr(providers._OPENER, "open", open_)
    get_provider("groq/llama-3.3-70b-versatile").complete("s", "u")
    assert seen and seen[0].startswith("sqleval/")
