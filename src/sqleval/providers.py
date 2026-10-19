"""Model providers behind one small interface. API keys come from environment variables only.

    github/<model>      GitHub Models (free tier, works inside GitHub Actions with the job's token)
    openai/<model>      OPENAI_API_KEY
    groq/<model>        GROQ_API_KEY
    anthropic/<model>   ANTHROPIC_API_KEY
    gold                returns the gold query (checks the harness end to end; always scores 100%)

Every call returns the reply text, latency and token counts. HTTP 429 and 5xx are retried with backoff,
honouring Retry-After; a provider's minimum interval keeps requests under free-tier rate limits.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class ProviderError(RuntimeError):
    """The provider could not produce a reply (after retries). Recorded and retried on the next run.

    `fatal` marks errors that will repeat for every question (unknown model, no access), so the run moves on.
    """

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


@dataclass
class Reply:
    text: str
    latency_s: float
    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass
class Provider:
    name: str  # e.g. "github/openai/gpt-4.1-mini"
    min_interval_s: float = 0.0
    _last: float = field(default=0.0, repr=False)

    def complete(self, system: str, user: str, gold: str = "") -> Reply:
        raise NotImplementedError

    def _pace(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirected POST silently becomes a GET with no body; fail loudly instead."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise ProviderError(f"HTTP {code} redirect from {req.full_url} to {newurl}", fatal=True)


_OPENER = urllib.request.build_opener(_NoRedirect)
USER_AGENT = "sqleval/0.1 (+https://github.com/AshokReddy010/text-to-sql-eval)"


def _post(url: str, payload: dict[str, Any], headers: dict[str, str], retries: int = 4, timeout: float = 90) -> Any:
    body = json.dumps(payload).encode()
    last = ""
    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            # Cloudflare-fronted APIs (Groq) refuse urllib's default User-Agent with "error code: 1010"
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers},
        )
        try:
            with _OPENER.open(req, timeout=timeout) as r:
                raw = r.read().decode(errors="replace")
                if not raw.strip():
                    raise ProviderError(
                        f"empty response (HTTP {r.status}, {r.headers.get('Content-Type')}) from {url}", fatal=True
                    )
                if raw.strip() == "OK" and "models.github.ai" in url:
                    raise ProviderError(
                        "GitHub Models answered 'OK' instead of a completion: it is not enabled for this account",
                        fatal=True,
                    )
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ProviderError(f"non-JSON response from {url}: {raw[:200]}", fatal=True) from error
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            last = f"HTTP {error.code}: {detail}"
            if error.code == 429 or error.code >= 500:
                retry_after = error.headers.get("Retry-After") if error.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt * 3
                if delay > 90:  # a daily quota: give up now, the next run resumes
                    raise ProviderError(f"rate limited for {delay:.0f}s: {detail}") from error
                time.sleep(delay)
                continue
            raise ProviderError(last, fatal=error.code in (400, 401, 403, 404)) from error
        except (urllib.error.URLError, TimeoutError) as error:
            last = f"{type(error).__name__}: {error}"
            time.sleep(2**attempt)
    raise ProviderError(last)


@dataclass
class OpenAICompatible(Provider):
    url: str = ""
    model: str = ""
    key_env: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)
    fallback: tuple[str, str] | None = None  # (url, model) tried once if the primary endpoint is unusable

    def complete(self, system: str, user: str, gold: str = "") -> Reply:
        key = os.environ.get(self.key_env, "")
        if not key:
            raise ProviderError(f"{self.key_env} is not set")
        self._pace()
        try:
            return self._call(key, system, user)
        except ProviderError as error:
            if not (error.fatal and self.fallback):
                raise
            self.url, self.model = self.fallback  # switch for the rest of the run
            self.fallback = None
            return self._call(key, system, user)

    def _call(self, key: str, system: str, user: str) -> Reply:
        t = time.monotonic()
        data = _post(
            self.url,
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 2000,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            {"Authorization": f"Bearer {key}", **self.extra_headers},
        )
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f"unexpected response: {str(data)[:200]}") from error
        usage = data.get("usage") or {}
        return Reply(text, round(time.monotonic() - t, 3), usage.get("prompt_tokens"), usage.get("completion_tokens"))


@dataclass
class Anthropic(Provider):
    model: str = ""

    def complete(self, system: str, user: str, gold: str = "") -> Reply:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self._pace()
        t = time.monotonic()
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {
                "model": self.model,
                "max_tokens": 2000,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return Reply(text, round(time.monotonic() - t, 3), usage.get("input_tokens"), usage.get("output_tokens"))


@dataclass
class Gold(Provider):
    def complete(self, system: str, user: str, gold: str = "") -> Reply:
        return Reply(f"```sql\n{gold}\n```", 0.0, 0, 0)


GITHUB_URL = "https://models.github.ai/inference/chat/completions"
GITHUB_LEGACY_URL = "https://models.inference.ai.azure.com/chat/completions"  # older endpoint, model names unprefixed


def get_provider(spec: str) -> Provider:
    kind, _, model = spec.partition("/")
    if kind == "gold":
        return Gold("gold")
    if not model:
        raise ValueError(f"provider spec must look like kind/model, got {spec!r}")
    if kind == "github":
        # free tier: about 15 requests a minute for most models
        return OpenAICompatible(
            spec,
            4.5,
            url=GITHUB_URL,
            model=model,
            key_env="GITHUB_TOKEN",
            extra_headers={"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
        )
    if kind == "openai":
        return OpenAICompatible(
            spec, 0.0, url="https://api.openai.com/v1/chat/completions", model=model, key_env="OPENAI_API_KEY"
        )
    if kind == "groq":
        return OpenAICompatible(
            spec, 2.0, url="https://api.groq.com/openai/v1/chat/completions", model=model, key_env="GROQ_API_KEY"
        )
    if kind == "anthropic":
        return Anthropic(spec, 0.0, model=model)
    raise ValueError(f"unknown provider {kind!r}")


# Models tried, in order, when the catalog is asked for "auto". Only those GitHub currently serves are used.
GITHUB_PREFERRED = [
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o-mini",
    "deepseek/DeepSeek-V3-0324",
    "meta/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "mistral-ai/Codestral-2501",
    "microsoft/Phi-4",
    "meta/Meta-Llama-3.1-8B-Instruct",
]


def github_catalog() -> list[str]:
    token = os.environ.get("GITHUB_TOKEN", "")
    req = urllib.request.Request(
        "https://models.github.ai/catalog/models",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode().strip()
    data = json.loads(body) if body else []
    items = data.get("models", data) if isinstance(data, dict) else data
    return [m["id"] for m in items if isinstance(m, dict) and "id" in m]


def auto_models(limit: int = 6) -> list[str]:
    """Preferred GitHub Models that the catalog lists; if the catalog cannot be read, try them all (a model
    that is not served fails on its first question and is skipped)."""
    try:
        available = set(github_catalog())
    except Exception:  # the catalog is a convenience, not a requirement
        available = set()
    chosen = [m for m in GITHUB_PREFERRED if m in available] if available else list(GITHUB_PREFERRED)
    return [f"github/{m}" for m in chosen][:limit]


# Groq retires and adds models often, so the workflow asks Groq which ones it serves today. Preferred families,
# best first; anything else that is a chat model follows. Speech, safety-classifier and agent models are skipped.
GROQ_PREFERRED = [
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2",
    "meta-llama/llama-4-maverick",
    "qwen/",
    "deepseek",
    "llama-3.3-70b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-4-scout",
    "llama-3.1-8b",
]
GROQ_SKIP = ("whisper", "guard", "tts", "playai", "orpheus", "compound", "distil", "safeguard", "embed")


def groq_catalog() -> list[str]:
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}", "User-Agent": USER_AGENT},
    )
    with _OPENER.open(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("active", True) and "id" in m]


def rank_groq(ids: list[str], limit: int) -> list[str]:
    usable = sorted(i for i in ids if not any(s in i.lower() for s in GROQ_SKIP))

    def key(model: str) -> int:
        return next((n for n, p in enumerate(GROQ_PREFERRED) if model.lower().startswith(p)), len(GROQ_PREFERRED))

    chosen: list[str] = []
    for model in sorted(usable, key=key):  # stable: alphabetical within a family
        family = next((p for p in GROQ_PREFERRED if model.lower().startswith(p)), model)
        if family == "qwen/" and any(c.startswith("qwen/") for c in chosen):
            continue  # one model per family keeps the board varied
        if family != model and any(c.lower().startswith(family) for c in chosen):
            continue
        chosen.append(model)
    return [f"groq/{m}" for m in chosen][:limit]


def expand(spec: str) -> list[str]:
    """'auto' -> GitHub Models; 'groq/auto' or 'groq/auto:N' -> the best N chat models Groq serves today."""
    if spec == "auto":
        return auto_models()
    if spec.startswith("groq/auto"):
        _, _, n = spec.partition(":")
        return rank_groq(groq_catalog(), int(n) if n else 5)
    return [spec]
