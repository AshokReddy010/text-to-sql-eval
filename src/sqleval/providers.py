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


def _post(url: str, payload: dict[str, Any], headers: dict[str, str], retries: int = 5, timeout: float = 120) -> Any:
    body = json.dumps(payload).encode()
    last = ""
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=body, method="POST", headers={"Content-Type": "application/json", **headers}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:300]
            last = f"HTTP {error.code}: {detail}"
            if error.code == 429 or error.code >= 500:
                retry_after = error.headers.get("Retry-After") if error.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt * 3
                if delay > 300:  # a daily quota: give up now, the next run resumes
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

    def complete(self, system: str, user: str, gold: str = "") -> Reply:
        key = os.environ.get(self.key_env, "")
        if not key:
            raise ProviderError(f"{self.key_env} is not set")
        self._pace()
        t = time.monotonic()
        data = _post(
            self.url,
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 1200,
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
                "max_tokens": 1200,
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
            extra_headers={"Accept": "application/vnd.github+json"},
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
