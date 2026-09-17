"""The LLM the portal uses to draft things — currently catalog entries.

Deliberately OpenAI-compatible and nothing more: one chat-completions call. That
covers the fleet's own LiteLLM proxy (the default, so a working install needs no
external account and no key leaving the building), plus OpenAI, Anthropic via a
gateway, vLLM directly, Ollama — anything speaking the same shape.

Settings live in the database rather than the environment because the operator
changes them from the Settings page. The API key is write-only across the API:
it goes in, it is never sent back out.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings as env
from ..models import AppSetting

log = logging.getLogger(__name__)

SETTING_KEY = "llm"


class LLMError(RuntimeError):
    """Anything that stopped us getting an answer, phrased for an operator."""


class LLMNotConfigured(LLMError):
    pass


@dataclass
class LLMConfig:
    """How to reach the drafting model."""
    enabled: bool = False
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout_s: float = 60.0
    # Long READMEs are common and most of what matters is near the top; this
    # caps both the bill and the blast radius of anything hidden further down.
    max_input_chars: int = 24000

    @property
    def effective_base_url(self) -> str:
        """Blank means the fleet's own proxy, which is the point of the default."""
        return (self.base_url or env.litellm_base_url).rstrip("/")

    @property
    def effective_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        # Talking to our own proxy: the master key is already ours.
        if not self.base_url:
            return env.litellm_master_key
        return ""


async def load_config(db: AsyncSession) -> LLMConfig:
    row = await db.get(AppSetting, SETTING_KEY)
    if row is None:
        return LLMConfig()
    data = dict(row.value or {})
    known = {f for f in LLMConfig.__dataclass_fields__}
    return LLMConfig(**{k: v for k, v in data.items() if k in known})


async def save_config(db: AsyncSession, cfg: LLMConfig, *, actor: str = "") -> LLMConfig:
    row = await db.get(AppSetting, SETTING_KEY)
    if row is None:
        row = AppSetting(key=SETTING_KEY)
        db.add(row)
    row.value = cfg.__dict__.copy()
    row.updated_by = actor
    await db.commit()
    return cfg


# ------------------------------------------------------------------ the call

@dataclass
class Completion:
    text: str
    model: str = ""
    usage: dict = field(default_factory=dict)


async def complete(
    cfg: LLMConfig,
    *,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> Completion:
    """One chat-completions round trip. Raises LLMError with something readable."""
    if not cfg.enabled:
        raise LLMNotConfigured(
            "No drafting model is configured. Set one up under Settings → LLM, "
            "then try again."
        )
    if not cfg.model:
        raise LLMNotConfigured("The LLM settings have no model name. Set one under Settings → LLM.")

    headers = {"Content-Type": "application/json"}
    if cfg.effective_api_key:
        headers["Authorization"] = f"Bearer {cfg.effective_api_key}"

    payload = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    url = f"{cfg.effective_base_url}/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
            r = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise LLMError(f"Could not reach the model at {cfg.effective_base_url}: {exc}") from exc

    if r.status_code >= 400:
        detail = r.text.strip().splitlines()[0][:240] if r.text.strip() else ""
        raise LLMError(f"The model returned HTTP {r.status_code}. {detail}".strip())

    try:
        body = r.json()
        choice = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError("The model's response was not in the expected format.") from exc

    return Completion(text=choice or "", model=body.get("model", cfg.model), usage=body.get("usage") or {})


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict:
    """Pull an object out of a reply that may be fenced or have prose around it.

    Models wrap JSON in markdown even when told not to, so this is not optional.
    """
    candidates = [m.group(1) for m in _FENCE.finditer(text)]
    candidates.append(text)
    for chunk in candidates:
        chunk = chunk.strip()
        start, end = chunk.find("{"), chunk.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            parsed = json.loads(chunk[start : end + 1])
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise LLMError("The model did not return usable JSON.")
