"""LiteLLM proxy integration.

Every healthy vLLM deployment is registered with the proxy as one entry in a
model group. Two deployments sharing a served_model_name become two members of
the same group, and LiteLLM load-balances across them — that is how you scale a
model horizontally here: deploy it a second time, nothing else to configure.
"""
from __future__ import annotations

import logging

import httpx

from ..config import settings

log = logging.getLogger(__name__)


class LiteLLMError(RuntimeError):
    pass


class LiteLLMClient:
    def __init__(self, base_url: str | None = None, master_key: str | None = None) -> None:
        self.base_url = (base_url or settings.litellm_base_url).rstrip("/")
        self.key = master_key or settings.litellm_master_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.key}"},
            timeout=15.0,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kw) -> dict:
        try:
            r = await self._client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise LiteLLMError(f"cannot reach LiteLLM at {self.base_url}: {exc}") from exc
        if r.status_code >= 400:
            raise LiteLLMError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        if not r.content:
            return {}
        try:
            return r.json()
        except ValueError:
            return {"raw": r.text}

    # ------------------------------------------------------------------ read

    async def health(self) -> dict:
        try:
            info = await self._request("GET", "/health/readiness")
            return {"reachable": True, "detail": info}
        except LiteLLMError as exc:
            return {"reachable": False, "detail": str(exc)}

    async def list_models(self) -> list[dict]:
        data = await self._request("GET", "/model/info")
        return data.get("data", []) if isinstance(data, dict) else []

    async def groups(self) -> list[dict]:
        """Collapse raw entries into model groups with their member endpoints."""
        entries = await self.list_models()
        grouped: dict[str, dict] = {}
        for e in entries:
            name = e.get("model_name", "?")
            params = e.get("litellm_params", {}) or {}
            info = e.get("model_info", {}) or {}
            g = grouped.setdefault(name, {"model_name": name, "members": []})
            g["members"].append(
                {
                    "id": info.get("id", ""),
                    "api_base": params.get("api_base", ""),
                    "upstream_model": params.get("model", ""),
                    "managed_by_dgxctl": str(info.get("dgxctl_deployment_id", "")) != "",
                    "deployment_id": info.get("dgxctl_deployment_id", ""),
                }
            )
        return sorted(grouped.values(), key=lambda g: g["model_name"])

    # ----------------------------------------------------------------- write

    async def register(
        self,
        model_name: str,
        api_base: str,
        upstream_model: str,
        deployment_id: str,
        rpm: int | None = None,
        tpm: int | None = None,
        extra_params: dict | None = None,
    ) -> str:
        params: dict = {
            # hosted_vllm/* tells LiteLLM to speak OpenAI-compatible to a vLLM server
            "model": f"hosted_vllm/{upstream_model}",
            "api_base": api_base.rstrip("/") + "/v1",
            "api_key": "none",
        }
        if rpm:
            params["rpm"] = rpm
        if tpm:
            params["tpm"] = tpm
        params.update(extra_params or {})

        body = {
            "model_name": model_name,
            "litellm_params": params,
            "model_info": {
                "id": deployment_id,
                "dgxctl_deployment_id": deployment_id,
                "mode": "chat",
            },
        }
        await self._request("POST", "/model/new", json=body)
        return deployment_id

    async def deregister(self, model_id: str) -> None:
        await self._request("POST", "/model/delete", json={"id": model_id})

    async def test_model(self, model_name: str) -> dict:
        """Round-trip a tiny completion so 'is it actually serving?' is one click."""
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
        }
        try:
            r = await self._client.post("/v1/chat/completions", json=body, timeout=60.0)
        except httpx.HTTPError as exc:
            return {"ok": False, "error": str(exc)}
        if r.status_code >= 400:
            return {"ok": False, "error": f"{r.status_code}: {r.text[:400]}"}
        data = r.json()
        choice = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return {"ok": True, "reply": choice, "usage": data.get("usage", {})}


async def reconcile(db) -> dict:
    """Make the proxy match the fleet, in both directions.

    Registers healthy deployments the proxy is missing, and removes entries
    whose deployment is no longer healthy — the second half matters because a
    stale entry means LiteLLM load-balances onto a backend that is gone, so a
    share of user requests fail. Entries dgxctl did not create (hosted APIs
    added by hand) are never touched.

    Returns what changed. Never raises: an unreachable proxy is reported, not
    escalated.
    """
    from sqlalchemy import select

    from ..models import ACTIVE_STATUSES, Deployment

    if not settings.litellm_auto_register:
        return {"added": [], "removed": [], "errors": ["auto-register disabled"]}

    client = LiteLLMClient()
    added: list[str] = []
    removed: list[str] = []
    errors: list[str] = []
    try:
        try:
            groups = await client.groups()
        except LiteLLMError as exc:
            return {"added": [], "removed": [], "errors": [str(exc)]}

        registered = {m["deployment_id"] for g in groups for m in g["members"] if m.get("deployment_id")}

        rows = await db.execute(select(Deployment).where(Deployment.status == "healthy"))
        healthy = list(rows.scalars().unique())
        healthy_ids = {d.id for d in healthy}

        for dep in healthy:
            if dep.id in registered:
                dep.litellm_registered = True
                continue
            ok, msg = await sync_deployment(dep, register=True)
            dep.litellm_registered = ok
            if ok:
                dep.litellm_model_id = dep.id
                added.append(f"{dep.served_model_name}@{dep.node.name}")
            else:
                errors.append(msg)

        for stale in registered - healthy_ids:
            try:
                await client.deregister(stale)
                removed.append(stale)
            except LiteLLMError as exc:
                errors.append(str(exc))

        rows = await db.execute(select(Deployment).where(Deployment.status.in_(ACTIVE_STATUSES)))
        for dep in rows.scalars().unique():
            if dep.status.value != "healthy":
                dep.litellm_registered = False

        return {"added": added, "removed": removed, "errors": errors}
    finally:
        await client.close()


async def sync_deployment(deployment, register: bool) -> tuple[bool, str]:
    """Register or deregister one deployment. Returns (ok, message).

    Never raises: LiteLLM being down must not block cluster operations, it just
    shows up as an unsynced deployment in the UI.
    """
    if not settings.litellm_auto_register:
        return False, "auto-register disabled"
    client = LiteLLMClient()
    try:
        if register:
            await client.register(
                model_name=deployment.served_model_name,
                api_base=f"http://{deployment.node.hostname}:{deployment.port}",
                upstream_model=deployment.served_model_name,
                deployment_id=deployment.id,
            )
            return True, f"registered {deployment.served_model_name} in LiteLLM"
        await client.deregister(deployment.id)
        return True, f"removed {deployment.served_model_name} from LiteLLM"
    except LiteLLMError as exc:
        log.warning("litellm sync failed: %s", exc)
        return False, str(exc)
    finally:
        await client.close()
