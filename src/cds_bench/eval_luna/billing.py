"""Reconcile completed requests against OpenRouter generation metadata."""

import asyncio
import json

import httpx

from .io import api_key, save


async def reconcile(raw_root):
    key = api_key()
    directory = raw_root / "billing"
    directory.mkdir(exist_ok=True)
    responses = [
        json.loads(p.read_text()) for p in (raw_root / "traces").glob("*/*/response-*.json")
    ]
    identifiers = sorted({r["id"] for r in responses if r.get("id")})
    gate = asyncio.Semaphore(10)
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:

        async def get(identifier):
            path = directory / (identifier + ".json")
            if path.exists():
                existing = json.loads(path.read_text())
                if existing.get("data", {}).get("total_cost") is not None:
                    return existing
            async with gate:
                try:
                    reply = await client.get(
                        "https://openrouter.ai/api/v1/generation",
                        params={"id": identifier},
                        headers={"Authorization": "Bearer " + key},
                    )
                    data = json.loads(reply.text.replace(key, "[REDACTED]"))
                    data["retrieval_http_status"] = reply.status_code
                except Exception as error:  # noqa: BLE001 - retain failed request metadata
                    data = {
                        "error": str(error).replace(key, "[REDACTED]"),
                        "generation_id": identifier,
                    }
                save(path, data)
                return data

        values = await asyncio.gather(*(get(identifier) for identifier in identifiers))
    known = [v["data"] for v in values if v.get("data", {}).get("total_cost") is not None]
    summary = {
        "generations": len(identifiers),
        "billing_records": len(known),
        "total_cost_usd": sum(v["total_cost"] for v in known),
        "providers": sorted({v["provider_name"] for v in known if v.get("provider_name")}),
        "models": sorted({v["model"] for v in known if v.get("model")}),
        "web_search_results": sum(v.get("num_search_results") or 0 for v in known),
        "web_fetches": sum(v.get("num_fetches") or 0 for v in known),
        "complete_for_identified_requests": len(known) == len(identifiers),
        "responses_without_generation_id": sum(not v.get("id") for v in responses),
    }
    save(raw_root / "billing-summary.json", summary)
    return summary
