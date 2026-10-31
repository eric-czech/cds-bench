"""Streaming Chat Completions with a wall deadline and preserved reasoning."""

import asyncio
import copy
import json
import re
import time

from cds_bench.eval_luna.io import save
from cds_bench.eval_luna.settings import REQUEST_TIMEOUT


def merge_reasoning(chunks):
    """Reassemble streamed fields without rewriting or dropping reasoning items."""
    merged = []
    positions = {}
    for item in chunks:
        identity = (item.get("index"), item.get("type"))
        if identity not in positions:
            positions[identity] = len(merged)
            merged.append(copy.deepcopy(item))
            continue
        target = merged[positions[identity]]
        for key, value in item.items():
            if key in ("text", "summary", "data", "signature") and isinstance(value, str):
                target[key] = (target.get(key) or "") + value
            elif value is not None:
                if key in target and target[key] is not None and target[key] != value:
                    raise RuntimeError(f"Conflicting streamed reasoning metadata: {key}")
                target[key] = value
    return merged


class RequestFailure(Exception):
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


async def api_call(client, directory, number, body, key):
    save(directory / f"request-{number:02}.json", body)
    started = time.monotonic()
    response = {"role": "assistant", "content": "", "tool_calls": [], "reasoning_details": []}
    text = []
    reasoning = []
    details = []
    calls = {}
    meta = {"request_number": number, "request_max_tokens": body["max_tokens"]}
    error = None
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT):
            async with client.stream(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                json=body,
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "CDS-bench open-weight 30-task pilot",
                },
            ) as stream:
                meta["http_status"] = stream.status_code
                if stream.status_code != 200:
                    data = (
                        (await stream.aread()).decode(errors="replace").replace(key, "[REDACTED]")
                    )
                    raise RuntimeError(f"HTTP {stream.status_code}: {data}")
                with (directory / f"response-{number:02}.sse").open("w") as log:
                    async for line in stream.aiter_lines():
                        line = line.replace(key, "[REDACTED]")
                        log.write(line + "\n")
                        log.flush()
                        if not line.startswith("data:"):
                            continue
                        value = line[5:].strip()
                        if value == "[DONE]":
                            break
                        if not value:
                            continue
                        event = json.loads(value)
                        if event.get("error"):
                            raise RuntimeError(str(event["error"]))
                        for field in ("id", "provider", "model", "usage"):
                            if event.get(field) is not None:
                                meta[field] = event[field]
                        for choice in event.get("choices", []):
                            if choice.get("error"):
                                raise RuntimeError(str(choice["error"]))
                            if choice.get("finish_reason") == "error":
                                raise RuntimeError("Provider ended the stream with finish_reason=error")
                            if choice.get("finish_reason"):
                                meta["finish_reason"] = choice["finish_reason"]
                            delta = choice.get("delta", {})
                            if delta.get("content"):
                                text.append(delta["content"])
                            if delta.get("reasoning"):
                                reasoning.append(delta["reasoning"])
                            details.extend(delta.get("reasoning_details") or [])
                            for part in delta.get("tool_calls") or []:
                                call = calls.setdefault(
                                    part["index"],
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if part.get("id"):
                                    call["id"] += part["id"]
                                fn = part.get("function", {})
                                for field in ("name", "arguments"):
                                    if fn.get(field):
                                        call["function"][field] += fn[field]
        normalized_provider = re.sub(r"[^a-z0-9]", "", (meta.get("provider") or "").casefold())
        expected_provider = re.sub(r"[^a-z0-9]", "", body["provider"]["only"][0].split("/")[0].casefold())
        if normalized_provider != expected_provider:
            raise RuntimeError(f"Unexpected provider: {meta.get('provider')}")
        if not meta.get("finish_reason") or not meta.get("usage"):
            raise RuntimeError("Stream ended without finish reason or usage")
    except TimeoutError:
        meta["request_timeout"] = True
        error = f"{REQUEST_TIMEOUT}-second API request deadline exceeded"
    except Exception as exc:  # noqa: BLE001 - retain failed attempts and tool errors
        error = f"{type(exc).__name__}: {exc}".replace(key, "[REDACTED]")
    response["content"] = "".join(text) or None
    response["tool_calls"] = [calls[k] for k in sorted(calls)]
    response["reasoning_details"] = merge_reasoning(details)
    if not response["reasoning_details"] and reasoning:
        response["reasoning"] = "".join(reasoning)
    meta.update(
        elapsed_seconds=time.monotonic() - started,
        message=response,
        reasoning_summary="".join(reasoning),
        error=error,
    )
    save(directory / f"response-{number:02}.json", meta)
    if error is not None:
        raise RequestFailure(error, meta)
    return meta
