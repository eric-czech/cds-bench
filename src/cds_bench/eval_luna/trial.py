"""One scored attempt with optional Python calls and an aggregate token budget."""

import asyncio
import copy
import json
import time
from datetime import UTC, datetime

from . import sandbox as executor
from . import score as scoring
from .client import RequestFailure, api_call
from .io import save
from .score import score_result
from .settings import MAX_TOOL_CALLS, PARAMETERS, TOKEN_BUDGET, TOOLS


async def trial(client, gate, index, record, with_tools, key, raw_root):
    condition = record["variant"] + ("-tools" if with_tools else "-no-tools")
    directory = (
        raw_root
        / "traces"
        / condition
        / f"{index:04}-{record['task_id']}-{record['instance_id'].rsplit('/', 1)[1]}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    # Never silently overwrite an interrupted attempt or issue its requests again.
    if (directory / "request-01.json").exists():
        raise RuntimeError(f"Interrupted attempt requires review: {directory}")
    async with gate:
        started = time.monotonic()
        save(directory / "case.json", record)
        (directory / "prompt.md").write_text(record["prompt"] + "\n")
        (directory / "expected.txt").write_text(record["answer"] + "\n")
        messages = [{"role": "user", "content": record["prompt"]}]
        responses = []
        tool_results = []
        remaining = TOKEN_BUDGET
        result = {
            "task_id": record["task_id"],
            "instance_id": record["instance_id"],
            "variant": record["variant"],
            "tools_enabled": with_tools,
            "condition": condition,
            "trace_directory": str(directory),
            "started_at": datetime.now(UTC).isoformat(),
        }
        final = ""
        finish = None
        try:
            while remaining > 0:
                body = copy.deepcopy(PARAMETERS)
                body.update(
                    messages=messages,
                    max_tokens=remaining,
                    tools=TOOLS if with_tools else [],
                    tool_choice="auto"
                    if with_tools and len(tool_results) < MAX_TOOL_CALLS
                    else "none",
                )
                try:
                    reply = await api_call(client, directory, len(responses) + 1, body, key)
                except RequestFailure as exc:
                    responses.append(exc.response)
                    result.update(
                        api_error=True, request_timeout=bool(exc.response.get("request_timeout"))
                    )
                    raise
                responses.append(reply)
                consumed = reply["usage"].get("completion_tokens")
                if not isinstance(consumed, int) or consumed < 0:
                    raise RuntimeError("Cannot enforce budget without completion token usage")
                remaining -= consumed
                assistant = reply["message"]
                finish = reply["finish_reason"]
                final = assistant.get("content") or ""
                calls = assistant.get("tool_calls") or []
                if not calls:
                    result.update(score_result(record, final, finish))
                    break
                if not with_tools:
                    raise RuntimeError("Unexpected tool call in no-tools condition")
                if finish == "length" or remaining <= 0:
                    result.update(
                        passed=False, truncated=True, error="Token budget ended during tool call"
                    )
                    break
                messages.append(
                    {
                        k: v
                        for k, v in assistant.items()
                        if v not in (None, [], "") or k == "content"
                    }
                )
                for call in calls:
                    if len(tool_results) >= MAX_TOOL_CALLS:
                        output = {
                            "error": "Maximum of 8 Python calls reached; provide your final answer."
                        }
                    else:
                        try:
                            if call["function"]["name"] != "python":
                                raise ValueError("Only python is available")
                            arguments = scoring.parse_json(call["function"]["arguments"])
                            if set(arguments) != {"code"} or not isinstance(arguments["code"], str):
                                raise ValueError("Expected exactly one string field: code")
                            tool_directory = (
                                executor.WORK_ROOT / raw_root.name / condition / directory.name
                            )
                            output = await asyncio.to_thread(
                                executor.execute, arguments["code"], tool_directory
                            )
                        except Exception as exc:  # noqa: BLE001 - retain failed attempts and tool errors
                            output = {"error": str(exc).replace(key, "[REDACTED]")}
                    tool_results.append(
                        {"tool_call_id": call["id"], "call": call, "result": output}
                    )
                    save(directory / f"tool-{len(tool_results):02}.json", tool_results[-1])
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(output, ensure_ascii=False),
                        }
                    )
                # At most one final request after the advertised call limit.
                if len(responses) > MAX_TOOL_CALLS + 1:
                    raise RuntimeError("Unexpected tool calls after disabling tool_choice")
            else:
                result.update(
                    passed=False, truncated=True, error="Aggregate token budget exhausted"
                )
        except Exception as exc:  # noqa: BLE001 - retain failed attempts and tool errors
            result.update(passed=False, error=str(exc).replace(key, "[REDACTED]"))
        usage = [r.get("usage") or {} for r in responses]
        result.update(
            elapsed_seconds=time.monotonic() - started,
            finish_reason=finish,
            request_count=len(responses),
            tool_calls=len(tool_results),
            tool_failures=sum(
                bool(
                    t["result"].get("error")
                    or t["result"].get("exit_code")
                    or t["result"].get("timed_out")
                )
                for t in tool_results
            ),
            reasoning_context_replayed=any(
                m.get("reasoning_details") for m in messages if m["role"] == "assistant"
            ),
            prompt_tokens=sum(u.get("prompt_tokens", 0) for u in usage),
            completion_tokens=sum(u.get("completion_tokens", 0) for u in usage),
            reasoning_tokens=sum(
                (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) for u in usage
            ),
            reported_cost_usd=sum(u.get("cost", 0) for u in usage),
            cost_complete=all(bool(r.get("usage")) for r in responses),
            providers=sorted({r.get("provider") for r in responses if r.get("provider")}),
            response_ids=[r["id"] for r in responses if r.get("id")],
        )
        (directory / "answer.txt").write_text(final)
        save(result_path, result)
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in [
                        "condition",
                        "task_id",
                        "passed",
                        "request_timeout",
                        "error",
                        "tool_calls",
                        "elapsed_seconds",
                        "reported_cost_usd",
                    ]
                }
            ),
            flush=True,
        )
        return result
