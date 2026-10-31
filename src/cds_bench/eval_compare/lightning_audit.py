"""Audit the saved full Lightning run without changing scores or retrying requests."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from cds_bench.eval_luna.io import api_key, save
from cds_bench.eval_luna.report import aggregate, records
from cds_bench.eval_luna.score import score_result
from cds_bench.eval_luna.settings import TOOLS

from .lightning_run import PILOT, RAW, ROOT, verify_qwen
from .run import OUTPUT, PLANNED, dataset, verify_luna


def read(path):
    return json.loads(path.read_text())


def audit(raw=RAW, output=OUTPUT, partial=False):
    config, job = read(raw / "config.json"), read(raw / "job-status.json")
    if not partial:
        assert job["state"] == "complete" and job["completed"] == 3260
    data = dataset()
    assert read(raw / "dataset.json") == data
    assert verify_luna(data) == config["luna_comparison"]
    assert verify_qwen(data) == config["qwen_comparison"]
    for name, digest in config["inference_source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    registrations = [raw / "recovery-registration.json"] if (raw / "recovery-registration.json").exists() else []
    registrations += sorted((raw / "operational-resumes").glob("*.json"))
    for registration in registrations:
        recovery = read(registration)
        for path, digest in recovery['source_sha256'].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
        for path, digest in recovery['retained_result_sha256'].items():
            assert hashlib.sha256((raw / path).read_bytes()).hexdigest() == digest, path
    pilot = read(PILOT / "config.json")
    assert hashlib.sha256((PILOT / "config.json").read_bytes()).hexdigest() == config["pilot_config_sha256"]
    route = config["parameters_no_tools"]["provider"]["only"]
    assert route in [["coreweave/bf16"], ["deepinfra/bf16"]]
    model_key = "deepinfra" if route == ["deepinfra/bf16"] else "lightning"
    expected_parameters = copy.deepcopy(pilot["parameters_no_tools"])
    expected_parameters["provider"]["only"] = route
    assert config["parameters_no_tools"] == expected_parameters
    assert config["tools_condition"]["tools"] == pilot["tools"] == TOOLS
    assert config["concurrency"] == 10 and config["request_timeout_seconds"] == 300
    assert config["aggregate_completion_token_budget"] == 32768 and config["max_tool_calls"] == 8
    expected = {(r["instance_id"], v): r for v, rs in data.items() for r in rs}
    counts, conditions, seen = Counter(), Counter(), set()
    over_budget, malformed, missing_usage = [], [], []
    rows = records(raw)
    pilot_ids = {identifier for row in records(PILOT) for identifier in row["response_ids"]}
    assert not (pilot_ids & {identifier for row in rows for identifier in row["response_ids"]})
    if model_key == "deepinfra":
        coreweave_ids = {identifier for row in records(RAW) for identifier in row["response_ids"]}
        assert not (coreweave_ids & {identifier for row in rows for identifier in row["response_ids"]})
    for result in rows:
        directory = Path(result["trace_directory"])
        case = read(directory / "case.json")
        assert case == expected[result["instance_id"], result["variant"]]
        identity = result["instance_id"], result["condition"]
        assert identity not in seen
        seen.add(identity)
        conditions[result["condition"]] += 1
        requests = sorted(directory.glob("request-*.json"))
        tools = [read(p) for p in sorted(directory.glob("tool-*.json"))]
        assert len(requests) == result["request_count"]
        assert len(tools) == result["tool_calls"]
        assert result["tools_enabled"] or not tools
        remaining, prior = 32768, []
        for path in requests:
            request, response = read(path), read(path.with_name(path.name.replace("request-", "response-")))
            for field in ["model", "provider", "reasoning", "plugins", "seed"]:
                assert request.get(field) == config["parameters_no_tools"].get(field), field
            assert "seed" not in request
            assert request["max_tokens"] == remaining and remaining > 0
            assert request["messages"][0] == {"role": "user", "content": case["prompt"]}
            assert sum(m["role"] == "user" for m in request["messages"]) == 1
            assert not any(m["role"] == "system" for m in request["messages"])
            assert [m for m in request["messages"] if m["role"] == "assistant"] == prior
            tool_messages = [m for m in request["messages"] if m["role"] == "tool"]
            for message, saved in zip(tool_messages, tools, strict=False):
                assert message["tool_call_id"] == saved["tool_call_id"]
                assert json.loads(message["content"]) == saved["result"]
            assert len(tool_messages) <= len(tools)
            assert request["tools"] == (TOOLS if result["tools_enabled"] else [])
            assert request["tool_choice"] == ("auto" if result["tools_enabled"] and len(tool_messages) < 8 else "none")
            prior.append({k: v for k, v in response["message"].items() if v not in (None, [], "") or k == "content"})
            if response.get("provider"):
                assert response["provider"].casefold() == route[0].split("/")[0]
            consumed = (response.get("usage") or {}).get("completion_tokens")
            if not isinstance(consumed, int):
                assert response.get("error")
                missing_usage.append({"instance_id": result["instance_id"], "condition": result["condition"], "request": path.name})
                consumed = 0
            remaining -= consumed
            if remaining < 0:
                assert path == requests[-1]
                over_budget.append({"instance_id": result["instance_id"], "condition": result["condition"], "requested": request["max_tokens"], "reported": consumed})
            counts["api_requests"] += 1
            counts["reasoning_replays"] += any(m.get("reasoning_details") or m.get("reasoning") for m in request["messages"] if m["role"] == "assistant")
        for number, tool in enumerate(tools, 1):
            if number > 8:
                assert tool["result"].get("error", "").startswith("Maximum of 8 Python calls reached")
                continue
            function = tool["call"]["function"]
            try:
                assert function["name"] == "python"
                args = json.loads(function["arguments"])
                assert set(args) == {"code"} and isinstance(args["code"], str)
            except (AssertionError, ValueError, TypeError):
                assert tool["result"].get("error") and "exit_code" not in tool["result"]
                malformed.append({"instance_id": result["instance_id"], "condition": result["condition"], "call": number})
        assert sum("exit_code" in t["result"] for t in tools) <= 8
        counts["python_calls"] += len(tools)
        if not result.get("api_error"):
            checked = score_result(case, (directory / "answer.txt").read_text(), result["finish_reason"])
            assert checked["passed"] == result["passed"]
        else:
            assert not result["passed"]
        counts["correct"] += bool(result["passed"])
    assert all(conditions[c] <= n for c, n in PLANNED.items())
    if not partial:
        assert dict(conditions) == PLANNED
        for path in (raw / "traces").glob("*/*/request-01.json"):
            assert (path.parent / "result.json").exists(), path
        public = [r for r in read(output / "results.json") if r["model"] == model_key]
        assert aggregate(public) == aggregate(rows)
    checks = read(raw / "sandbox-validation.json")
    assert all(checks[name]["exit_code"] != 0 for name in ["network", "source", "private_file", "biopython", "write_outside", "subprocess"])
    assert read(raw / "preflight-result.json")["passed"]
    billing = read(raw / "billing-summary.json") if (raw / "billing-summary.json").exists() else {}
    assert not billing.get("web_search_results") and not billing.get("web_fetches")
    if not partial:
        key = api_key().encode()
        for path in raw.rglob("*"):
            if path.is_file() and path.suffix not in {".lock", ".tmp"}:
                assert key not in path.read_bytes(), path
    result = {"complete": not partial, "attempts_audited": len(rows), "conditions": dict(conditions), **counts,
              "source_hashes_verified": True, "matched_luna_qwen_cases_verified": 3260,
              "pilot_settings_verified_except_declared_provider": True, "provider": route[0], "no_pilot_response_reuse_or_inference_retries": True,
              "request_settings_verified": True, "reasoning_and_tool_result_replay_verified": True,
              "requested_token_budgets_verified": True, "token_budgets_verified": not over_budget,
              "provider_reported_budget_overruns": over_budget, "requests_without_usage": missing_usage,
              "malformed_calls_rejected_without_execution": malformed,
              "scores_verified": True, "sandbox_restrictions_verified": True, "web_usage": 0,
              "credential_scan_passed": not partial, "totals": aggregate(rows)}
    if model_key == "deepinfra":
        result["no_coreweave_full_response_reuse"] = True
    save(raw / ("partial-audit.json" if partial else "audit.json"), result)
    if not partial:
        save(output / ("deepinfra-audit.json" if model_key == "deepinfra" else "lightning-audit.json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.raw_root.resolve(), args.output.resolve(), args.partial), indent=2))
