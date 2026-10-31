"""Audit saved full-run requests, scores, reasoning replay, and isolation."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from cds_bench.eval_luna.io import api_key, save
from cds_bench.eval_luna.score import score_result

from .run import LUNA_RESULTS, OUTPUT, PLANNED, RAW, ROOT, dataset, verify_luna
from .collector import records


def read(path):
    return json.loads(path.read_text())


def audit(raw=RAW, output=OUTPUT, partial=False):
    config = read(raw / "config.json")
    job = read(raw / "job-status.json")
    if not partial:
        assert job["state"] == "complete" and job["completed"] == 3260
    data = dataset()
    assert read(raw / "dataset.json") == data
    assert verify_luna(data) == config["luna_comparison"]
    assert hashlib.sha256(LUNA_RESULTS.read_bytes()).hexdigest() == config["luna_comparison"]["published_results_sha256"]
    for name, checksum in config["inference_source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum, name
    cases = {(r["instance_id"], variant): r for variant, rows in data.items() for r in rows}
    counts = Counter()
    budget_overruns = []
    recovered = {(r["instance_id"], r["condition"]): r for r in records(raw)}
    conditions = Counter()
    seen = set()
    for path in sorted((raw / "traces").glob("*/*/result.json")):
        result = read(path)
        case = read(path.parent / "case.json")
        assert case == cases[result["instance_id"], result["variant"]]
        identity = result["instance_id"], result["condition"]
        assert identity not in seen
        seen.add(identity)
        assert Path(result["trace_directory"]).resolve() == path.parent.resolve()
        conditions[result["condition"]] += 1
        remaining, prior = 32768, []
        request_paths = sorted(path.parent.glob("request-*.json"))
        for request_path in request_paths:
            request = read(request_path)
            response = read(request_path.with_name(request_path.name.replace("request-", "response-")))
            for field in ["model", "provider", "reasoning", "seed", "plugins"]:
                assert request[field] == config["parameters_no_tools"][field]
            assert request["max_tokens"] == remaining
            assert request["messages"][0] == {"role": "user", "content": case["prompt"]}
            assert sum(m["role"] == "user" for m in request["messages"]) == 1
            assert [m for m in request["messages"] if m["role"] == "assistant"] == prior
            assert request["tools"] == (config["tools_condition"]["tools"] if result["tools_enabled"] else [])
            assert request["tool_choice"] in (["auto", "none"] if result["tools_enabled"] else ["none"])
            prior.append({k: v for k, v in response["message"].items() if v not in (None, [], "") or k == "content"})
            remaining -= (response.get("usage") or {}).get("completion_tokens", 0)
            if remaining < 0:
                assert request_path == request_paths[-1], "Request sent after the reported budget was exhausted"
                budget_overruns.append({"instance_id": result["instance_id"],
                                        "condition": result["condition"],
                                        "requested_tokens": request["max_tokens"],
                                        "reported_tokens": (response.get("usage") or {}).get("completion_tokens", 0)})
            if response.get("provider"):
                assert response["provider"].casefold() == "alibaba"
            if response.get("id") and (raw / "billing" / (response["id"] + ".json")).exists():
                billing = read(raw / "billing" / (response["id"] + ".json")).get("data", {})
                assert not billing.get("num_search_results") and not billing.get("num_fetches")
            counts["api_requests"] += 1
            counts["reasoning_replays"] += any(m.get("reasoning_details") for m in request["messages"][1:] if m["role"] == "assistant")
        tools = [read(p) for p in path.parent.glob("tool-*.json")]
        assert len(tools) == result["tool_calls"]
        assert sum("exit_code" in t["result"] for t in tools) <= 8
        counts["python_calls"] += len(tools)
        reported = recovered[identity]
        assert reported["passed"] == result["passed"]
        assert reported["request_count"] == len(list(path.parent.glob("request-*.json")))
        if reported.get("api_error"):
            assert not result["passed"]
            assert any(read(p).get("error") for p in path.parent.glob("response-*.json"))
        else:
            checked = score_result(case, (path.parent / "answer.txt").read_text(), result["finish_reason"])
            assert checked["passed"] == result["passed"]
        counts["recovered_failed_request_metadata"] += bool(reported.get("request_metadata_recovered"))
        counts["correct"] += bool(result["passed"])
    assert all(conditions[c] <= n for c, n in PLANNED.items())
    if not partial:
        assert dict(conditions) == PLANNED
        public = [r for r in read(output / "results.json") if r["model"] == "qwen"]
        assert len(public) == 3260 and sum(r["passed"] for r in public) == counts["correct"]
    sandbox = read(raw / "sandbox-validation.json")
    for item in ["network", "source", "private_file", "biopython", "write_outside", "subprocess"]:
        assert sandbox[item]["exit_code"] != 0
    assert read(raw / "preflight-result.json")["passed"]
    key = api_key().encode()
    for path in raw.rglob("*"):
        if path.is_file() and path.suffix not in (".lock", ".tmp"):
            try:
                assert key not in path.read_bytes(), path
            except FileNotFoundError:
                if not partial:
                    raise
    document = {"complete": not partial, "attempts_audited": sum(conditions.values()),
                "conditions": dict(conditions), **counts, "matched_luna_cases_verified": 3260,
                "historical_luna_results_unchanged": True, "source_hashes_verified": True,
                "prompt_and_answer_identity_verified": True, "request_settings_verified": True,
                "reasoning_replay_verified": True,
                "requested_token_budgets_verified": True,
                "token_budgets_verified": not budget_overruns,
                "provider_reported_budget_overruns": budget_overruns,
                "scores_verified": True, "sandbox_restrictions_verified": True,
                "web_usage": 0, "credential_scan_passed": True}
    save(raw / ("partial-audit.json" if partial else "audit.json"), document)
    if not partial:
        save(output / "audit.json", document)
    print(json.dumps(document, indent=2))
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    audit(args.raw_root.resolve(), args.output.resolve(), args.partial)


if __name__ == "__main__":
    main()
