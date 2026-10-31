"""Fresh CoreWeave Nemotron Lightning full evaluation; no pilot response reuse."""

import argparse
import asyncio
import copy
import fcntl
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import httpx

from cds_bench.eval_luna import sandbox
from cds_bench.eval_luna.billing import reconcile
from cds_bench.eval_luna.io import api_key, save
from cds_bench.eval_luna.report import records
from cds_bench.eval_luna.run import now, status, validate_sandbox
from cds_bench.eval_luna.settings import TOOLS

from .lightning_client import RequestFailure, api_call
from .run import CONDITIONS, HASHES, LUNA_RAW, PLANNED, REMOVED_HARD, REVISION, ROOT, dataset, verify_luna

RAW = ROOT / "scratch/nemotron-lightning-coreweave-full-20261030-01"
PILOT = ROOT / "scratch/open-weights-pilot-20261029-01/models/nemotron-lightning-coreweave"
QWEN_RAW = ROOT / "scratch/qwen37-flash-high-full-20261029-01"
runtime = importlib.import_module("cds_bench.eval_luna.trial")


def parameters():
    # The successful pilot is the exact request contract, including omitted seed.
    return copy.deepcopy(json.loads((PILOT / "config.json").read_text())["parameters_no_tools"])


def verify_qwen(rows):
    expected = {(r["instance_id"], variant): r for variant, cases in rows.items() for r in cases}
    count = 0
    for path in (QWEN_RAW / "traces").glob("*/*/case.json"):
        case = json.loads(path.read_text())
        assert case == expected[case["instance_id"], case["variant"]]
        assert (path.parent / "result.json").exists()
        count += 1
    assert count == 3260
    return {"retained_attempts": count, "every_retained_case_verified": True,
            "source_config_sha256": hashlib.sha256((QWEN_RAW / "config.json").read_bytes()).hexdigest()}


def prepare(raw):
    raw.mkdir(parents=True, exist_ok=True)
    rows = dataset()
    pilot = json.loads((PILOT / "config.json").read_text())
    value = parameters()
    assert value["model"] == "nvidia/nemotron-3.5-lightning"
    assert value["provider"] == {"only": ["coreweave/bf16"], "allow_fallbacks": False, "require_parameters": True}
    assert value["reasoning"] == {"enabled": True, "exclude": False} and "seed" not in value
    assert pilot["tools"] == TOOLS
    files = [Path(__file__), Path(__file__).with_name("lightning_client.py"), Path(__file__).with_name("run.py")]
    files += [ROOT / "src/cds_bench/eval_luna" / name for name in
              ("settings.py", "sandbox.py", "trial.py", "score.py", "billing.py", "io.py", "run.py")]
    config = {
        "run_id": raw.name, "planned_attempts": 3260, "planned_per_condition": PLANNED,
        "condition_order": list(PLANNED), "model": value["model"], "provider_quantization": "bf16",
        "parameters_no_tools": value, "tools_condition": {"tools": TOOLS, "tool_choice": "auto"},
        "request_timeout_seconds": 300, "aggregate_completion_token_budget": 32768,
        "max_tool_calls": 8, "concurrency": 10, "minimum_request_spacing_seconds": 0,
        "dataset": {"repo": "open-athena/cds-bench", "revision": REVISION, "parquet_sha256": HASHES,
                    "rows_per_split": {v: len(rs) for v, rs in rows.items()}, "removed_hard_tasks": sorted(REMOVED_HARD)},
        "luna_comparison": verify_luna(rows), "qwen_comparison": verify_qwen(rows),
        "execution_environment": json.loads((LUNA_RAW / "config.json").read_text())["execution_environment"],
        "sampling": "One fresh request sequence per released case/condition. No pilot response reuse, inference retries or correctness feedback. Model/provider selected after the pilot.",
        "settings_difference": "Reasoning enabled without an advertised effort selector; seed omitted, matching the CoreWeave pilot. Same prompts, scorer, original JSON Python function, budgets and sandbox as prior full evaluations.",
        "pilot_config_sha256": hashlib.sha256((PILOT / "config.json").read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "inference_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    path = raw / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        for key in config:
            if key != "source_commit":
                assert old[key] == config[key], f"Frozen configuration changed: {key}"
        assert json.loads((raw / "dataset.json").read_text()) == rows
    else:
        save(path, config)
        save(raw / "dataset.json", rows)
    runtime.PARAMETERS = value
    runtime.RequestFailure = RequestFailure
    runtime.api_call = api_call
    sandbox.WORK_ROOT = ROOT / "build/luna-tool-work" / raw.name
    return rows


async def preflight(client, raw, key):
    response = await client.get("https://openrouter.ai/api/v1/models/nvidia/nemotron-3.5-lightning/endpoints")
    response.raise_for_status()
    save(raw / "model-endpoints.json", response.json())
    native = next(e for e in response.json()["data"]["endpoints"] if e["tag"] == "coreweave/bf16")
    assert {"reasoning", "tools", "tool_choice", "max_tokens"} <= set(native["supported_parameters"])
    assert native["max_completion_tokens"] >= 32768
    case = {"id": "protocol/01/easy", "instance_id": "protocol/01", "task_id": "protocol",
            "variant": "easy", "output_format": "scalar",
            "prompt": "Use the python tool to calculate 123456789 * 987654321. You must execute Python before answering. After any explanation, write Final answer: on its own line, then the integer result on the next line.",
            "answer": "121932631112635269"}
    result = await runtime.trial(client, asyncio.Semaphore(1), 1, case, True, key, raw / "preflight")
    assert result["passed"] and result["tool_calls"] >= 1
    assert result["reasoning_tokens"] > 0 or result["reasoning_context_replayed"]
    save(raw / "preflight-result.json", result)
    await reconcile(raw / "preflight")


async def evaluate(raw, rows, preflight_only=False):
    key = api_key()
    gate = asyncio.Semaphore(10)
    # Saved results are retained on an operational restart. Only unissued cases
    # enter the queue; an interrupted request always requires explicit review.
    existing = {(r["instance_id"], r["condition"]): r for r in records(raw)}
    async with httpx.AsyncClient(timeout=300, trust_env=False,
                                limits=httpx.Limits(max_connections=10, max_keepalive_connections=10)) as client:
        status(raw, state="preflight", planned=3260, completed=len(existing), error=None)
        await preflight(client, raw, key)
        if preflight_only:
            status(raw, state="preflight-passed", completed=len(existing))
            return
        for variant, tools in CONDITIONS:
            condition = variant + ("-tools" if tools else "-no-tools")
            pending = [(i, r) for i, r in enumerate(rows[variant], 1) if (r["instance_id"], condition) not in existing]
            count = sum(c == condition for _, c in existing)
            status(raw, state="running", phase=condition, completed=len(existing), phase_completed=count,
                   phase_planned=len(rows[variant]), planned=3260, error=None)
            stop = asyncio.Event()

            async def attempt(index, row):
                async with gate:
                    if stop.is_set():
                        return None
                    return await runtime.trial(client, asyncio.Semaphore(1), index, row, tools, key, raw)

            errors = 0
            for future in asyncio.as_completed([attempt(i, r) for i, r in pending]):
                result = await future
                if result is None:
                    continue
                identity = result["instance_id"], result["condition"]
                assert identity not in existing
                existing[identity] = result
                count += 1
                errors = errors + 1 if result.get("api_error") and not result.get("request_timeout") else 0
                if errors >= 5:
                    stop.set()
                status(raw, completed=len(existing), phase_completed=count,
                       streamed_cost_usd=sum(r["reported_cost_usd"] for r in existing.values()))
            await reconcile(raw)
            if stop.is_set():
                raise RuntimeError("Stopped after five consecutive API failures; issued results retained, review before resuming")
            assert count == len(rows[variant])
            status(raw, last_completed_phase=condition)
    assert len(existing) == 3260
    save(raw / "results.json", list(existing.values()))
    status(raw, state="complete", phase="complete", completed=3260, finished_at=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            rows = prepare(raw)
            if args.prepare:
                print("Prepared 3,260 fresh Lightning attempts; historical Luna/Qwen cases match.")
                return
            status(raw, state="validating-sandbox", planned=3260, error=None)
            validate_sandbox(raw)
            asyncio.run(evaluate(raw, rows, args.preflight_only))
        except BaseException as error:
            status(raw, state="failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
