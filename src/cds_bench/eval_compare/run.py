"""Fresh Qwen full evaluation; historical Luna responses are never rerun."""

import argparse
import asyncio
import copy
import fcntl
import hashlib
import importlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from cds_bench.eval_luna import sandbox
from cds_bench.eval_luna.billing import reconcile
from cds_bench.eval_luna.io import api_key, save
from cds_bench.eval_luna.run import now, status, validate_sandbox
from cds_bench.eval_luna.settings import PARAMETERS, TOOLS

from .client import api_call

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "scratch/qwen37-flash-high-full-20261029-01"
OUTPUT = ROOT / "evaluations/luna-qwen-high-20261029"
LUNA_RAW = ROOT / "scratch/luna-high-full-20261029-01"
LUNA_RESULTS = ROOT / "evaluations/luna-high-20261029/results.json"
REVISION = "26f63e8e488022bed14e12af0f1e44d87a28e14f"
HASHES = {
    "easy": "d756b2655e4242812045def8ff4e4485f945280e5e6af2b4a845223102b7460e",
    "hard": "aa724480707b2132abbe8c4f21e979701866a5ada36812476e2b7bb89b6cc1fa",
}
REMOVED_HARD = {
    "coords.nt_to_codon", "variant.apply", "variant.frame_status",
    "rna.dotbracket_to_pairs", "construct.inframe_deletion",
}
CONDITIONS = [("easy", False), ("hard", False), ("easy", True), ("hard", True)]
PLANNED = {"easy-no-tools": 840, "hard-no-tools": 790, "easy-tools": 840, "hard-tools": 790}
runtime = importlib.import_module("cds_bench.eval_luna.trial")


def parameters():
    value = copy.deepcopy(PARAMETERS)
    value["model"] = "qwen/qwen3.7-flash"
    value["provider"] = {
        "only": ["alibaba/us-east-1"], "allow_fallbacks": False, "require_parameters": True,
    }
    return value


def dataset():
    rows = {}
    for variant, checksum in HASHES.items():
        path = Path(hf_hub_download(
            "open-athena/cds-bench", filename=f"data/{variant}.parquet", repo_type="dataset",
            revision=REVISION, cache_dir=ROOT / "build/hf-cache", token=False,
        ))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
        rows[variant] = pq.read_table(path).to_pylist()
        counts = Counter(r["task_id"] for r in rows[variant])
        assert len(rows[variant]) == (840 if variant == "easy" else 790)
        assert set(counts.values()) == {10}
        assert len({r["id"] for r in rows[variant]}) == len(rows[variant])
        assert all(r["variant"] == variant for r in rows[variant])
    assert set(r["task_id"] for r in rows["easy"]) - set(r["task_id"] for r in rows["hard"]) == REMOVED_HARD
    easy = {r["instance_id"]: r for r in rows["easy"]}
    assert all(r["answer"] == easy[r["instance_id"]]["answer"] and r["inputs_json"] == easy[r["instance_id"]]["inputs_json"] for r in rows["hard"])
    return rows


def verify_luna(rows):
    """Check every retained historical prompt and answer, not just matching IDs."""
    current = {(r["instance_id"], variant): r for variant, cases in rows.items() for r in cases}
    results = json.loads(LUNA_RESULTS.read_text())
    retained = [r for r in results if (r["instance_id"], r["variant"]) in current]
    assert len(retained) == 3260 and len(results) == 3360
    raw_cases = {}
    for p in (LUNA_RAW / "traces").glob("*/*/case.json"):
        old = json.loads(p.read_text())
        key = old["instance_id"], old["variant"]
        if key not in current:
            continue
        assert old == current[key], f"Historical prompt/answer differs: {key}"
        raw_cases[key, p.parent.parent.name] = str(p)
    assert len(raw_cases) == 3260
    assert Counter(r["condition"] for r in retained) == PLANNED
    return {"original_attempts": 3360, "retained_attempts": 3260, "excluded_duplicate_hard_attempts": 100,
            "every_retained_case_verified": True, "published_results_sha256": hashlib.sha256(LUNA_RESULTS.read_bytes()).hexdigest()}


def prepare(raw):
    raw.mkdir(parents=True, exist_ok=True)
    rows = dataset()
    files = [Path(__file__), Path(__file__).with_name("client.py")]
    files += [ROOT / "src/cds_bench/eval_luna" / n for n in
              ("settings.py", "sandbox.py", "trial.py", "score.py", "billing.py", "io.py", "run.py")]
    config = {
        "run_id": raw.name, "planned_attempts": 3260, "planned_per_condition": PLANNED,
        "condition_order": list(PLANNED), "model": "qwen/qwen3.7-flash",
        "parameters_no_tools": parameters(), "tools_condition": {"tools": TOOLS, "tool_choice": "auto"},
        "request_timeout_seconds": 300, "aggregate_completion_token_budget": 32768,
        "max_tool_calls": 8, "concurrency": 10,
        "dataset": {"repo": "open-athena/cds-bench", "revision": REVISION, "parquet_sha256": HASHES,
                    "rows_per_split": {k: len(v) for k, v in rows.items()}, "removed_hard_tasks": sorted(REMOVED_HARD)},
        "luna_comparison": verify_luna(rows),
        "execution_environment": json.loads((LUNA_RAW / "config.json").read_text())["execution_environment"],
        "sampling": "One fresh request sequence per released case/condition; no pilot reuse, inference retries, or correctness feedback. Model chosen after the earlier pilot.",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "inference_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    path = raw / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        for field in config:
            if field != "source_commit":
                assert old[field] == config[field], f"Frozen configuration changed: {field}"
    else:
        save(path, config)
        save(raw / "dataset.json", rows)
    # Reuse exactly the established scorer, budgets, sandbox and tool loop.
    # Only the requested model/provider and the model-neutral HTTP client differ.
    runtime.PARAMETERS = parameters()
    runtime.api_call = api_call
    sandbox.WORK_ROOT = ROOT / "build/luna-tool-work" / raw.name
    return rows


async def preflight(client, raw, key):
    case = {"id": "protocol/01/easy", "instance_id": "protocol/01", "task_id": "protocol",
            "variant": "easy", "output_format": "scalar",
            "prompt": "Use the python tool to calculate 123456789 * 987654321. You must execute Python before answering. After any explanation, write Final answer: on its own line, then the integer result on the next line.",
            "answer": "121932631112635269"}
    result = await runtime.trial(client, asyncio.Semaphore(1), 1, case, True, key, raw / "preflight")
    assert result["passed"] and result["tool_calls"] >= 1, result
    assert result["reasoning_tokens"] > 0 or result["reasoning_context_replayed"], result
    save(raw / "preflight-result.json", result)
    await reconcile(raw / "preflight")


async def evaluate(raw, rows, preflight_only=False):
    key = api_key()
    gate = asyncio.Semaphore(10)
    results = []
    async with httpx.AsyncClient(timeout=300, trust_env=False,
                                limits=httpx.Limits(max_connections=10, max_keepalive_connections=10)) as client:
        status(raw, state="preflight", planned=3260)
        await preflight(client, raw, key)
        if preflight_only:
            status(raw, state="preflight-passed", completed=0)
            return
        for variant, tools in CONDITIONS:
            condition = variant + ("-tools" if tools else "-no-tools")
            status(raw, state="running", phase=condition, completed=len(results), phase_completed=0,
                   phase_planned=len(rows[variant]), planned=3260, error=None)
            stop = asyncio.Event()

            async def attempt(index, row):
                async with gate:
                    if stop.is_set():
                        return None
                    return await runtime.trial(client, asyncio.Semaphore(1), index, row, tools, key, raw)

            count, errors = 0, 0
            for future in asyncio.as_completed([attempt(i, row) for i, row in enumerate(rows[variant], 1)]):
                result = await future
                if result is None:
                    continue
                results.append(result)
                count += 1
                errors = errors + 1 if result.get("api_error") and not result.get("request_timeout") else 0
                if errors >= 5:
                    stop.set()
                status(raw, completed=len(results), phase_completed=count,
                       streamed_cost_usd=sum(r["reported_cost_usd"] for r in results))
            await reconcile(raw)
            if stop.is_set():
                raise RuntimeError("Stopped after five consecutive API failures; review saved attempts before resuming")
            assert count == len(rows[variant])
            status(raw, last_completed_phase=condition)
    assert len(results) == 3260
    save(raw / "results.json", results)
    for path in raw.rglob("*"):
        if path.is_file() and path.suffix != ".lock":
            assert key.encode() not in path.read_bytes(), path
    status(raw, state="complete", phase="complete", completed=3260, finished_at=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-sandbox", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            rows = prepare(raw)
            if args.prepare:
                print("Prepared 3,260 fresh Qwen attempts and verified the matching Luna cases.")
                return
            status(raw, state="validating-sandbox", planned=3260, error=None)
            validate_sandbox(raw)
            if args.check_sandbox:
                print("Sandbox acceptance checks passed.")
                return
            asyncio.run(evaluate(raw, rows, args.preflight_only))
        except BaseException as error:
            status(raw, state="failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
