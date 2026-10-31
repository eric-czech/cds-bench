"""Run four fresh baseline passes. Pilot results are never loaded or reused."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from cds_bench.catalog import SPEC_SHA256, TASKS

from . import sandbox
from .billing import reconcile
from .io import api_key, save
from .settings import CONCURRENCY, MAX_TOOL_CALLS, PARAMETERS, REQUEST_TIMEOUT, TOKEN_BUDGET, TOOLS
from .trial import trial

ROOT = Path(__file__).resolve().parents[3]
CONDITIONS = [("easy", False), ("hard", False), ("easy", True), ("hard", True)]
HF_REVISION = "b48333ec757cf66fd09d758d120c4f47a0256b5d"


def now():
    return datetime.now(UTC).isoformat()


def dataset():
    release = json.loads((ROOT / "generation/release.json").read_text())
    if SPEC_SHA256 != release["spec_sha256"]:
        raise RuntimeError("Benchmark specification differs from the released dataset")
    rows = {}
    hashes = {}
    for variant in ("easy", "hard"):
        path = ROOT / f"build/split-dataset/data/{variant}.parquet"
        if not path.exists():
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id="open-athena/cds-bench",
                repo_type="dataset",
                filename=f"data/{variant}.parquet",
                revision=HF_REVISION,
                cache_dir=ROOT / "build/hf-cache",
                token=False,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(Path(downloaded).read_bytes())
        hashes[variant] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[variant] != release["splits"][variant]["parquet_sha256"]:
            raise RuntimeError(f"Released {variant} Parquet checksum mismatch")
        rows[variant] = pq.read_table(path).to_pylist()
        counts = Counter(row["task_id"] for row in rows[variant])
        assert set(counts) == set(TASKS) and set(counts.values()) == {10}
        assert len(rows[variant]) == len({row["id"] for row in rows[variant]}) == 840
        assert all(row["variant"] == variant and "{{" not in row["prompt"] for row in rows[variant])
    easy = {r["instance_id"]: r for r in rows["easy"]}
    for hard in rows["hard"]:
        paired = easy[hard["instance_id"]]
        assert paired["answer"] == hard["answer"] and paired["inputs_json"] == hard["inputs_json"]
    return rows, hashes


def prepare(raw_root):
    raw_root.mkdir(parents=True, exist_ok=True)
    rows, hashes = dataset()
    modules = ["settings.py", "sandbox.py", "client.py", "trial.py", "score.py", "run.py"]
    config = {
        "run_id": raw_root.name,
        "planned_attempts": 3360,
        "tasks": 84,
        "instances_per_task_per_condition": 10,
        "condition_order": [v + ("-tools" if t else "-no-tools") for v, t in CONDITIONS],
        "model": PARAMETERS["model"],
        "parameters_no_tools": PARAMETERS,
        "tools_condition": {"tools": TOOLS, "tool_choice": "auto"},
        "request_timeout_seconds": REQUEST_TIMEOUT,
        "aggregate_completion_token_budget": TOKEN_BUDGET,
        "max_tool_calls": MAX_TOOL_CALLS,
        "concurrency": CONCURRENCY,
        "dataset": {
            "repo": "open-athena/cds-bench",
            "revision": HF_REVISION,
            "parquet_sha256": hashes,
            "spec_sha256": SPEC_SHA256,
        },
        "execution_environment": {
            "python": "3.12.9",
            "numpy": "2.5.4",
            "biopython": False,
            "network": "macOS sandbox deny network*; not an OpenRouter request parameter",
            "subprocesses": "denied",
            "fresh_interpreter_per_call": True,
            "tool_wall_seconds": sandbox.TIMEOUT_SECONDS,
            "stdout_stderr_limit_each": sandbox.OUTPUT_LIMIT,
            "files": "Runtime files and current example workdir only; workdir persists within example",
        },
        "sampling": "One fresh attempt per released prompt and condition; no pilot reuse, retries, or correctness feedback",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "inference_source_sha256": {
            n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest() for n in modules
        },
    }
    path = raw_root / "config.json"
    if path.exists():
        existing = json.loads(path.read_text())
        for key in ("parameters_no_tools", "tools_condition", "dataset", "inference_source_sha256"):
            if existing[key] != config[key]:
                raise RuntimeError(f"Frozen run configuration changed: {key}")
        config = existing
    else:
        save(path, config)
    return rows, config


def validate_sandbox(raw_root):
    version = subprocess.check_output(
        [
            str(sandbox.PYTHON),
            "-I",
            "-B",
            "-c",
            "import sys,numpy; print('.'.join(map(str,sys.version_info[:3]))); print(numpy.__version__)",
        ],
        text=True,
    ).splitlines()
    assert version == ["3.12.9", "2.5.4"], version
    sentinel = raw_root / "sandbox-private-sentinel.txt"
    sentinel.write_text("This file must be inaccessible to model code.\n")
    tests = {
        "numpy": "import numpy as np; print(np.arange(5).sum())",
        "network": "import socket; socket.create_connection(('1.1.1.1',443),timeout=2)",
        "source": f"open({str(ROOT / 'src/cds_bench/solve.py')!r}).read(1)",
        "private_file": f"open({str(sentinel)!r}).read(1)",
        "biopython": "import Bio",
        "write_outside": f"open({str(raw_root / 'should-not-exist')!r},'w').write('no')",
        "subprocess": "import subprocess; subprocess.run(['/bin/echo','no'],check=True)",
        "workdir": "open('test.txt','w').write('ok'); print(open('test.txt').read())",
    }
    results = {name: sandbox.execute(code) for name, code in tests.items()}
    assert results["numpy"]["stdout"].strip() == "10"
    assert results["workdir"]["stdout"].strip() == "ok"
    assert all(
        results[name]["exit_code"] != 0 for name in tests if name not in ("numpy", "workdir")
    )
    save(raw_root / "sandbox-validation.json", results)


def status(raw_root, **updates):
    path = raw_root / "job-status.json"
    value = json.loads(path.read_text()) if path.exists() else {"started_at": now()}
    value.update(updates, updated_at=now())
    save(path, value)


async def evaluate(args, rows):
    from .report import build

    raw_root, output = args.raw_root, args.output
    key = api_key()
    gate = asyncio.Semaphore(CONCURRENCY)
    all_results = []
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, limits=limits, trust_env=False) as client:
        for variant, with_tools in CONDITIONS:
            condition = variant + ("-tools" if with_tools else "-no-tools")
            status(
                raw_root, state="running", phase=condition, completed=len(all_results), planned=3360
            )
            stop = asyncio.Event()
            consecutive_errors = 0

            async def attempt(index, row, stop=stop, with_tools=with_tools):
                if stop.is_set():
                    return None
                # The task wrapper checks stop after the shared gate is acquired.
                async with gate:
                    if stop.is_set():
                        return None
                    local_gate = asyncio.Semaphore(1)
                    return await trial(client, local_gate, index, row, with_tools, key, raw_root)

            pending = [attempt(i, row) for i, row in enumerate(rows[variant], 1)]
            phase_count = 0
            for future in asyncio.as_completed(pending):
                result = await future
                if result is None:
                    continue
                all_results.append(result)
                phase_count += 1
                if result.get("api_error") and not result.get("request_timeout"):
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0
                if consecutive_errors >= 5:
                    stop.set()
                status(
                    raw_root,
                    completed=len(all_results),
                    phase_completed=phase_count,
                    streamed_cost_usd=sum(r["reported_cost_usd"] for r in all_results),
                    incomplete_usage_examples=sum(not r["cost_complete"] for r in all_results),
                )
                if phase_count % 25 == 0:
                    await asyncio.to_thread(build, raw_root, output)
            await reconcile(raw_root)
            await asyncio.to_thread(build, raw_root, output)
            if stop.is_set():
                raise RuntimeError(
                    "Stopped after five consecutive API failures; review saved attempts before resuming"
                )
            assert phase_count == 840
            status(raw_root, last_completed_phase=condition)
    assert len(all_results) == 3360
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix not in (".lock",):
            assert key.encode() not in path.read_bytes(), path
    status(raw_root, state="complete", phase="complete", completed=3360, finished_at=now())
    await asyncio.to_thread(build, raw_root, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path, default=ROOT / "scratch/luna-high-full-20261029-01"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "evaluations/luna-high-20261029")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check-sandbox", action="store_true")
    args = parser.parse_args()
    args.raw_root = args.raw_root.resolve()
    args.output = args.output.resolve()
    args.raw_root.mkdir(parents=True, exist_ok=True)
    with (args.raw_root / "driver.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, _ = prepare(args.raw_root)
        if args.prepare:
            print("Prepared 3,360 attempts from the released easy/hard splits. No API calls.")
            return
        validate_sandbox(args.raw_root)
        if args.check_sandbox:
            print("Python/NumPy runtime and sandbox acceptance checks passed.")
            return
        try:
            asyncio.run(evaluate(args, rows))
        except BaseException as error:
            status(args.raw_root, state="failed", error=str(error), failed_at=now())
            raise


if __name__ == "__main__":
    main()
