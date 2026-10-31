"""Resume unissued cases after temporary DeepInfra rate limits; never retry a case."""

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from cds_bench.eval_luna.io import api_key, save
from cds_bench.eval_luna.report import records
from cds_bench.eval_luna.run import now, status, validate_sandbox

from . import deepinfra_run as driver
from .lightning_client import RequestFailure, api_call

RAW = driver.RAW


def latest_failure_was_rate_limit(raw):
    errors = [r for r in records(raw) if r.get("api_error") and not r.get("request_timeout")]
    errors.sort(key=lambda r: datetime.fromisoformat(r["started_at"]).timestamp() + r["elapsed_seconds"])
    return len(errors) >= 5 and all("HTTP 429" in r.get("error", "") for r in errors[-5:])


async def available(raw):
    directory = raw / "availability-checks" / now().replace(":", "-")
    directory.mkdir(parents=True)
    body = copy.deepcopy(driver.parameters())
    body.update(messages=[{"role": "user", "content": "Reply with the word OK."}], max_tokens=128)
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        try:
            reply = await api_call(client, directory, 1, body, api_key())
            result = {"checked_at": now(), "available": True, "http_status": reply.get("http_status"),
                      "cost_usd": (reply.get("usage") or {}).get("cost"), "provider": reply.get("provider")}
        except RequestFailure as error:
            result = {"checked_at": now(), "available": False, "http_status": error.response.get("http_status"),
                      "rate_limited": "429" in str(error)}
    save(directory / "check.json", result)
    return result["available"]


def recovery(raw, **updates):
    path = raw / "recovery-status.json"
    value = json.loads(path.read_text()) if path.exists() else {}
    value.update(updates, updated_at=now())
    save(path, value)


async def supervise(raw, rows):
    # At most six outage episodes and six health checks per outage. No benchmark
    # request is repeated; health checks are separate, unscored diagnostic calls.
    for episode in range(7):
        recovery(raw, state="running", outage_episode=episode, next_check_at=None)
        try:
            await driver.evaluate(raw, rows)
        except RuntimeError as error:
            if "Stopped after five consecutive API failures" not in str(error) or not latest_failure_was_rate_limit(raw):
                raise
            if episode == 6:
                raise RuntimeError("DeepInfra shared-pool rate limits persisted through six recovery episodes; all issued attempts retained") from error
            recovery(raw, state="waiting-provider", cause="DeepInfra upstream shared-pool rate limit (HTTP 429)")
            for check in range(6):
                delay = min(120 * 2 ** check, 600)
                next_check = (datetime.now(UTC) + timedelta(seconds=delay)).isoformat()
                recovery(raw, next_check_at=next_check, health_check=check + 1)
                status(raw, state="waiting-provider", error=None,
                       pause_reason="DeepInfra returned HTTP 429; waiting before a separate availability check")
                await asyncio.sleep(delay)
                if await available(raw):
                    break
            else:
                raise RuntimeError("DeepInfra remained unavailable after six spaced health checks; benchmark cases were not retried") from error
            # evaluate() loads all saved results, including failures, and queues
            # only identities with no saved result. Its frozen code is unchanged.
        else:
            recovery(raw, state="complete", next_check_at=None)
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    with (raw / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registration = raw / "recovery-registration.json"
        if not registration.exists():
            initial = json.loads((raw / "job-status.json").read_text())
            assert initial["state"] in {"preflight-passed", "failed"}
            if initial["state"] == "failed":
                assert latest_failure_was_rate_limit(raw)
            else:
                assert not records(raw), "A fresh full run must not reuse scored attempts"
            for path in (raw / "traces").glob("*/*/request-01.json"):
                assert (path.parent / "result.json").exists(), "An interrupted request needs review"
            save(registration, {"registered_at": now(), "initial_job_status": initial,
                 "retained_attempts": len(records(raw)), "concurrency_unchanged": 10,
                 "model_request_settings_unchanged": True, "no_scored_request_retries": True,
                 "source_sha256": {str(Path(__file__).relative_to(driver.ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                 "retained_result_sha256": {str(p.relative_to(raw)): hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in (raw / "traces").glob("*/*/result.json")}})
        else:
            prior = json.loads(registration.read_text())
            assert prior["source_sha256"][str(Path(__file__).relative_to(driver.ROOT))] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        try:
            rows = driver.prepare(raw)
            validate_sandbox(raw)
            asyncio.run(supervise(raw, rows))
        except BaseException as error:
            recovery(raw, state="failed", next_check_at=None)
            status(raw, state="failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
