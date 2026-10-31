"""Finite local DeepInfra monitor: refresh reports, audit completion, publish PR OP."""

import argparse
import asyncio
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path

from cds_bench.eval_luna.billing import reconcile
from cds_bench.eval_luna.io import save
from cds_bench.eval_luna.run import now

from .lightning_audit import audit
from .deepinfra_run import RAW, ROOT
from .report import BRANCH, PR_TITLE, REPOSITORY, build
from .run import OUTPUT


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def publish(raw, output, number):
    assert command("git", "branch", "--show-current") == BRANCH
    allowed = str(output.relative_to(ROOT))
    staged = command("git", "diff", "--cached", "--name-only").splitlines()
    assert all(p.startswith(allowed + "/") for p in staged), "Unrelated staged files; not publishing"
    snapshot_root = raw / "publication-snapshot"
    snapshot = snapshot_root / allowed
    # Preserve the completed Qwen audit and other existing report attachments.
    shutil.copytree(output, snapshot, dirs_exist_ok=True)
    summary = build(output=snapshot, deepinfra_raw=raw)
    if (raw / "audit.json").exists():
        shutil.copy2(raw / "audit.json", snapshot / "deepinfra-audit.json")
    shutil.copytree(snapshot, output, dirs_exist_ok=True)
    subprocess.run(["git", "--git-dir", str(ROOT / ".git"), "--work-tree", str(snapshot_root),
                    "-C", str(snapshot_root), "add", "--", allowed], check=True)
    if command("git", "diff", "--cached", "--name-only"):
        subprocess.run(["git", "commit", "-m", f"Update full DeepInfra Lightning evaluation: {summary['deepinfra_completed']}/3260 attempts"], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", BRANCH], cwd=ROOT, check=True)
    subprocess.run(["gh", "pr", "edit", str(number), "--repo", REPOSITORY,
                    "--title", PR_TITLE, "--body-file", str(output / "pr-body.md")], cwd=ROOT, check=True)
    result = {"deepinfra_completed": summary["deepinfra_completed"], "state": summary["state"],
              "cost_usd": summary["totals"]["deepinfra"]["cost_usd"], "published_at": now(), "pr_number": number}
    save(raw / "publication.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--pr", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    raw, output = args.raw_root.resolve(), args.output.resolve()
    with (raw / "watcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_built, last_phase, last_state, last_time = -25, None, None, 0
        publication = json.loads((raw / "publication.json").read_text()) if (raw / "publication.json").exists() else {}
        while True:
            job = json.loads((raw / "job-status.json").read_text())
            count = job.get("completed", 0)
            terminal = job["state"] in {"complete", "failed", "stopped"}
            phase_changed = last_phase is not None and job.get("phase") != last_phase
            state_changed = last_state is not None and job["state"] != last_state
            due = count >= publication.get("deepinfra_completed", 0) + 250 or terminal or phase_changed or state_changed or args.once
            try:
                if due:
                    asyncio.run(reconcile(raw))
                if terminal and job["state"] == "complete" and not (raw / "audit.json").exists():
                    build(output=output, deepinfra_raw=raw)
                    audit(raw, output)
                rebuilt = False
                if due and not args.no_publish:
                    publication = publish(raw, output, args.pr)
                    rebuilt = True
                elif count >= last_built + 25 or (count > last_built and time.monotonic() - last_time >= 60) or phase_changed or state_changed or terminal or args.once:
                    build(output=output, deepinfra_raw=raw)
                    rebuilt = True
                save(raw / "watch-status.json", {"state": "finished" if terminal else "watching", "completed": count,
                                                 "updated_at": now(), "error": None})
                if rebuilt:
                    last_built, last_time = count, time.monotonic()
                last_phase = job.get("phase")
                last_state = job["state"]
                if terminal or args.once:
                    return
            except Exception as error:
                save(raw / "watch-status.json", {"state": "attention", "completed": count,
                     "updated_at": now(), "error": f"{type(error).__name__}: {error}"})
                print(f"Checkpoint error: {type(error).__name__}: {error}", flush=True)
                # A publishing outage must not freeze local inference progress.
                try:
                    build(output=output, deepinfra_raw=raw)
                except Exception as refresh_error:
                    print(f"Local refresh error: {type(refresh_error).__name__}: {refresh_error}", flush=True)
                if args.once:
                    raise
            time.sleep(20)


if __name__ == "__main__":
    main()
