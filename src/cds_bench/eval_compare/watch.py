"""Refresh the combined report and update only the evaluation PR's original post."""

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

from .audit import audit
from .report import PR_TITLE, BRANCH, REPOSITORY, build
from .run import OUTPUT, RAW, ROOT


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def publish(raw, output, number):
    assert command("git", "branch", "--show-current") == BRANCH, "Evaluation branch changed; publication stopped"
    allowed = str(output.relative_to(ROOT))
    staged = command("git", "diff", "--cached", "--name-only").splitlines()
    assert all(p.startswith(allowed + "/") for p in staged), "Unrelated staged files; publication stopped"
    snapshot_root = raw / "publication-snapshot"
    snapshot = snapshot_root / allowed
    summary = build(raw, snapshot)
    if (raw / "audit.json").exists():
        shutil.copy2(raw / "audit.json", snapshot / "audit.json")
    # Keep the browser's live directory identical to the published snapshot.
    shutil.copytree(snapshot, output, dirs_exist_ok=True)
    subprocess.run(["git", "--git-dir", str(ROOT / ".git"), "--work-tree", str(snapshot_root),
                    "-C", str(snapshot_root), "add", "--", allowed], check=True)
    if command("git", "diff", "--cached", "--name-only"):
        subprocess.run(["git", "commit", "-m", f"Update matched Luna/Qwen evaluation: {summary['qwen_completed']}/3260 Qwen attempts"], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", BRANCH], cwd=ROOT, check=True)
    subprocess.run(["gh", "pr", "edit", str(number), "--repo", REPOSITORY,
                    "--title", PR_TITLE,
                    "--body-file", str(output / "pr-body.md")], cwd=ROOT, check=True)
    value = {"qwen_completed": summary["qwen_completed"], "state": summary["state"],
             "qwen_cost_usd": summary["totals"]["qwen"]["cost_usd"], "published_at": now(), "pr_number": number}
    save(raw / "publication.json", value)
    return value


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
        last_built, last_phase, last_built_at = -25, None, 0
        publication = json.loads((raw / "publication.json").read_text()) if (raw / "publication.json").exists() else {}
        while True:
            job = json.loads((raw / "job-status.json").read_text())
            completed = job.get("completed", 0)
            terminal = job["state"] in ("complete", "failed", "stopped")
            phase_changed = last_phase is not None and job.get("phase") != last_phase
            try:
                rebuilt = False
                due = completed >= publication.get("qwen_completed", 0) + 250 or terminal or phase_changed or args.once
                if due:
                    asyncio.run(reconcile(raw))
                if terminal and job["state"] == "complete" and not (raw / "audit.json").exists():
                    build(raw, output)
                    audit(raw, output)
                if due and not args.no_publish:
                    publication = publish(raw, output, args.pr)
                    rebuilt = True
                elif completed >= last_built + 25 or (completed > last_built and time.monotonic() - last_built_at >= 60) or phase_changed or terminal or args.once:
                    build(raw, output)
                    rebuilt = True
                save(raw / "watch-status.json", {"state": "watching" if not terminal else "finished", "completed": completed, "updated_at": now(), "error": None})
                if rebuilt:
                    last_built = completed
                    last_built_at = time.monotonic()
                last_phase = job.get("phase")
                if terminal or args.once:
                    return
            except Exception as error:
                save(raw / "watch-status.json", {"state": "attention", "completed": completed, "updated_at": now(), "error": f"{type(error).__name__}: {error}"})
                print(f"Checkpoint error: {type(error).__name__}: {error}", flush=True)
                if args.once:
                    raise
            time.sleep(20)


if __name__ == "__main__":
    main()
