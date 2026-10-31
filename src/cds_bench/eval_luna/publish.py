"""Publish report checkpoints to the existing PR; never touches raw traces."""

import argparse
import fcntl
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from .io import save
from .report import build

ROOT = Path(__file__).resolve().parents[3]
BRANCH = "eval/luna-high-baseline"
REPOSITORY = "Open-Athena/cds-bench"


def command(*args, capture=False):
    return (
        subprocess.check_output(args, cwd=ROOT, text=True).strip()
        if capture
        else subprocess.run(args, cwd=ROOT, check=True)
    )


def publish(raw_root, output, number):
    if command("git", "branch", "--show-current", capture=True) != BRANCH:
        raise RuntimeError("Evaluation branch is no longer checked out; publication stopped")
    allowed = str(output.relative_to(ROOT))
    # Stage a separate, finished snapshot. The live runner may keep regenerating
    # its report while this process commits and pushes the snapshot.
    staging_root = raw_root / "publication-worktree"
    snapshot = staging_root / allowed
    build(raw_root, snapshot)
    with (raw_root / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        staged = command("git", "diff", "--cached", "--name-only", capture=True).splitlines()
        if any(not path.startswith(allowed + "/") for path in staged):
            raise RuntimeError(
                "Unrelated staged files found; publication stopped without committing them"
            )
        command(
            "git",
            "--git-dir",
            str(ROOT / ".git"),
            "--work-tree",
            str(staging_root),
            "-C",
            str(staging_root),
            "add",
            "--",
            allowed,
        )
        changed = command("git", "diff", "--cached", "--name-only", capture=True)
        summary = json.loads((snapshot / "summary.json").read_text())
        body = raw_root / "published-pr-body.md"
        body.write_text((snapshot / "pr-body.md").read_text())
        if changed:
            command(
                "git",
                "commit",
                "-m",
                f"Update Luna baseline: {summary['completed_attempts']}/3360 attempts",
            )
    command("git", "push", "origin", BRANCH)
    # Updating the OP keeps one concise post; no PR comments are created.
    command("gh", "pr", "edit", str(number), "--repo", REPOSITORY, "--body-file", str(body))
    value = {
        "completed": summary["completed_attempts"],
        "state": summary["state"],
        "cost_usd": summary["total"]["cost_usd"],
        "published_at": datetime.now(UTC).isoformat(),
        "pr_number": number,
    }
    save(raw_root / "publication.json", value)
    print(json.dumps(value), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path, default=ROOT / "scratch/luna-high-full-20261029-01"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "evaluations/luna-high-20261029")
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.raw_root, args.output = args.raw_root.resolve(), args.output.resolve()
    with (args.raw_root / "publisher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.once:
            publish(args.raw_root, args.output, args.pr)
            return
        previous = {}
        state_path = args.raw_root / "publication.json"
        if state_path.exists():
            previous = json.loads(state_path.read_text())
        last_phase = None
        while True:
            job = json.loads((args.raw_root / "job-status.json").read_text())
            completed = job.get("completed", 0)
            terminal = job["state"] in ("complete", "failed")
            due = completed >= previous.get("completed", 0) + 250 or terminal
            if last_phase is not None and job.get("phase") != last_phase:
                due = True
            if due:
                previous = publish(args.raw_root, args.output, args.pr)
                if terminal:
                    return
            last_phase = job.get("phase")
            time.sleep(20)


if __name__ == "__main__":
    main()
