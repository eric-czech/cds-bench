"""macOS-sandboxed Python + NumPy; no networking or benchmark files."""

import json
import os
import signal
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "build/luna-python-runtime"
PYTHON = RUNTIME / "bin/python"
PYTHON_BASE = PYTHON.resolve().parents[1]
WORK_ROOT = ROOT / "build/luna-tool-work"
TIMEOUT_SECONDS = 30
OUTPUT_LIMIT = 16384
LAUNCHER = """import resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(30,30))
resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))
resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
code=sys.stdin.read()
exec(compile(code,"<python-tool>","exec"),{"__name__":"__main__"})
"""


def profile(directory):
    q = lambda p: json.dumps(str(Path(p).resolve()))
    readable = [
        PYTHON_BASE,
        RUNTIME,
        directory,
        "/System",
        "/usr/lib",
        "/usr/share",
        "/System/Volumes/Preboot/Cryptexes/OS/usr/lib",
        "/private/var/db/dyld",
        "/dev",
    ]
    exclusions = "\n".join("(require-not (subpath " + q(p) + "))" for p in readable)
    return f"""(version 1)
(allow default)
(deny network*)
(deny process-fork)
(deny file-read-data (require-all (require-not (literal "/")) {exclusions}))
(deny file-write* (require-all
    (require-not (subpath {q(directory)}))
    (require-not (literal "/dev/null"))))
"""


def execute(code, directory=None, timeout=TIMEOUT_SECONDS):
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    directory = Path(directory) if directory else WORK_ROOT / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sandbox.sb"
    path.write_text(profile(directory))
    # The runtime sees neither the runner environment nor the repository in
    # sys.path. All other file content and network access is blocked by macOS.
    environment = {
        "PATH": str(RUNTIME / "bin") + ":/usr/bin:/bin",
        "HOME": str(directory),
        "TMPDIR": str(directory),
        "LANG": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    timed_out = False
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        child = subprocess.Popen(
            ["/usr/bin/sandbox-exec", "-f", str(path), str(PYTHON), "-I", "-B", "-c", LAUNCHER],
            cwd=directory,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        try:
            child.communicate(code.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
    output = stdout_path.read_bytes()
    errors = stderr_path.read_bytes()
    return {
        "stdout": output[:OUTPUT_LIMIT].decode(errors="replace"),
        "stderr": errors[:OUTPUT_LIMIT].decode(errors="replace"),
        "exit_code": child.returncode,
        "timed_out": timed_out,
        "output_truncated": len(output) > OUTPUT_LIMIT or len(errors) > OUTPUT_LIMIT,
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(execute(sys.stdin.read()), indent=2))
