# Luna baseline runner

This evaluator sends the released CDS-bench prompts to OpenRouter Chat Completions using `openai/gpt-6-luna`, the standard OpenAI provider, and reasoning effort `high`. It runs four fresh passes: easy without tools, hard without tools, easy with Python, then hard with Python. No pilot results are reused.

The benchmark specification, generated data, and reference answers are unchanged. The evaluator never imports the reference solvers.

## Setup and execution

Run from the repository root on macOS. The Python executor uses the operating system's `sandbox-exec`; it intentionally stops if that sandbox is unavailable.

```sh
uv sync --extra eval --locked
uv venv --python 3.12.9 --no-project build/luna-python-runtime
uv pip install --python build/luna-python-runtime/bin/python numpy==2.5.4

uv run --extra eval python -m cds_bench.eval_luna.run --prepare
uv run --extra eval python -m cds_bench.eval_luna.run --check-sandbox
uv run --extra eval python -m cds_bench.eval_luna.run
```

The controller reads `OPENROUTER_API_KEY` from its environment or a literal assignment in the user's zsh profile. Credentials are never saved in requests or passed to Python tools.

The default output directory is `evaluations/luna-high-20261029`. Raw requests, responses, reasoning records, tool programs/results, and billing metadata are saved in the **untracked** directory `scratch/luna-high-full-20261029-01`. The generated HTML report is self-contained and can be downloaded and opened offline. Published artifacts contain metrics and final predictions, not raw traces.

If local Parquet files are missing, the runner downloads the pinned Hugging Face revision. Both split checksums and the benchmark specification checksum must match `generation/release.json`. The run freezes inference-source hashes and refuses to continue if those files or request settings change.

A file lock prevents duplicate drivers for the same run. Finished attempts are reused on restart, but an interrupted attempt with a saved request and no result stops the runner for review. Failed model/API attempts are preserved and never automatically retried. Five consecutive API errors stop the phase. Request timeouts are recorded separately and do not trigger that outage check.

## Tool execution and budgets

`settings.py` contains the exact request parameters and tool schema. There is one function, `python(code: string)`. It exposes Python 3.12.9, its standard library, and NumPy 2.5.4; Biopython is not installed. Each call starts a fresh interpreter. Files persist only within the current example's workdir.

The macOS sandbox denies networking, subprocess creation, benchmark/answer/credential reads, and writes outside that workdir. Runtime files remain readable. The startup checks verify these restrictions before any paid requests.

Each example receives 32,768 generated tokens across all model turns, including reasoning and tool arguments. Each next request gets the remaining budget. Every individual API request has a hard 300-second wall deadline. At most eight Python calls can run, each with a 30-second wall/CPU limit and 16,384-character output limit per stream. The client returns only execution output and errors, never scoring feedback.

## Reports and costs

The report refreshes after every 25 completed examples and at phase boundaries. It shows task-level and family-level accuracy, costs, error categories, time, optional tool use, paired comparisons, and final predictions. The PR description is generated as `pr-body.md`, including the exact tool definition in a details section.

Streaming usage is logged per API request. At each phase boundary the runner retrieves generation metadata, including charges for cancelled requests, and checks provider identity and reported web activity. Costs without final billing metadata are explicitly marked. Network isolation is an executor setting, not an OpenRouter request parameter.

To rebuild a report without sending inference requests:

```sh
uv run --extra eval python -m cds_bench.eval_luna.report \
  scratch/luna-high-full-20261029-01 evaluations/luna-high-20261029
```

To publish report checkpoints to the existing evaluation PR, run the companion process:

```sh
uv run --extra eval python -m cds_bench.eval_luna.publish --pr 2
```

It updates only the report directory and the PR's original post after each 250 examples, at phase changes, and on completion or failure. It never creates comments. It refuses to commit if another branch is checked out or unrelated files are staged.

Raw traces remain local for subsequent analysis. Do not merge this evaluation branch.
