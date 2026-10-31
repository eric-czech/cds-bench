# Full model comparison

The shared report compares GPT-6 Luna (high), Nemotron 3.5 Lightning,
Qwen 3.7 Flash (high) and Snowball v0.1. All four full runs are complete,
with 3,260 attempts per model on the same pinned dataset and no pilot-response
reuse. Models are displayed in descending pooled mean task accuracy.
The Snowball runner and local watcher are documented below.

## Completed OpenRouter runs

The Nemotron run evaluated `nvidia/nemotron-3.5-lightning` on `deepinfra/bf16`, with no provider fallback. It restarted all 3,260 attempts over the same pinned release used by the completed Luna/Qwen comparison. No CoreWeave or pilot responses were reused. Reasoning was enabled without an effort selector, seed was omitted, and all other request, tool and scoring settings matched the CoreWeave pilot. Each OpenRouter full run used ten concurrent examples.

```sh
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.deepinfra_run --prepare
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.deepinfra_run --preflight-only
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.deepinfra_supervise
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.deepinfra_watch --pr 2
```

The DeepInfra watcher is a finite local process. It refreshes the combined report at least once a minute when results change, reconciles billing and publishes after 250 new completions or a phase/state change, and performs the final audit before publishing completion. It updates only PR #2's title and original post, never comments or merges. It creates no scheduled chat messages. `--once` performs an immediate checkpoint; `--no-publish` refreshes locally. A stopped or failed driver remains partial and is not silently retried.

The CoreWeave run and watcher were stopped at the user's request. Its 353 completed attempts, including failures, are retained as an exploratory extension in `evaluations/model-pilot-20261029/coreweave-extra.html`, separately from its original 116-attempt pilot. Ten interrupted attempts are excluded from scoring and timing; their traces are preserved. Earlier recovery from CoreWeave rate limits and its original 111-result registration also remain intact. DeepInfra starts fresh rather than continuing or pooling these results.

The DeepInfra supervisor keeps ten-way parallelism and frozen request settings. If repeated HTTP 429s halt scheduling, it runs separate unscored availability checks with increasing delays and bounded recovery attempts, then resumes only unissued cases. It never repeats a scored request or switches provider. Recovery registration preserves source and retained-result hashes; the final audit verifies them.

Raw requests, reasoning and tool programs/results from the completed Nemotron run are untracked in `scratch/nemotron-lightning-deepinfra-full-20261030-01`; the stopped CoreWeave run remains in `scratch/nemotron-lightning-coreweave-full-20261030-01`. The shared report remains in `evaluations/luna-qwen-high-20261029` to preserve existing links. Its per-task metrics, failure counts, costs, settings and plots include all four full models. CoreWeave is excluded from the main tables, plots and exports and appears only with the pilot results. Historical predictions and scores are unchanged. A complete DeepInfra audit is published as `deepinfra-audit.json`; the older Qwen audit remains `audit.json`.

```sh
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.lightning_audit --raw-root scratch/nemotron-lightning-deepinfra-full-20261030-01 --partial
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.lightning_audit --raw-root scratch/nemotron-lightning-deepinfra-full-20261030-01
```

The commands below reproduce the earlier Qwen run and its original watcher.

The Qwen run makes 3,260 fresh attempts over the pinned current release: 840 easy and 790 hard prompts, each without tools and with offline Python. It requests `high` reasoning on `qwen/qwen3.7-flash`, pinned to `alibaba/us-east-1`, using the same budgets, scorer, and Python tool as Luna. Pilot responses are not reused.

The comparison retains 3,260 historical Luna attempts whose full prompt and reference-answer records match the current release. It excludes only the 100 historical attempts corresponding to five removed duplicate hard variants. The original Luna reports and raw responses remain intact.

```sh
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.run --prepare
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.run --preflight-only
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.run
```

Run the watcher separately to refresh the report, reconcile costs at publication checkpoints, and update only the existing evaluation PR's original post:

```sh
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.watch --pr 2
```

The watcher publishes after 250 new completions, phase changes, or termination. Use `--once` for an immediate checkpoint or `--no-publish` to update locally. It refuses to publish if another branch is checked out or unrelated changes are staged. It never creates PR comments or merges the PR.

During inference, the local report refreshes at least once a minute when new attempts complete. The browser checks for newer snapshots every 30 seconds and preserves expanded sections, selected instances, the task filter, and scroll position when refreshing. Each snapshot displays its generation time; downloaded reports still work offline.

Raw requests, streaming responses, reasoning, and Python programs/results remain untracked in `scratch/qwen37-flash-high-full-20261029-01`. Public final predictions, metrics, figures, exact request settings, and a self-contained HTML report go in `evaluations/luna-qwen-high-20261029`.

There are no inference retries. A lock prevents duplicate drivers. Completed results are reused on restart, but interrupted requests without a result require review. Five consecutive non-timeout API errors halt scheduling. Each API request has a 300-second wall deadline; all model turns share 32,768 generated tokens per example. Eight Python executions are allowed, with a fresh interpreter and a 30-second limit per call. The macOS sandbox denies networking, subprocesses, and access to benchmark answers or credentials.

To rebuild the report or audit the finished run without new inference:

```sh
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.report
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.audit
```

The PR also includes the earlier 30-task model/provider pilot. Publish its saved results
before rebuilding the full report, so the PR's original post includes its summary table:

```sh
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.pilot
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.report
```

The pilot publisher reads the saved local comparison and verifies prompts, reference
answers, scores and aggregate metrics before exporting final predictions, settings and a
self-contained report to `evaluations/model-pilot-20261029`. It makes no inference
requests. Interrupted attempts and discarded duplicate-hard cases remain excluded.
Provider diagnostics and earlier ten-example checks stay separate from the pilot scores.
Original requests, reasoning and Python traces remain untracked locally. Sanitized
copies are published in the public [PR #2 trace archive](https://huggingface.co/datasets/eczech/cds-bench-pr-2-traces).
`tables.py` defines
the shared result columns used by both reports and the PR original post.

`audit --partial` checks only completed attempts during a run. A separately labeled diagnostic accepts inline final-answer markers and outer JSON fences; it never changes primary scores. The endpoint's quantization and public-weight availability are not established by this evaluation.

The Qwen client and legacy trial have different request-exception classes. Failed requests remain failures, but their raw result files omit some request metadata. `collector.py` reconstructs counts, timeout flags, usage, and charges from saved response files for reporting. It leaves raw traces and scores untouched. The frozen driver's circuit breaker does not recognize these omitted error flags; the completion check monitors operational failures separately.

The controller requests a maximum of 32,768 generated tokens across all turns. Some Qwen responses report usage above that allowance. The audit lists each exception, verifies the requested allowance and that no later request follows exhaustion, and reports `token_budgets_verified=false` when exceptions exist. Actual provider-reported usage is therefore not strictly matched to Luna's requested cap.
## Full Snowball v0.1 evaluation

Use the existing private Marin service; this runner does not deploy a model.
Pass the local access file without copying its capability URL into the repository:

```sh
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.snowball_run --access-file /path/to/access.json
```

The pinned `open-athena/Marin-Snowball-67B-A2B-v0.1` release at
`adc0078a19a4383bc998fcdb43867b3360baf5f3` receives 3,260 fresh attempts.
The four passes contain 840 easy and 790 hard cases, without tools and with
the existing offline Python function. Pilot responses are not reused.
The validated native thinking/tool transport, temperature 0.7, seed 20261029,
32,768 aggregate generated tokens, 1,800-second API wall deadline and concurrency
60 are retained. The scorer, Python/NumPy sandbox and eight-call limit are unchanged.
Requests are not retried; an interrupted attempt requires review before continuing.
The access file and credentials remain private; sanitized traces are in the public
archive. The API exposes token usage but no
per-request monetary price.

The local watcher refreshes the existing report and runs the completion audit:

```sh
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.snowball_watch
```

It reads saved run files and creates no scheduled chat messages. It does not
publish remotely; reviewed snapshots are committed and the existing PR post is
updated separately.

The completed Snowball run passed `snowball-v01-full-audit.json`: all 3,260
prompts, reference answers, scores, exported predictions, frozen inference
source hashes and token budgets were verified. The audit also checks the
sandbox, native tool round trips and response IDs distinct from the pilot.
All earlier model predictions remain unchanged. A connection-timeout burst
caused one operational resume after 477 attempts; only unissued cases resumed,
with unchanged settings and no scored retries. Failures remain in the results.

The report includes task scatter plots for Nemotron versus Snowball, using
the radar group colors and identical completed cases on both axes. The PR
exports the two Easy conditions in one figure with a shared legend.

## Public trace archive

[eczech/cds-bench-pr-2-traces](https://huggingface.co/datasets/eczech/cds-bench-pr-2-traces)
preserves full runs, pilots, smoke checks, explicit retries, interrupted attempts
and provider/tool diagnostics. Its `requests`, `attempts`, `tools`, `runs` and
`artifacts` configurations support analysis; compressed bundles retain the
sanitized source JSON, SSE streams, programs, outputs and operational records.
The dataset includes the exporter, validator, bundle checksums and coverage audit.
Original local files stay untracked and unchanged. Future exports for PR #2
should update the same dataset, retaining source-relative IDs and Hub revisions.
