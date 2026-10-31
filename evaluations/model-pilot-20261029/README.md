# CDS-bench: 30-task model pilot

Download [report.html](report.html) and open it in a browser. The report is self-contained and includes model/provider settings, four-condition scores, failure categories, timing, task-level results and GPT-OSS compatibility checks.

Complete runs contain 116 attempts: one instance from 30 task types, easy/hard, without tools/with offline Python, excluding two duplicate hard variants. Partial runs are marked explicitly. The stopped CoreWeave expansion is retained as 353 additional exploratory attempts in [its own report](coreweave-extra.html), separate from the original 116-attempt pilot. Scores and raw traces are unchanged. Earlier ten-example checks and GPT-OSS diagnostics are separate.

[Summary and settings](summary.json) · [Per-task outcomes](by-task.csv) · [Final predictions](results.json) · [Prompt instances](cases.json) · [GPT-OSS diagnostics](gpt-oss-diagnostics.json)

Sanitized API requests, reasoning and Python programs/results are in the [public HF trace archive](https://huggingface.co/datasets/eczech/cds-bench-pr-2-traces); originals remain untracked locally. Costs exclude preflight and interrupted requests; unknown charges are not assumed to be zero.

Rebuild from the saved local pilot, with no inference:

```sh
PYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.pilot
```

Snowball RC (2026.09.21): [Grug-67B-A2B-Datakit-SFT-262K-2026.09.21](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21/tree/b8c07f7df1df65525abbfdbcd1572318ba11c42f), `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21` at revision `b8c07f7df1df65525abbfdbcd1572318ba11c42f`. Uses the internal API with native thinking/tool transport, a 16,384-token aggregate budget, 1,800-second request deadline, temperature 0.7 and concurrency 60. Monetary cost is unavailable. [Pilot audit](snowball-audit.json).

Snowball v0.1: [Marin-Snowball-67B-A2B-v0.1](https://huggingface.co/open-athena/Marin-Snowball-67B-A2B-v0.1/tree/adc0078a19a4383bc998fcdb43867b3360baf5f3), `open-athena/Marin-Snowball-67B-A2B-v0.1` at revision `adc0078a19a4383bc998fcdb43867b3360baf5f3`. Uses the internal API with native thinking/tool transport, a 32,768-token aggregate budget, 1,800-second request deadline, temperature 0.7 and concurrency 60. Monetary cost is unavailable. [Pilot audit](snowball-v01-audit.json).
