# Full CDS-bench model evaluation

Snowball v0.1: 3260/3,260 fresh attempts (complete). GPT-6 Luna, Nemotron 3.5 Lightning and Qwen 3.7 Flash each have 3,260 matched completed attempts.

Download [report.html](report.html) and open it in a browser. It is self-contained.

[Summary](summary.json) · [Per-task metrics](by-task.csv) · [Final predictions](results.json) · [Settings](config.json)

[Snowball completion audit](snowball-v01-full-audit.json)

Snowball v0.1 is a 67B MoE model with 2B active parameters per token. The full run uses 32,768 aggregate generated tokens, a 1,800-second API request deadline, temperature 0.7 and concurrency 60. Its native thinking/tool transport and private access configuration match the validated pilot. No per-request monetary price is exposed.

Checkpoint: [open-athena/Marin-Snowball-67B-A2B-v0.1](https://huggingface.co/open-athena/Marin-Snowball-67B-A2B-v0.1/tree/adc0078a19a4383bc998fcdb43867b3360baf5f3), revision `adc0078a19a4383bc998fcdb43867b3360baf5f3`.

Accuracy by condition: [PNG](overall.png) · [SVG](overall.svg).

With the same Snowball weights, the required-execution variant improves accuracy from 26.3% to 37.6% on Easy/Python and 22.8% to 32.2% on Hard/Python. Python invocation rises from 10.9% to 95.9%.

This changes the tool-use protocol, combining a demonstration with mandatory execution and constrained code generation with thinking disabled. The gain is not an isolated one-shot prompting effect.

[Pooled task radar](task-radar.svg) · [PNG](task-radar.png) · [Task order and scores](task-radar.json). The report selector also shows each of the four evaluation conditions. Task codes and group colors stay fixed across views, with links to the full-name/ID key and existing details.

Nemotron versus Snowball, Easy conditions: [PNG](nemotron-snowball-easy.png) · [SVG](nemotron-snowball-easy.svg). The two panels share a legend and axes. Each point compares the same completed examples for one task; all failures count. The interactive report includes all five condition views and links points to task details.

[30-task model/provider pilot](../model-pilot-20261029/README.md). Sanitized requests, reasoning and Python traces are in the [public HF trace archive](https://huggingface.co/datasets/eczech/cds-bench-pr-2-traces); originals remain untracked locally. The original Luna evaluation is preserved in `../luna-high-20261029/`.

Failed Qwen request counts, timeout flags, usage, and charges are reconstructed from saved response metadata where the collector omitted them. Raw files and scores are unchanged; no requests are retried.
