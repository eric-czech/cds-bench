Full CDS-bench baselines for **GPT-6 Luna (high)**, **Nemotron 3.5 Lightning**, **Qwen 3.7 Flash (high)** and **Snowball v0.1**, with and without offline Python.

Evaluation-only PR. **Do not merge.**

## Full evaluation

The dataset has ten examples per task: **84 easy task types and 79 hard task types**. Each original baseline has 3,260 matching completed attempts, or **13,040 attempts** across the four models. Retained prompts and reference answers are identical; pilot responses are not reused. The main chart also includes a separate 1,630-attempt Snowball tools-only follow-up with modified prompting and execution control.

[Full report (HTML)](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/report.html) · [Per-task results (CSV)](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/by-task.csv) · [Final predictions (JSON)](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/results.json) · [Traces (HF)](https://huggingface.co/datasets/eczech/cds-bench-pr-2-traces)

The HTML reports are self-contained; download and open them in a browser.

### Accuracy by condition

![CDS-bench accuracy by model and evaluation condition](https://raw.githubusercontent.com/Open-Athena/cds-bench/refs/heads/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/overall.png?v=3a2bb17e2722)

Exports: [PNG](https://raw.githubusercontent.com/Open-Athena/cds-bench/refs/heads/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/overall.png?v=3a2bb17e2722) · [SVG](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/overall.svg)

Hatched bars show the separate **Snowball v0.1 (1-shot tools)** follow-up in the two Python conditions. Its prompt and execution changes are explained under [Snowball tool use](#snowball-tool-use).

<details>
<summary>Full results: accuracy, median time, cost and truncation</summary>


| Model | Condition | Correct | Accuracy | Median | Cost | Truncated |
|---|---|---:|---:|---:|---:|---:|
| GPT-6&nbsp;Luna&nbsp;(high) | Easy,&nbsp;no&nbsp;tools | 738/840 | 87.9% | 8.6s | $0.63658 | 1 |
| GPT-6&nbsp;Luna&nbsp;(high) | Hard,&nbsp;no&nbsp;tools | 679/790 | 85.9% | 9.5s | $0.59889 | 0 |
| GPT-6&nbsp;Luna&nbsp;(high) | Easy,&nbsp;Python | 793/840 | 94.4% | 7.6s | $0.43532 | 0 |
| GPT-6&nbsp;Luna&nbsp;(high) | Hard,&nbsp;Python | 710/790 | 89.9% | 8.0s | $0.39479 | 0 |
| Nemotron&nbsp;3.5&nbsp;Lightning | Easy,&nbsp;no&nbsp;tools | 203/840 | 24.2% | 161.5s | $2.92877 | 124 |
| Nemotron&nbsp;3.5&nbsp;Lightning | Hard,&nbsp;no&nbsp;tools | 185/790 | 23.4% | 157.4s | $3.10986 | 163 |
| Nemotron&nbsp;3.5&nbsp;Lightning | Easy,&nbsp;Python | 537/840 | 63.9% | 28.3s | $1.83524 | 4 |
| Nemotron&nbsp;3.5&nbsp;Lightning | Hard,&nbsp;Python | 397/790 | 50.3% | 40.7s | $2.37400 | 7 |
| Qwen&nbsp;3.7&nbsp;Flash&nbsp;(high) | Easy,&nbsp;no&nbsp;tools | 227/840 | 27.0% | 235.3s | $3.12415 | 293 |
| Qwen&nbsp;3.7&nbsp;Flash&nbsp;(high) | Hard,&nbsp;no&nbsp;tools | 199/790 | 25.2% | 250.8s | $2.99366 | 310 |
| Qwen&nbsp;3.7&nbsp;Flash&nbsp;(high) | Easy,&nbsp;Python | 386/840 | 46.0% | 52.2s | $1.27409 | 13 |
| Qwen&nbsp;3.7&nbsp;Flash&nbsp;(high) | Hard,&nbsp;Python | 332/790 | 42.0% | 70.6s | $1.49165 | 23 |
| Snowball&nbsp;v0.1 | Easy,&nbsp;no&nbsp;tools | 179/840 | 21.3% | 109.6s | Unavailable | 109 |
| Snowball&nbsp;v0.1 | Hard,&nbsp;no&nbsp;tools | 158/790 | 20.0% | 119.5s | Unavailable | 103 |
| Snowball&nbsp;v0.1 | Easy,&nbsp;Python | 221/840 | 26.3% | 110.4s | Unavailable | 91 |
| Snowball&nbsp;v0.1 | Hard,&nbsp;Python | 180/790 | 22.8% | 122.2s | Unavailable | 96 |

Models are ordered by pooled mean task accuracy, giving each task equal weight; the order stays fixed across conditions. All failures remain in the denominator. Median time includes tool round trips and failures. **Truncated** counts token-cap failures; costs cover scored attempts and exclude the pilot.

</details>

Scores count incorrect answers, formatting failures, timeouts, token caps and API errors as failures. Their counts and billing caveats are below.

<details>
<summary>Full-evaluation failures, billing and token-budget exceptions</summary>

| Model | Condition | Wrong | Format | Timeout | Cap | API |
|---|---|---:|---:|---:|---:|---:|
| GPT-6&nbsp;Luna&nbsp;(high) | Easy,&nbsp;no&nbsp;tools | 90 | 3 | 8 | 1 | 0 |
| GPT-6&nbsp;Luna&nbsp;(high) | Hard,&nbsp;no&nbsp;tools | 103 | 2 | 6 | 0 | 0 |
| GPT-6&nbsp;Luna&nbsp;(high) | Easy,&nbsp;Python | 42 | 4 | 0 | 0 | 1 |
| GPT-6&nbsp;Luna&nbsp;(high) | Hard,&nbsp;Python | 63 | 1 | 0 | 0 | 16 |
| Nemotron&nbsp;3.5&nbsp;Lightning | Easy,&nbsp;no&nbsp;tools | 251 | 139 | 86 | 124 | 37 |
| Nemotron&nbsp;3.5&nbsp;Lightning | Hard,&nbsp;no&nbsp;tools | 260 | 137 | 43 | 163 | 2 |
| Nemotron&nbsp;3.5&nbsp;Lightning | Easy,&nbsp;Python | 87 | 194 | 3 | 4 | 15 |
| Nemotron&nbsp;3.5&nbsp;Lightning | Hard,&nbsp;Python | 146 | 237 | 3 | 7 | 0 |
| Qwen&nbsp;3.7&nbsp;Flash&nbsp;(high) | Easy,&nbsp;no&nbsp;tools | 72 | 198 | 17 | 293 | 33 |
| Qwen&nbsp;3.7&nbsp;Flash&nbsp;(high) | Hard,&nbsp;no&nbsp;tools | 65 | 163 | 44 | 310 | 9 |
| Qwen&nbsp;3.7&nbsp;Flash&nbsp;(high) | Easy,&nbsp;Python | 75 | 359 | 3 | 13 | 4 |
| Qwen&nbsp;3.7&nbsp;Flash&nbsp;(high) | Hard,&nbsp;Python | 112 | 309 | 4 | 23 | 10 |
| Snowball&nbsp;v0.1 | Easy,&nbsp;no&nbsp;tools | 153 | 382 | 0 | 109 | 17 |
| Snowball&nbsp;v0.1 | Hard,&nbsp;no&nbsp;tools | 145 | 384 | 0 | 103 | 0 |
| Snowball&nbsp;v0.1 | Easy,&nbsp;Python | 167 | 355 | 0 | 91 | 6 |
| Snowball&nbsp;v0.1 | Hard,&nbsp;Python | 161 | 351 | 0 | 96 | 2 |

Recorded charges: **$10.24787 Nemotron 3.5 Lightning**, **$8.88355 Qwen**, **$2.06559 retained Luna**. Billing is incomplete for 53 Nemotron 3.5 Lightning, 29 Qwen and 16 retained Luna attempts. Unknown charges are not assumed to be zero.

Snowball’s internal API exposes token usage but no per-request monetary price. Its cost is unavailable, not zero; GPU allocation cost is not attributed to individual requests.

Qwen reports usage beyond the requested 32,768-token budget for 24 attempts, up to 64,923 tokens. The [audit](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/audit.json) lists these exceptions. Actual provider-reported usage is not strictly matched to Luna’s requested cap.

</details>

### Task performance: All conditions (pooled)

![CDS-bench pooled task accuracy for all 84 task types, with one radar panel per model](https://raw.githubusercontent.com/Open-Athena/cds-bench/refs/heads/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/task-radar.png?v=89fc0b242a9d)

Exports: [PNG](https://raw.githubusercontent.com/Open-Athena/cds-bench/refs/heads/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/task-radar.png?v=89fc0b242a9d) · [SVG](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/task-radar.svg)

Each spoke pools completed attempts for one task. Subtitles average available task accuracies equally; five tasks have only easy variants. Missing results are unscored, not zero. Groups and tasks share the same order across models. The [report’s condition selector and task key](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/report.html#task-radar) show the four separate conditions and map codes to full task names and IDs.

### Nemotron versus Snowball: Easy examples

![Task accuracy for Nemotron and Snowball: Easy without tools and Easy with Python](https://raw.githubusercontent.com/Open-Athena/cds-bench/refs/heads/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/nemotron-snowball-easy.png?v=6a10f0fd7a02)

Exports: [PNG](https://raw.githubusercontent.com/Open-Athena/cds-bench/refs/heads/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/nemotron-snowball-easy.png?v=6a10f0fd7a02) · [SVG](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/nemotron-snowball-easy.svg)

Each point is a task, colored by the same groups as the radars. Both axes use identical completed cases. Above the dashed equality line Snowball scores higher; below it Nemotron does. Numbered markers combine overlapping tasks. The [interactive report](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/report.html#task-scatter) shows task names, IDs and scores when you select a point.

## Snowball tool use

### Optional tool use

With Python available, Snowball invoked it on 177/1,630 examples (10.9%), versus 1,544/1,630 (94.7%) for Nemotron. These are optional tool-use conditions; supplying a tool does not require the model to call it. In the one-shot pilot, 7 of 22 final responses missing the required final-answer marker still ended with the correct answer; two also prefixed it with the demonstration's arithmetic result.

For Easy/Python, 33 task types have at least two examples with a Python call and two without one. Averaging accuracy equally across these same task types gives 54.8% with calls versus 23.2% without (+31.6 percentage points). This compares different examples within each task and is observational, not a causal estimate of tool benefit. [Per-task breakdown](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/report.html#snowball-tools).

<details>
<summary>Snowball follow-up: one tool-use demonstration on the matched pilot</summary>

The one-demonstration Snowball pilot used Python on 10/58 examples versus 5/58 originally. Correct answers: 12/58 (20.7%) versus 15/58 (25.9%). This small matched repeat does not show an accuracy benefit.

Fresh attempts on the original 30 easy and 28 hard pilot questions. Each benchmark user message is unchanged byte for byte, but the full API message history differs: an arithmetic question, assistant Python call, tool result and assistant example answer are prepended before it. The demonstration shows both tool use and final-answer formatting. Baseline request transformation, response parser and scorer are unchanged. One draw per question; no causal isolation of formatting versus tool-selection effects.

| Condition | Original correct | With demo | Original tool users | With demo |
|---|---:|---:|---:|---:|
| Easy, Python | 7/30 (23.3%) | 8/30 (26.7%) | 3/30 | 7/30 |
| Hard, Python | 8/28 (28.6%) | 4/28 (14.3%) | 2/28 | 3/28 |

Same native XML transformation/parser, optional Python tool, 32,768 aggregate generated tokens, temperature 0.7, seed 20261029, concurrency 60 and 1,800-second API deadline. The demonstration is the only change to model-facing context/settings. Its call is excluded from measured tool use. Cost is unavailable. The full baselines and pilot model-ranking table are unchanged.

Uses the full-run client with the identical native request transformation and response parser. Relative to the original pilot, it adds error-stage metadata and a 15-second connection-setup timeout instead of the generic 1,800-second timeout. No connection failures occurred in this follow-up.

[Follow-up results and exact demonstration](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/report.html#snowball-oneshot). No scored retries; all 58 attempts passed the source/prompt/budget/scoring audit.

</details>

### Required Python execution follow-up

Hatched bars show Snowball v0.1 (1-shot tools): 316/840 (37.6%) on Easy/Python and 254/790 (32.2%) on Hard/Python. The same weights receive one arithmetic tool-use demonstration plus instructions requiring a complete Python solution. Initial code generation is schema constrained with thinking disabled; thinking resumes after successful execution. The model writes execution-error repairs within the same eight-execution limit. Python was invoked on 1,563/1,630 examples (95.9%); the sandbox, 32,768-token budget and final scorer are unchanged. This is directed tool use, so the improvement cannot be attributed to the demonstration alone. No no-tools cases were run for this variant. It appears only in this chart; the tables, radars and scatter plots retain the four original baselines.

<details>
<summary>Snowball v0.1 (1-shot tools): prompt and execution changes</summary>


[Exact example request](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/snowball-v01-one-shot-tools-example.txt) · [Aggregates, prompt instructions and schema](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/snowball-v01-one-shot-tools.json) · [Completion audit](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/snowball-v01-one-shot-tools-audit.json)

An arithmetic question, assistant Python call, tool result and assistant answer precede each current question. Generic instructions are added before and after the original benchmark prompt. The original sequence inputs, references, answer and scoring rules are unchanged. The first response must contain exactly one `code` string; the controller executes it verbatim and returns the result. Failed executions prompt model-written complete repair programs; no gold-answer feedback or controller repair is supplied. After successful execution, native thinking and optional XML Python calls resume. The final answer has no decoding grammar and is scored as originally specified.

The full run contains 1,630 fresh attempts, separate from calibration, with no scored retries. Temperature 0.7, seed 20261029, concurrency 60 and a 1,800-second request deadline match the original Snowball setup. Only the main chart includes this follow-up. Its raw traces remain local and are not yet in the linked HF archive.

Same 67B total / 2B active checkpoint: [open-athena/Marin-Snowball-67B-A2B-v0.1](https://huggingface.co/open-athena/Marin-Snowball-67B-A2B-v0.1/tree/adc0078a19a4383bc998fcdb43867b3360baf5f3), revision `adc0078a19a4383bc998fcdb43867b3360baf5f3`.

</details>

## 30-task pilot

The pilot tested **13 models across 14 model/provider runs**, including **DeepSeek V4.1 Flash 1017**, **Beam 1.0 501B A23B**, **Gemma 4 31B** and **GPT-OSS 20B**. One judgment-selected instance per task, each in easy/hard form and with/without Python. Two duplicate hard variants are omitted, leaving **30 easy + 28 hard prompts and 116 attempts per complete model**. The pilot checks accuracy, cost, latency and tool compatibility. It is too small for a definitive model ranking.

[Pilot report (HTML)](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/model-pilot-20261029/report.html) · [Pilot per-task results (CSV)](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/model-pilot-20261029/by-task.csv). The report covers provider behavior, tool compatibility, timing and failure categories. Partial runs and differences in token limits remain explicit.

**Internal API exceptions:** both Snowball models have **67B total parameters and 2B active per token**, and use **1,800 seconds/API request**, with concurrency 60. **Snowball RC (2026.09.21) uses 16,384 aggregate tokens; Snowball v0.1 uses 32,768.** RC means release candidate. The older checkpoint’s original 32,768-token/300-second trial is preserved separately and excluded.

<details>
<summary>Pilot results: 14 model/provider runs, with accuracy, timing and costs</summary>

| Model | Condition | Correct | Accuracy | Median | Cost | Truncated |
|---|---|---:|---:|---:|---:|---:|
| GPT-6&nbsp;Luna&nbsp;(high) | All&nbsp;four | 107/116 | 92.2% | 8.6s | $0.07805 | 0 |
| DeepSeek&nbsp;Flash | Partial | 93/101 | 92.1% | 13.5s | $0.12730 | 2 |
| Beam&nbsp;501B&nbsp;A23B | All&nbsp;four | 99/116 | 85.3% | 6.9s | $0.62241 | 5 |
| Nemotron&nbsp;Ultra | Partial | 69/100 | 69.0% | 130.3s | $0.00000 | 0 |
| Nemotron&nbsp;3.5&nbsp;Lightning&nbsp;CoreWeave | All&nbsp;four | 45/116 | 38.8% | 45.9s | $0.46213 | 35 |
| Qwen&nbsp;3.7&nbsp;Flash&nbsp;(high) | All&nbsp;four | 47/116 | 40.5% | 143.0s | $0.30059 | 16 |
| North&nbsp;Mini&nbsp;Code | Partial | 13/33 | 39.4% | 159.4s | $0.00000 | 1 |
| GPT-OSS&nbsp;20B | All&nbsp;four | 37/116 | 31.9% | 26.7s | $0.35233 | 19 |
| Nemotron&nbsp;3.5&nbsp;Lightning&nbsp;io.net | Partial | 21/81 | 25.9% | 154.7s | $0.12087 | 2 |
| Snowball&nbsp;v0.1 | All&nbsp;four | 27/116 | 23.3% | 110.3s | Unavailable | 11 |
| Gemma&nbsp;31B | Partial | 5/24 | 20.8% | 300.0s | $0.04795 | 0 |
| Snowball&nbsp;RC&nbsp;(2026.09.21) | All&nbsp;four | 16/116 | 13.8% | 154.8s | Unavailable | 61 |
| LFM&nbsp;24B&nbsp;A2B | All&nbsp;four | 9/116 | 7.8% | 20.8s | $0.65148 | 99 |
| Gemma&nbsp;26B&nbsp;A4B | Partial | 8/100 | 8.0% | 220.2s | $0.35711 | 6 |
| Nemotron&nbsp;3.5&nbsp;Lightning&nbsp;CoreWeave&nbsp;extra | Easy,&nbsp;no&nbsp;tools | 75/353 | 21.2% | 95.5s | $1.50838 | 134 |

**All four** pools easy/hard with/without Python. **Partial** means the run was stopped early and pools only completed attempts across the conditions reached. These subsets are not directly comparable to complete runs. North and Gemma 31B never reached a scored Python condition. The CoreWeave extra row retains all 353 completed attempts from the stopped expansion, separately from its original 116-attempt pilot. Ten interrupted attempts are excluded. [Inspect the additional samples](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/model-pilot-20261029/report.html#coreweave-extra). Full model names, provider routing and failure counts are in the pilot report.

LFM used the approved **8,192-token** budget and exhausted it on 99/116 attempts. GPT-OSS/Bedrock’s tool scores include **24 API/protocol failures**; separate CoreWeave/Groq checks did not establish a reliable OpenRouter tool setup. Gemma, North, LFM and Nemotron 3.5 Lightning enable reasoning without an effort selector; the remaining OpenRouter models request `high`. Nemotron 3.5 Lightning’s io.net run was stopped at the user’s request because it was slow; a separate fresh CoreWeave BF16 run uses the same prompts and settings. Both routes have no fallback and remain separate. io.net initially used concurrency ten, then continued only unissued cases at concurrency three with five-second request spacing; all completed failures are retained.

Costs cover these scored subsets, excluding compatibility checks, removed duplicate-hard cases, interrupted requests and follow-up diagnostics. Missing charges remain explicit. The reports separate incorrect values, formatting failures, timeouts, token caps and API/protocol errors. Earlier ten-example GLM/Luna checks are a separate report appendix, not pooled here.

**Snowball RC (2026.09.21)** uses the internal Marin/vLLM API, native thinking and tool-call tags, and the same offline Python function. A real tool round trip was verified before scoring. Uses 16,384 aggregate generated tokens/example and a 1,800-second API deadline, temperature 0.7 and concurrency 60; no verified effort selector or per-request price. Full checkpoint: [Grug-67B-A2B-Datakit-SFT-262K-2026.09.21](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21/tree/b8c07f7df1df65525abbfdbcd1572318ba11c42f), `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21` at revision `b8c07f7df1df65525abbfdbcd1572318ba11c42f`. Its transport and execution checks are documented in the pilot report.

**Snowball v0.1** uses the internal Marin/vLLM API, native thinking and tool-call tags, and the same offline Python function. A real tool round trip was verified before scoring. Uses 32,768 aggregate generated tokens/example and a 1,800-second API deadline, temperature 0.7 and concurrency 60; no verified effort selector or per-request price. Full checkpoint: [Marin-Snowball-67B-A2B-v0.1](https://huggingface.co/open-athena/Marin-Snowball-67B-A2B-v0.1/tree/adc0078a19a4383bc998fcdb43867b3360baf5f3), `open-athena/Marin-Snowball-67B-A2B-v0.1` at revision `adc0078a19a4383bc998fcdb43867b3360baf5f3`. Its transport and execution checks are documented in the pilot report.

</details>

## Settings and validation

Snowball v0.1 is a 67B MoE model with 2B active parameters per token. Full checkpoint: [open-athena/Marin-Snowball-67B-A2B-v0.1](https://huggingface.co/open-athena/Marin-Snowball-67B-A2B-v0.1/tree/adc0078a19a4383bc998fcdb43867b3360baf5f3), revision `adc0078a19a4383bc998fcdb43867b3360baf5f3`. Its full run uses the validated native API setup; no new deployment or pilot-response reuse.

| Setting | Value |
|---|---|
| Full models / providers | Luna / OpenAI; Qwen / `alibaba/us-east-1`; Nemotron / `deepinfra/bf16`; Snowball v0.1 / internal Marin API. Exact model identifiers are below; no fallback |
| Full reasoning | Luna/Qwen: `high`; Nemotron: reasoning enabled; Snowball: native `enable_thinking=true`. No effort selector is verified for Nemotron or Snowball |
| Full output / timeout | 32,768 aggregate generated tokens/example. OpenRouter: **300 seconds/API request**; Snowball: **1,800 seconds/API request** |
| Tools / web | One local Python function; no web tools. OpenRouter uses `auto` or an empty tools array with `none`. Snowball uses its native tool-call format |
| Executor | **`/usr/bin/sandbox-exec`**, `(deny network*)`; Python 3.12.9 + NumPy 2.5.4; **no Biopython** |
| Tool limits | Eight calls/example; 30 seconds/call; fresh interpreter; files persist within example |
| Sampling | OpenRouter: concurrency 10, provider-default temperature. Snowball: concurrency 60, temperature 0.7. Seed 20261029 for Luna/Qwen/Snowball; omitted for Nemotron |

<details>
<summary>Exact Python function supplied to the models</summary>

```json
[
  {
    "type": "function",
    "function": {
      "name": "python",
      "description": "Execute Python 3.12.9 code with the standard library and NumPy 2.5.4. Biopython is not installed. Networking and subprocesses are disabled; only runtime files and this example's working directory are readable. Use print() to return results. Each call starts a fresh interpreter; files in the working directory persist within this example. Limits: 30 seconds per call, 16384 characters each of stdout/stderr, and at most 8 calls per example. Returns stdout, stderr, exit_code, timed_out, and output_truncated.",
      "strict": true,
      "parameters": {
        "type": "object",
        "properties": {
          "code": {
            "type": "string",
            "description": "Python code to execute."
          }
        },
        "required": [
          "code"
        ],
        "additionalProperties": false
      }
    }
  }
]
```

OpenRouter receives an empty tools array for no-tools cases. Snowball omits the function definitions from its chat template. No hosted shell or web tool is registered.

</details>

<details>
<summary>Exact full-evaluation request settings</summary>

```json
{
  "luna": {
    "model": "openai/gpt-6-luna",
    "max_tokens": 32768,
    "seed": 20261029,
    "reasoning": {
      "effort": "high",
      "exclude": false
    },
    "tool_choice": "none",
    "tools": [],
    "plugins": [
      {
        "id": "web",
        "enabled": false
      },
      {
        "id": "response-healing",
        "enabled": false
      },
      {
        "id": "file-parser",
        "enabled": false
      }
    ],
    "provider": {
      "only": [
        "openai"
      ],
      "ignore": [
        "openai/flex",
        "openai/fast"
      ],
      "allow_fallbacks": false,
      "require_parameters": true
    },
    "stream": true,
    "stream_options": {
      "include_usage": true
    }
  },
  "qwen": {
    "model": "qwen/qwen3.7-flash",
    "max_tokens": 32768,
    "seed": 20261029,
    "reasoning": {
      "effort": "high",
      "exclude": false
    },
    "tool_choice": "none",
    "tools": [],
    "plugins": [
      {
        "id": "web",
        "enabled": false
      },
      {
        "id": "response-healing",
        "enabled": false
      },
      {
        "id": "file-parser",
        "enabled": false
      }
    ],
    "provider": {
      "only": [
        "alibaba/us-east-1"
      ],
      "allow_fallbacks": false,
      "require_parameters": true
    },
    "stream": true,
    "stream_options": {
      "include_usage": true
    }
  },
  "deepinfra": {
    "model": "nvidia/nemotron-3.5-lightning",
    "max_tokens": 32768,
    "reasoning": {
      "enabled": true,
      "exclude": false
    },
    "tool_choice": "none",
    "tools": [],
    "plugins": [
      {
        "id": "web",
        "enabled": false
      },
      {
        "id": "response-healing",
        "enabled": false
      },
      {
        "id": "file-parser",
        "enabled": false
      }
    ],
    "provider": {
      "only": [
        "deepinfra/bf16"
      ],
      "allow_fallbacks": false,
      "require_parameters": true
    },
    "stream": true,
    "stream_options": {
      "include_usage": true
    }
  },
  "snowball-v01": {
    "model": "[private serving model ID]",
    "max_tokens": 32768,
    "temperature": 0.7,
    "seed": 20261029,
    "stream": true,
    "stream_options": {
      "include_usage": true
    },
    "skip_special_tokens": false,
    "chat_template_kwargs": {
      "enable_thinking": true
    }
  }
}
```

OpenRouter adds the function above and uses `auto`, then `none` after the call limit. Snowball supplies the same function through its native chat template, as detailed below. Each round receives the remaining token budget. Returned reasoning is preserved between turns. Private Snowball access identifiers are redacted.

</details>

<details>
<summary>Local Python execution and network isolation</summary>

The controller executes returned Python programs locally and sends stdout/stderr, exit code and timeout status back as tool results. Network isolation is an OS policy, not an OpenRouter parameter; the controller retains API access.

```sh
/usr/bin/sandbox-exec -f "$workdir/sandbox.sb" \
  "$runtime/bin/python" -I -B -c "$launcher"
```

The policy denies `network*` and `process-fork`, restricts reads to runtime/system paths and the current example workdir, and restricts writes to that workdir. Benchmark answers and credentials are inaccessible. Startup checks verify these restrictions.

</details>

<details>
<summary>Dataset matching, scoring, validation and saved artifacts</summary>

Full-evaluation dataset revision: `26f63e8e488022bed14e12af0f1e44d87a28e14f`. Five duplicated hard variants were removed after the original Luna run. Its original 3,360-attempt artifacts remain intact; the combined comparison excludes only those 100 historical attempts. Retained prompts and reference answers match byte for byte. The pilot’s 30 task types contain two of those duplicated hard variants, so its comparison retains 116 attempts/model.

The deterministic scorer and benchmark definitions are unchanged. No scored inference requests are retried. A separate formatting diagnostic accepts an inline final-answer marker and removes one outer JSON fence; it never changes primary scores. GPT-OSS exploratory compatibility checks are separate from scored pilot attempts.

[Snowball completion audit passed](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-qwen-high-20261029/snowball-v01-full-audit.json): all 3,260 cases verified, with no scored retries or pilot-response reuse.

Failed Qwen request metadata omitted by the legacy trial is reconstructed from saved response and billing files for reporting. Original traces, final answers and scores are unchanged. Public artifacts include prompts, final predictions, settings and metrics. Sanitized API requests, reasoning and tool programs/results are archived on HF; originals remain untracked locally.

[Public request, reasoning and tool traces (HF)](https://huggingface.co/datasets/eczech/cds-bench-pr-2-traces) cover the original full runs, pilots, retries, interrupted attempts and provider diagnostics. Credentials and private endpoints are redacted. The required-execution follow-up’s traces remain local, as noted above.

</details>

<details>
<summary>Snowball tool calling: process, exact API tool definition and limitations</summary>

The service is not configured for standard automatic function parsing; the initial standard auto-tool probe returned HTTP 400. The controller passes the exact function below through chat_template_kwargs.xml_tools, with enable_thinking=true and skip_special_tokens=false. The checkpoint's own template describes calls as JSON objects with name and arguments fields between opening and closing tool_call tags. Generation stops at the closing tool_call tag, including that delimiter. The client parses the JSON, executes code verbatim locally, and supplies the result as a tool-role message containing stdout, stderr, exit_code, timed_out and output_truncated. Prior native reasoning and calls are replayed before the model continues. Tool use is optional; strict:true is embedded in the definition, but this native route does not enforce a decoding grammar. At eight calls the definitions are omitted.

The original full run recorded 240 successful program executions and 14 program failures. Exit zero is not evidence that the program or final answer was correct.

A later review found nine complete JSON calls inside unfinished thinking blocks. The original parser treated them as reasoning and returned no call; seven contain syntactically valid Python and two contain placeholder programs. These remain failed original attempts, including in the no-call subgroup. No scores were changed. This is a known adapter limitation.

The checkpoint template also advertises a direct Python-code channel using the python_tag and eom_id special tokens, with ipython replies. That alternative was not used or tested here. It advertises a stateful notebook, whereas this evaluation uses a fresh interpreter per call. Server-side parsing and a forced initial Python call are other possible diagnostics, not features of this baseline. The private capability URL and serving model ID are omitted. A real execution/result round trip passed before scoring.

Exact tool-related API fields, including the complete Python function definition:

```json
{
  "skip_special_tokens": false,
  "chat_template_kwargs": {
    "enable_thinking": true,
    "xml_tools": [
      {
        "type": "function",
        "function": {
          "name": "python",
          "description": "Execute Python 3.12.9 code with the standard library and NumPy 2.5.4. Biopython is not installed. Networking and subprocesses are disabled; only runtime files and this example's working directory are readable. Use print() to return results. Each call starts a fresh interpreter; files in the working directory persist within this example. Limits: 30 seconds per call, 16384 characters each of stdout/stderr, and at most 8 calls per example. Returns stdout, stderr, exit_code, timed_out, and output_truncated.",
          "strict": true,
          "parameters": {
            "type": "object",
            "properties": {
              "code": {
                "type": "string",
                "description": "Python code to execute."
              }
            },
            "required": [
              "code"
            ],
            "additionalProperties": false
          }
        }
      }
    ]
  },
  "stop": [
    "</tool_call>"
  ],
  "include_stop_str_in_output": true
}
```

The full run has a separate 15-second connection-setup limit. After a connection-timeout burst, only unissued cases resumed with unchanged settings; earlier failures remain counted.

</details>
