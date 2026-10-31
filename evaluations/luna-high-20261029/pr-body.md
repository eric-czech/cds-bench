Run GPT-6 Luna with **high** reasoning through OpenRouter's standard OpenAI provider on all 840 easy and 840 hard prompts, with and without a local Python tool restricted by macOS `sandbox-exec`. This is an evaluation-only branch; **do not merge**.

**3,360/3,360 attempts completed.** Recorded charges: **$2.088340**.

16 connection failures returned no generation ID, so their individual charges cannot be verified. All identified generation charges are shown separately in the report's billing metadata. These failures remain in the denominator; no requests were retried.

| Condition | Correct / attempted | Accuracy | Recorded cost (USD) | Median time |
|---|---:|---:|---:|---:|
| Easy, no tools | 738/840 | 87.86% | $0.636581 | 8.6s |
| Hard, no tools | 728/840 | 86.67% | $0.610633 | 9.1s |
| Easy, Python | 793/840 | 94.40% | $0.435323 | 7.6s |
| Hard, Python | 760/840 | 90.48% | $0.405802 | 7.7s |

Failures are separated below. Incorrect means a complete, correctly formatted answer that does not match the reference. Format means the final answer is missing or malformed. Timeouts are the 300-second API request deadline; token caps exhaust the generated-token budget. API errors are provider or connection failures. Categories are mutually exclusive, and all remain in the accuracy denominator.

| Condition | Incorrect | Format | Timeout | Token cap | API error |
|---|---:|---:|---:|---:|---:|
| Easy, no tools | 90 | 3 | 8 | 1 | 0 |
| Hard, no tools | 104 | 2 | 6 | 0 | 0 |
| Easy, Python | 42 | 4 | 0 | 0 | 1 |
| Hard, Python | 63 | 1 | 0 | 0 | 16 |

[HTML report](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-high-20261029/report.html) · [Per-task metrics](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-high-20261029/by-task.csv) · [Final predictions](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-high-20261029/results.json)

![Accuracy by condition](https://raw.githubusercontent.com/Open-Athena/cds-bench/eval/luna-high-baseline/evaluations/luna-high-20261029/overall.png)

| Request / execution setting | Value |
|---|---|
| Model / provider | `openai/gpt-6-luna`; standard OpenAI only, no fallbacks |
| Reasoning | `{"effort":"high","exclude":false}` |
| Output budget | 32,768 generated tokens per example, including reasoning and all tool turns |
| Request timeout | 300 seconds per API request, enforced by the client |
| Tool choice | `auto` with one Python function; no-tools uses `tools=[]`, `tool_choice="none"` |
| Web | Web plugin disabled; no search/fetch tool registered |
| Executor networking | Local `/usr/bin/sandbox-exec` with `(deny network*)`; **not an OpenRouter parameter** |
| Environment | Python 3.12.9 standard library + NumPy 2.5.4; no Biopython |
| Tool limits | 30 seconds/call, eight calls/example, 16,384 output characters per stream |
| Sampling / parallelism | Seed 20261029; provider-default sampling; 10 concurrent examples |

<details>
<summary>Exact tools JSON passed to OpenRouter</summary>

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

The no-tools condition sends `"tools": []`. No hosted shell is used.

</details>

<details>
<summary>Local Python execution and macOS sandbox policy</summary>

When Luna returns a `python` function call, the evaluator executes the supplied program locally and sends the execution result back through OpenRouter. Each call launches Python using this command structure; the runner supplies the runtime path, example workdir, and launcher, then sends the model's program through stdin:

```sh
/usr/bin/sandbox-exec -f "$workdir/sandbox.sb" \
  "$runtime/bin/python" -I -B -c "$launcher"
```

The generated policy includes these literal restrictions:

```scheme
(deny network*)
(deny process-fork)
```

Additional rules restrict file-content reads to the Python runtime, required system paths, and the current example workdir. Writes are restricted to that workdir and `/dev/null`. Python receives a minimal environment without API credentials. Each call starts a fresh interpreter; workdir files persist within an example. The launcher enforces resource limits, and the evaluator kills executions exceeding 30 seconds.

The network restriction applies to the Python child process. The evaluator retains network access to communicate with OpenRouter. Before inference, acceptance checks confirmed that NumPy and workdir reads/writes worked, while a socket connection, benchmark-source read, private-file read, outside-workdir write, and subprocess launch failed. Biopython was also confirmed unavailable.

[Executor and full policy](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/src/cds_bench/eval_luna/sandbox.py) · [Acceptance checks](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/src/cds_bench/eval_luna/run.py#L123)

</details>

<details>
<summary>Request settings and trace handling</summary>

```json
{
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
}
```

In the tools condition, `tools` is the definition above and `tool_choice` is `auto`. Each next request receives the remaining generated-token budget and unchanged, reconstructed reasoning records. The task prompt itself is unchanged.

The model receives stdout, stderr, exit code, timeout status, and truncation status from the local executor. No correctness feedback is returned.

Raw requests, SSE responses, reasoning summaries/encrypted records, and tool programs/results remain untracked. Final predictions and metrics are included in the report.

</details>

<details>
<summary>Dataset, scoring, costs, and per-task results</summary>

The released [open-athena/cds-bench](https://huggingface.co/datasets/open-athena/cds-bench) easy/hard splits each contain ten instances of 84 task types. Corresponding variants use identical inputs and answers. This run makes one fresh attempt per condition; no pilot results are reused. Prompts, tasks, and reference answers are unchanged.

Final answers are checked deterministically. JSON is parsed with duplicate keys rejected; object key order does not matter and array order does. Numeric scalars are compared as decimal values; other scalar answers ignore whitespace. Wrong answers, formatting failures, token caps, request timeouts, and API errors are shown separately. There are no inference retries or correctness hints.

Costs are logged per request and example, then reconciled with OpenRouter generation metadata after each phase. The report aggregates cost and performance by condition, task, and family. Values in an unfinished run cover completed attempts, with missing billing records called out.

![Per-task accuracy](https://raw.githubusercontent.com/Open-Athena/cds-bench/eval/luna-high-baseline/evaluations/luna-high-20261029/by-task.png)

</details>

<details>
<summary>Final validation and collection issues</summary>

All 3,360 cases, original prompts, request settings, saved final answers, scores, and reasoning replays were checked. The benchmark definitions and released dataset are unchanged.

The run retained 14 five-minute timeouts, one token-limit failure, one provider 502, and 16 connection failures as failures. The collector lost the exception class on those 16 connection failures and subsequently recorded a `usage` error. Error handling was fixed after the run, with a regression test; no model responses or scores were changed, and nothing was retried.

[Audit](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-high-20261029/audit.json) · [Collector fix](https://github.com/Open-Athena/cds-bench/blob/eval/luna-high-baseline/evaluations/luna-high-20261029/collector-note.json)

</details>
