"""Portable HTML, per-task metrics, figures, and one PR description."""

import argparse
import base64
import csv
import fcntl
import html
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from cds_bench.catalog import TASKS

from .io import save
from .settings import PARAMETERS, TOOLS

CONDITIONS = ["easy-no-tools", "hard-no-tools", "easy-tools", "hard-tools"]
LABELS = ["Easy, no tools", "Hard, no tools", "Easy, Python", "Hard, Python"]
REPORT_PATH = "evaluations/luna-high-20261029"
BRANCH = "eval/luna-high-baseline"
GITHUB = "https://github.com/Open-Athena/cds-bench"


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def outcome(row):
    if row.get("passed"):
        return "correct"
    if row.get("request_timeout"):
        return "timeout"
    if (
        row.get("api_error")
        or row.get("error")
        and "format_ok" not in row
        and not row.get("truncated")
    ):
        return "error"
    if row.get("truncated"):
        return "token_cap"
    if not row.get("format_ok"):
        return "format"
    return "wrong"


def aggregate(rows):
    counts = Counter(outcome(row) for row in rows)
    durations = [row["elapsed_seconds"] for row in rows]
    return {
        "attempted": len(rows),
        "correct": counts["correct"],
        "wrong": counts["wrong"],
        "format": counts["format"],
        "timeout": counts["timeout"],
        "token_cap": counts["token_cap"],
        "error": counts["error"],
        "accuracy": counts["correct"] / len(rows) if rows else None,
        "median_seconds": statistics.median(durations) if rows else None,
        "total_example_seconds": sum(durations),
        "examples_using_python": sum(row["tool_calls"] > 0 for row in rows),
        "python_calls": sum(row["tool_calls"] for row in rows),
        "python_failures": sum(row["tool_failures"] for row in rows),
        "api_requests": sum(row["request_count"] for row in rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
        "reasoning_tokens": sum(row["reasoning_tokens"] for row in rows),
        "cost_usd": sum(row["cost_usd"] for row in rows),
        "unreconciled_examples": sum(not row["billing_complete"] for row in rows),
    }


def records(raw_root):
    rows = []
    for path in sorted((raw_root / "traces").glob("*/*/result.json")):
        row = load(path)
        case = load(path.parent / "case.json")
        charges = []
        for identifier in row["response_ids"]:
            charges.append(
                load(raw_root / "billing" / (identifier + ".json"), {})
                .get("data", {})
                .get("total_cost")
            )
        complete = len(charges) == row["request_count"] and all(c is not None for c in charges)
        row["billing_complete"] = complete
        row["cost_usd"] = sum(charges) if complete else row["reported_cost_usd"]
        row["connection_failure"] = False
        if row.get("error") == "'usage'":
            replies = [load(p) for p in path.parent.glob("response-*.json")]
            if any(
                not r.get("id") and r.get("http_status") is None and r.get("error") == ""
                for r in replies
            ):
                # Clarify the recorded collector error without changing the
                # original trace, final answer, or score.
                row["connection_failure"] = True
                row["api_error"] = True
                row["error"] = (
                    "Request failed before HTTP response headers. The original collector lost the empty exception message and recorded KeyError('usage')."
                )
        row["expected_answer"] = case["answer"]
        row["outcome"] = outcome(row)
        row["output_format"] = case["output_format"]
        rows.append(row)
    return rows


def public_record(row):
    # Raw responses, plaintext reasoning, encrypted reasoning, requests, and tool
    # programs remain untracked. Only final predictions and metrics are published.
    fields = [
        "instance_id",
        "task_id",
        "variant",
        "condition",
        "tools_enabled",
        "api_error",
        "connection_failure",
        "passed",
        "format_ok",
        "request_timeout",
        "truncated",
        "error",
        "finish_reason",
        "tool_calls",
        "tool_failures",
        "request_count",
        "elapsed_seconds",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cost_usd",
        "billing_complete",
        "final_answer",
        "expected_answer",
        "outcome",
    ]
    return {key: row.get(key) for key in fields}


def plots(output, summaries, by_task):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(8.7, 3.7), layout="constrained")
    values = [100 * (summaries[c]["accuracy"] or 0) for c in CONDITIONS]
    bars = ax.bar(
        np.arange(4), values, color=["#476987", "#7891a9", "#337b67", "#7ca797"], width=0.62
    )
    for bar, value, c in zip(bars, values, CONDITIONS):
        r = summaries[c]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{r['correct']}/{r['attempted']}",
            ha="center",
            va="bottom",
        )
    ax.set_xticks(np.arange(4), LABELS)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Correct answers (%) among completed attempts")
    ax.set_title("CDS-bench · GPT-6 Luna · high reasoning", loc="left", pad=16)
    ax.grid(axis="y", color="#e4e7eb")
    ax.set_axisbelow(True)
    fig.savefig(output / "overall.png", dpi=180)
    plt.close(fig)

    ids = list(TASKS)
    matrix = np.array(
        [
            [
                by_task[t][c]["accuracy"] if by_task[t][c]["attempted"] else np.nan
                for c in CONDITIONS
            ]
            for t in ids
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 25), layout="constrained")
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#f1f3f5")
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(ids)), ids, fontsize=8, fontfamily="monospace")
    ax.set_xticks(np.arange(4), LABELS)
    ax.xaxis.tick_top()
    for i, task in enumerate(ids):
        for j, condition in enumerate(CONDITIONS):
            r = by_task[task][condition]
            label = f"{r['correct']}/{r['attempted']}" if r["attempted"] else "—"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if r["accuracy"] and r["accuracy"] > 0.65 else "#243240",
            )
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Correct / completed attempts; 10 planned per task and condition", pad=24, loc="left"
    )
    fig.savefig(output / "by-task.png", dpi=140)
    plt.close(fig)


def build(raw_root, output):
    # The publisher takes the same lock when staging a consistent report snapshot.
    with (raw_root / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build(raw_root, output)


def _build(raw_root, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = records(raw_root)
    summaries = {c: aggregate([row for row in rows if row["condition"] == c]) for c in CONDITIONS}
    by_task = {
        task: {
            c: aggregate([row for row in rows if row["task_id"] == task and row["condition"] == c])
            for c in CONDITIONS
        }
        for task in TASKS
    }
    families = sorted({v.get("family", "composition") for v in TASKS.values()})
    by_family = {
        family: {
            c: aggregate(
                [
                    r
                    for r in rows
                    if r["condition"] == c
                    and TASKS[r["task_id"]].get("family", "composition") == family
                ]
            )
            for c in CONDITIONS
        }
        for family in families
    }
    config = load(raw_root / "config.json")
    job = load(raw_root / "job-status.json", {"state": "prepared", "completed": 0})
    billing = load(raw_root / "billing-summary.json", {})
    paired = {}
    lookup = {(r["instance_id"], r["condition"]): r for r in rows}
    for variant in ("easy", "hard"):
        counter = Counter()
        for instance in {r["instance_id"] for r in rows}:
            a = lookup.get((instance, variant + "-no-tools"))
            b = lookup.get((instance, variant + "-tools"))
            if a and b:
                counter[(bool(a["passed"]), bool(b["passed"]))] += 1
        paired[variant] = {
            "both_correct": counter[(True, True)],
            "helped": counter[(False, True)],
            "hurt": counter[(True, False)],
            "both_failed": counter[(False, False)],
        }
    summary = {
        "planned_attempts": 3360,
        "completed_attempts": len(rows),
        "state": job["state"],
        "conditions": summaries,
        "by_task": by_task,
        "by_family": by_family,
        "tool_comparisons": paired,
        "total": aggregate(rows),
        "billing": billing,
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    }
    grouped_instances = {}
    for row in rows:
        grouped_instances.setdefault(row["instance_id"], []).append(row)
    summary["failed_all_conditions"] = [
        {
            "instance_id": instance,
            "task_id": values[0]["task_id"],
            "outcomes": {r["condition"]: r["outcome"] for r in values},
        }
        for instance, values in grouped_instances.items()
        if len(values) == 4 and not any(r["passed"] for r in values)
    ]
    save(output / "summary.json", summary)
    save(output / "results.json", [public_record(r) for r in rows])
    save(output / "config.json", config)
    for name in ("audit.json", "collector-note.json"):
        if (raw_root / name).exists():
            save(output / name, load(raw_root / name))
    table_rows = []
    for task in TASKS:
        for condition in CONDITIONS:
            table_rows.append({"task_id": task, "condition": condition, **by_task[task][condition]})
    with (output / "by-task.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table_rows)
    plots(output, summaries, by_task)
    render_html(output, rows, summary, config)
    render_pr(output, summary, config)
    title = "# GPT-6 Luna high: full CDS-bench baseline\n\n"
    description = f"{len(rows):,}/3,360 attempts complete. Four fresh passes over all 84 task types and ten instances per task.\n\n"
    (output / "README.md").write_text(
        title
        + description
        + "[Download the HTML report](report.html), then open it in a browser. It is self-contained and works offline.\n\n[Summary](summary.json) · [Per-task metrics](by-task.csv) · [Final predictions](results.json) · [Request configuration](config.json) · [Final audit](audit.json)\n\nRaw API responses, reasoning records, and Python traces are retained locally under the run's untracked scratch directory. The recorded inference source is identified in config.json; the small error-handler repair made after inference is documented in [collector-note.json](collector-note.json).\n"
    )
    return summary


def render_pr(output, summary, config):
    total = summary["total"]
    lines = [
        "Run GPT-6 Luna with **high** reasoning through OpenRouter's standard OpenAI provider on all 840 easy and 840 hard prompts, with and without a local Python tool restricted by macOS `sandbox-exec`. This is an evaluation-only branch; **do not merge**.",
        f"**{summary['completed_attempts']:,}/3,360 attempts completed.** Recorded charges: **${total['cost_usd']:.6f}**.",
        "| Condition | Correct / attempted | Accuracy | Recorded cost (USD) | Median time |",
        "|---|---:|---:|---:|---:|",
    ]
    unknown = summary["billing"].get("responses_without_generation_id", 0)
    if unknown:
        lines.insert(
            2,
            f"{unknown} connection failures returned no generation ID, so their individual charges cannot be verified. All identified generation charges are shown separately in the report's billing metadata. These failures remain in the denominator; no requests were retried.",
        )
    for c, label in zip(CONDITIONS, LABELS):
        r = summary["conditions"][c]
        lines.append(
            f"| {label} | {r['correct']}/{r['attempted']} | {100 * (r['accuracy'] or 0):.2f}% | ${r['cost_usd']:.6f} | {r['median_seconds'] or 0:.1f}s |"
        )
    lines += [
        "Failures are separated below. Incorrect means a complete, correctly formatted answer that does not match the reference. Format means the final answer is missing or malformed. Timeouts are the 300-second API request deadline; token caps exhaust the generated-token budget. API errors are provider or connection failures. Categories are mutually exclusive, and all remain in the accuracy denominator.",
        "| Condition | Incorrect | Format | Timeout | Token cap | API error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c, label in zip(CONDITIONS, LABELS):
        r = summary["conditions"][c]
        lines.append(
            f"| {label} | {r['wrong']} | {r['format']} | {r['timeout']} | {r['token_cap']} | {r['error']} |"
        )
    lines += [
        f"[HTML report]({GITHUB}/blob/{BRANCH}/{REPORT_PATH}/report.html) · [Per-task metrics]({GITHUB}/blob/{BRANCH}/{REPORT_PATH}/by-task.csv) · [Final predictions]({GITHUB}/blob/{BRANCH}/{REPORT_PATH}/results.json)",
        f"![Accuracy by condition](https://raw.githubusercontent.com/Open-Athena/cds-bench/{BRANCH}/{REPORT_PATH}/overall.png)",
        "| Request / execution setting | Value |",
        "|---|---|",
        "| Model / provider | `openai/gpt-6-luna`; standard OpenAI only, no fallbacks |",
        '| Reasoning | `{"effort":"high","exclude":false}` |',
        "| Output budget | 32,768 generated tokens per example, including reasoning and all tool turns |",
        "| Request timeout | 300 seconds per API request, enforced by the client |",
        '| Tool choice | `auto` with one Python function; no-tools uses `tools=[]`, `tool_choice="none"` |',
        "| Web | Web plugin disabled; no search/fetch tool registered |",
        "| Executor networking | Local `/usr/bin/sandbox-exec` with `(deny network*)`; **not an OpenRouter parameter** |",
        "| Environment | Python 3.12.9 standard library + NumPy 2.5.4; no Biopython |",
        "| Tool limits | 30 seconds/call, eight calls/example, 16,384 output characters per stream |",
        "| Sampling / parallelism | Seed 20261029; provider-default sampling; 10 concurrent examples |",
        "<details>\n<summary>Exact tools JSON passed to OpenRouter</summary>\n\n```json\n"
        + json.dumps(TOOLS, indent=2)
        + '\n```\n\nThe no-tools condition sends `"tools": []`. No hosted shell is used.\n\n</details>',
        (
            "<details>\n<summary>Local Python execution and macOS sandbox policy</summary>\n\n"
            "When Luna returns a `python` function call, the evaluator executes the supplied program locally and sends the execution result back through OpenRouter. Each call launches Python using this command structure; the runner supplies the runtime path, example workdir, and launcher, then sends the model's program through stdin:\n\n"
            '```sh\n/usr/bin/sandbox-exec -f "$workdir/sandbox.sb" \\\n  "$runtime/bin/python" -I -B -c "$launcher"\n```\n\n'
            "The generated policy includes these literal restrictions:\n\n"
            "```scheme\n(deny network*)\n(deny process-fork)\n```\n\n"
            "Additional rules restrict file-content reads to the Python runtime, required system paths, and the current example workdir. Writes are restricted to that workdir and `/dev/null`. Python receives a minimal environment without API credentials. Each call starts a fresh interpreter; workdir files persist within an example. The launcher enforces resource limits, and the evaluator kills executions exceeding 30 seconds.\n\n"
            "The network restriction applies to the Python child process. The evaluator retains network access to communicate with OpenRouter. Before inference, acceptance checks confirmed that NumPy and workdir reads/writes worked, while a socket connection, benchmark-source read, private-file read, outside-workdir write, and subprocess launch failed. Biopython was also confirmed unavailable.\n\n"
            f"[Executor and full policy]({GITHUB}/blob/{BRANCH}/src/cds_bench/eval_luna/sandbox.py) · [Acceptance checks]({GITHUB}/blob/{BRANCH}/src/cds_bench/eval_luna/run.py#L123)\n\n</details>"
        ),
        "<details>\n<summary>Request settings and trace handling</summary>\n\n```json\n"
        + json.dumps(PARAMETERS, indent=2)
        + "\n```\n\nIn the tools condition, `tools` is the definition above and `tool_choice` is `auto`. Each next request receives the remaining generated-token budget and unchanged, reconstructed reasoning records. The task prompt itself is unchanged.\n\nThe model receives stdout, stderr, exit code, timeout status, and truncation status from the local executor. No correctness feedback is returned.\n\nRaw requests, SSE responses, reasoning summaries/encrypted records, and tool programs/results remain untracked. Final predictions and metrics are included in the report.\n\n</details>",
        "<details>\n<summary>Dataset, scoring, costs, and per-task results</summary>\n\nThe released [open-athena/cds-bench](https://huggingface.co/datasets/open-athena/cds-bench) easy/hard splits each contain ten instances of 84 task types. Corresponding variants use identical inputs and answers. This run makes one fresh attempt per condition; no pilot results are reused. Prompts, tasks, and reference answers are unchanged.\n\nFinal answers are checked deterministically. JSON is parsed with duplicate keys rejected; object key order does not matter and array order does. Numeric scalars are compared as decimal values; other scalar answers ignore whitespace. Wrong answers, formatting failures, token caps, request timeouts, and API errors are shown separately. There are no inference retries or correctness hints.\n\nCosts are logged per request and example, then reconciled with OpenRouter generation metadata after each phase. The report aggregates cost and performance by condition, task, and family. Values in an unfinished run cover completed attempts, with missing billing records called out.\n\n![Per-task accuracy](https://raw.githubusercontent.com/Open-Athena/cds-bench/"
        + BRANCH
        + "/"
        + REPORT_PATH
        + "/by-task.png)\n\n</details>",
    ]
    if (output / "audit.json").exists():
        lines.append(
            "<details>\n<summary>Final validation and collection issues</summary>\n\n"
            "All 3,360 cases, original prompts, request settings, saved final answers, scores, and reasoning replays were checked. The benchmark definitions and released dataset are unchanged.\n\n"
            "The run retained 14 five-minute timeouts, one token-limit failure, one provider 502, and 16 connection failures as failures. The collector lost the exception class on those 16 connection failures and subsequently recorded a `usage` error. Error handling was fixed after the run, with a regression test; no model responses or scores were changed, and nothing was retried.\n\n"
            f"[Audit]({GITHUB}/blob/{BRANCH}/{REPORT_PATH}/audit.json) · [Collector fix]({GITHUB}/blob/{BRANCH}/{REPORT_PATH}/collector-note.json)\n\n</details>"
        )
    body = "\n\n".join(lines)
    body = re.sub(r"(?m)(^\|.*\|)\n\n(?=\|)", r"\1\n", body)
    (output / "pr-body.md").write_text(body + "\n")


def render_html(output, rows, summary, config):
    esc = html.escape
    encoded_plot = base64.b64encode((output / "overall.png").read_bytes()).decode()
    payload = {
        "records": [public_record(r) for r in rows],
        "conditions": CONDITIONS,
        "tasks": {
            task: {"title": data["title"], "family": data.get("family", "composition")}
            for task, data in TASKS.items()
        },
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    parts = [
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CDS-bench · Luna baseline</title><style>
body{font:15px/1.5 system-ui,sans-serif;color:#243240;background:white;margin:0}main{max-width:1250px;margin:32px auto;padding:0 24px}h1{font-size:27px;margin-bottom:6px}h2{font-size:20px;margin-top:32px}h3{font-size:16px}p{max-width:1050px}a{color:#1d5c89}table{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}th,td{text-align:left;padding:9px 11px;border-bottom:1px solid #dce1e6;vertical-align:top}th{background:#f3f5f7}td small{display:block;color:#687480}pre,code{font:12px/1.5 ui-monospace,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;border:1px solid #dce1e6;padding:12px;max-height:350px;overflow:auto}summary,.task-row{cursor:pointer}details{border-top:1px solid #dce1e6;padding:10px 0}.muted{color:#687480}.scroll{overflow-x:auto}img{max-width:100%;height:auto}.controls{display:flex;gap:20px;align-items:center;margin:16px 0}input[type=search]{font:inherit;padding:7px 9px;border:1px solid #acb7c2;width:330px}.task-row:hover{background:#fafbfc}.detail-row{display:none}.detail-row.open{display:table-row}[hidden]{display:none!important}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}.pill{padding:2px 6px;border-radius:3px;white-space:nowrap;font-size:12px}.correct{background:#dcefe6}.wrong{background:#f5dadd}.timeout,.token_cap{background:#f6e7bd}.format{background:#e8dff2}.error{background:#d8dce3}.pending{background:#f1f3f5}@media(max-width:850px){.two{display:block}.controls{display:block}main{padding:0 14px}td,th{padding:7px}}
</style><main><h1>CDS-bench: Luna baseline</h1>""",
        f'<p class="muted">GPT-6 Luna · high reasoning · {summary["completed_attempts"]:,}/3,360 attempts complete · {esc(summary["state"])}</p>',
        '<nav><a href="#overview">Overview</a> · <a href="#tasks">By task</a> · <a href="#groups">Aggregations</a> · <a href="#settings">Settings</a></nav>',
        '<h2 id="overview">Overview</h2><p>All 84 task types, with ten instances per task in each of four conditions. Easy and hard variants share the same input and expected answer. Each prompt receives one fresh attempt. Tools are optional when available.</p>',
        '<p class="muted">Failures, including timeouts and connection errors, remain in the denominator. Each task has ten instances per condition.</p>',
        f'<img src="data:image/png;base64,{encoded_plot}" alt="Accuracy by condition">',
        '<div class="scroll"><table><tr><th>Condition</th><th>Correct / attempted</th><th>Accuracy</th><th>Wrong</th><th>Format</th><th>Token cap</th><th>Timeout</th><th>Other error</th><th>Used Python</th><th>Median time</th><th>Recorded cost</th></tr>',
    ]
    for c, label in zip(CONDITIONS, LABELS):
        r = summary["conditions"][c]
        parts.append(
            f"<tr><td>{label}</td><td><b>{r['correct']}/{r['attempted']}</b></td><td>{100 * (r['accuracy'] or 0):.2f}%</td><td>{r['wrong']}</td><td>{r['format']}</td><td>{r['token_cap']}</td><td>{r['timeout']}</td><td>{r['error']}</td><td>{r['examples_using_python']}</td><td>{r['median_seconds'] or 0:.1f}s</td><td>${r['cost_usd']:.5f}</td></tr>"
        )
    total = summary["total"]
    parts += [
        "</table></div>",
        f"<p>Recorded charges: <b>${total['cost_usd']:.6f}</b>. {total['unreconciled_examples']} examples lack complete billing metadata. {summary['billing'].get('responses_without_generation_id', 0)} requests returned no generation ID, so their individual charges cannot be verified. Cancelled requests are included when identified.</p>",
        '<h2 id="tasks">By task</h2><p>Each cell gives correct / completed attempts, with ten planned. Click a row to compare the ten instances and their final answers. Family and per-task costs are available below and in the downloadable metrics.</p>',
        '<div class="controls"><input id="filter" type="search" placeholder="Filter task names or identifiers"><label><input id="failures" type="checkbox"> Only tasks with a failure</label></div>',
        '<div class="scroll"><table id="tasks-table"><thead><tr><th>Task</th>'
        + "".join(f"<th>{label}</th>" for label in LABELS)
        + "</tr></thead><tbody>",
    ]
    for task, definition in TASKS.items():
        values = summary["by_task"][task]
        failures = any(r["correct"] < r["attempted"] for r in values.values())
        cells = []
        for c in CONDITIONS:
            r = values[c]
            background = (
                "#f1f3f5"
                if not r["attempted"]
                else "#dcefe6"
                if r["correct"] == r["attempted"]
                else "#f6e7bd"
                if r["correct"] >= r["attempted"] / 2
                else "#f5dadd"
            )
            cells.append(
                f'<td><span class="pill" style="background:{background}">{r["correct"]}/{r["attempted"]}</span><small>${r["cost_usd"]:.4f}</small></td>'
            )
        parts.append(
            f'<tr class="task-row" id="task-{task}" data-task="{task}" data-search="{esc((task + " " + definition["title"]).lower())}" data-failed="{int(failures)}" tabindex="0" aria-expanded="false"><td><a href="#task-{task}"><b>{esc(definition["title"])}</b></a><small><code>{task}</code></small></td>{"".join(cells)}</tr><tr class="detail-row"><td colspan="5"></td></tr>'
        )
    parts += [
        "</tbody></table></div>",
        "<details><summary>Instances missed in all four conditions</summary><p>These are candidates for inspecting shared model errors or task wording; a shared failure does not establish a specification problem.</p><ul>"
        + "".join(
            f'<li><a href="#task-{v["task_id"]}"><code>{esc(v["instance_id"])}</code></a></li>'
            for v in summary["failed_all_conditions"]
        )
        + "</ul></details>",
        '<h2 id="groups">Compact aggregations</h2><div class="two"><section><h3>Task families</h3><div class="scroll"><table><tr><th>Family</th>'
        + "".join(f"<th>{l}</th>" for l in LABELS)
        + "</tr>",
    ]
    for family, values in summary["by_family"].items():
        parts.append(
            f"<tr><td>{family}</td>"
            + "".join(
                f"<td>{values[c]['correct']}/{values[c]['attempted']}</td>" for c in CONDITIONS
            )
            + "</tr>"
        )
    parts += [
        "</table></div></section><section><h3>Paired effect of offering Python</h3><table><tr><th>Variant</th><th>Both correct</th><th>Helped</th><th>Hurt</th><th>Both failed</th></tr>"
    ]
    for variant, counts in summary["tool_comparisons"].items():
        parts.append(
            f"<tr><td>{variant.title()}</td><td>{counts['both_correct']}</td><td>{counts['helped']}</td><td>{counts['hurt']}</td><td>{counts['both_failed']}</td></tr>"
        )
    parts += [
        '</table><p class="muted">Only instances completed in both conditions are compared. Tool availability can change a response even if Python is not used. These counts do not establish a causal effect from repeated sampling.</p></section></div>',
        '<h2 id="settings">Settings and methods</h2>',
        "<p>OpenRouter Chat Completions routes to the standard OpenAI provider, with reasoning effort <code>high</code>. The aggregate budget is 32,768 generated tokens per example, including reasoning and all tool turns. Each API request has a hard 300-second deadline. There are no inference retries or correctness hints.</p>",
        "<p>Python 3.12.9 provides its standard library and NumPy 2.5.4. Biopython is absent. macOS denies network access, subprocess creation, source/answer/credential reads, and writes outside the example workdir. Each call starts a fresh interpreter; files persist within an example. Limits are 30 seconds per call, eight calls per example, and 16,384 output characters per stream. This network restriction is enforced locally, not by an OpenRouter parameter.</p>",
        "<p>Scoring requires the final-answer marker. JSON is parsed with duplicate keys rejected; object key order is ignored and array order is preserved. Numeric scalar values are compared as decimals; whitespace is ignored in other scalar answers. Truncation is a failure. Wrong answers, format failures, token caps, request timeouts, and API errors are reported separately.</p>",
        "<p>Raw prompts, API/SSE responses, reasoning records, and Python traces remain local and untracked. The report contains final predictions and metrics. OpenAI reasoning summaries are not the full private reasoning trace; returned encrypted records are preserved during tool continuations.</p>",
        "<details><summary>Exact Python tool definition passed to OpenRouter</summary><pre>"
        + esc(json.dumps(TOOLS, indent=2))
        + "</pre></details>",
        "<details><summary>Complete configuration and dataset checksums</summary><pre>"
        + esc(json.dumps(config, indent=2))
        + "</pre></details>",
        "<details><summary>OpenRouter billing and web-use metadata</summary><pre>"
        + esc(json.dumps(summary["billing"], indent=2))
        + "</pre></details>",
        "<p>The initial run retained connection failures and the provider error as failures. Raw traces are unchanged. The empty-exception collector fix was applied after inference completed.</p>"
        if (output / "collector-note.json").exists()
        else "",
        '<p><a href="summary.json">Summary JSON</a> · <a href="by-task.csv">Per-task CSV</a> · <a href="results.json">Final predictions</a> · <a href="config.json">Configuration</a></p>',
        '</main><script id="report-data" type="application/json">' + data + "</script>",
        """<script>
const report=JSON.parse(document.getElementById('report-data').textContent);
const labels=['Easy, no tools','Hard, no tools','Easy, Python','Hard, Python'];
function element(tag,text){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e}
function expand(row,force){const detail=row.nextElementSibling;const open=force===undefined?!detail.classList.contains('open'):force;detail.classList.toggle('open',open);row.setAttribute('aria-expanded',open);if(!open||detail.dataset.loaded)return;detail.dataset.loaded='1';const cell=detail.firstElementChild;const task=row.dataset.task;const family=report.tasks[task].family;const url='https://openathena.ai/cds-bench/#'+(family==='composition'?'composition-':'task-')+task;const link=element('a','Benchmark task definition');link.href=url;cell.append(link);const list=report.records.filter(r=>r.task_id===task);const instances=[...new Set(list.map(r=>r.instance_id))].sort();for(const id of instances){const section=element('details');section.append(element('summary',id));const subset=list.filter(r=>r.instance_id===id);section.append(element('h3','Expected answer'),element('pre',subset[0].expected_answer));for(let j=0;j<report.conditions.length;j++){const r=subset.find(r=>r.condition===report.conditions[j]);const child=element('details');child.append(element('summary',labels[j]+': '+(r?r.outcome:'pending')));if(r){child.append(element('pre',r.final_answer||r.error||'No final answer'));child.append(element('p',r.elapsed_seconds.toFixed(1)+' seconds; '+r.tool_calls+' Python calls; $'+r.cost_usd.toFixed(6)+(r.billing_complete?' (reconciled)':' (streamed usage)')))}section.append(child)}cell.append(section)}}
for(const row of document.querySelectorAll('.task-row')){row.addEventListener('click',e=>{if(e.target.closest('a')){expand(row,true)}else{expand(row)}});row.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&!e.target.closest('a')){e.preventDefault();expand(row)}})}
function filter(){const q=document.querySelector('#filter').value.toLowerCase();const failures=document.querySelector('#failures').checked;for(const row of document.querySelectorAll('.task-row')){const hide=!row.dataset.search.includes(q)||(failures&&row.dataset.failed!=='1');row.hidden=hide;row.nextElementSibling.hidden=hide}}document.querySelector('#filter').addEventListener('input',filter);document.querySelector('#failures').addEventListener('change',filter);
function anchor(){const hash=decodeURIComponent(location.hash.slice(1));const row=document.getElementById(hash);if(row&&row.classList.contains('task-row')){expand(row,true);row.scrollIntoView()}}window.addEventListener('hashchange',anchor);anchor();
</script></html>""",
    ]
    temporary = output / "report.html.tmp"
    temporary.write_text("\n".join(parts))
    temporary.replace(output / "report.html")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = build(args.raw_root, args.output)
    print(
        f"{summary['completed_attempts']}/3,360 complete; cost ${summary['total']['cost_usd']:.6f}"
    )


if __name__ == "__main__":
    main()
