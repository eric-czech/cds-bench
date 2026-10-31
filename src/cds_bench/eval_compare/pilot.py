"""Publish the saved 30-task pilot without inference, raw reasoning or tool traces."""

import argparse
import csv
import hashlib
import html
import json
import re
import statistics
from pathlib import Path

from cds_bench.catalog import TASKS
from cds_bench.eval_luna.io import save
from cds_bench.eval_luna.report import aggregate, public_record
from cds_bench.eval_luna.score import score_result

from .tables import html_results, models_by_task_accuracy
from . import pilot_extension
from . import snowball_pilot

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scratch/open-weights-pilot-20261029-01"
OUTPUT = ROOT / "evaluations/model-pilot-20261029"
CONDITIONS = ["easy-no-tools", "hard-no-tools", "easy-tools", "hard-tools"]
LABELS = ["Easy, no tools", "Hard, no tools", "Easy, Python", "Hard, Python"]
MODELS = {
    "luna": "GPT-6 Luna (high)", "beam": "Beam 1.0 501B A23B",
    "deepseek": "DeepSeek V4.1 Flash 1017", "nemotron": "Nemotron 3 Ultra 550B A55B",
    "nemotron-lightning": "Nemotron 3.5 Lightning (io.net)",
    "nemotron-lightning-coreweave": "Nemotron 3.5 Lightning (CoreWeave)",
    "qwen-flash": "Qwen 3.7 Flash (high)", "north-mini-code": "North Mini Code",
    "gpt-oss-20b": "GPT-OSS 20B", "gemma": "Gemma 4 31B",
    "gemma-moe": "Gemma 4 26B A4B", "liquid": "LFM 2.5 24B A2B",
}
OMITTED = {"coords.nt_to_codon", "variant.frame_status"}


def read(path):
    return json.loads(path.read_text())


def format_only_recovery(case, row, text):
    if row["outcome"] != "format":
        return False
    matches = list(re.finditer(r"(?m)^Final answer:[ \t]*", text))
    if not matches:
        return False
    answer = text[matches[-1].end():].strip()
    fence = re.fullmatch(r"```(?:json)?[ \t]*\n(.*?)\n```", answer, re.DOTALL)
    if case["output_format"] == "json" and fence:
        answer = fence[1]
    return bool(score_result(case, "Final answer:\n" + answer, row["finish_reason"])["passed"])


def build(source=SOURCE, output=OUTPUT):
    original = read(source / "summary.json")
    originals = read(source / "results.json")
    selected_path = ROOT / "scratch/luna-baseline-pilot/selected.json"
    selected = [
        {k: v for k, v in r.items() if k != "selection_rationale"}
        for r in read(selected_path)
        if r["variant"] != "hard" or r["task_id"] not in OMITTED
    ]
    cases = {(r["instance_id"], r["variant"]): r for r in selected}
    assert len(cases) == 58
    assert len({(r["model_key"], r["instance_id"], r["condition"]) for r in originals}) == len(originals)
    configs, rows, totals, conditions, sources = {}, [], {}, {}, {}
    for model, name in MODELS.items():
        raw = source / "models" / model if model != "luna" else ROOT / "scratch/luna-baseline-pilot"
        config = read(raw / "config.json")
        parameters = config.get("parameters_no_tools", config.get("parameters"))
        definition = config.get("model") or {"model": parameters["model"], "endpoint": parameters["provider"]["only"][0], "reasoning": parameters["reasoning"]}
        interrupted = read(raw / "user-interrupted.json") if (raw / "user-interrupted.json").exists() else {}
        continuation = read(raw / "continuation.json") if (raw / "continuation.json").exists() else None
        excluded = {(r["instance_id"], r["condition"]) for r in interrupted.get("interrupted_results_excluded", [])}
        configs[model] = {
            "name": name, "model": definition["model"], "provider": parameters["provider"],
            "reasoning": parameters["reasoning"], "parameters_no_tools": parameters,
            "tools": config.get("tools", config.get("tools_condition", {}).get("tools")),
            "aggregate_completion_token_budget": config["aggregate_completion_token_budget"],
            "request_timeout_seconds": config["request_timeout_seconds"],
            "concurrency": config["concurrency"],
            "dataset_revision": config.get("dataset_revision"),
            "dataset_parquet_sha256": config.get("dataset_parquet_sha256"),
            "minimum_request_spacing_seconds": config.get("minimum_request_spacing_seconds", 4 if model == "nemotron" else 0),
            "state": original["states"][model]["state"],
            "stop_reason": original["states"][model].get("stop_reason"),
            "settings_difference": config.get("settings_difference"),
            "scheduling_continuation": {
                k: continuation[k] for k in ["initial_completed", "concurrency", "minimum_request_spacing_seconds", "reason"]
            } if continuation else None,
            "interrupted_original_attempts": len(interrupted.get("interrupted_results_excluded", [])),
            "source_billing_summary": read(raw / "billing-summary.json") if (raw / "billing-summary.json").exists() else {},
        }
        durations = []
        for row in [r for r in originals if r["model_key"] == model]:
            assert (row["instance_id"], row["condition"]) not in excluded
            directory = Path(row["trace_directory"])
            case = read(directory / "case.json")
            assert {k: v for k, v in case.items() if k != "selection_rationale"} == cases[row["instance_id"], row["variant"]]
            assert row["expected_answer"] == case["answer"]
            text = (directory / "answer.txt").read_text()
            if not row.get("api_error"):
                assert score_result(case, text, row["finish_reason"])["passed"] == row["passed"]
            replies = [read(p) for p in sorted(directory.glob("response-*.json"))]
            assert len(replies) == row["request_count"]
            durations.extend(r["elapsed_seconds"] for r in replies)
            # Absent format_ok distinguishes a protocol failure from a malformed answer.
            exported = {k: v for k, v in public_record(row).items() if k in row}
            if exported.get("error"):
                exported["error"] = {
                    "error": "API or tool-protocol failure", "timeout": "API deadline exceeded",
                    "format": "Final answer does not satisfy the required format",
                    "token_cap": "Generated-token budget exhausted", "wrong": "Incorrect value",
                }[row["outcome"]]
            rows.append(dict(exported, model=model,
                             format_diagnostic_exact=format_only_recovery(case, row, text)))
        model_rows = [r for r in rows if r["model"] == model]
        totals[model] = aggregate(model_rows)
        assert totals[model] == original["totals"][model], model
        totals[model]["median_api_request_seconds"] = statistics.median(durations) if durations else None
        totals[model]["format_diagnostic_additional_exact"] = sum(r["format_diagnostic_exact"] for r in model_rows)
        conditions[model] = {c: aggregate([r for r in model_rows if r["condition"] == c]) for c in CONDITIONS}
        assert conditions[model] == original["conditions"][model]
        sources[str((raw / "config.json").relative_to(ROOT))] = hashlib.sha256((raw / "config.json").read_bytes()).hexdigest()
    models = dict(MODELS)
    for model, definition in snowball_pilot.MODELS.items():
        snowball = snowball_pilot.build(output, cases, format_only_recovery, model)
        if snowball:
            model_rows, totals[model], conditions[model], configs[model] = snowball
            rows.extend(model_rows)
            models[model] = definition['name']
    models, model_means = models_by_task_accuracy(models, rows)
    task_ids = [r["task_id"] for r in selected if r["variant"] == "easy"]
    source_diagnostics = read(source / "gpt-oss-reliability-summary.json")
    diagnostics = {
        "conclusion": source_diagnostics["conclusion"],
        "diagnostic_subsets_only": True,
        "configurations": [{k: v for k, v in r.items() if k != "root"} for r in source_diagnostics["configurations"]],
        "responses_api_checks": {
            "coreweave": "Failed: provider content validation error during message replay.",
            "groq": "Failed: provider reported a required tool call was not made.",
        },
        "references": source_diagnostics["references"],
    }
    appendix = []
    for folder, name in [("glm-5.3-smoke", "GLM 5.3 / Morph FP8"), ("glm-5.3-wafer-us", "GLM 5.3 / Wafer US"), ("gpt-6-luna-high-smoke", "GPT-6 Luna (high) / OpenAI")]:
        raw = ROOT / "scratch" / folder
        if not (raw / "summary.json").exists():
            continue
        s, c = read(raw / "summary.json"), read(raw / "config.json")
        appendix.append({"name": name, "attempted": s["completed"], "correct": s["strict_passed"],
                         "cost_usd": s["reported_cost_usd"], "median_seconds": statistics.median(r["elapsed_seconds"] for r in s["cases"]),
                         "token_caps": s["truncated"], "parameters": c["parameters"]})
    summary = {"models": models, "model_mean_task_accuracy":model_means,
               "model_order_metric":"Descending mean task accuracy over completed attempts, weighting each available task equally; partial subsets remain non-comparable.",
               "planned_per_condition": dict(zip(CONDITIONS, [30, 28, 30, 28])),
               "omitted_duplicate_hard_tasks": sorted(OMITTED), "task_ids": task_ids,
               "totals": totals, "conditions": conditions, "configurations": configs,
               "earlier_ten_example_checks": appendix,
               "cost_scope": "Completed retained pilot attempts only; capability checks, discarded duplicate-hard cases, interrupted requests and GPT-OSS follow-ups are excluded. Missing charges are not zero.",
               "timing_scope": "Median wall time per completed attempt, including failed attempts, API timeouts, tool round trips and any request pacing. User-interrupted requests are excluded.",
               "validation": {"original_scores_and_aggregates_verified": True, "prompts_and_answers_verified": True,
                              "no_inference_or_retries": True, "raw_reasoning_and_tool_traces_excluded": True,
                              "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source / "summary.json", source / "results.json", source / "gpt-oss-reliability-summary.json", selected_path]},
                              "configuration_sha256": sources}}
    summary["provider_reported_budget_overruns"] = [
        {"model": r["model"], "instance_id": r["instance_id"], "condition": r["condition"],
         "requested_tokens": configs[r["model"]]["aggregate_completion_token_budget"],
         "reported_tokens": r["completion_tokens"]}
        for r in rows
        if r["completion_tokens"] > configs[r["model"]]["aggregate_completion_token_budget"]
    ]
    output.mkdir(parents=True, exist_ok=True)
    summary["coreweave_extension"] = pilot_extension.build(output, rows)
    save(output / "summary.json", summary)
    save(output / "results.json", rows)
    save(output / "cases.json", selected)
    save(output / "gpt-oss-diagnostics.json", diagnostics)
    with (output / "by-task.csv").open("w") as f:
        fields = ["model", "task_id", "instance_id", "condition", "outcome", "passed", "elapsed_seconds", "cost_usd", "tool_calls", "tool_failures"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    render(output, summary, rows, selected, diagnostics)
    (output / "README.md").write_text(
        "# CDS-bench: 30-task model pilot\n\nDownload [report.html](report.html) and open it in a browser. The report is self-contained and includes model/provider settings, four-condition scores, failure categories, timing, task-level results and GPT-OSS compatibility checks.\n\n"
        "Complete runs contain 116 attempts: one instance from 30 task types, easy/hard, without tools/with offline Python, excluding two duplicate hard variants. Partial runs are marked explicitly. The stopped CoreWeave expansion is retained as 353 additional exploratory attempts in [its own report](coreweave-extra.html), separate from the original 116-attempt pilot. Scores and raw traces are unchanged. Earlier ten-example checks and GPT-OSS diagnostics are separate.\n\n"
        "[Summary and settings](summary.json) · [Per-task outcomes](by-task.csv) · [Final predictions](results.json) · [Prompt instances](cases.json) · [GPT-OSS diagnostics](gpt-oss-diagnostics.json)\n\n"
        "Sanitized API requests, reasoning and Python programs/results are in the [public HF trace archive](https://huggingface.co/datasets/eczech/cds-bench-pr-2-traces); originals remain untracked locally. Costs exclude preflight and interrupted requests; unknown charges are not assumed to be zero.\n\n"
        "Rebuild from the saved local pilot, with no inference:\n\n```sh\nPYTHONPATH=src .venv/bin/python -m cds_bench.eval_compare.pilot\n```\n"
    )
    for model, definition in snowball_pilot.MODELS.items():
        if model in configs:
            c = configs[model]
            p = c['provenance']
            with (output / 'README.md').open('a') as file:
                file.write(f"\n{c['name']}: [{c['full_name']}]({p['url']}), `{p['repo']}` at revision `{p['revision']}`. Uses the internal API with native thinking/tool transport, a {c['aggregate_completion_token_budget']:,}-token aggregate budget, 1,800-second request deadline, temperature 0.7 and concurrency 60. Monetary cost is unavailable. [Pilot audit]({definition['audit_file']}).\n")
    return summary


def render(output, summary, rows, cases, diagnostics):
    esc = html.escape
    MODELS = summary['models']
    configs, totals = summary["configurations"], summary["totals"]
    parts = ['''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CDS-bench: 30-task model pilot</title><style>
body{font:15px/1.5 system-ui,sans-serif;color:#25313c;background:#fff;margin:0}main{max-width:1400px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:20px;margin-top:30px}h3{font-size:16px}a{color:#205d89}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #dce2e7;text-align:left;vertical-align:top}th{background:#f4f6f8}.scroll{overflow-x:auto}small,.muted{color:#64727e}small{display:block}details{border-top:1px solid #dce2e7;padding:10px 0}summary,.task-row{cursor:pointer}.task-row:hover{background:#f4f7f9}.task-detail{display:none}.task-detail.open{display:table-row}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto;background:#f5f7f9;padding:12px;font:12px/1.5 ui-monospace,monospace;border:1px solid #dce2e7}input,select{font:inherit;padding:6px;margin:10px 12px 10px 0}nav{margin:12px 0 20px}[hidden]{display:none!important}.partial{font-weight:600}
</style></head><body><main><h1>CDS-bench: 30-task model pilot</h1><nav><a href="#overview">Overview</a> · <a href="#conditions">Four conditions</a> · <a href="#coreweave-extra">Additional CoreWeave samples</a> · <a href="#models">Model/provider notes</a> · <a href="#gpt-oss">GPT-OSS tools</a> · <a href="#tasks">By task</a> · <a href="#settings">Settings</a> · <a href="../luna-qwen-high-20261029/report.html">Full evaluation</a></nav>
<p>One instance from each of 30 judgment-selected task types, covering translation, RNA, sequence statistics, constrained design, assays and compositions. Each instance is evaluated in easy/hard form, without tools and with offline Python. Two identical hard variants are omitted, leaving 30 easy and 28 hard prompts, or 116 attempts per complete model. This is a small model/provider comparison, not a representative estimate over the full benchmark. Full-evaluation requests are separate.</p>
<h2 id="overview">Overview</h2><p>All failures remain in the denominator. An asterisk marks a stopped run, whose completed subset is not directly comparable to a complete run. Times are complete-example medians, including failures and tool round trips. Token budgets include reasoning and all tool turns.</p>''']
    top = [
        (MODELS[m] + (" *" if configs[m]["state"] == "stopped" else
                      "" if configs[m]["state"] in {"complete", "prior pilot"} else
                      " (" + configs[m]["state"] + ")"),
         totals[m], configs[m]["aggregate_completion_token_budget"])
        for m in MODELS
    ]
    parts.append(html_results(top))
    parts.append('<p class="muted">Recorded cost covers only the retained completed attempts. Compatibility checks, interrupted requests, discarded duplicate-hard cases and GPT-OSS follow-ups are excluded. Missing billing records are not assumed to be zero. OpenRouter pilot requests had a 300-second deadline. Both internal Marin checkpoints use a <b>1,800-second request deadline</b>: Snowball RC (2026.09.21) has a <b>16,384-token</b> aggregate budget and Snowball v0.1 has <b>32,768 tokens</b>. Both are mixture-of-experts models with <b>67B total parameters and 2B active per token</b>. RC means release candidate. Neither API exposes per-request pricing. Unavailable cost is not zero.</p>')
    parts.append('<p class="muted">Models are ordered by mean task accuracy over their completed cases, giving each available task equal weight. Partial runs cover different subsets, so this display order is not a matched model ranking.</p>')
    parts.append('<p><a href="https://huggingface.co/datasets/eczech/cds-bench-pr-2-traces">Public request, reasoning and tool traces</a> include the saved pilots, follow-up checks, interrupted attempts and provider diagnostics. Credentials and private endpoints are redacted.</p>')
    if summary["provider_reported_budget_overruns"]:
        parts.append('<details><summary>Provider-reported token-budget exceptions</summary><p>Budgets are requested maxima. North reports 32,791 generated tokens on one attempt, 27 above the requested 32,768. This is retained explicitly; requests and scores are unchanged.</p><pre>'+esc(json.dumps(summary["provider_reported_budget_overruns"],indent=2))+'</pre></details>')
    failures = '<table><tr><th>Model</th><th>Incorrect value</th><th>Format</th><th>Timeout</th><th>Token cap</th><th>API/protocol</th><th>Unreconciled examples</th></tr>'
    for m in MODELS:
        failures += '<tr><td>'+esc(MODELS[m])+'</td>'+''.join(f'<td>{totals[m][k]}</td>' for k in ['wrong','format','timeout','token_cap','error','unreconciled_examples'])+'</tr>'
    parts.append('<details><summary>Failure categories and billing coverage</summary><div class="scroll">'+failures+'</table></div><p>Categories are mutually exclusive: correct, timeout, API/protocol failure, token cap, malformed final answer, then incorrect value. Valid-looking content in a malformed answer is not accepted by the primary scorer.</p></details>')
    parts.append('<details><summary>Individual API request times and Python use</summary><table><tr><th>Model</th><th>API requests</th><th>Median request</th><th>Median example</th><th>Examples using Python</th><th>Python calls</th><th>Failed calls</th></tr>')
    for m in MODELS:
        r=totals[m]
        request_time = f'{r["median_api_request_seconds"]:.1f}s' if r['median_api_request_seconds'] is not None else 'Not reached'
        example_time = f'{r["median_seconds"]:.1f}s' if r['median_seconds'] is not None else 'Not reached'
        parts.append(f'<tr><td>{esc(MODELS[m])}</td><td>{r["api_requests"]}</td><td>{request_time}</td><td>{example_time}</td><td>{r["examples_using_python"]}</td><td>{r["python_calls"]}</td><td>{r["python_failures"]}</td></tr>')
    parts.append('</table><p>Request timing runs from sending a request through response completion or failure. Example timing includes all turns, local tool execution and any pacing. Fast protocol rejections lower the GPT-OSS request median. North and Nemotron starts were spaced for free-endpoint limits; that waiting is included in example times. Attempted Python calls include malformed or excess calls rejected without execution.</p></details>')
    parts.append('<h2 id="conditions">Four conditions</h2><p>Each model receives the same retained prompts and reference answers. Easy supplies biological references; hard omits those references. Python is optional when exposed. No web tool is supplied.</p><table><tr><th>Model</th>'+''.join('<th>'+s+'</th>' for s in LABELS)+'</tr>')
    for m in MODELS:
        parts.append('<tr><td>'+esc(MODELS[m])+'</td>'+''.join(f'<td>{r["correct"]}/{r["attempted"]}</td>' if r['attempted'] else '<td>Not reached</td>' for r in summary['conditions'][m].values())+'</tr>')
    parts.append('</table><details><summary>Detailed metrics for every model and condition</summary>')
    parts.append(html_results([(MODELS[m]+': '+label,summary['conditions'][m][c],configs[m]['aggregate_completion_token_budget']) for m in MODELS for c,label in zip(CONDITIONS,LABELS)]))
    parts.append('</details>')
    parts.append(pilot_extension.section(summary['coreweave_extension']))
    parts.append('<h2 id="models">Model/provider notes</h2>')
    notes = {
        'luna': 'Completed all four conditions. This is the original pilot, restricted to the same 116 retained cases; the historical responses are unchanged.',
        'beam': f'Completed all four conditions. {totals["beam"]["python_failures"]} of {totals["beam"]["python_calls"]} attempted Python calls failed in the retained cases. Returned programs sometimes contained trailing PY delimiters, XML argument tags or stray quotes. Programs were executed verbatim. The pilot does not separate model behavior from provider parsing. All six answer-format failures become exact matches under the separate formatting diagnostic.',
        'deepseek': 'Stopped to switch models after the complete easy/no-tools, hard/no-tools and easy/Python passes, plus 13 hard/Python attempts. The partial hard/Python subset should not be treated as a full-pass score.',
        'nemotron': 'Stopped at the user’s request because the free endpoint was too slow. The first three passes completed; only 12 hard/Python attempts finished. Free-endpoint pacing contributes to example time.',
        'nemotron-lightning': 'Uses io.net only, with no provider fallback. Reasoning is enabled without an advertised effort selector; seed is omitted because io.net does not advertise support. Quantization is undisclosed. The same 32,768-token aggregate budget, 300-second request deadline, original JSON Python function and offline sandbox apply. The collector also detects nested streaming provider errors and preserves raw reasoning strings when structured reasoning details are absent; no answer or tool-program repair is applied.',
        'nemotron-lightning-coreweave': 'A separate fresh run of the same Lightning model on CoreWeave BF16 only, with no fallback. Uses the same retained prompts, reasoning enabled, omitted seed, 32,768-token aggregate budget, 300-second request deadline, collector, original JSON Python function and offline sandbox as the io.net run. Seed remains omitted to hold it fixed across providers, although CoreWeave advertises seed support. Starts with ten concurrent examples and no pacing. io.net results are retained separately and are not pooled with this run.',
        'qwen-flash': 'Completed all four conditions. Python improved scores and reduced example time, but many final answers still violated the required format. Formatting-only recoveries are reported separately; primary scores are unchanged. The later full evaluation uses fresh requests, not these responses.',
        'north-mini-code': 'Stopped at the user’s request because it was too slow. All 30 easy/no-tools cases and only three hard/no-tools cases finished. No scored Python condition was reached; its tool preflight succeeded separately. It exposes reasoning enable/disable, not an effort-level selector.',
        'gpt-oss-20b': 'Completed on Amazon Bedrock, with 24 tool-protocol/API failures: 19 HTTP 400 tool-message rejections and five failures from continued calls after the eight-call limit. Some function names contained Harmony channel markup; other calls contained invalid JSON arguments. The initial preflight passed, but scored tool cases were unreliable. See the separate provider diagnostics below.',
        'gemma': 'Stopped because it was too slow/flaky, after 24 easy/no-tools cases. No hard or scored Python condition was reached. The earlier Crusoe preflight was rate-limited; scored requests used Novita BF16.',
        'gemma-moe': 'Stopped at the user’s request because it was too slow. The first three passes completed, with only 12 hard/Python cases finished. Fifty answers failed the required format; many match the reference after limited formatting normalization. There were also five-minute timeouts and HTTP 502/504 errors. NextBit BF16 was used.',
        'liquid': 'Completed all four conditions using the approved 8,192-token aggregate budget, the endpoint’s output limit, rather than 32,768. Reasoning was enabled without an effort selector. Ninety-nine attempts exhausted that budget. All 55 emitted tool calls had valid names and JSON arguments; 47 executions succeeded and eight had program errors, mostly undefined variables in a fresh interpreter. This does not establish performance at a larger budget.',
        'snowball': 'Snowball RC (2026.09.21), the older release candidate, uses the internal Marin/vLLM BF16 service rather than OpenRouter. Its Delphi chat template enables thinking; there is no verified high-effort selector. The server lacks automatic function parsing, so the client supplies the unchanged Python function via the native xml_tools template argument, parses only complete JSON tool_call blocks, and stops generation at the closing tool-call tag. Native thinking is preserved and replayed. Generated Python and final answers are not repaired. A real two-request sandboxed Python round trip passed before scoring. At the user’s request, this model uses a 16,384-token aggregate budget and 1,800-second API deadline. The eight-call limit and offline Python/NumPy environment are unchanged. The earlier 32,768-token/300-second trial is preserved separately and is excluded from these scores. Temperature is explicitly 0.7, seed 20261029; sixty examples run concurrently across all four conditions. API usage reports tokens but no monetary charges, so cost is unavailable rather than zero.',
    }
    notes['snowball-v01'] = ('Uses the newer Marin-Snowball-67B-A2B-v0.1 checkpoint on the existing internal Marin/vLLM BF16 service. The same native Delphi transport, exact Python function and offline sandbox apply, with thinking enabled, temperature 0.7, seed 20261029 and concurrency 60. This pilot restores the 32,768-token aggregate budget while retaining the 1,800-second API deadline. The older Snowball RC (2026.09.21) used 16,384 tokens, so this comparison changes both checkpoint and output budget. No effort-level selector is verified. The unscored preflight executed Python and returned its correct result, but appended a stray Markdown fence; its strict formatting failure is preserved. The actual execution and returned-result delivery were verified separately without retrying or repairing that answer. Primary pilot scoring is unchanged. Costs and reasoning-only token counts are unavailable.')
    lightning = configs['nemotron-lightning']
    coreweave = totals['nemotron-lightning-coreweave']
    if configs['nemotron-lightning-coreweave']['state'] == 'complete':
        notes['nemotron-lightning-coreweave'] = (
            f"Completed all 116 attempts: {coreweave['correct']} correct ({coreweave['accuracy']:.1%})."
            f" No API/protocol errors or request timeouts; {coreweave['token_cap']} token caps, all without tools."
            " Python improved the score from 11/58 without tools to 34/58 with tools."
            " All 169 calls had valid function names and JSON arguments; 136 executions succeeded and 33 had program errors. "
            + notes['nemotron-lightning-coreweave']
        )
    if lightning['state'] == 'stopped':
        partial = totals['nemotron-lightning']
        notes['nemotron-lightning'] = (
            f"Stopped at the user’s request because it was too slow. {partial['attempted']}/116 attempts finished,"
            f" with {partial['correct']} correct ({partial['accuracy']:.1%}), a {partial['median_seconds']:.1f}-second median"
            f" and ${partial['cost_usd']:.5f} in recorded charges. Failures: {partial['wrong']} incorrect values,"
            f" {partial['format']} formatting failures, {partial['timeout']} timeouts, {partial['token_cap']} token caps"
            f" and {partial['error']} API/protocol errors. {lightning['interrupted_original_attempts']} interrupted attempts"
            " are preserved but excluded from scoring and timing. The hard/Python pass was not reached. "
            + notes['nemotron-lightning']
        )
    if lightning['scheduling_continuation']:
        scheduling = lightning['scheduling_continuation']
        notes['nemotron-lightning'] += (
            f" Repeated provider errors stopped the initial run after {scheduling['initial_completed']} attempts."
            f" Only unissued cases then continued with {scheduling['concurrency']} concurrent examples"
            f" and {scheduling['minimum_request_spacing_seconds']:g}-second request spacing."
            " All initial failures remain unchanged; no scored request was retried."
        )
        if lightning['stop_reason']:
            notes['nemotron-lightning'] += ' ' + lightning['stop_reason']
    for m in MODELS:
        c=configs[m]
        parts.append('<details><summary>'+esc(MODELS[m])+' · '+esc(', '.join(c['provider']['only']))+'</summary><p>'+esc(notes[m])+'</p><p><code>'+esc(c['model'])+'</code>; reasoning <code>'+esc(json.dumps(c['reasoning'],separators=(',',':')))+'</code>; no provider fallback.</p></details>')
        if m in snowball_pilot.MODELS:
            p = c['provenance']
            parts.append('<p>'+esc(MODELS[m])+' provenance: <a href="'+esc(p['url'])+'">'+esc(c['full_name'])+'</a>, <code>'+esc(p['repo'])+'</code> at revision <code>'+esc(p['revision'])+'</code>. <a href="'+snowball_pilot.MODELS[m]['audit_file']+'">Pilot audit</a>.</p>')
            a = c['tool_audit']
            parts.append(f"<p>Tool verification: {a['successful_executions']} successful Python executions, {a['program_errors']} program failures and {a['invalid_calls']} rejected calls. {a['native_protocol_failures']} attempts returned malformed native tool JSON or tags; {a['call_limit_failures']} continued calling after the limit. These remain failed attempts. Scored HTTP failures: {a['http_failures']}.</p>")
            if a.get('incomplete_streams'):
                parts.append(f"<p>{a['incomplete_streams']} request ended before its response stream was complete. It remains an API/connection failure, with no retry.</p>")
    parts.append('<details><summary>Formatting diagnostic; primary scores unchanged</summary><p>Only an inline Final answer: marker and one outer JSON fence are normalized. Missing markers and answer content are not repaired.</p><table><tr><th>Model</th><th>Strict correct</th><th>Additional exact matches</th></tr>')
    for m in MODELS:
        parts.append(f'<tr><td>{esc(MODELS[m])}</td><td>{totals[m]["correct"]}/{totals[m]["attempted"]}</td><td>{totals[m]["format_diagnostic_additional_exact"]}</td></tr>')
    parts.append('</table></details><h2 id="gpt-oss">GPT-OSS: no reliable OpenRouter tool setup established</h2><p>On ten matched easy instances, CoreWeave with a renamed JSON function improved from 3/10 without tools to 5/10 with tools, but four tool attempts failed at the protocol level. The local sandbox successfully executed many valid programs. These exploratory checks used subsets selected partly for earlier failures, so they are not independent model accuracy estimates and are not pooled into the pilot table.</p><p>Variants tried renaming the function, accepting raw Python arguments without repairing code, explicit fresh-script/print instructions, non-streaming requests, lower temperature, a forced first tool call, hidden returned reasoning, high/medium effort, and Groq/DeepInfra endpoints. DeepInfra was rate-limited and was not substantively evaluated. Responses API checks also failed. No hosted interpreter or web tool was enabled.</p><details><summary>Diagnostic configurations and execution counts</summary><div class="scroll"><table><tr><th>Configuration</th><th>Correct / attempted</th><th>Successful Python executions / attempted calls</th><th>Rejected calls</th></tr>')
    for entry in diagnostics['configurations']:
        counts='; '.join(c['condition']+f': {c["correct"]}/{c["attempted"]}' for c in entry['conditions'])
        parts.append(f'<tr><td>{esc(entry["configuration"])}</td><td>{esc(counts)}</td><td>{entry["successful_python_executions"]}/{entry["tool_calls"]}</td><td>{entry["rejected_calls"]}</td></tr>')
    parts.append('</table></div><p>One older non-streaming Groq trace contains a nested provider error that the diagnostic client originally classified as a formatting failure. The JSON flags that case; original scores remain unchanged. Rejected calls include malformed arguments and calls beyond the limit. CoreWeave’s Responses API check failed on provider content validation; Groq reported a required-call failure.</p><p>A <a href="https://arxiv.org/html/2604.00362v1">native Harmony harness</a> is a documented candidate, not a locally verified fix. The tested OpenRouter configurations do not establish a general limit on GPT-OSS tool capability.</p></details>')
    parts.append('<h2 id="tasks">By task</h2><p>Choose a model, then expand a row for actual prompts, expected answers and final predictions. Raw reasoning and tool programs/results are not included.</p><label>Model <select id="model">'+''.join(f'<option value="{m}">{esc(name)}</option>' for m,name in MODELS.items())+'</select></label><label>Task <input id="filter" type="search" placeholder="Filter task name or ID"></label><table><thead><tr><th>Task</th>'+''.join('<th>'+s+'</th>' for s in LABELS)+'<th></th></tr></thead><tbody>')
    for t in summary['task_ids']:
        title=TASKS[t]['title']
        parts.append(f'<tr class="task-row" id="task-{t}" data-task="{t}" data-search="{esc((t+" "+title).lower())}" tabindex="0" aria-expanded="false"><td><a href="#task-{t}">{esc(title)}</a><small>{esc(t)}</small></td>'+''.join(f'<td data-condition="{c}"></td>' for c in CONDITIONS)+'<td class="toggle" aria-hidden="true">+</td></tr><tr class="task-detail"><td colspan="6"></td></tr>')
    parts.append('</tbody></table><h2 id="settings">Settings, execution and saved data</h2><p>High reasoning was requested where supported. Gemma, North, Liquid and Nemotron Lightning expose reasoning enable/disable without an effort selector. OpenRouter pilot calls used provider-default temperature, a seed where supported (omitted on Bedrock and both Lightning runs), streaming responses and initially ten concurrent examples. North and Nemotron Ultra starts were paced for free-endpoint limits. After provider errors, io.net continued only unissued cases at concurrency three with five-second request spacing; its timing includes that scheduling change.</p><p>Both Snowball models have 67B total parameters and 2B active per token. They use native thinking and tool-call transport, temperature 0.7, seed 20261029 and concurrency 60; their checkpoint provenance, budgets and transport differences are recorded above. Python 3.12.9 + NumPy 2.5.4, no Biopython. The custom function runs locally under <code>/usr/bin/sandbox-exec</code>, with networking and process forks denied, and restricted file access. Each call starts a fresh interpreter; files persist within the example. Limits: eight Python calls/example, 30 seconds/call, 16,384 characters each of stdout/stderr. The no-tools condition supplies no functions. Web plugins are disabled. Network denial is an executor policy, not an OpenRouter parameter.</p>')
    parts.append('<details><summary>Exact Python function supplied to the models</summary><pre>'+esc(json.dumps(configs['luna']['tools'],indent=2))+'</pre></details>')
    parts.append('<details><summary>Exact request settings, provider routing and billing coverage</summary><p>Some source billing summaries also cover interrupted or removed duplicate-hard attempts; their totals are not substituted for the scored-subset costs above.</p><pre>'+esc(json.dumps(configs,indent=2))+'</pre></details>')
    parts.append('<details><summary>Earlier ten-example checks, excluded from the 30-task pilot</summary><p>Easy-only, no-tools smoke checks. GLM requested max reasoning with temperature 1.0 and top_p 0.95; Luna requested high reasoning. These earlier checks did not use the later pilot’s uniform five-minute wall deadline. All had a 32,768-token cap.</p><table><tr><th>Model / provider</th><th>Correct</th><th>Recorded cost</th><th>Median example</th><th>Token caps</th></tr>')
    for r in summary['earlier_ten_example_checks']:
        parts.append(f'<tr><td>{esc(r["name"])}</td><td>{r["correct"]}/{r["attempted"]}</td><td>${r["cost_usd"]:.5f}</td><td>{r["median_seconds"]:.1f}s</td><td>{r["token_caps"]}</td></tr>')
    parts.append('</table></details><p><a href="summary.json">Summary and settings</a> · <a href="by-task.csv">Per-task CSV</a> · <a href="results.json">Final predictions</a> · <a href="cases.json">Prompt instances</a> · <a href="gpt-oss-diagnostics.json">GPT-OSS diagnostics</a></p>')
    payload={'models':MODELS,'conditions':CONDITIONS,'labels':LABELS,'rows':rows,'cases':cases}
    parts.append('<script id="data" type="application/json">'+json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')+'</script>')
    parts.append('''<script>
const d=JSON.parse(document.getElementById('data').textContent),lookup=new Map(),cases=new Map();for(const r of d.rows)lookup.set(r.model+'|'+r.task_id+'|'+r.condition,r);for(const c of d.cases)cases.set(c.task_id+'|'+c.variant,c);const names={correct:'Correct',wrong:'Incorrect',format:'Format',timeout:'Timeout',token_cap:'Token cap',error:'API/protocol'};function node(tag,text){const x=document.createElement(tag);if(text!==undefined)x.textContent=text;return x}function section(label,text){const x=node('details');x.append(node('summary',label),node('pre',text));return x}
function expand(row,force){const child=row.nextElementSibling,open=force===undefined?!child.classList.contains('open'):force;child.classList.toggle('open',open);row.setAttribute('aria-expanded',open);row.querySelector('.toggle').textContent=open?'×':'+';if(!open||child.dataset.loaded)return;child.dataset.loaded='1';const cell=child.firstElementChild;for(const v of ['easy','hard']){const c=cases.get(row.dataset.task+'|'+v);if(c)cell.append(section(v+' prompt',c.prompt),section(v+' expected answer',c.answer))}for(let i=0;i<d.conditions.length;i++){const condition=d.conditions[i],group=node('details');group.append(node('summary',d.labels[i]));for(const [m,name]of Object.entries(d.models)){const r=lookup.get(m+'|'+row.dataset.task+'|'+condition);group.append(section(name+': '+(r?names[r.outcome]:(cases.has(row.dataset.task+'|'+condition.split('-')[0])?'Not reached':'Variant omitted')),r?((r.final_answer??'No final answer')+'\\n\\n'+r.elapsed_seconds.toFixed(1)+'s; '+r.tool_calls+' Python calls; '+(r.cost_usd===null?'cost unavailable':'$'+r.cost_usd.toFixed(6))):'No scored attempt'))}cell.append(group)}}
function update(){const m=document.getElementById('model').value;for(const row of document.querySelectorAll('.task-row'))for(const cell of row.querySelectorAll('[data-condition]')){const r=lookup.get(m+'|'+row.dataset.task+'|'+cell.dataset.condition);cell.textContent=r?names[r.outcome]:(cases.has(row.dataset.task+'|'+cell.dataset.condition.split('-')[0])?'Not reached':'Variant omitted')}}
for(const row of document.querySelectorAll('.task-row')){row.addEventListener('click',e=>expand(row,e.target.closest('a')?true:undefined));row.addEventListener('keydown',e=>{if(e.target===row&&(e.key==='Enter'||e.key===' ')){e.preventDefault();expand(row)}})}document.getElementById('model').addEventListener('change',update);document.getElementById('filter').addEventListener('input',e=>{for(const row of document.querySelectorAll('.task-row')){row.hidden=!row.dataset.search.includes(e.target.value.toLowerCase());row.nextElementSibling.hidden=row.hidden}});function anchor(){const row=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(row?.classList.contains('task-row'))expand(row,true)}addEventListener('hashchange',anchor);update();anchor();
</script></main></body></html>''')
    (output / 'report.html').write_text('\n'.join(parts))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    result=build(args.source.resolve(),args.output.resolve())
    print({m:(r['correct'],r['attempted']) for m,r in result['totals'].items()})
