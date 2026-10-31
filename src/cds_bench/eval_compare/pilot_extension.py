"""Retain the stopped CoreWeave expansion as exploratory pilot data."""

import csv
import html
import json
from pathlib import Path

from cds_bench.eval_luna.io import save
from cds_bench.eval_luna.report import aggregate, public_record
from cds_bench.eval_luna.score import score_result
from .collector import records
from .lightning_run import RAW, ROOT
from .tables import html_results

NAME = 'Nemotron 3.5 Lightning / CoreWeave: additional samples'


def build(output, pilot_rows):
    source = records(RAW)
    config = json.loads((RAW / 'config.json').read_text())
    stopped = json.loads((RAW / 'user-interrupted.json').read_text())
    pilot_ids = {(r['instance_id'], r['condition']) for r in pilot_rows
                 if r['model'] == 'nemotron-lightning-coreweave'}
    rows, cases = [], {}
    for row in source:
        directory = Path(row['trace_directory'])
        case = json.loads((directory / 'case.json').read_text())
        assert case['answer'] == row['expected_answer']
        if not row.get('api_error'):
            assert score_result(case, (directory / 'answer.txt').read_text(), row['finish_reason'])['passed'] == row['passed']
        exported = {k: v for k, v in public_record(row).items() if k in row}
        if exported.get('error'):
            exported['error'] = {'error': 'API failure', 'timeout': 'API deadline exceeded',
                                 'format': 'Final answer does not satisfy the required format',
                                 'token_cap': 'Generated-token budget exhausted', 'wrong': 'Incorrect value'}[row['outcome']]
        rows.append(exported)
        cases[row['instance_id']] = case
    totals = aggregate(rows)
    assert totals == aggregate(source) and totals['attempted'] == 353
    summary = {'name': NAME, 'state': 'stopped', 'totals': totals,
               'conditions': {c: aggregate([r for r in rows if r['condition'] == c])
                              for c in config['planned_per_condition']},
               'task_count': len({r['task_id'] for r in rows}),
               'interrupted_excluded': len(stopped['interrupted_results_excluded']),
               'repeated_pilot_case_conditions': sum((r['instance_id'], r['condition']) in pilot_ids for r in rows),
               'parameters_no_tools': config['parameters_no_tools'], 'concurrency': config['concurrency'],
               'request_timeout_seconds': config['request_timeout_seconds'],
               'aggregate_completion_token_budget': config['aggregate_completion_token_budget'],
               'dataset': config['dataset'], 'pooled_with_original_pilot': False,
               'validation': {'original_scores_verified': True, 'aggregate_metrics_verified': True,
                              'no_inference_or_retries': True}}
    payload = {'summary': summary, 'results': rows, 'cases': list(cases.values())}
    save(output / 'coreweave-extra.json', payload)
    with (output / 'coreweave-extra-by-task.csv').open('w') as f:
        fields = ['task_id', 'attempted', 'correct', 'accuracy', 'wrong', 'format', 'timeout', 'token_cap', 'error', 'median_seconds', 'cost_usd']
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for task in sorted({r['task_id'] for r in rows}):
            metrics = aggregate([r for r in rows if r['task_id'] == task])
            writer.writerow({'task_id': task, **{k: metrics[k] for k in fields if k != 'task_id'}})
    esc = html.escape
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoreWeave Nemotron: additional pilot samples</title><style>body{font:15px/1.5 system-ui;color:#25313c;max-width:1200px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%}th,td{padding:8px;text-align:left;border-bottom:1px solid #dce2e7}summary{cursor:pointer}details{padding:8px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:12px;max-height:360px;overflow:auto}a{color:#205d89}</style><h1>CoreWeave Nemotron: additional pilot samples</h1><p><a href="report.html#coreweave-extra">Back to the model pilot</a></p>']
    parts.append('<p>The stopped expansion contains 353 completed Easy/no-tools attempts, beyond the original 116-attempt pilot. Ten interrupted attempts are excluded; raw traces remain preserved. These samples are exploratory and are excluded from the main full-evaluation comparison.</p>')
    parts.append(f'<p>{summary["repeated_pilot_case_conditions"]} case/condition pairs repeat the original pilot in independent requests. The two runs are reported separately.</p>')
    parts.append(html_results([(NAME, totals, 32768)]))
    parts.append('<p>'+ '; '.join(f'{k}: {totals[k]}' for k in ['wrong','format','timeout','token_cap','error','unreconciled_examples']) + '.</p>')
    parts.append('<details><summary>Exact settings and retained sample metadata</summary><pre>'+esc(json.dumps(summary,indent=2))+'</pre></details><h2>All additional attempts</h2>')
    for task in sorted({r['task_id'] for r in rows}):
        selected = [r for r in rows if r['task_id'] == task]
        metrics = aggregate(selected)
        parts.append(f'<details><summary>{esc(task)}: {metrics["correct"]}/{metrics["attempted"]}</summary>')
        for row in sorted(selected, key=lambda r:r['instance_id']):
            case = cases[row['instance_id']]
            parts.append(f'<details><summary>{esc(row["instance_id"])} · {esc(row["outcome"])} · {row["elapsed_seconds"]:.1f}s · ${row["cost_usd"]:.5f}</summary>')
            for title, text in [('Exact prompt', case['prompt']), ('Expected answer', case['answer']), ('Final prediction', row.get('final_answer') or row.get('error') or 'No final answer')]:
                parts.append('<h3>'+title+'</h3><pre>'+esc(str(text))+'</pre>')
            parts.append('</details>')
        parts.append('</details>')
    parts.append('<p><a href="coreweave-extra.json">Predictions, prompts and settings</a> · <a href="coreweave-extra-by-task.csv">Per-task metrics</a></p></html>')
    (output / 'coreweave-extra.html').write_text('\n'.join(parts))
    return summary


def section(summary):
    totals = summary['totals']
    return ('<h2 id="coreweave-extra">CoreWeave Nemotron: additional pilot samples</h2>'
            '<p>The original 116-attempt pilot is unchanged. The stopped expansion adds 353 completed Easy/no-tools attempts; ten interrupted attempts are excluded. This expansion is exploratory and is excluded from the full evaluation. Repeated pilot cases are kept separate.</p>'
            + html_results([(NAME, totals, 32768)])
            + '<p>' + '; '.join(f'{k}: {totals[k]}' for k in ['wrong','format','timeout','token_cap','error']) + '.</p>'
            + '<p><a href="coreweave-extra.html">Inspect all 353 additional attempts</a> · <a href="coreweave-extra-by-task.csv">Per-task metrics</a> · <a href="coreweave-extra.json">Predictions, prompts and settings</a></p>')
