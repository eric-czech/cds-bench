"""Describe saved tool use without changing inference or primary scores."""

import ast
import csv
import html
import json
import re
from collections import defaultdict
from statistics import mean

from cds_bench.catalog import TASKS
from cds_bench.eval_luna.io import save

from .snowball_full import RAW, ROOT

ONESHOT = ROOT/'scratch/snowball-v01-oneshot-pilot-20261031-01'


def one_shot():
    """Keep the prompted follow-up separate from all original model scores."""
    path=ONESHOT/'summary.json'
    if not path.exists():
        return None
    summary=json.loads(path.read_text())
    if summary['state']!='complete':
        return None
    audit=json.loads((ONESHOT/'audit.json').read_text())
    assert audit['passed'] and audit['attempts']==58
    config=json.loads((ONESHOT/'config.json').read_text())
    preflight=json.loads((ONESHOT/'preflight-result.json').read_text())
    return {'attempts':58, 'original_conditions':summary['original_conditions'],
            'conditions':summary['conditions'], 'paired_comparison':summary['paired_comparison'],
            'demonstration':json.loads((ONESHOT/'demonstration.json').read_text()),
            'provenance':config['provenance'], 'settings':{k:config[k] for k in
                ['concurrency','aggregate_completion_token_budget','request_timeout_seconds',
                 'connection_timeout_seconds','max_tool_calls','adapter','sampling','cost_note']},
            'transport_note':'Uses the full-run client with the identical native request transformation and response parser. Relative to the original pilot, it adds error-stage metadata and a 15-second connection-setup timeout instead of the generic 1,800-second timeout. No connection failures occurred in this follow-up.',
            'preflight':{'real_tool_round_trip':bool(preflight['tool_calls'] and not preflight['tool_failures']),
                         'strict_answer_passed':preflight['passed']},
            'audit':audit,
            'interpretation':'Fresh attempts on the original 30 easy and 28 hard pilot questions. Each benchmark user message is unchanged byte for byte, but the full API message history differs: an arithmetic question, assistant Python call, tool result and assistant example answer are prepended before it. The demonstration shows both tool use and final-answer formatting. Baseline request transformation, response parser and scorer are unchanged. One draw per question; no causal isolation of formatting versus tool-selection effects.'}


def within_tasks(rows, minimum=2):
    """Match task types, retaining >= minimum examples in each observed group."""
    grouped = defaultdict(list)
    for row in rows:
        if row['model'] == 'snowball-v01' and row['condition'] == 'easy-tools':
            grouped[row['task_id']].append(row)
    tasks = []
    for task, examples in sorted(grouped.items()):
        called = [r for r in examples if r['tool_calls'] > 0]
        uncalled = [r for r in examples if r['tool_calls'] == 0]
        if min(len(called), len(uncalled)) < minimum:
            continue
        a, b = mean(bool(r['passed']) for r in called), mean(bool(r['passed']) for r in uncalled)
        tasks.append({'task_id':task, 'name':TASKS[task]['title'],
                      'called_correct':sum(bool(r['passed']) for r in called), 'called_n':len(called),
                      'uncalled_correct':sum(bool(r['passed']) for r in uncalled), 'uncalled_n':len(uncalled),
                      'called_accuracy':a, 'uncalled_accuracy':b, 'difference':a-b})
    return {'condition':'easy-tools', 'minimum_examples_per_group_per_task':minimum,
            'eligible_tasks':len(tasks), 'tasks':tasks,
            'called_mean_task_accuracy':mean(r['called_accuracy'] for r in tasks) if tasks else None,
            'uncalled_mean_task_accuracy':mean(r['uncalled_accuracy'] for r in tasks) if tasks else None,
            'mean_within_task_difference':mean(r['difference'] for r in tasks) if tasks else None,
            'called_correct':sum(r['called_correct'] for r in tasks),
            'called_n':sum(r['called_n'] for r in tasks),
            'uncalled_correct':sum(r['uncalled_correct'] for r in tasks),
            'uncalled_n':sum(r['uncalled_n'] for r in tasks),
            'interpretation':'Observational comparison of different examples within the same task types, not a causal tool effect. All failures count. Call means at least one recorded Python invocation, including program failures.'}


def build(output, rows, models):
    uptake = {}
    for model, name in models.items():
        selected = [r for r in rows if r['model']==model and r['condition'].endswith('-tools') and not r['condition'].endswith('-no-tools')]
        uptake[model] = {'name':name, 'available':len(selected),
                         'called':sum(r['tool_calls']>0 for r in selected)}
    missed = []
    for condition in ['easy-tools','hard-tools']:
        for path in sorted((RAW/'traces'/condition).glob('*/response-*.json')):
            reply = json.loads(path.read_text())
            if reply.get('message',{}).get('tool_calls'):
                continue
            raw = reply.get('raw_content','')
            if '<|start_think|>' not in raw or '<|end_think|>' in raw or not raw.rstrip().endswith('</tool_call>'):
                continue
            for match in re.finditer(r'<tool_call>(.*?)</tool_call>',raw,re.DOTALL):
                try:
                    call = json.loads(match[1])
                except (ValueError, TypeError):
                    continue
                if not isinstance(call,dict) or set(call)!={'name','arguments'} or call['name']!='python':
                    continue
                arguments = call['arguments']
                if not isinstance(arguments,dict) or set(arguments)!={'code'} or not isinstance(arguments['code'],str):
                    continue
                try:
                    ast.parse(arguments['code'])
                    syntax = True
                except SyntaxError:
                    syntax = False
                case = json.loads((path.parent/'case.json').read_text())
                missed.append({'instance_id':case['instance_id'], 'condition':condition,
                               'request_number':reply['request_number'], 'valid_python_syntax':syntax})
    audit = json.loads((RAW/'audit.json').read_text())
    result = {'tool_invocation':uptake, 'easy_python_within_tasks':within_tasks(rows),
              'successful_program_executions':audit['successful_python_executions'],
              'program_failures':audit['python_program_failures'],
              'missed_complete_json_handoffs':missed,
              'missed_handoff_note':'The original parser returned no calls for these complete JSON calls inside unfinished thinking. Seven contain syntactically valid Python; two are placeholder programs. No programs were executed again and no scores were changed.'}
    result['one_shot_pilot']=one_shot()
    save(output/'snowball-tool-use.json',result)
    tasks = result['easy_python_within_tasks']['tasks']
    if tasks:
        with (output/'snowball-tool-use-by-task.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=list(tasks[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(tasks)
    return result


def short_text(stats):
    a = stats['tool_invocation']['snowball-v01']
    b = stats['tool_invocation']['deepinfra']
    text = (f"With Python available, Snowball invoked it on {a['called']:,}/{a['available']:,} examples "
            f"({a['called']/a['available']:.1%}), versus {b['called']:,}/{b['available']:,} "
            f"({b['called']/b['available']:.1%}) for Nemotron. "
            "These are optional tool-use conditions; supplying a tool does not require the model to call it.")
    if stats.get('one_shot_pilot'):
        text += (" In the one-shot pilot, 7 of 22 final responses missing the required final-answer marker "
                 "still ended with the correct answer; two also prefixed it with the demonstration's arithmetic result.")
    return text


def comparison_text(stats):
    s = stats['easy_python_within_tasks']
    return (f"For Easy/Python, {s['eligible_tasks']} task types have at least two examples with a Python call and "
            "two without one. Averaging accuracy equally across these same task types gives "
            f"{s['called_mean_task_accuracy']:.1%} with calls versus {s['uncalled_mean_task_accuracy']:.1%} without "
            f"({100*s['mean_within_task_difference']:+.1f} percentage points). "
            "This compares different examples within each task and is observational, not a causal estimate of tool benefit.")


def native_explanation(stats):
    return ("The service is not configured for standard automatic function parsing; the initial standard auto-tool probe returned HTTP 400. "
            "The controller passes the exact function below through chat_template_kwargs.xml_tools, with enable_thinking=true and skip_special_tokens=false. "
            "The checkpoint's own template describes calls as JSON objects with name and arguments fields between opening and closing tool_call tags. "
            "Generation stops at the closing tool_call tag, including that delimiter. The client parses the JSON, executes code verbatim locally, "
            "and supplies the result as a tool-role message containing stdout, stderr, exit_code, timed_out and output_truncated. "
            "Prior native reasoning and calls are replayed before the model continues. Tool use is optional; strict:true is embedded in the definition, "
            "but this native route does not enforce a decoding grammar. At eight calls the definitions are omitted.\n\n"
            f"The original full run recorded {stats['successful_program_executions']} successful program executions and {stats['program_failures']} program failures. "
            "Exit zero is not evidence that the program or final answer was correct.\n\n"
            "A later review found nine complete JSON calls inside unfinished thinking blocks. The original parser treated them as reasoning and returned no call; "
            "seven contain syntactically valid Python and two contain placeholder programs. These remain failed original attempts, including in the no-call subgroup. "
            "No scores were changed. This is a known adapter limitation.\n\n"
            "The checkpoint template also advertises a direct Python-code channel using the python_tag and eom_id special tokens, with ipython replies. "
            "That alternative was not used or tested here. It advertises a stateful notebook, whereas this evaluation uses a fresh interpreter per call. "
            "Server-side parsing and a forced initial Python call are other possible diagnostics, not features of this baseline. "
            "The private capability URL and serving model ID are omitted. A real execution/result round trip passed before scoring.")


def one_shot_text(pilot):
    old,new=pilot['original_conditions'].values(),pilot['conditions'].values()
    old_calls,new_calls=sum(r['tool_users'] for r in old),sum(r['tool_users'] for r in new)
    old_correct,new_correct=sum(r['correct'] for r in old),sum(r['correct'] for r in new)
    n=pilot['attempts']
    return (f"The one-demonstration Snowball pilot used Python on {new_calls}/{n} examples versus {old_calls}/{n} originally. "
            f"Correct answers: {new_correct}/{n} ({new_correct/n:.1%}) versus {old_correct}/{n} ({old_correct/n:.1%}). "
            "This small matched repeat does not show an accuracy benefit.")


def one_shot_markdown(stats, base):
    pilot=stats.get('one_shot_pilot')
    if not pilot:
        return []
    lines=['\n<details>\n<summary>Snowball follow-up: one tool-use demonstration on the matched pilot</summary>\n',
           one_shot_text(pilot),
           '\n'+pilot['interpretation'],
           '\n| Condition | Original correct | With demo | Original tool users | With demo |',
           '|---|---:|---:|---:|---:|']
    for condition,label in [('easy-tools','Easy, Python'),('hard-tools','Hard, Python')]:
        a,b=pilot['original_conditions'][condition],pilot['conditions'][condition]
        lines.append(f'| {label} | {a["correct"]}/{a["n"]} ({a["accuracy"]:.1%}) | {b["correct"]}/{b["n"]} ({b["accuracy"]:.1%}) | {a["tool_users"]}/{a["n"]} | {b["tool_users"]}/{b["n"]} |')
    lines.extend(['\nSame native XML transformation/parser, optional Python tool, 32,768 aggregate generated tokens, temperature 0.7, seed 20261029, concurrency 60 and 1,800-second API deadline. The demonstration is the only change to model-facing context/settings. Its call is excluded from measured tool use. Cost is unavailable. The full baselines and pilot model-ranking table are unchanged.',
                  '\n'+pilot['transport_note'],
                  f'\n[Follow-up results and exact demonstration]({base}/report.html#snowball-oneshot). No scored retries; all 58 attempts passed the source/prompt/budget/scoring audit.',
                  '\n</details>'])
    return lines


def html_section(stats):
    esc=html.escape
    s=stats['easy_python_within_tasks']
    parts=['<h2 id="snowball-tools">Snowball tool use</h2>', '<p>'+esc(short_text(stats))+'</p>',
           '<p>'+esc(comparison_text(stats))+'</p>',
           '<details><summary>Easy/Python accuracy within the same task types</summary>',
           f"<p>Eligible examples pooled: calls {s['called_correct']}/{s['called_n']}; no calls {s['uncalled_correct']}/{s['uncalled_n']}. "
           "The headline percentages instead give each task equal weight. Program failures count as calls; all answer and infrastructure failures count against accuracy.</p>",
           '<table><tr><th>Task</th><th>Calls: correct / n</th><th>No calls: correct / n</th><th>Difference</th></tr>']
    for r in s['tasks']:
        parts.append(f'<tr><td><a href="#task-{esc(r["task_id"])}">{esc(r["name"])}</a><small>{esc(r["task_id"])}</small></td>'
                     f'<td>{r["called_correct"]}/{r["called_n"]} ({r["called_accuracy"]:.1%})</td>'
                     f'<td>{r["uncalled_correct"]}/{r["uncalled_n"]} ({r["uncalled_accuracy"]:.1%})</td>'
                     f'<td>{100*r["difference"]:+.1f} pp</td></tr>')
    parts.append('</table><p><a href="snowball-tool-use.json">Analysis JSON</a> · <a href="snowball-tool-use-by-task.csv">Per-task CSV</a></p></details>')
    pilot=stats.get('one_shot_pilot')
    if pilot:
        parts.extend(['<h3 id="snowball-oneshot">One tool-use demonstration: matched pilot</h3>',
            '<p>'+esc(pilot['interpretation'])+'</p>',
            '<p>'+esc(one_shot_text(pilot))+'</p>',
            '<table><tr><th>Condition</th><th>Original correct</th><th>With demonstration</th><th>Original tool users</th><th>With demonstration</th></tr>'])
        for condition,label in [('easy-tools','Easy, Python'),('hard-tools','Hard, Python')]:
            old,new=pilot['original_conditions'][condition],pilot['conditions'][condition]
            parts.append(f'<tr><td>{label}</td><td>{old["correct"]}/{old["n"]} ({old["accuracy"]:.1%})</td>'
                f'<td>{new["correct"]}/{new["n"]} ({new["accuracy"]:.1%})</td>'
                f'<td>{old["tool_users"]}/{old["n"]}</td><td>{new["tool_users"]}/{new["n"]}</td></tr>')
        parts.append('</table><p>Same native XML adapter and optional Python tool, 32,768 aggregate generated tokens, temperature 0.7, seed 20261029, concurrency 60 and 1,800-second API deadline. The demonstration is context; its call is excluded from measured tool use. No baseline scores or model rankings include this follow-up. Cost is unavailable.</p>')
        parts.append('<details><summary>Exact demonstration messages and follow-up audit</summary><pre>'+esc(json.dumps(pilot['demonstration']['messages'],indent=2))+'</pre><p>'+esc(pilot['transport_note'])+'</p><p>Unscored preflight executed Python and delivered its result; strict final-answer score: '+str(pilot['preflight']['strict_answer_passed']).lower()+'. All 58 scored attempts passed the prompt, source-hash, token-budget and scoring audit. No scored retries.</p></details>')
    return ''.join(parts)
