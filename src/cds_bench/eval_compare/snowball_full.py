"""Saved full Snowball records, public settings and independent completion audit."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from cds_bench.eval_luna.io import save
from cds_bench.eval_luna.report import outcome
from cds_bench.eval_luna.score import score_result

from .snowball_run import MODEL_KEY, MODEL_REPO, MODEL_REVISION, PILOT, PLANNED, RAW, ROOT


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def records(raw=RAW):
    rows = []
    for original in read(raw / 'results.json', []):
        row = dict(original)
        directory = Path(row['trace_directory'])
        case = read(directory / 'case.json')
        row.update(expected_answer=case['answer'], output_format=case['output_format'],
                   billing_complete=False, cost_usd=0, reasoning_tokens=0)
        row['outcome'] = outcome(row)
        rows.append(row)
    return rows


def public_config(raw=RAW):
    config = read(raw / 'config.json')
    if not config:
        return None
    fields = ['run_id','model_key','name','full_name','provenance','total_parameters',
              'active_parameters_per_token','planned_attempts','planned_per_condition',
              'condition_order','tools_condition','native_request_fields','transport',
              'reasoning','request_timeout_seconds','aggregate_completion_token_budget',
              'max_tool_calls','concurrency','dataset','luna_comparison','execution_environment',
              'sampling','settings_difference','pilot_config_sha256','source_commit',
              'inference_source_sha256','cost_note']
    result = {k:config[k] for k in fields}
    result['parameters_no_tools'] = {**config['parameters_no_tools'], 'model':'[private serving model ID]',
        'skip_special_tokens':False, 'chat_template_kwargs':{'enable_thinking':True}}
    result['token_count_note'] = 'Completion tokens include thinking; reasoning-only counts are not separately exposed.'
    result['connection_setup_timeout_seconds'] = 15
    result['operational_resumes'] = [
        {k:event[k] for k in ['at','completed_before_resume','scored_retries','settings_changed']}
        for event in (read(p) for p in sorted(raw.glob('resume-*.json')))
    ]
    return result


def public_row(row):
    row = dict(row, cost_usd=None, reasoning_tokens=None)
    if row.get('error'):
        row['error'] = {'error':'API, connection or native tool-protocol failure',
            'timeout':'API deadline exceeded','format':'Final answer does not satisfy the required format',
            'token_cap':'Generated-token budget exhausted','wrong':'Incorrect value'}[row['outcome']]
    return row


def audit(raw=RAW, output=None):
    config = read(raw / 'config.json')
    job = read(raw / 'job-status.json')
    assert job['state']=='complete' and job['completed']==3260
    assert config['provenance']['repo']==MODEL_REPO and config['provenance']['revision']==MODEL_REVISION
    assert config['aggregate_completion_token_budget']==32768 and config['request_timeout_seconds']==1800
    for name,digest in config['inference_source_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest()==digest, name
    dataset = read(raw / 'dataset.json')
    cases = {(c['instance_id'],variant):c for variant,rs in dataset.items() for c in rs}
    rows = records(raw)
    assert len(rows)==len({(r['instance_id'],r['condition']) for r in rows})==3260
    assert Counter(r['condition'] for r in rows)==PLANNED
    pilot_ids = {i for r in read(PILOT/'results.json', []) for i in r.get('response_ids', []) if i}
    full_ids = {i for r in rows for i in r.get('response_ids', []) if i}
    assert pilot_ids and full_ids and not pilot_ids.intersection(full_ids)
    expected = {(c['instance_id'],v+('-tools' if t else '-no-tools'))
                for v,rs in dataset.items() for c in rs for t in [False,True]}
    assert {(r['instance_id'],r['condition']) for r in rows}==expected
    sandbox = read(raw / 'sandbox-validation.json')
    assert sandbox and sandbox['numpy']['stdout'].strip()=='10'
    assert sandbox['workdir']['stdout'].strip()=='ok'
    assert all(r['exit_code']!=0 for name,r in sandbox.items() if name not in {'numpy','workdir'})
    preflight = read(raw / 'tool-round-trip-verification.json')
    assert preflight['passed'] and preflight['tool_result_delivered']
    successes = programs_failed = rejected = native_failures = network_failures = 0
    request_count = reasoning_replays = tool_replays = 0
    for row in rows:
        directory = Path(row['trace_directory'])
        assert directory.resolve().is_relative_to(raw.resolve()/'traces')
        case = cases[row['instance_id'],row['variant']]
        assert read(directory / 'case.json')==case
        assert (directory / 'prompt.md').read_text()==case['prompt']+'\n'
        assert (directory / 'expected.txt').read_text()==case['answer']+'\n'
        if row['outcome'] in {'correct','wrong','format'}:
            scored = score_result(case,(directory/'answer.txt').read_text(),row['finish_reason'])
            assert scored['passed']==row['passed']
            assert scored.get('format_ok')==row.get('format_ok')
        replies = sorted(directory.glob('response-*.json'))
        requests = sorted(directory.glob('request-*.json'))
        assert len(replies)==len(requests)==row['request_count']
        remaining = 32768
        for number,(request_path,response_path) in enumerate(zip(requests,replies,strict=True),1):
            assert request_path.name==f'request-{number:02}.json'
            assert response_path.name==f'response-{number:02}.json'
            request, response = read(request_path),read(response_path)
            assert request['messages'][0]=={'role':'user','content':case['prompt']}
            assert request['model']==config['parameters_no_tools']['model']
            assert request['max_tokens']==remaining and 0<remaining<=32768
            assert request['temperature']==0.7 and request['seed']==20261029
            assert request['skip_special_tokens'] is False
            assert request['stream'] is True and request['stream_options']['include_usage'] is True
            assert request['chat_template_kwargs']['enable_thinking'] is True
            assert 'tools' not in request and 'tool_choice' not in request
            tools = request['chat_template_kwargs'].get('xml_tools')
            if row['tools_enabled'] and number==1:
                assert tools==config['tools_condition']['tools']
            if tools:
                assert row['tools_enabled'] and tools==config['tools_condition']['tools']
                assert request['stop']==['</tool_call>'] and request['include_stop_str_in_output'] is True
            if not row['tools_enabled']:
                assert not tools
            reasoning_replays += sum(bool(m.get('reasoning_content')) for m in request['messages'] if m['role']=='assistant')
            tool_replays += sum(m['role']=='tool' for m in request['messages'])
            usage = (response.get('usage') or {}).get('completion_tokens',0)
            assert isinstance(usage,int) and usage>=0
            remaining -= usage
            native_failures += response.get('error_stage')=='native_protocol'
            network_failures += response.get('error_stage')=='transport'
            request_count += 1
        assert row['completion_tokens']==32768-remaining<=32768
        assert len(list(directory.glob('tool-*.json')))==row['tool_calls']
        for path in directory.glob('tool-*.json'):
            tool = read(path)
            result = tool['result']
            if result.get('error'):
                rejected += 1
            else:
                assert tool['call']['function']['name']=='python'
                if result.get('exit_code') or result.get('timed_out'):
                    programs_failed += 1
                else:
                    successes += 1
    assert successes>0 and reasoning_replays>0 and tool_replays>0
    if output:
        exported = [r for r in read(output/'results.json') if r['model']==MODEL_KEY]
        assert len(exported)==3260
        lookup = {(r['instance_id'],r['condition']):r for r in rows}
        for row in exported:
            source = lookup[row['instance_id'],row['condition']]
            assert row['passed']==source['passed'] and row['outcome']==source['outcome']
            assert row['expected_answer']==source['expected_answer']
            assert row['cost_usd'] is None and row['reasoning_tokens'] is None
        old = read(raw/'prior-full-results.json')
        assert [r for r in read(output/'results.json') if r['model']!=MODEL_KEY]==old
    result = {'complete':True,'attempts':3260,'planned_per_condition':PLANNED,
              'source_hashes_verified':True,'all_prompts_answers_and_scores_verified':True,
              'no_pilot_response_reuse':True,'scored_retries':0,'exported_predictions_verified':bool(output),
              'token_budget_verified':True,'sandbox_verified':True,'api_requests':request_count,
              'provenance':config['provenance'],'successful_python_executions':successes,
              'python_program_failures':programs_failed,'rejected_calls':rejected,
              'native_protocol_failures':native_failures,'transport_failures':network_failures,
              'reasoning_replayed':True,'tool_results_delivered':True,'cost_usd':None}
    result['operational_resumes'] = public_config(raw)['operational_resumes']
    save(raw/'audit.json',result)
    if output:
        save(output/'snowball-v01-full-audit.json',result)
    return result

if __name__ == '__main__':
    import argparse
    from .run import OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,default=RAW)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args = parser.parse_args()
    result = audit(args.raw_root.resolve(),args.output.resolve())
    print(json.dumps({'complete':result['complete'],'attempts':result['attempts']}))

