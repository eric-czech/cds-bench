"""Audit and publish the saved internal-API Snowball pilot, without inference."""

import hashlib
import json
import statistics
from pathlib import Path

from cds_bench.eval_luna.io import save
from cds_bench.eval_luna.report import aggregate, outcome, public_record
from cds_bench.eval_luna.score import score_result

ROOT = Path(__file__).resolve().parents[3]
MODELS = {
    'snowball': {'name':'Snowball RC (2026.09.21)', 'source':'scratch/snowball-pilot-20261031-02',
                 'budget':16384, 'audit_file':'snowball-audit.json'},
    'snowball-v01': {'name':'Snowball v0.1', 'source':'scratch/snowball-v01-pilot-20261031-01',
                     'budget':32768, 'audit_file':'snowball-v01-audit.json'},
}


def read(path):
    return json.loads(path.read_text())


def build(output, cases, format_diagnostic, model_key='snowball'):
    """Return one optional model, requiring complete and independently rescored data."""
    definition = MODELS[model_key]
    source = ROOT / definition['source']
    budget = definition['budget']
    if not (source / 'summary.json').exists():
        return None
    status = read(source / 'summary.json')
    if status['state'] != 'complete':
        return None
    config = read(source / 'config.json')
    assert config['aggregate_completion_token_budget'] == budget
    assert config['request_timeout_seconds'] == 1800
    preflight = read(source / 'preflight-result.json')
    assert preflight['tool_calls'] and not preflight['tool_failures']
    if not preflight['passed']:
        verification = read(source / 'tool-round-trip-verification.json')
        assert verification['passed'] and verification['tool_result_delivered']
        protocol = Path(preflight['trace_directory'])
        tool = read(protocol / 'tool-01.json')
        expected = read(protocol / 'case.json')['answer']
        assert tool['result']['exit_code'] == 0 and tool['result']['stdout'].strip() == expected
        assert expected in (protocol / 'answer.txt').read_text()
        messages = read(protocol / 'request-02.json')['messages']
        assert any(m['role']=='tool' and m['tool_call_id']==tool['tool_call_id'] and
                   json.loads(m['content'])==tool['result'] for m in messages)
    for path, digest in config['source_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    originals = read(source / 'results.json')
    assert len(originals) == len({(r['instance_id'], r['condition']) for r in originals}) == 116
    rows, durations = [], []
    successful_executions = tool_program_errors = invalid_calls = 0
    protocol_failures = http_failures = 0
    transport_failures = 0
    reasoning_replays = tool_result_replays = 0
    for original in originals:
        directory = Path(original['trace_directory'])
        case = read(directory / 'case.json')
        assert case == cases[original['instance_id'], original['variant']]
        text = (directory / 'answer.txt').read_text()
        if not original.get('api_error'):
            assert score_result(case, text, original['finish_reason'])['passed'] == original['passed']
        replies = [read(p) for p in sorted(directory.glob('response-*.json'))]
        assert len(replies) == original['request_count']
        durations.extend(r['elapsed_seconds'] for r in replies)
        assert original['completion_tokens'] <= budget
        if not original['tools_enabled']:
            assert original['tool_calls'] == 0
        for response in replies:
            protocol_failures += (response.get('error') or '').startswith(('ValueError:', 'JSONDecodeError:'))
            transport_failures += (response.get('error') or '').startswith('RemoteProtocolError:')
            http_failures += response.get('http_status') not in (None, 200)
            request = read(directory / f"request-{response['request_number']:02}.json")
            reasoning_replays += sum(bool(m.get('reasoning_content')) for m in request['messages'] if m['role'] == 'assistant')
            tool_result_replays += sum(m['role'] == 'tool' for m in request['messages'])
            assert request['chat_template_kwargs']['enable_thinking'] is True
            assert request['temperature'] == 0.7 and request['seed'] == 20261029
            assert 'tools' not in request and 'tool_choice' not in request
            if original['tools_enabled'] and request['chat_template_kwargs'].get('xml_tools'):
                assert request['chat_template_kwargs']['xml_tools'] == config['tools']
            if not original['tools_enabled']:
                assert 'xml_tools' not in request['chat_template_kwargs']
        for p in directory.glob('tool-*.json'):
            result = read(p)['result']
            if result.get('error'):
                invalid_calls += 1
            elif result.get('exit_code') or result.get('timed_out'):
                tool_program_errors += 1
            else:
                successful_executions += 1
        row = dict(original, expected_answer=case['answer'], output_format=case['output_format'],
                   cost_usd=0, billing_complete=False)
        row['outcome'] = outcome(row)
        exported = {k:v for k,v in public_record(row).items() if k in row}
        if exported.get('error'):
            exported['error'] = {'error':'API or native tool-protocol failure', 'timeout':'API deadline exceeded',
                                 'format':'Final answer does not satisfy the required format',
                                 'token_cap':'Generated-token budget exhausted', 'wrong':'Incorrect value'}[row['outcome']]
        exported['cost_usd'] = None
        exported['reasoning_tokens'] = None
        rows.append(dict(exported, model=model_key, format_diagnostic_exact=format_diagnostic(case, row, text)))
    def metrics(subset):
        result = aggregate([dict(r, cost_usd=0, reasoning_tokens=0) for r in subset])
        result['cost_usd'] = None
        result['reasoning_tokens'] = None
        return result
    totals = metrics(rows)
    totals['median_api_request_seconds'] = statistics.median(durations)
    totals['format_diagnostic_additional_exact'] = sum(r['format_diagnostic_exact'] for r in rows)
    conditions = {c:metrics([r for r in rows if r['condition'] == c]) for c in status['conditions']}
    assert [v['attempted'] for v in conditions.values()] == [30,28,30,28]
    assert successful_executions > 0 and reasoning_replays > 0 and tool_result_replays > 0
    settings = {
        'name':definition['name'], 'model':config['provenance']['repo'], 'full_name':config['full_name'],
        'provenance':config['provenance'], 'provider':{'only':['Marin internal API'], 'allow_fallbacks':False},
        'reasoning':config['reasoning'],
        'parameters_no_tools':{**config['parameters_no_tools'], 'model':'[private serving model ID]',
                               'skip_special_tokens':False, 'chat_template_kwargs':{'enable_thinking':True}},
        'tools':config['tools'], 'native_request_fields':config['native_request_fields'],
        'transport':config['transport'], 'backend':'Marin/vLLM BF16 on H100x8, data-parallel size 8 with expert parallelism',
        'aggregate_completion_token_budget':budget, 'request_timeout_seconds':1800,
        'concurrency':config['concurrency'], 'schedule':config['schedule'],
        'dataset_revision':config['dataset_revision'], 'state':'complete', 'stop_reason':None,
        'cost_note':config['cost_note'],
        'token_count_note':'Completion tokens include native thinking; reasoning-only counts are not separately reported.',
        'tool_audit':{'preflight_passed':True,'successful_executions':successful_executions,
                      'program_errors':tool_program_errors,'invalid_calls':invalid_calls,
                      'native_protocol_failures':protocol_failures,'http_failures':http_failures,
                      'native_reasoning_replayed':True,'tool_results_returned_to_model':True,
                      'call_limit_failures':sum(r.get('error') == 'Unexpected tool calls after disabling tool_choice' for r in originals)},
    }
    if not preflight['passed']:
        settings['preflight_note'] = verification['note']
        settings['tool_audit']['strict_preflight_answer_passed'] = False
    if transport_failures:
        settings['tool_audit']['incomplete_streams'] = transport_failures
    save(output / definition['audit_file'], {'passed':True, 'attempts':116,
        'prompts_answers_and_scoring_verified':True, 'tools_and_thinking_verified':True,
        'scored_retries':0, 'source_hashes_verified':True, 'raw_traces_excluded':True,
        'provenance':config['provenance'], 'tool_execution':settings['tool_audit']})
    return rows, totals, conditions, settings
