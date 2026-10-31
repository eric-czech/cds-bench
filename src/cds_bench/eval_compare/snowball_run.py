"""Full Snowball v0.1 evaluation through the existing private Marin API."""

import argparse
import asyncio
import copy
import fcntl
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import httpx

from cds_bench.eval_luna import sandbox
from cds_bench.eval_luna.io import save
from cds_bench.eval_luna.run import now, status, validate_sandbox
from cds_bench.eval_luna.settings import TOOLS

from . import snowball_client
from .run import CONDITIONS, HASHES, LUNA_RAW, PLANNED, REMOVED_HARD, REVISION, ROOT, dataset, verify_luna

RAW = ROOT / 'scratch/snowball-v01-full-20261031-01'
PILOT = ROOT / 'scratch/snowball-v01-pilot-20261031-01'
MODEL_KEY = 'snowball-v01'
MODEL_REPO = 'open-athena/Marin-Snowball-67B-A2B-v0.1'
MODEL_REVISION = 'adc0078a19a4383bc998fcdb43867b3360baf5f3'
CONCURRENCY = 60
runtime = importlib.import_module('cds_bench.eval_luna.trial')


def read(path):
    return json.loads(path.read_text())


def configure(access_file):
    access = read(access_file)
    assert access['source_model'] == MODEL_REPO and access['source_revision'] == MODEL_REVISION
    pilot = read(PILOT / 'config.json')
    parameters = copy.deepcopy(pilot['parameters_no_tools'])
    assert parameters['model'] == access['model_id']
    assert parameters['max_tokens'] == 32768 and parameters['temperature'] == 0.7
    assert parameters['seed'] == 20261029 and pilot['tools'] == TOOLS
    runtime.PARAMETERS = parameters
    runtime.TOKEN_BUDGET = 32768
    runtime.RequestFailure = snowball_client.RequestFailure
    runtime.api_call = snowball_client.api_call
    snowball_client.configure_access(access_file)
    return access, parameters, pilot


def prepare(raw, access_file):
    raw.mkdir(parents=True, exist_ok=True)
    access, parameters, pilot = configure(access_file)
    rows = dataset()
    files = [Path(__file__), Path(__file__).with_name('snowball_client.py')]
    files += [ROOT / 'src/cds_bench/eval_luna' / n for n in
              ['trial.py', 'sandbox.py', 'score.py', 'settings.py', 'io.py', 'run.py']]
    config = {
        'run_id':raw.name, 'model_key':MODEL_KEY, 'name':'Snowball v0.1',
        'full_name':'Marin-Snowball-67B-A2B-v0.1',
        'provenance':{'repo':MODEL_REPO, 'revision':MODEL_REVISION,
                      'url':f'https://huggingface.co/{MODEL_REPO}/tree/{MODEL_REVISION}'},
        'total_parameters':67000000000, 'active_parameters_per_token':2000000000,
        'planned_attempts':3260, 'planned_per_condition':PLANNED, 'condition_order':list(PLANNED),
        'parameters_no_tools':parameters, 'tools_condition':{'tools':TOOLS, 'tool_choice':'auto'},
        'native_request_fields':pilot['native_request_fields'], 'transport':pilot['transport'],
        'reasoning':pilot['reasoning'], 'request_timeout_seconds':1800,
        'aggregate_completion_token_budget':32768, 'max_tool_calls':8, 'concurrency':CONCURRENCY,
        'dataset':{'repo':'open-athena/cds-bench', 'revision':REVISION, 'parquet_sha256':HASHES,
                   'rows_per_split':{v:len(rs) for v,rs in rows.items()},
                   'removed_hard_tasks':sorted(REMOVED_HARD)},
        'luna_comparison':verify_luna(rows),
        'execution_environment':read(LUNA_RAW / 'config.json')['execution_environment'],
        'sampling':'One fresh attempt per case/condition; pilot responses are not reused. No scored inference retries, prompt changes, answer repair or correctness feedback.',
        'settings_difference':'Native Marin thinking/tool transport, explicit temperature 0.7 and seed 20261029, concurrency 60 and a 1800-second API wall deadline. Same 32768 aggregate generated tokens, exact Python function, sandbox and scorer as the validated v0.1 pilot.',
        'pilot_config_sha256':hashlib.sha256((PILOT / 'config.json').read_bytes()).hexdigest(),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'inference_source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'cost_note':'No per-request monetary price is exposed. Token usage is recorded; server allocation cost is not attributed to requests.',
        'access_file':str(access_file.resolve()),
    }
    path = raw / 'config.json'
    if path.exists():
        old = read(path)
        assert {k:v for k,v in old.items() if k!='source_commit'} == {k:v for k,v in config.items() if k!='source_commit'}, 'Frozen run configuration changed'
        assert read(raw / 'dataset.json') == rows
    else:
        save(path, config)
        save(raw / 'dataset.json', rows)
    sandbox.WORK_ROOT = ROOT / 'build/luna-tool-work' / raw.name
    return rows


def saved_results(raw):
    result = {}
    for path in (raw / 'traces').glob('*/*/result.json'):
        row = read(path)
        key = row['instance_id'], row['condition']
        assert key not in result
        result[key] = row
    for path in (raw / 'traces').glob('*/*/request-01.json'):
        if not (path.parent / 'result.json').exists():
            raise RuntimeError(f'Interrupted request needs review; no automatic retry: {path.parent.name}')
    return result


async def preflight(client, raw):
    case = {'id':'protocol/01/easy', 'instance_id':'protocol/01', 'task_id':'protocol',
            'variant':'easy', 'output_format':'scalar',
            'prompt':'Use the python tool to calculate 123456789 * 987654321. You must execute Python before answering. After any explanation, write Final answer: on its own line, then the integer result on the next line.',
            'answer':'121932631112635269'}
    result = await runtime.trial(client, asyncio.Semaphore(1), 1, case, True,
                                 'snowball-no-api-key', raw / 'preflight')
    save(raw / 'preflight-result.json', result)
    assert result['tool_calls'] and result['tool_failures']==0
    directory = Path(result['trace_directory'])
    tool = read(directory / 'tool-01.json')
    assert tool['result']['exit_code']==0 and tool['result']['stdout'].strip()==case['answer']
    assert case['answer'] in (directory / 'answer.txt').read_text()
    messages = read(directory / 'request-02.json')['messages']
    assert any(m['role']=='tool' and json.loads(m['content'])==tool['result'] for m in messages)
    save(raw / 'tool-round-trip-verification.json', {'passed':True,
         'successful_python_execution':True, 'tool_result_delivered':True,
         'strict_preflight_answer_passed':result['passed'],
         'note':'Tool health is verified independently of answer formatting. The unscored preflight answer and its strict score are retained without repair or retry.'})


async def evaluate(raw, rows):
    existing = saved_results(raw)
    save(raw / 'results.json', list(existing.values()))
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800,connect=15), trust_env=False,
        limits=httpx.Limits(max_connections=CONCURRENCY,max_keepalive_connections=CONCURRENCY)) as client:
        status(raw,state='preflight',planned=3260,completed=len(existing))
        await preflight(client, raw)
        for variant, tools in CONDITIONS:
            condition = variant + ('-tools' if tools else '-no-tools')
            count = sum(c==condition for _,c in existing)
            pending = [(i,case) for i,case in enumerate(rows[variant],1) if (case['instance_id'],condition) not in existing]
            status(raw,state='running',phase=condition,planned=3260,completed=len(existing),
                   phase_completed=count,phase_planned=PLANNED[condition],error=None)
            stop = asyncio.Event()
            gate = asyncio.Semaphore(CONCURRENCY)
            async def attempt(index,case):
                async with gate:
                    if stop.is_set():
                        return None
                    return await runtime.trial(client,asyncio.Semaphore(1),index,case,tools,
                                               'snowball-no-api-key',raw)
            errors = 0
            for completed in asyncio.as_completed([attempt(i,c) for i,c in pending]):
                row = await completed
                if row is None:
                    continue
                key = row['instance_id'], condition
                assert key not in existing
                existing[key] = row
                count += 1
                replies = sorted(Path(row['trace_directory']).glob('response-*.json'))
                last = read(replies[-1]) if replies else {}
                infrastructure_error = row.get('api_error') and not row.get('request_timeout') and last.get('error_stage')=='transport'
                errors = errors+1 if infrastructure_error else 0
                if errors>=5:
                    stop.set()
                save(raw / 'results.json',list(existing.values()))
                status(raw,completed=len(existing),phase_completed=count)
            if stop.is_set():
                raise RuntimeError('Stopped after five consecutive transport/API failures; completed attempts retained, unissued cases may resume after review')
            assert count == PLANNED[condition]
            status(raw,last_completed_phase=condition)
    assert len(existing)==3260
    status(raw,state='complete',phase='complete',completed=3260,finished_at=now())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,default=RAW)
    parser.add_argument('--access-file',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    raw=args.raw_root.resolve()
    raw.mkdir(parents=True,exist_ok=True)
    with (raw/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            rows=prepare(raw,args.access_file)
            status(raw,state='prepared',planned=3260,completed=len(saved_results(raw)),error=None)
            if args.prepare:
                return
            validate_sandbox(raw)
            asyncio.run(evaluate(raw,rows))
        except BaseException as exc:
            status(raw,state='failed',error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':
    main()
