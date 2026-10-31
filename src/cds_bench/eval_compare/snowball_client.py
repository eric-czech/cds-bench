"""Snowball's native Delphi protocol over its existing streaming chat API.

The server lacks automatic tool parsing. Supply the exact eval function through
the checkpoint's xml_tools template argument and parse only its documented tags.
No prompt, argument, generated Python or final-answer repair is performed.
"""
import asyncio
import copy
import json
import re
import time
from pathlib import Path

from cds_bench.eval_luna.io import save
from cds_bench.eval_luna.score import no_duplicate_keys

ACCESS_PATH = None


def configure_access(path):
    global ACCESS_PATH
    ACCESS_PATH = Path(path)
REQUEST_TIMEOUT = 1800
THINK_START, THINK_END = '<|start_think|>', '<|end_think|>'
CALL_START, CALL_END = '<tool_call>', '</tool_call>'


class RequestFailure(Exception):
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


def transport_body(body):
    payload = copy.deepcopy(body)
    tools = payload.pop('tools', [])
    choice = payload.pop('tool_choice', 'none')
    assert choice in {'auto', 'none'}
    kwargs = payload.setdefault('chat_template_kwargs', {})
    kwargs['enable_thinking'] = True
    if choice == 'auto' and tools:
        kwargs['xml_tools'] = tools
        payload['stop'] = [CALL_END]
        payload['include_stop_str_in_output'] = True
    payload['skip_special_tokens'] = False
    return payload


def parse_message(raw, number):
    text = raw
    reasoning = ''
    if THINK_START in text:
        prefix, text = text.split(THINK_START, 1)
        if prefix.strip():
            raise ValueError('Unexpected content before native thinking block')
        if THINK_END not in text:
            return {'role': 'assistant', 'content': None, 'reasoning_content': text, 'tool_calls': []}
        reasoning, text = text.split(THINK_END, 1)
    elif THINK_END in text:
        reasoning, text = text.split(THINK_END, 1)
    # Only remove the checkpoint's exact terminal EOS token, if returned.
    if text.endswith('<|eot_id|>'):
        text = text[:-len('<|eot_id|>')]
    calls = []
    def call(match):
        value = json.loads(match.group(1), object_pairs_hook=no_duplicate_keys)
        if not isinstance(value, dict) or set(value) != {'name', 'arguments'}:
            raise ValueError('Native tool call requires name and arguments')
        if not isinstance(value['name'], str) or not isinstance(value['arguments'], dict):
            raise ValueError('Invalid native tool-call types')
        calls.append({'id': f'call_{number}_{len(calls)}', 'type': 'function',
                      'function': {'name': value['name'], 'arguments': json.dumps(value['arguments'])}})
        return ''
    text = re.sub(re.escape(CALL_START) + r'(.*?)' + re.escape(CALL_END), call, text, flags=re.DOTALL)
    if CALL_START in text or CALL_END in text:
        raise ValueError('Incomplete native tool-call block')
    return {'role': 'assistant', 'content': text.strip() or None,
            'reasoning_content': reasoning or None, 'tool_calls': calls}


async def api_call(client, directory, number, body, key):
    if ACCESS_PATH is None:
        raise RuntimeError('A private access file must be configured')
    access = json.loads(ACCESS_PATH.read_text())
    base = access['base_url'].rstrip('/')
    def clean(value):
        return str(value).replace(base, '[PRIVATE_API]').replace(access['ui_url'].rstrip('/'), '[PRIVATE_API]')
    payload = transport_body(body)
    save(directory / f'request-{number:02}.json', payload)
    started = time.monotonic()
    meta = {'request_number': number, 'request_max_tokens': payload['max_tokens']}
    content, structured_reasoning = [], []
    error = None
    terminal = False
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT):
            async with client.stream('POST', base + '/chat/completions', json=payload) as stream:
                meta['http_status'] = stream.status_code
                if stream.status_code != 200:
                    raise RuntimeError(f'HTTP {stream.status_code}: ' + clean((await stream.aread()).decode()))
                with (directory / f'response-{number:02}.sse').open('w') as log:
                    async for line in stream.aiter_lines():
                        line = clean(line)
                        log.write(line + '\n')
                        log.flush()
                        if not line.startswith('data:'):
                            continue
                        value = line[5:].strip()
                        if value == '[DONE]':
                            terminal = True
                            break
                        if not value:
                            continue
                        event = json.loads(value)
                        if event.get('error'):
                            raise RuntimeError(str(event['error']))
                        for field in ['id', 'model', 'usage']:
                            if event.get(field) is not None:
                                meta[field] = event[field]
                        for choice in event.get('choices', []):
                            if choice.get('finish_reason'):
                                meta['finish_reason'] = choice['finish_reason']
                            delta = choice.get('delta') or {}
                            content.append(delta.get('content') or '')
                            structured_reasoning.append(delta.get('reasoning_content') or '')
                            if delta.get('tool_calls'):
                                raise RuntimeError('Unexpected structured calls from native-protocol route')
        if not terminal or not meta.get('finish_reason') or not meta.get('usage'):
            raise RuntimeError('Stream ended without DONE, finish reason or usage')
        if meta.get('model') != access['model_id']:
            raise RuntimeError('Unexpected served model ID')
    except TimeoutError:
        meta['error_stage'] = 'transport'
        meta['request_timeout'] = True
        error = '1800-second API request deadline exceeded'
    except Exception as exc:
        meta['error_stage'] = 'transport'
        error = clean(f'{type(exc).__name__}: {exc}')
    raw = ''.join(content)
    message = {'role': 'assistant', 'content': None, 'tool_calls': []}
    try:
        message = parse_message(raw, number)
        if structured_reasoning:
            message['reasoning_content'] = ''.join(structured_reasoning) or message.get('reasoning_content')
        if message['tool_calls']:
            meta['server_finish_reason'] = meta.get('finish_reason')
            meta['finish_reason'] = 'tool_calls' if meta.get('finish_reason') != 'length' else 'length'
    except Exception as exc:
        if meta.get('finish_reason') != 'length' and not error:
            meta['error_stage'] = 'native_protocol'
            error = clean(f'{type(exc).__name__}: {exc}')
    meta.update(elapsed_seconds=time.monotonic()-started, raw_content=raw,
                reasoning_summary=message.get('reasoning_content') or '', message=message,
                native_protocol=True, error=error)
    save(directory / f'response-{number:02}.json', meta)
    if error:
        raise RequestFailure(error, meta)
    return meta
