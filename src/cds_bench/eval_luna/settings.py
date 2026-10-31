"""The same request and tool settings validated by the local pilot."""

PARAMETERS = {
    "model": "openai/gpt-6-luna",
    "max_tokens": 32768,
    "seed": 20261029,
    "reasoning": {"effort": "high", "exclude": False},
    "tool_choice": "none",
    "tools": [],
    "plugins": [
        {"id": "web", "enabled": False},
        {"id": "response-healing", "enabled": False},
        {"id": "file-parser", "enabled": False},
    ],
    "provider": {
        "only": ["openai"],
        "ignore": ["openai/flex", "openai/fast"],
        "allow_fallbacks": False,
        "require_parameters": True,
    },
    "stream": True,
    "stream_options": {"include_usage": True},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute Python 3.12.9 code with the standard library and NumPy 2.5.4. Biopython is not installed. Networking and subprocesses are disabled; only runtime files and this example's working directory are readable. Use print() to return results. Each call starts a fresh interpreter; files in the working directory persist within this example. Limits: 30 seconds per call, 16384 characters each of stdout/stderr, and at most 8 calls per example. Returns stdout, stderr, exit_code, timed_out, and output_truncated.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute."}
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    }
]

REQUEST_TIMEOUT = 300
TOKEN_BUDGET = 32768
MAX_TOOL_CALLS = 8
CONCURRENCY = 10
