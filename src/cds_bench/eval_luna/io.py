"""JSON artifacts and authorized local API-key lookup."""

import json
import os
import re
import shlex
from pathlib import Path


def api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    # Read only the authorized literal assignment; do not execute the profile or
    # save/print its contents, which may contain unrelated credentials.
    for name in [".zprofile", ".zshrc", ".zshenv", ".zsh_profile"]:
        path = Path.home() / name
        if not path.is_file():
            continue
        for value in re.findall(
            r"^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*?)\s*$", path.read_text(), re.MULTILINE
        ):
            tokens = shlex.split(value, comments=True)
            if tokens and tokens[0].startswith("sk-"):
                return tokens[0]
    raise RuntimeError("No literal OPENROUTER_API_KEY assignment found in the zsh profile.")


def save(path, value):
    """Replace a JSON artifact atomically so progress readers never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)
