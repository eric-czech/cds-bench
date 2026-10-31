"""Deterministic final-answer scoring; no reference solvers are imported."""

import json
import re
from decimal import Decimal


def no_duplicate_keys(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError(f"duplicate object key: {key}")
        out[key] = value
    return out


def parse_json(text):
    return json.loads(
        text,
        object_pairs_hook=no_duplicate_keys,
        parse_float=Decimal,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("non-finite JSON number")),
    )


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) == type(b) and a == b
    if isinstance(a, (int, Decimal)) and isinstance(b, (int, Decimal)):
        return a == b
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(u, v) for u, v in zip(a, b))
    return a == b


def score(record, content):
    markers = list(re.finditer(r"^Final answer:[ \t]*$", content, re.MULTILINE))
    if not markers:
        return {"passed": False, "format_ok": False, "error": "Missing exact final-answer marker"}
    final = content[markers[-1].end() :].strip()
    try:
        if record["output_format"] == "json":
            got = parse_json(final)
            want = parse_json(record["answer"])
            return {
                "passed": same(got, want),
                "format_ok": True,
                "final_answer": final,
                "parsed_answer": got,
            }
        try:
            want = parse_json(record["answer"])
        except (ValueError, TypeError):
            want = record["answer"]
        if isinstance(want, (int, Decimal)) and not isinstance(want, bool):
            got = Decimal(final)
            return {
                "passed": got.is_finite() and got == Decimal(want),
                "format_ok": True,
                "final_answer": final,
            }
        got = "".join(final.split())
        return {
            "passed": got == "".join(record["answer"].split()),
            "format_ok": True,
            "final_answer": final,
        }
    except (ValueError, TypeError, ArithmeticError) as error:
        return {"passed": False, "format_ok": False, "error": str(error), "final_answer": final}


def score_result(record, text, finish):
    result = score(record, text)
    diagnostic = dict(result)
    match = re.search(
        r"(?m)^Final answer:[ \t]*\n\s*```(?:json)?\s*\n(.*?)\n```\s*$", text, re.DOTALL
    )
    if match and record["output_format"] == "json":
        diagnostic = score(record, "Final answer:\n" + match[1])
    result["content_passed_after_outer_json_fence_removal"] = diagnostic.get("passed", False)
    if finish == "length":
        result.update(passed=False, truncated=True)
    return result
