"""Recover failed-request bookkeeping from immutable saved response metadata."""

import json
from pathlib import Path

from cds_bench.eval_luna.report import outcome, records as original_records


def records(raw):
    rows = original_records(raw)
    for row in rows:
        directory = Path(row["trace_directory"])
        paths = sorted(directory.glob("response-*.json"))
        if len(paths) == row["request_count"]:
            continue
        # The Qwen client raises its own RequestFailure class. The legacy trial
        # catches it at the outer handler, preserving failure and the response
        # file but omitting the failed response from its bookkeeping list.
        assert len(paths) == row["request_count"] + 1 and not row["passed"]
        replies = [json.loads(p.read_text()) for p in paths]
        assert replies[-1].get("error")
        usage = [r.get("usage") or {} for r in replies]
        row.update(
            api_error=True,
            request_timeout=any(r.get("request_timeout") for r in replies),
            request_count=len(replies),
            response_ids=[r["id"] for r in replies if r.get("id")],
            prompt_tokens=sum(u.get("prompt_tokens") or 0 for u in usage),
            completion_tokens=sum(u.get("completion_tokens") or 0 for u in usage),
            reasoning_tokens=sum(
                (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                for u in usage
            ),
            reported_cost_usd=sum(u.get("cost") or 0 for u in usage),
            request_metadata_recovered=True,
        )
        charges = []
        for response in replies:
            path = raw / "billing" / (str(response.get("id")) + ".json")
            charge = json.loads(path.read_text()).get("data", {}).get("total_cost") if path.exists() else None
            charges.append(charge)
        row["billing_complete"] = all(c is not None for c in charges)
        row["cost_usd"] = sum(
            charge if charge is not None else (reply.get("usage") or {}).get("cost") or 0
            for charge, reply in zip(charges, replies)
        )
        row["outcome"] = outcome(row)
    return rows
