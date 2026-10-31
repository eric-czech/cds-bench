"""Compact GitHub tables and detailed HTML report tables."""

import html
from collections import defaultdict
from statistics import mean


def models_by_task_accuracy(names, rows):
    """Use one pooled, equally weighted task mean for every model display."""
    tasks = defaultdict(list)
    for row in rows:
        if row['model'] in names:
            tasks[row['model'], row['task_id']].append(row['passed'])
    scores = {model: mean(mean(values) for (m, _), values in tasks.items() if m == model)
              if any(m == model for m, _ in tasks) else None for model in names}
    order = sorted(names, key=lambda model: (-(scores[model] if scores[model] is not None else -1), names[model]))
    return {model: names[model] for model in order}, scores


HEADERS = [
    "Model / condition", "Correct / finished", "Accuracy", "Median / example",
    "Recorded cost", "Token budget", "Timeouts", "Token caps", "Format", "API/protocol",
]
PR_HEADERS = ["Model", "Condition", "Correct", "Accuracy", "Median", "Cost", "Truncated"]


def cells(label, metric, budget):
    return [
        label,
        f"{metric['correct']}/{metric['attempted']}",
        f"{100 * metric['accuracy']:.1f}%" if metric["attempted"] else "n/a",
        f"{metric['median_seconds']:.1f}s" if metric["attempted"] else "n/a",
        f"${metric['cost_usd']:.5f}" if metric['cost_usd'] is not None else "Unavailable", f"{budget:,}",
        str(metric["timeout"]), str(metric["token_cap"]), str(metric["format"]), str(metric["error"]),
    ]


def markdown_table(headers, rows):
    # GitHub strips table CSS. Nonbreaking spaces keep short labels together.
    def nowrap(value):
        return html.escape(str(value)).replace(" ", "&nbsp;")

    lines = ["| " + " | ".join(headers) + " |", "|---|---|" + "---:|" * (len(headers) - 2)]
    for row in rows:
        lines.append("| " + " | ".join(nowrap(value) for value in row) + " |")
    return "\n".join(lines)


def markdown_results(rows):
    return markdown_table(PR_HEADERS, [
        [model, condition, *cells("", metric, budget)[1:5], metric["token_cap"]]
        for (model, condition), metric, budget in rows
    ])


def html_results(rows):
    head = "".join(f"<th>{html.escape(h)}</th>" for h in HEADERS)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells(*row)) + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def html_grouped_table(headers, groups):
    """Render two row-label columns, with the first spanning each group."""
    escape = lambda value: html.escape(str(value))
    head = "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    parts = [f'<div class="scroll"><table class="indexed-table"><thead><tr>{head}</tr></thead>']
    for name, rows in groups:
        parts.append('<tbody>')
        for index, row in enumerate(rows):
            parts.append('<tr>')
            if index == 0:
                parts.append(f'<th class="index-group" scope="rowgroup" rowspan="{len(rows)}">{escape(name)}</th>')
            parts.append(f'<th class="index-item" scope="row">{escape(row[0])}</th>')
            parts.extend(f'<td>{escape(value)}</td>' for value in row[1:])
            parts.append('</tr>')
        parts.append('</tbody>')
    parts.append('</table></div>')
    return ''.join(parts)
