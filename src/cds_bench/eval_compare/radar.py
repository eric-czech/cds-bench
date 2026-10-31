"""Task-level radar panels from saved metrics, with a shared linked key."""

import html
import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from statistics import mean

from cds_bench.catalog import TASKS


FAMILIES = {
    "sequence": ("Sequence operations", "#4477aa"),
    "variation": ("Variant effects", "#ee6677"),
    "measure": ("Sequence statistics", "#228833"),
    "protein": ("Protein operations", "#b49b30"),
    "rna": ("RNA operations", "#55aacc"),
    "redesign": ("Synonymous design", "#aa3377"),
    "construct": ("Construct design", "#dd7733"),
    "assay": ("Assay design", "#009988"),
    "overlap": ("Overlapping frames", "#777777"),
    "composition": ("Composition tasks", "#332288"),
}
SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
VIEWS = {
    "all": "All conditions (pooled)",
    "easy-no-tools": "Easy, no tools",
    "hard-no-tools": "Hard, no tools",
    "easy-tools": "Easy, Python",
    "hard-tools": "Hard, Python",
}

def _model_summary(values):
    values = [p for p in values if p is not None]
    return {"task_count": len(values), "mean_accuracy": mean(values) if values else None}


def _task_metrics(summary, task_id, models, condition):
    metrics = {}
    for model in models:
        saved = summary["by_task"][task_id][model]
        selected = saved.values() if condition == "all" else [saved[condition]]
        correct = sum(r["correct"] for r in selected)
        attempted = sum(r["attempted"] for r in selected)
        metrics[model] = {"correct": correct, "attempted": attempted,
                          "accuracy": correct / attempted if attempted else None}
    return metrics


def prepare(summary, models, condition="all", reference=None):
    assert condition in VIEWS
    if condition != "all":
        data = deepcopy(reference or prepare(summary, models))
        groups = data["groups"]
        for group in groups:
            for task in group["tasks"]:
                task["models"] = _task_metrics(summary, task["task_id"], models, condition)
                scores = [m["accuracy"] for m in task["models"].values() if m["accuracy"] is not None]
                task["mean_accuracy"] = mean(scores) if scores else None
            scores = [t["mean_accuracy"] for t in group["tasks"] if t["mean_accuracy"] is not None]
            group["mean_accuracy"] = mean(scores) if scores else None
        return _view_data(data, models, condition)
    tasks = []
    for task_id, task in TASKS.items():
        metrics = _task_metrics(summary, task_id, models, condition)
        values = [m["accuracy"] for m in metrics.values() if m["accuracy"] is not None]
        tasks.append({"task_id": task_id, "title": task["title"],
                      "family": task.get("family", "composition"), "models": metrics,
                      "mean_accuracy": mean(values) if values else None})
    groups = []
    rank = lambda value: -(value if value is not None else -1)
    for family, (name, color) in FAMILIES.items():
        members = sorted((t for t in tasks if t["family"] == family),
                         key=lambda t: (rank(t["mean_accuracy"]), t["task_id"]))
        if members:
            values = [t["mean_accuracy"] for t in members if t["mean_accuracy"] is not None]
            groups.append({"family": family, "name": name, "color": color,
                           "mean_accuracy": mean(values) if values else None, "tasks": members})
    groups.sort(key=lambda g: (rank(g["mean_accuracy"]), g["family"]))
    for index, group in enumerate(groups):
        group["code"] = chr(ord("A") + index)
        for number, task in enumerate(group["tasks"], 1):
            task["code"] = f'{group["code"]}{number:02}'
    assert sum(len(g["tasks"]) for g in groups) == len(TASKS) == 84
    return _view_data({
        "models": models,
        "ordering": "Fixed from the pooled view: descending mean task accuracy across models, first by group then within group; ties use family or task ID.",
        "group_average": "Unweighted mean of task means within each group.",
        "groups": groups,
    }, models, condition)


def _view_data(data, models, condition):
    tasks = [t for g in data["groups"] for t in g["tasks"]]
    data.update({
        "condition": condition, "condition_label": VIEWS[condition],
        "available_views": VIEWS, "ordering_reference": "all",
        "score": "Correct / attempted for each task; all failures count. " + (
            "All retained conditions are pooled within each task." if condition == "all" else "Only the selected condition contributes."),
        "model_summaries": {m: _model_summary(t["models"][m]["accuracy"] for t in tasks) for m in models},
    })
    return data


def stem(condition):
    return "task-radar" if condition == "all" else f"task-radar-{condition}"


def plot(output, summary, models):
    reference = prepare(summary, models)
    views = {}
    for condition in VIEWS:
        data = reference if condition == "all" else prepare(summary, models, condition, reference)
        _plot_view(output, data)
        views[condition] = data
    return views


def _plot_view(output, data):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    models = data["models"]
    condition = data["condition"]
    filename = stem(condition)
    groups = data["groups"]
    tasks = [t for group in groups for t in group["tasks"]]
    gap = 1.2
    unit = 2 * math.pi / (len(tasks) + gap * len(groups))
    cursor, angles = 0, []
    for group in groups:
        group["angle_start"] = cursor * unit
        for task in group["tasks"]:
            task["angle"] = (cursor + 0.5) * unit
            angles.append(task["angle"])
            cursor += 1
        group["angle_end"] = cursor * unit
        cursor += gap

    with mpl.rc_context({"svg.fonttype": "none", "svg.hashsalt": f"cds-bench-task-radar-{condition}"}):
        columns = len(models) if len(models)<=3 else 2
        rows = math.ceil(len(models)/columns)
        fig, axes = plt.subplots(rows, columns, figsize=(6*columns, 7.2 if rows==1 else 6.2*rows),
                                 squeeze=False, subplot_kw={"projection": "polar"})
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.18 if rows==1 else 0.11,
                            top=0.88 if rows==1 else 0.91, wspace=0.11, hspace=0.30)
        axes = axes.ravel()
        for unused in axes[len(models):]:
            unused.set_visible(False)
        axes = axes[:len(models)]
        for ax, (model, name) in zip(axes, models.items(), strict=True):
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(0, 1.22)
            ax.set_xticks([])
            ax.set_yticks([0.25, 0.5, 0.75, 1], ["", "", "", ""])
            for radius in [0.25, 0.5, 0.75, 1]:
                ax.text(math.pi / 2, radius - 0.015, f"{radius:.0%}", ha="right", va="center",
                        fontsize=8, color="#64727e", zorder=6,
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.2})
            ax.grid(color="#dce2e7", linewidth=0.6)
            ax.spines["polar"].set_visible(False)
            ax.set_axisbelow(True)
            for group in groups:
                start, end = group["angle_start"], group["angle_end"]
                center, width = (start + end) / 2, end - start
                ax.bar(center, 1, width=width, bottom=0, color=group["color"], alpha=0.035, linewidth=0, zorder=0)
                ax.bar(center, 0.035, width=width, bottom=1.025, color=group["color"], linewidth=0, zorder=3)
                for angle in [start, end]:
                    ax.plot([angle, angle], [0, 1.06], color="#cbd3da", linewidth=0.6, zorder=0)
                for task in group["tasks"]:
                    angle = task["angle"]
                    ax.plot([angle, angle], [0, 1], color="#dce2e7", linewidth=0.3, zorder=0)
                    value = task["models"][model]["accuracy"]
                    if value is not None:
                        ax.scatter([angle], [value], s=10, color=group["color"], edgecolors="white", linewidths=0.25, zorder=5)
                    degrees = math.degrees(angle) % 360
                    rotation = 90 - degrees
                    if 180 < degrees < 360:
                        rotation += 180
                    label = ax.text(angle, 1.135, task["code"], fontsize=9,
                                    color="#334155" if value is not None else "#aab2b9",
                                    rotation=rotation, rotation_mode="anchor", ha="center", va="center")
                    label.set_gid(f'radar-label-{model}-{task["task_id"]}')
                    label.set_url(f'#task-{task["task_id"]}')
            values = [t["models"][model]["accuracy"] for t in tasks]
            contour = [v if v is not None else np.nan for v in values]
            ax.plot(angles + [angles[0] + 2 * math.pi], contour + contour[:1], color="#344554", linewidth=1.1, zorder=4)
            ax.fill(angles + [angles[0] + 2 * math.pi], contour + contour[:1], color="#344554", alpha=0.055, zorder=2)
            center = (ax.get_position().x0 + ax.get_position().x1) / 2
            title_y = 0.967 if rows==1 else ax.get_position().y1+0.045
            subtitle_y = 0.925 if rows==1 else ax.get_position().y1+0.017
            fig.text(center, title_y, name, ha="center", va="top", fontsize=13, color="#25313c")
            stats = data["model_summaries"][model]
            subtitle = (f'Mean task accuracy: {stats["mean_accuracy"]:.1%}'
                        if stats["task_count"] else "No completed attempts")
            expected_count = 79 if condition.startswith('hard') else 84
            if 0<stats['task_count']<expected_count:
                subtitle += f' ({stats["task_count"]}/{expected_count} tasks)'
            fig.text(center, subtitle_y, subtitle, ha="center", fontsize=10, color="#64727e")
        handles = [Patch(facecolor=g["color"], label=f'{g["code"]}  {g["name"]}') for g in groups]
        # Matplotlib fills legend columns; keep the displayed group order row-first.
        handles = [handles[i] for column in range(5) for i in range(column, len(handles), 5)]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=5,
                   frameon=False, fontsize=9.5, handlelength=0.9, columnspacing=1.8)
        count = next(iter(data["model_summaries"].values()))["task_count"]
        partial = any(t['models'][m]['attempted']<max(r['attempted'] for r in t['models'].values()) for t in tasks for m in models)
        footer = (f'{data["condition_label"]} · completed attempts only · fixed pooled task order · 0–100% scale'
                  + (" · gray labels and gaps indicate N/A" if count<84 or partial else ""))
        fig.text(0.5, 0.016, footer,
                 ha="center", fontsize=9, color="#64727e")
        fig.savefig(output / f"{filename}.png", dpi=200, facecolor="white")
        fig.savefig(output / f"{filename}.svg", metadata={"Date": None})
        plt.close(fig)

    # Keep actual text in the SVG, and make each compact label an accessible link.
    ET.register_namespace("", SVG)
    ET.register_namespace("xlink", XLINK)
    tree = ET.parse(output / f"{filename}.svg")
    root = tree.getroot()
    root.set("role", "group")
    root.set("data-radar-condition", condition)
    root.set("aria-label", f'{data["condition_label"]}: 84 task axes, completed attempts only, with one radar panel per model and linked task codes')
    lookup = {t["task_id"]: t for t in tasks}
    for element in root.iter(f"{{{SVG}}}g"):
        identity = element.get("id", "")
        if not identity.startswith("radar-label-"):
            continue
        model = next(m for m in sorted(models,key=len,reverse=True) if identity.startswith(f'radar-label-{m}-'))
        task_id = identity.removeprefix(f'radar-label-{model}-')
        task = lookup[task_id]
        link = element.find(f"{{{SVG}}}a")
        score = task["models"][model]
        missing_hard = condition.startswith('hard') and all(r['attempted']==0 for r in task['models'].values())
        value = f'{score["correct"]}/{score["attempted"]}' if score["attempted"] else 'N/A: no retained hard variant' if missing_hard else 'N/A: no completed attempts'
        description = f'{task["code"]}: {task["title"]} ({task_id}). {models[model]}: {value}.'
        link.set("href", f"report.html#task-{task_id}")
        link.set(f"{{{XLINK}}}href", f"report.html#task-{task_id}")
        link.set("target", "_self")
        link.set("aria-label", description)
        ET.SubElement(link, f"{{{SVG}}}title").text = description
    # Five SVGs share a document. Prefix every internal ID and reference.
    ids = {e.get("id"): f'radar-{condition}-{e.get("id")}' for e in root.iter() if e.get("id")}
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if key == "id":
                element.set(key, ids[value])
            elif value.startswith("#") and value[1:] in ids:
                element.set(key, "#" + ids[value[1:]])
            elif "url(#" in value:
                for old, new in ids.items():
                    value = value.replace(f"url(#{old})", f"url(#{new})")
                element.set(key, value)
    tree.write(output / f"{filename}.svg", encoding="unicode")
    (output / f"{filename}.json").write_text(json.dumps(data, indent=2) + "\n")


def section(output, summary, models):
    esc = html.escape
    parts = ['<h2 id="task-radar">Task performance by condition</h2>',
             '<label class="radar-control" for="radar-condition">Evaluation condition <select id="radar-condition">'
             + ''.join(f'<option value="{key}">{esc(label)}</option>' for key, label in VIEWS.items()) + '</select></label>',
             '<p>Groups and tasks are ordered by mean accuracy across models in the pooled view. Task order, codes and group colors stay fixed when you change conditions.</p>']
    for condition in VIEWS:
        data = json.loads((output / f"{stem(condition)}.json").read_text())
        parts.append(_view_section(output, data))
    parts.append('''<script>
(()=>{
  const select=document.getElementById('radar-condition');
  const views=[...document.querySelectorAll('[data-radar-view]')];
  const key='cds-radar-condition:'+location.pathname;
  function show(){
    for(const view of views)view.hidden=view.dataset.radarView!==select.value;
    try{sessionStorage.setItem(key,select.value)}catch{}
  }
  try{const saved=sessionStorage.getItem(key);if([...select.options].some(o=>o.value===saved))select.value=saved}catch{}
  select.addEventListener('change',show);show();
})();
</script>''')
    return ''.join(parts)


def _view_section(output, data):
    esc = html.escape
    condition, filename = data["condition"], stem(data["condition"])
    svg = ((output / f"{filename}.svg").read_text()
           .replace('<svg ', '<svg class="task-radar-svg" ', 1)
           .replace('href="report.html#task-', 'href="#task-'))
    partial = any(t['models'][m]['attempted']<max(r['attempted'] for r in t['models'].values()) for g in data['groups'] for t in g['tasks'] for m in data['models'])
    if condition == "all":
        description = '84 task types. A completed model pools 40 attempts per task, or 20 for the five tasks without a hard variant.'
    elif condition.startswith("hard"):
        description = '79 retained task types, with ten attempts per task for a completed model. The five tasks without a hard variant are N/A, not zero: their labels are gray and their points are omitted.'
    else:
        description = '84 task types, with ten attempts per task for a completed model.'
    if partial:
        description += ' In-progress models include only finished attempts; missing results are unscored and appear as gray labels and gaps.'
    hidden = '' if condition == 'all' else ' hidden'
    parts = [f'<section data-radar-view="{condition}" aria-label="{esc(data["condition_label"])}"{hidden}>',
             f'<p class="radar-view-description"><strong>{esc(data["condition_label"])}</strong>. {description} All failures count.</p>',
             '<figure class="task-radar-figure"><div class="scroll">' + svg + '</div><figcaption>Colors mark task groups. Hover a code for the full name, ID and score; click it to open the existing task details. On a narrow screen, scroll horizontally to see all panels.</figcaption></figure>',
             f'<p><a href="{filename}.svg">SVG</a> · <a href="{filename}.png">PNG</a> · <a href="{filename}.json">Task order and scores</a></p>',
             f'<details id="radar-task-key-{condition}"><summary>Task key: all 84 codes, full names and IDs</summary>',
             '<div class="scroll"><table class="radar-key"><thead><tr><th>Code</th><th>Task</th><th>Mean accuracy</th></tr></thead>']
    for group in data["groups"]:
        parts.append(f'<tbody><tr class="radar-key-group"><th colspan="3"><span class="radar-swatch" style="background:{group["color"]}"></span>{group["code"]} · {esc(group["name"])}</th></tr>')
        for task in group["tasks"]:
            score = f'{task["mean_accuracy"]:.1%}' if task['mean_accuracy'] is not None else 'N/A'
            parts.append(f'<tr><th scope="row">{task["code"]}</th><td><a href="#task-{task["task_id"]}">{esc(task["title"])}</a><small><code>{esc(task["task_id"])}</code></small></td><td>{score}</td></tr>')
        parts.append('</tbody>')
    parts.append('</table></div></details></section>')
    return ''.join(parts)
