"""Matched task accuracies for Nemotron and Snowball, using the radar colors."""

import html
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from .radar import SVG, VIEWS


def points(rows, radar, condition):
    definitions = {t['task_id']: (t, g) for g in radar['groups'] for t in g['tasks']}
    nemotron = {(r['instance_id'], r['condition']): r for r in rows if r['model'] == 'deepinfra'}
    by_task = defaultdict(list)
    for row in rows:
        if row['model'] == 'snowball-v01' and (condition == 'all' or row['condition'] == condition):
            by_task[row['task_id']].append((nemotron[row['instance_id'], row['condition']], row))
    result = []
    for task_id, pairs in by_task.items():
        task, group = definitions[task_id]
        n = len(pairs)
        x, y = sum(a['passed'] for a, b in pairs), sum(b['passed'] for a, b in pairs)
        result.append({'task_id': task_id, 'title': task['title'], 'code': task['code'],
                       'group': group['name'], 'color': group['color'], 'attempted': n,
                       'nemotron_correct': x, 'snowball_correct': y, 'x': 100*x/n, 'y': 100*y/n})
    return sorted(result, key=lambda task: task['code'])


def clusters(data, condition):
    grouped = defaultdict(list)
    for task in data:
        grouped[round(task['x'], 9), round(task['y'], 9)].append(task)
    return [{'id': f'scatter-{condition}-{i}', 'x': x, 'y': y, 'tasks': tasks}
            for i, ((x, y), tasks) in enumerate(sorted(grouped.items()))]


def _draw(ax, data, condition):
    from matplotlib.patches import Circle, Wedge

    ax.set_aspect('equal')
    ax.set_xlim(-4, 104)
    ax.set_ylim(-4, 104)
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))
    ax.set_axisbelow(True)
    ax.grid(color='#e1e5e9', linewidth=.7)
    ax.plot([0, 100], [0, 100], color='#94a3ad', linestyle='--', linewidth=1, zorder=1)
    for edge in ['top', 'right']:
        ax.spines[edge].set_visible(False)
    for cluster in clusters(data, condition):
        x, y, tasks = cluster['x'], cluster['y'], cluster['tasks']
        radius = 1.25 if len(tasks) == 1 else 1.7 + .25*math.log2(len(tasks))
        angle = 90
        for color, n in Counter(task['color'] for task in tasks).items():
            end = angle + 360*n/len(tasks)
            ax.add_patch(Wedge((x, y), radius, angle, end, facecolor=color,
                               edgecolor='white', linewidth=.55, zorder=3))
            angle = end
        if len(tasks) > 1:
            ax.add_patch(Circle((x, y), radius*.57, color='white', zorder=4))
            ax.text(x, y, str(len(tasks)), ha='center', va='center', fontsize=7.5,
                    zorder=5, color='#26313b')
        hit = Circle((x, y), max(radius, 2.2), facecolor='white', alpha=0, zorder=6)
        hit.set_gid(cluster['id'])
        ax.add_patch(hit)
    if not data:
        ax.text(50, 55, 'No completed Snowball attempts\nfor this condition yet.',
                ha='center', va='center', color='#64727e',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 8})


def _handles(radar):
    from matplotlib.lines import Line2D
    return [Line2D([], [], marker='o', linestyle='', color=group['color'], markersize=6,
                   label=group['code']+'  '+group['name']) for group in radar['groups']]


def _accessible(svg_path, data, condition):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    root.set('class', 'task-scatter-svg')
    root.set('role', 'group')
    root.set('aria-label', VIEWS[condition]+': Nemotron versus Snowball task accuracy')
    lookup = {cluster['id']: cluster for cluster in clusters(data, condition)}
    for element in root.iter():
        if element.get('id') in lookup:
            cluster = lookup[element.get('id')]
            label = '; '.join(f"{t['code']}: {t['title']} ({t['task_id']}); Nemotron {t['nemotron_correct']}/{t['attempted']}, Snowball {t['snowball_correct']}/{t['attempted']}"
                              for t in cluster['tasks'])
            element.set('data-scatter-point', cluster['id'])
            element.set('tabindex', '0')
            element.set('role', 'button')
            element.set('aria-label', label)
            for child in element.iter():
                child.set('pointer-events', 'all')
            ET.SubElement(element, '{'+SVG+'}title').text = label
    ids = {e.get('id'): 'comparison-'+condition+'-'+e.get('id')
           for e in root.iter() if e.get('id')}
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if key == 'id':
                element.set(key, ids[value])
            elif value.startswith('#') and value[1:] in ids:
                element.set(key, '#'+ids[value[1:]])
            elif 'url(#' in value:
                for old, new in ids.items():
                    value = value.replace('url(#'+old+')', 'url(#'+new+')')
                element.set(key, value)
    ET.register_namespace('', SVG)
    tree.write(svg_path, encoding='unicode')


def plot(output, rows, radar):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    views = {condition: points(rows, radar, condition) for condition in VIEWS}
    for condition, data in views.items():
        name = 'task-scatter-'+condition
        with mpl.rc_context({'svg.fonttype': 'none', 'svg.hashsalt': name}):
            fig, ax = plt.subplots(figsize=(10.4, 6.7))
            fig.subplots_adjust(left=.09, right=.69, bottom=.13, top=.94)
            _draw(ax, data, condition)
            ax.set_xlabel('Nemotron 3.5 Lightning · task accuracy (%)', labelpad=10)
            ax.set_ylabel('Snowball v0.1 · task accuracy (%)', labelpad=10)
            ax.legend(handles=_handles(radar), loc='center left', bbox_to_anchor=(1.04, .5),
                      frameon=False, fontsize=9.5, labelspacing=1.1, handletextpad=.6)
            fig.savefig(output/(name+'.png'), dpi=170, facecolor='white')
            fig.savefig(output/(name+'.svg'), metadata={'Date': None})
            plt.close(fig)
        _accessible(output/(name+'.svg'), data, condition)
        (output/(name+'.json')).write_text(json.dumps(data, indent=2)+'\n')

    # One shared legend for the two Easy conditions shown in the PR.
    name = 'nemotron-snowball-easy'
    with mpl.rc_context({'svg.fonttype': 'none', 'svg.hashsalt': name}):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.7))
        fig.subplots_adjust(left=.065, right=.985, bottom=.27, top=.88, wspace=.14)
        for ax, condition in zip(axes, ['easy-no-tools', 'easy-tools'], strict=True):
            data = views[condition]
            _draw(ax, data, condition)
            ax.set_title(VIEWS[condition]+f"\n{len(data)} tasks · {sum(t['attempted'] for t in data):,} matched attempts/model",
                         fontsize=11, pad=13, linespacing=1.7)
        fig.supxlabel('Nemotron 3.5 Lightning · task accuracy (%)', y=.19, fontsize=11)
        fig.supylabel('Snowball v0.1 · task accuracy (%)', x=.018, fontsize=11)
        handles = _handles(radar)
        handles = [handles[i] for column in range(5) for i in range(column, len(handles), 5)]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .035), ncol=5,
                   frameon=False, fontsize=9.5, handletextpad=.5, columnspacing=1.4, labelspacing=1.1)
        fig.savefig(output/(name+'.png'), dpi=200, facecolor='white')
        fig.savefig(output/(name+'.svg'), metadata={'Date': None})
        plt.close(fig)
    svg_path = output/(name+'.svg')
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
    return views


def _inline(path):
    root = ET.parse(path).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.get('id', '').endswith('legend_1'):
                parent.remove(child)
    box = root.get('viewBox').split()
    box[2] = str(float(box[2])*.73)
    root.set('viewBox', ' '.join(box))
    root.set('width', box[2]+'pt')
    return ET.tostring(root, encoding='unicode')


def section(output):
    radar = json.loads((output/'task-radar.json').read_text())
    esc = html.escape
    parts = ['<h2 id="task-scatter">Nemotron versus Snowball by task</h2>',
             '<p>Each point is one task. Colors and task codes match the radar charts. Both axes use the same completed cases; while Snowball is running, Nemotron is restricted to those cases too. All failures count.</p>',
             '<label class="radar-control">Evaluation condition <select id="scatter-condition">',
             *[f'<option value="{c}">{esc(label)}</option>' for c, label in VIEWS.items()],
             '</select></label>']
    all_clusters = []
    for condition, label in VIEWS.items():
        name = 'task-scatter-'+condition
        data = json.loads((output/(name+'.json')).read_text())
        all_clusters.extend(clusters(data, condition))
        legend = '<ul class="scatter-legend">'+''.join('<li><span style="background:'+g['color']+'"></span>'+g['code']+' '+esc(g['name'])+'</li>' for g in radar['groups'])+'</ul>'
        parts.append(f'<section data-scatter-view="{condition}"'+(' hidden' if condition != 'all' else '')+f'><p><strong>{esc(label)}</strong> · {len(data)}/84 tasks · {sum(t["attempted"] for t in data):,} matched attempts per model.</p>'+_inline(output/(name+'.svg'))+legend+
                     '<p class="muted">Above the dashed equality line, Snowball scores higher; below it, Nemotron does. Overlapping tasks share one marker: the number shows the task count and colored sectors show their groups. Positions are not jittered. Hover or select a marker to see the tasks below.</p>'+
                     f'<p><a href="{name}.svg">SVG</a> · <a href="{name}.png">PNG</a></p></section>')
    parts.append('<div id="scatter-selection" aria-live="polite"><p class="muted">Select a marker to see task names, IDs and scores.</p></div>')
    payload = json.dumps({c['id']: c['tasks'] for c in all_clusters}).replace('<', '\\u003c')
    parts.append('<script id="scatter-data" type="application/json">'+payload+'</script>')
    parts.append('''<style>.task-scatter-svg{display:block;width:100%;max-width:760px;height:auto}.scatter-legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px 20px;list-style:none;padding:0;margin:8px 0 20px;font-size:13px}.scatter-legend span{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}.task-scatter-svg [data-scatter-point],.task-scatter-svg [data-scatter-point] *{cursor:pointer;pointer-events:all}.task-scatter-svg [data-scatter-point]:focus{outline:2px solid #475569}#scatter-selection{padding:4px 16px;border-left:3px solid #dce2e7;margin:16px 0 28px}#scatter-selection small{display:block;color:#64727e}</style>
<script>(()=>{
const select=document.getElementById('scatter-condition'),radar=document.getElementById('radar-condition'),data=JSON.parse(document.getElementById('scatter-data').textContent),panel=document.getElementById('scatter-selection');
function show(){for(const section of document.querySelectorAll('[data-scatter-view]'))section.hidden=section.dataset.scatterView!==select.value;panel.replaceChildren();const p=document.createElement('p');p.className='muted';p.textContent='Select a marker to see task names, IDs and scores.';panel.append(p)}
select.value=radar.value;radar.addEventListener('change',()=>{select.value=radar.value;show()});select.addEventListener('change',()=>{radar.value=select.value;radar.dispatchEvent(new Event('change'))});show();
function inspect(marker){panel.replaceChildren();for(const t of data[marker.dataset.scatterPoint]){const p=document.createElement('p'),a=document.createElement('a'),small=document.createElement('small');a.href='#task-'+t.task_id;a.textContent=t.code+' · '+t.title;small.textContent=t.task_id+' · '+t.group+' · Nemotron '+t.nemotron_correct+'/'+t.attempted+' ('+t.x.toFixed(1)+'%) · Snowball '+t.snowball_correct+'/'+t.attempted+' ('+t.y.toFixed(1)+'%)';p.append(a,small);panel.append(p)}}
for(const marker of document.querySelectorAll('[data-scatter-point]')){for(const event of ['mouseenter','focus','click'])marker.addEventListener(event,()=>inspect(marker));marker.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();inspect(marker)}})}
})();</script>''')
    return ''.join(parts)
