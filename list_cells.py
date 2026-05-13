import json

with open('Obesity_Last.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
for i, c in enumerate(cells):
    ct = c['cell_type']
    src = ''.join(c['source']).strip()
    preview = src[:200].replace('\n', ' | ')
    print(f"[{i:3d}] {ct:8s} | {preview}")
