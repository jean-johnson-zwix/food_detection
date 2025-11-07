from pathlib import Path
import numpy as np, json, collections
root = Path("data/processed/uecfood100/labels/train")
counts = collections.Counter()
bad = 0
for p in root.glob("*.txt"):
    rows = [r.split() for r in p.read_text().splitlines() if r.strip()]
    if not rows: bad += 1
    for r in rows:
        c, cx, cy, w, h = map(float, r)
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1): bad += 1
        counts[int(c)] += 1
print("empty_or_bad_label_files:", bad)
print("top10_classes:", counts.most_common(10))
