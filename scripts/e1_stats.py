"""Summarize user-supplied region hold-out measurements.

Usage: python scripts/e1_stats.py --input runs/holdout_scores.json
Input: a JSON object keyed by transfer direction. Each direction contains
gsa_pool, sc_pool, gsa_hold, sc_hold, gsa_L, sc_L, gsa_M and sc_M arrays,
with paired seed order and at least two IoU values in each array.
"""
import argparse
import json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, help="JSON file containing your own per-seed IoU measurements")
args = parser.parse_args()

import numpy as np
from scipy import stats

with open(args.input, encoding="utf-8") as f:
    D = json.load(f)
for direction, values in D.items():
    for source, target in (("gsa_pool", "sc_pool"), ("gsa_hold", "sc_hold"), ("gsa_L", "sc_L"), ("gsa_M", "sc_M")):
        a, b = np.asarray(values[source], dtype=float), np.asarray(values[target], dtype=float)
        if a.ndim != 1 or b.ndim != 1 or a.size < 2 or a.shape != b.shape:
            parser.error(f"{direction}: {source} and {target} must be paired one-dimensional arrays with at least two values")
        if not (np.isfinite(a).all() and np.isfinite(b).all() and ((0 <= a) & (a <= 1)).all() and ((0 <= b) & (b <= 1)).all()):
            parser.error(f"{direction}: IoU values must be finite and between 0 and 1")
def ms(x): x=np.array(x); return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"
for d, v in D.items():
    print(f"== {d}")
    for k in ["gsa_pool","sc_pool","gsa_hold","sc_hold","gsa_L","sc_L","gsa_M","sc_M"]:
        print(f"  {k:9s} {ms(v[k])}")
    for a,b,lab in [("gsa_pool","sc_pool","pool"),("gsa_hold","sc_hold","hold"),("gsa_L","sc_L","Longxi"),("gsa_M","sc_M","Moxi")]:
        diff = np.array(v[b])-np.array(v[a])
        t,p = stats.ttest_rel(v[b], v[a])
        wp = stats.wilcoxon(v[b], v[a], alternative="two-sided").pvalue if np.any(diff) else 1.0
        print(f"  diff {lab:6s} {diff.mean()*100:+.1f} ± {diff.std(ddof=1)*100:.1f} pts  pos {int((diff>0).sum())}/{len(diff)}  t p={p:.4f}  wilcoxon p={wp:.3f}  per-seed {[f'{x*100:+.1f}' for x in diff]}")
