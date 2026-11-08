"""Compare pool and hold-out IoU from a user-supplied JSON file.

Usage: python scripts/e1b_stats.py --input runs/holdout_variant_scores.json
Input keys: gsa_p, sc_p, gsa_h and sc_h. Each value is an array of per-seed
IoU measurements; source and adapted arrays must use the same seed order.
"""
import argparse
import json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, help="JSON file containing your own per-seed IoU measurements")
args = parser.parse_args()

import numpy as np
from scipy import stats

with open(args.input, encoding="utf-8") as f:
    scores = json.load(f)
gsa_p, sc_p, gsa_h, sc_h = (np.asarray(scores[k], dtype=float) for k in ("gsa_p", "sc_p", "gsa_h", "sc_h"))
for label, a, b in (("pool", gsa_p, sc_p), ("hold", gsa_h, sc_h)):
    if a.ndim != 1 or b.ndim != 1 or a.size < 2 or a.shape != b.shape:
        parser.error(f"{label}: arrays must be paired, one-dimensional and contain at least two values")
    if not (np.isfinite(a).all() and np.isfinite(b).all() and ((0 <= a) & (a <= 1)).all() and ((0 <= b) & (b <= 1)).all()):
        parser.error(f"{label}: IoU values must be finite and between 0 and 1")
ms=lambda x: f"{np.mean(x):.3f} ± {np.std(x,ddof=1):.3f}"
for lab,a,b in (("pool",gsa_p,sc_p),("hold",gsa_h,sc_h)):
    d=np.array(b)-np.array(a); t,p=stats.ttest_rel(b,a)
    print(f"{lab}: Source+GSA {ms(a)}  SCALAR {ms(b)}  diff {d.mean()*100:+.1f} ± {d.std(ddof=1)*100:.1f} pts, pos {(d>0).sum()}/{len(d)}, p={p:.4f}")
