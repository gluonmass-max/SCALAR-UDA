"""Region-cluster bootstrap with Longxi merged into the Wenchuan cluster (UAV->SAT); same seed/order as bootstrap.py."""
import csv, os, sys, numpy as np
R = sys.argv[1]; B = 2000
def load(n):
    p = os.path.join(R, n, "tiles.csv"); rows = list(csv.DictReader(open(p)))
    tp = np.array([int(r["tp_t05"]) for r in rows]); fp = np.array([int(r["fp_t05"]) for r in rows]); fn = np.array([int(r["fn_t05"]) for r in rows])
    reg = np.array([("Wenchuan" if r["region"] in ("Longxi", "Wenchuan") else r["region"]) for r in rows]); key = np.array([r["key"] for r in rows])
    return dict(tp=tp, fp=fp, fn=fn, reg=reg, key=key)
def iou(d, idx=None):
    if idx is None: idx = slice(None)
    tp, fp, fn = d["tp"][idx].sum(), d["fp"][idx].sum(), d["fn"][idx].sum(); return tp / max(tp + fp + fn, 1)
def boot_cluster(d, e=None, seed=0):
    rng = np.random.default_rng(seed); regs = np.unique(d["reg"]); groups = {r: np.where(d["reg"] == r)[0] for r in regs}; vals = []
    for _ in range(B):
        pick = rng.choice(regs, len(regs), replace=True); idx = np.concatenate([groups[r] for r in pick])
        vals.append(iou(d, idx) - (iou(e, idx) if e is not None else 0))
    v = np.array(vals); return np.percentile(v, 2.5), np.percentile(v, 97.5), len(regs)
d = "uav2sat"
nat, m1, ots, sc = load(f"{d}_native_s0"), load(f"{d}_M1_s0"), load(f"{d}_otsuM_s0"), load(f"{d}_SCALAR_s0")
print("clusters:", sorted(set(nat["reg"])))
for name, x in (("native", nat), ("Source+GSA", m1), ("GSA+LOC", ots), ("SCALAR", sc)):
    lo, hi, k = boot_cluster(x); print(f"{name}: {iou(x):.3f} region95 [{lo:.3f}, {hi:.3f}] (k={k})")
for name, a, b in (("Source+GSA - native", m1, nat), ("GSA+LOC - Source+GSA", ots, m1), ("SCALAR - GSA+LOC", sc, ots), ("SCALAR - Source+GSA", sc, m1)):
    assert (a["key"] == b["key"]).all(); lo, hi, k = boot_cluster(a, b); print(f"{name}: {(iou(a)-iou(b))*100:+.1f} region95 [{lo*100:+.1f}, {hi*100:+.1f}]")
