"""R2-M9: tile-level bootstrap CIs (and region-cluster bootstrap) for pooled IoU@0.5 and paired differences, from E3 tiles.csv."""
import csv, os, sys, numpy as np
R = sys.argv[1]; rng = np.random.default_rng(0); B = 2000
def load(n):
    p = os.path.join(R, n, "tiles.csv")
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p)))
    tp = np.array([int(r["tp_t05"]) for r in rows]); fp = np.array([int(r["fp_t05"]) for r in rows]); fn = np.array([int(r["fn_t05"]) for r in rows])
    reg = np.array([r["region"] for r in rows]); key = np.array([r["key"] for r in rows])
    return dict(tp=tp, fp=fp, fn=fn, reg=reg, key=key)
def iou(d, idx=None):
    if idx is None: idx = slice(None)
    tp, fp, fn = d["tp"][idx].sum(), d["fp"][idx].sum(), d["fn"][idx].sum()
    return tp / max(tp + fp + fn, 1)
def boot(d, e=None, cluster=False):
    n = len(d["tp"]); vals = []
    if cluster:
        regs = np.unique(d["reg"]); groups = {r: np.where(d["reg"] == r)[0] for r in regs}
        for _ in range(B):
            pick = rng.choice(regs, len(regs), replace=True); idx = np.concatenate([groups[r] for r in pick])
            vals.append(iou(d, idx) - (iou(e, idx) if e is not None else 0))
    else:
        for _ in range(B):
            idx = rng.integers(0, n, n); vals.append(iou(d, idx) - (iou(e, idx) if e is not None else 0))
    v = np.array(vals); return np.percentile(v, 2.5), np.percentile(v, 97.5)
def aligned(a, b):
    assert (a["key"] == b["key"]).all(), "tile order differs"
rows = []
for d in ("sat2uav", "uav2sat"):
    nat, m1, ots, sc = load(f"{d}_native_s0"), load(f"{d}_M1_s0"), load(f"{d}_otsuM_s0"), load(f"{d}_SCALAR_s0")
    for name, x in (("原生源 s0", nat), ("Source+GSA s0", m1), ("GSA+LOC s0", ots), ("SCALAR s0", sc)):
        lo, hi = boot(x); clo, chi = boot(x, cluster=True)
        rows.append((d, name, f"{iou(x):.3f}", f"[{lo:.3f}, {hi:.3f}]", f"[{clo:.3f}, {chi:.3f}]"))
    for name, a, b in (("Source+GSA − 原生源", m1, nat), ("GSA+LOC − Source+GSA", ots, m1), ("SCALAR − GSA+LOC", sc, ots), ("SCALAR − Source+GSA", sc, m1)):
        aligned(a, b); lo, hi = boot(a, b); clo, chi = boot(a, b, cluster=True)
        rows.append((d, name, f"{(iou(a)-iou(b))*100:+.1f}", f"[{lo*100:+.1f}, {hi*100:+.1f}]", f"[{clo*100:+.1f}, {chi*100:+.1f}]"))
    for cmp_, seeds in (("DACS", [1,2,3,4]), ("FDA", [1,2,3,4]), ("LandsDA", [1,2,3,4]), ("MIC", [1,2,3,4])):
        for s_ in seeds:
            c = load(f"{d}_{cmp_}_s{s_}")
            if c is None: continue
            aligned(sc, c); lo, hi = boot(sc, c)
            rows.append((d, f"SCALAR s0 − {cmp_} s{s_}", f"{(iou(sc)-iou(c))*100:+.1f}", f"[{lo*100:+.1f}, {hi*100:+.1f}]", "—")); break
# E1 hold-out: SCALAR vs Source+GSA, seed 0 and per seed
for d in ("sat2uav", "uav2sat"):
    for s_ in range(5):
        a, b = load(f"E1_{d}_asym_s{s_}_hold"), load(f"E1_{d}_M1_g0.5_s{s_}_hold")
        if a is None or b is None: continue
        aligned(a, b); lo, hi = boot(a, b)
        rows.append((f"E1 留出 {d}", f"SCALAR − Source+GSA, 种子 {s_}", f"{(iou(a)-iou(b))*100:+.1f}", f"[{lo*100:+.1f}, {hi*100:+.1f}]", "—"))
print("| 设定 | 量 | 点估计 | 切片 bootstrap 95 % | 区域簇 bootstrap 95 % |"); print("|---|---|---|---|---|")
for r in rows: print("| " + " | ".join(r) + " |")
