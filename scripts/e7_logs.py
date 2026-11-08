"""E7 log analysis: per-epoch LOC statistics of every self-training run (no GPU). Usage: python -I e7_logs.py <runs dir> <out csv>"""
import csv, json, os, re, sys
R, OUT = sys.argv[1], sys.argv[2]
rx_ep = re.compile(r"\[(otsu|prior)\] epoch (\d+): teacher mean prob ([\d.]+) \| EM prior ([\d.]+) -> thr ([\d.]+) \| logit-Otsu thr ([\d.]+) -> using ([\d.]+) \| pi_hat ([\d.]+)")
rx_tr = re.compile(r"^ep (\d+)/(\d+) .*? thr ([\d.]+) pl_kept ([\d.]+) .*?src-val IoU ([\d.]+)")
rx_test = re.compile(r"^TEST@0\.5\s+IoU ([\d.]+)")
def test_iou(name):
    p = os.path.join(R, name, "log.txt")
    if not os.path.exists(p): return None
    v = [float(m.group(1)) for m in map(rx_test.match, open(p, errors="ignore")) if m]
    return v[-1] if v else None
rows = []
for name in sorted(os.listdir(R)):
    p = os.path.join(R, name, "log.txt")
    if not os.path.exists(p): continue
    lines = open(p, errors="ignore").read().splitlines()
    if not lines or not lines[0].startswith("args "): continue
    try: args = json.loads(lines[0][5:])
    except Exception: continue
    eps = [m for m in (rx_ep.search(l) for l in lines) if m]
    if not eps: continue
    trs = [m for m in (rx_tr.match(l) for l in lines) if m]
    init = args.get("init", ""); init_name = init.split("/")[-2] if "/" in init else ""
    thr = [float(m.group(7)) for m in eps]; otsu = [float(m.group(6)) for m in eps]; pih = [float(m.group(8)) for m in eps]; pm = [float(m.group(3)) for m in eps]; em = [float(m.group(5)) for m in eps]
    kept = [float(m.group(4)) for m in trs]
    fin = test_iou(name); ini = test_iou(init_name) if init_name else None
    rows.append(dict(run=name, mode=eps[0].group(1), pl_mode=args.get("pl_mode"), margin_pos=args.get("pl_margin_pos"), margin_neg=args.get("pl_margin_neg"),
        uda_mode=args.get("uda_mode"), init=init_name, n_ep=len(eps),
        thr1=thr[0], thr_last=thr[-1], thr_min=min(thr), thr_max=max(thr), otsu1=otsu[0], em1=em[0], pmean1=pm[0], pihat1=pih[0], pihat_last=pih[-1],
        clip_hits=sum(1 for t in otsu if t <= 0.02 + 1e-9 or t >= 0.98 - 1e-9), kept_min=min(kept) if kept else None, kept_last=kept[-1] if kept else None,
        test_init=ini, test_final=fin, gain=(None if fin is None or ini is None else round(fin - ini, 4))))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), "self-training runs with LOC statistics ->", OUT)
# ---- summaries
import statistics as st
allth = [t for r in rows for t in (r["thr_min"], r["thr_max"])]
print(f"theta range over all runs/epochs: {min(r['thr_min'] for r in rows):.3f} .. {max(r['thr_max'] for r in rows):.3f}; runs with a clip hit: {sum(1 for r in rows if r['clip_hits'])}/{len(rows)}")
print(f"pl_kept min over runs: {min(r['kept_min'] for r in rows if r['kept_min'] is not None):.3f}")
def fam(n): return re.sub(r"_s\d+(_rerun)?$", "", n)
fams = {}
for r in rows: fams.setdefault(fam(r["run"]), []).append(r)
print("\nfamily                                   n  thr1 mean  thr_last mean  pihat1 mean  gain mean (n with gain)")
for k, rs in sorted(fams.items()):
    g = [r["gain"] for r in rs if r["gain"] is not None]
    print(f"{k:40s} {len(rs):2d}  {st.mean(r['thr1'] for r in rs):.3f}      {st.mean(r['thr_last'] for r in rs):.3f}          {st.mean(r['pihat1'] for r in rs):.3f}       {('%+.3f' % st.mean(g)) if g else '  -  '} ({len(g)})")
print("\nnegative-gain runs (final test IoU below the init model):")
for r in rows:
    if r["gain"] is not None and r["gain"] < 0:
        print(f"  {r['run']:42s} init {r['test_init']:.4f} -> {r['test_final']:.4f} ({r['gain']:+.4f})  thr1 {r['thr1']:.3f} pihat1 {r['pihat1']:.3f} pmean1 {r['pmean1']:.4f} kept_min {r['kept_min']}")
