"""Collect test metrics from runs/*/metrics_test.json into paper CSVs (CPU only).

Outputs (in --out):
  S1b_f1pr.csv        one row per run: IoU/F1/P/R/mIoU/OA at 0.5 for ALL, plus the run's Otsu-threshold row if present
  S1c_per_subset.csv  long format: run, key (sid:/region:/platform:), IoU, F1, P, R
  S5_loro.csv         LORO runs (B_loro_*) with per-subset breakdown
  seeds_summary.csv   mean ± sd over seeds for run families (name with _s<k> stripped)
Usage: python -m cas.collect_metrics [--runs runs] [--out paper/collected] [--pattern 'C_*']
"""
import argparse, csv, glob, json, os, re
from collections import defaultdict
from statistics import mean, stdev


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="paper/collected")
    ap.add_argument("--pattern", default="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows_all, rows_sub, rows_loro = [], [], []
    fam = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(a.runs, a.pattern))):
        name = os.path.basename(d)
        m = load(os.path.join(d, "metrics_test.json"))
        if not m:
            continue
        keys = [k for k in m if k.startswith("at_")]
        base = m.get("at_0.5") or m[keys[0]]
        if "ALL" not in base:
            continue
        r = base["ALL"]
        otsu_key = next((k for k in keys if "otsu" in k.lower()), None)
        o = m[otsu_key]["ALL"] if otsu_key and "ALL" in m[otsu_key] else {}
        rows_all.append(dict(run=name, IoU=r["IoU"], F1=r["F1"], P=r["Precision"], R=r["Recall"],
                             mIoU=r.get("mIoU"), OA=r.get("OA"),
                             otsu_thr=(otsu_key or "").replace("at_", ""), IoU_otsu=o.get("IoU"), F1_otsu=o.get("F1"),
                             P_otsu=o.get("Precision"), R_otsu=o.get("Recall")))
        for k, v in base.items():
            if k == "ALL":
                continue
            rows_sub.append(dict(run=name, key=k, IoU=v["IoU"], F1=v["F1"], P=v["Precision"], R=v["Recall"]))
            if name.startswith("B_loro_"):
                rows_loro.append(dict(run=name, key=k, IoU=v["IoU"], F1=v["F1"], P=v["Precision"], R=v["Recall"]))
        if name.startswith("B_loro_"):
            rows_loro.append(dict(run=name, key="ALL", IoU=r["IoU"], F1=r["F1"], P=r["Precision"], R=r["Recall"]))
        ms = re.match(r"^(.*)_s(\d+)$", name)
        if ms:
            fam[ms.group(1)].append((int(ms.group(2)), r["IoU"], r["F1"], r["Precision"], r["Recall"]))

    def dump(fn, rows):
        if not rows:
            return
        with open(os.path.join(a.out, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for x in rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in x.items()})

    dump("S1b_f1pr.csv", rows_all)
    dump("S1c_per_subset.csv", rows_sub)
    dump("S5_loro.csv", rows_loro)
    seeds = []
    for k, v in sorted(fam.items()):
        v.sort()
        ious = [x[1] for x in v]
        seeds.append(dict(family=k, n=len(v), seeds=" ".join(str(x[0]) for x in v),
                          IoU_mean=mean(ious), IoU_sd=(stdev(ious) if len(ious) > 1 else 0.0),
                          F1_mean=mean(x[2] for x in v), P_mean=mean(x[3] for x in v), R_mean=mean(x[4] for x in v),
                          IoU_list=" ".join(f"{x:.4f}" for x in ious)))
    dump("seeds_summary.csv", seeds)
    print(f"runs={len(rows_all)} subset_rows={len(rows_sub)} loro_rows={len(rows_loro)} families={len(seeds)} -> {a.out}")


if __name__ == "__main__":
    main()
