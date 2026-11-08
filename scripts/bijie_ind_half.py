"""IoU of the Bijie zero-shot and transductive models on the inductive test half (seed 0),
from the per-tile counts of runs/E3/bj_*/tiles.csv (same halving as train_uda --uda_mode inductive). Read-only.
Usage: python scripts/bijie_ind_half.py X_bj_sat.json bj_sat_native_s0 bj_sat_M1_s0 ..."""
import csv, json, sys, os
sys.path.insert(0, os.getcwd())
from cas.train_uda import split_target
sp = json.load(open("splits/" + sys.argv[1]))
index = {f"{r['sid']}/{r['stem']}": r for r in csv.DictReader(open("splits/index_bijie.csv"))}
tk = sp.get("target") or sp.get("test")
adapt, test = split_target(tk, index, "inductive", 0)
test = set(test); print("target", len(tk), "adapt", len(adapt), "test", len(test))
for e in sys.argv[2:]:
    f = f"runs/E3/{e}/tiles.csv"
    if not os.path.exists(f): print(e, "no tiles.csv"); continue
    tp = fp = fn = 0; n = 0
    for r in csv.DictReader(open(f)):
        if r["key"] in test:
            tp += float(r["tp_t05"]); fp += float(r["fp_t05"]); fn += float(r["fn_t05"]); n += 1
    print(e, "tiles", n, "IoU@0.5 on test half", round(tp / (tp + fp + fn), 4))
