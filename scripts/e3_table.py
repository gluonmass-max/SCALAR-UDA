"""E3 tables from summary.json + hist.npz: IoU at 0.5 / inference-only Otsu / prior / oracle threshold, AP, object-level."""
import json, os, sys, numpy as np
R = sys.argv[1]
def oracle(h):
    hp, hn = h["pos"].sum(0).astype(float), h["neg"].sum(0).astype(float)
    tp = np.cumsum(hp[::-1])[::-1]; fp = np.cumsum(hn[::-1])[::-1]; fn = hp.sum() - tp
    iou = tp / np.maximum(tp + fp + fn, 1); i = int(np.argmax(iou)); return float(iou[i]), i / len(hp)
rows = []
for n in sorted(os.listdir(R)):
    d = os.path.join(R, n)
    if not os.path.exists(os.path.join(d, "summary.json")): continue
    s = json.load(open(os.path.join(d, "summary.json"))); h = np.load(os.path.join(d, "hist.npz"))
    px = s["pixel"]; ob = s["objects"]["t05"]["ALL"]; oo = s["objects"]["otsu"]["ALL"]
    orc, t_orc = oracle(h)
    rows.append((n, px["t05"]["ALL"]["IoU"], s["thresholds"]["otsu"], px["otsu"]["ALL"]["IoU"], px["prior"]["ALL"]["IoU"], orc, t_orc, s["AP"]["ALL"]["AP"],
                 ob["omission_any"], ob["omission_50"], ob["commission_any"], oo["omission_50"], oo["commission_any"], s["pool"]["otsu_eta"], s["pool"]["frac_above_otsu"], s["pool"]["mean_prob"]))
print(f"{'model':30s} IoU@.5  θotsu IoU@θ  IoU@prior IoU@orc(t)   AP     om_any om_50 com_any | @θ: om_50 com_any | η   pihat pmean")
for r in rows:
    print(f"{r[0]:30s} {r[1]:.4f} {r[2]:.3f} {r[3]:.4f}  {r[4]:.4f}   {r[5]:.4f}({r[6]:.2f}) {r[7]:.4f}  {r[8]:.3f}  {r[9]:.3f} {r[10]:.3f}  |  {r[11]:.3f} {r[12]:.3f}  | {r[13]:.2f} {r[14]:.3f} {r[15]:.3f}")
