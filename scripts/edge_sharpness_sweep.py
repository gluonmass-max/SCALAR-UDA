"""R2.5 robustness of the edge-sharpness estimate (CPU only): same method as scripts/edge_sharpness.py
(Zhuo & Sim 2011 re-blur gradient ratio at Canny edges), repeated for re-blur scales s0 = 0.7, 1.0, 1.5 px, with the
per-subset interquartile range over tiles. Same 60 random tiles per subset (seed 0).
Writes runs/edge_sharpness_sweep_20261022.txt."""
import csv, random, os, sys
import numpy as np, cv2
sys.path.insert(0, os.getcwd())
from cas.train import read_img
from scipy.stats import spearmanr
cv2.setNumThreads(4)
idx = list(csv.DictReader(open("splits/index.csv")))
by = {}
for r in idx:
    by.setdefault(r["sid"], []).append(r)
def tile_sigma(path, s0):
    g = read_img(path).astype(np.float32)
    g = (0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]) if g.ndim == 3 else g
    u8 = np.clip(g, 0, 255).astype(np.uint8)
    med = float(np.median(u8)); e = cv2.Canny(u8, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)) or 50) > 0
    if e.sum() < 200:
        return None
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    b = cv2.GaussianBlur(g, (0, 0), s0)
    bx, by_ = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    m1, m2 = np.hypot(gx, gy)[e], np.hypot(bx, by_)[e]
    ok = (m2 > 1e-3)
    R = m1[ok] / m2[ok]; R = R[R > 1.01]
    if R.size < 100:
        return None
    return float(np.median(s0 / np.sqrt(R ** 2 - 1)))
out = []
for s0 in (0.7, 1.0, 1.5):
    rows = []
    for sid, rs in sorted(by.items()):
        rnd = random.Random(0); sample = rnd.sample(rs, min(60, len(rs)))
        sig = [s for s in (tile_sigma(os.path.join("data/CAS_raw", r["img"]), s0) for r in sample) if s is not None]
        q1, q2, q3 = np.percentile(sig, [25, 50, 75])
        rows.append(dict(sid=sid, platform=rs[0]["platform"], gsd=float(rs[0]["gsd_m"]), n=len(sig), q1=q1, med=q2, q3=q3))
    nat = float(np.median([r["med"] for r in rows if r["platform"] == "UAV"]))
    for r in rows:
        r["ratio"] = r["med"] / nat; r["eff"] = r["gsd"] * max(1.0, r["ratio"])
    rho = spearmanr([r["gsd"] for r in rows], [r["eff"] for r in rows]).correlation
    out.append(f"s0 = {s0} px: sigma_native {nat:.3f} px, Spearman(nominal, effective) {rho:.3f}")
    for r in rows:
        out.append(f"  {r['sid']:20s} nominal {r['gsd']:4.1f} m  sigma {r['med']:.2f} px (IQR {r['q1']:.2f}-{r['q3']:.2f})  ratio {r['ratio']:.2f}  effective {r['eff']:.2f} m  n={r['n']}")
    print(out[-len(rows) - 1], flush=True)
open("runs/edge_sharpness_sweep_20261022.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
