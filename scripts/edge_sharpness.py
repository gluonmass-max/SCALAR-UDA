"""R2.5 follow-up: effective-resolution check from edge sharpness (CPU only).
Blur at edges by the re-blur gradient-ratio method (Zhuo & Sim 2011, defocus map): at Canny edge pixels,
R = |grad I| / |grad (I * G_s0)| and sigma = s0 / sqrt(R^2 - 1), in pixels of the released grid. A grid resampled from
coarser data (e.g. a Sentinel-2 scene put on a 0.6 m grid) should show a large sigma in pixels; a native grid about
0.5-1 px. Per tile: median over edge pixels; per subset: median over 60 random tiles.
Effective GSD estimate = nominal GSD x max(1, sigma / sigma_native), sigma_native = median sigma over the UAV subsets.
Writes runs/edge_sharpness_20261022.{csv,txt}."""
import csv, math, random, os, sys
import numpy as np, cv2
sys.path.insert(0, os.getcwd())
from cas.train import read_img
cv2.setNumThreads(4)
S0 = 1.0
idx = list(csv.DictReader(open("splits/index.csv")))
by = {}
for r in idx:
    by.setdefault(r["sid"], []).append(r)
def tile_sigma(path):
    g = read_img(path).astype(np.float32)
    g = (0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]) if g.ndim == 3 else g
    u8 = np.clip(g, 0, 255).astype(np.uint8)
    med = float(np.median(u8)); e = cv2.Canny(u8, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)) or 50) > 0
    if e.sum() < 200:
        return None
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    b = cv2.GaussianBlur(g, (0, 0), S0)
    bx, by_ = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    m1, m2 = np.hypot(gx, gy)[e], np.hypot(bx, by_)[e]
    ok = (m2 > 1e-3)
    R = m1[ok] / m2[ok]; R = R[R > 1.01]
    if R.size < 100:
        return None
    return float(np.median(S0 / np.sqrt(R ** 2 - 1)))
rows = []
for sid, rs in sorted(by.items()):
    rnd = random.Random(0); sample = rnd.sample(rs, min(60, len(rs)))
    sig = [s for s in (tile_sigma(os.path.join("data/CAS_raw", r["img"])) for r in sample) if s is not None]
    rows.append(dict(sid=sid, platform=rs[0]["platform"], sensor=rs[0]["sensor"], gsd=float(rs[0]["gsd_m"]),
                     n=len(sig), sigma_px=float(np.median(sig)) if sig else float("nan")))
    print(sid, rows[-1]["gsd"], rows[-1]["n"], round(rows[-1]["sigma_px"], 3), flush=True)
nat = float(np.median([r["sigma_px"] for r in rows if r["platform"] == "UAV"]))
for r in rows:
    r["sigma_m"] = r["sigma_px"] * r["gsd"]; r["eff_gsd"] = r["gsd"] * max(1.0, r["sigma_px"] / nat)
from scipy.stats import spearmanr
rho = spearmanr([r["gsd"] for r in rows], [r["eff_gsd"] for r in rows]).correlation
with open("runs/edge_sharpness_20261022.csv", "w", newline="") as f_:
    w = csv.DictWriter(f_, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open("runs/edge_sharpness_20261022.txt", "w") as f_:
    f_.write(f"sigma_native (median over UAV subsets) = {nat:.3f} px; Spearman(nominal, effective) = {rho:.3f}\n")
    for r in rows:
        f_.write(f"{r['sid']:20s} {r['sensor']:16s} nominal {r['gsd']:4.1f} m  sigma {r['sigma_px']:.2f} px  ({r['sigma_m']:.2f} m)  effective {r['eff_gsd']:.2f} m  n={r['n']}\n")
print("done", nat, rho)
