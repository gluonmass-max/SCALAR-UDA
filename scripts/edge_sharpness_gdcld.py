"""R2.5 follow-up: the edge-sharpness estimate of scripts/edge_sharpness.py (s0 = 1 px, 60 random tiles per scene,
seed 0) applied to the GDCLD external scenes, calibrated with the CAS UAV value sigma_native = 0.634 px.
Writes runs/edge_sharpness_gdcld_20261022.txt."""
import csv, random, os, sys
import numpy as np, cv2
sys.path.insert(0, os.getcwd())
from cas.train import read_img
cv2.setNumThreads(4)
S0, NAT = 1.0, 0.634
by = {}
for r in csv.DictReader(open("splits/index_gdcld.csv")):
    if r["platform"] == "EXT":
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
out = [f"GDCLD scenes, s0 = {S0} px, sigma_native = {NAT} px (CAS UAV median)"]
for sid, rs in sorted(by.items()):
    rnd = random.Random(0); sample = rnd.sample(rs, min(60, len(rs)))
    sig = [s for s in (tile_sigma(os.path.join("data/CAS_raw", r["img"])) for r in sample) if s is not None]
    q1, q2, q3 = np.percentile(sig, [25, 50, 75]); g = float(rs[0]["gsd_m"])
    out.append(f"  {sid:16s} nominal {g:4.1f} m  sigma {q2:.2f} px (IQR {q1:.2f}-{q3:.2f})  ratio {q2 / NAT:.2f}  effective {g * max(1, q2 / NAT):.2f} m  n={len(sig)}")
    print(out[-1], flush=True)
open("runs/edge_sharpness_gdcld_20261022.txt", "w").write("\n".join(out) + "\n")
