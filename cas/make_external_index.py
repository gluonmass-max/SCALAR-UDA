"""Build an index + split for an EXTERNAL target domain (GDCLD val tiles) so train_uda.py / train.py can use it.

  index_ext.csv          = CAS index rows + one row per GDCLD val tile (sid GDCLD_val, region GDCLD, platform EXT,
                           gsd_m 1.0 as a placeholder: per-tile GSD is not published, so M1 is identity here)
  X_cas2gdcld.json       = train/val of A_712 (all CAS), test = all GDCLD val tiles  (transductive / inductive via --uda_mode)
  X_cas2gdcld_indtest.json = same train/val, test = the INDUCTIVE test half (split_target seed 0) for eval-only baselines
"""
import csv, json, os, sys, glob
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cas.train_uda import split_target  # noqa: E402

HY = "."; RAW = "data/CAS_raw"; GD = "data/GDCLD"
rows = list(csv.DictReader(open(f"{HY}/splits/index.csv")))
fields = list(rows[0].keys())
ext = []
for img in sorted(glob.glob(f"{GD}/val_data/*.tif")):
    stem = os.path.basename(img)[:-4]
    lab = f"{GD}/val_label/{stem}.tif"
    m = cv2.imread(lab, cv2.IMREAD_UNCHANGED)
    if m is None:
        continue
    if m.ndim == 3:
        m = m[..., 0]
    h, w = m.shape
    ext.append({"sid": "GDCLD_val", "region": "GDCLD", "platform": "EXT", "sensor": "mixed", "gsd_m": "1.0", "stem": stem,
                "img": os.path.relpath(img, RAW), "mask": os.path.relpath(lab, RAW), "ls_frac": f"{(m > 0).mean():.5f}", "h": h, "w": w})
with open(f"{HY}/splits/index_ext.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader()
    for r in rows + ext:
        wr.writerow({k: r.get(k, "") for k in fields})
index = {f"{r['sid']}/{r['stem']}": r for r in rows + ext}
keys = [f"GDCLD_val/{r['stem']}" for r in ext]
a712 = json.load(open(f"{HY}/splits/A_712.json"))
adapt, test = split_target(keys, index, "inductive", 0)
json.dump({"note": "CAS (A_712 train/val) -> GDCLD val tiles as external unlabelled target", "n": {"train": len(a712["train"]), "val": len(a712["val"]), "test": len(keys)},
           "train": a712["train"], "val": a712["val"], "test": keys}, open(f"{HY}/splits/X_cas2gdcld.json", "w"))
json.dump({"note": "inductive TEST half of GDCLD val (split_target seed 0); for eval-only baseline rows", "n": {"train": len(a712["train"]), "val": len(a712["val"]), "test": len(test)},
           "train": a712["train"], "val": a712["val"], "test": test}, open(f"{HY}/splits/X_cas2gdcld_indtest.json", "w"))
fr = np.array([float(r["ls_frac"]) for r in ext])
print(f"GDCLD val tiles {len(ext)}  adapt {len(adapt)} test {len(test)}  landslide frac {fr.mean():.4f}  empty tiles {(fr == 0).mean():.3f}")
