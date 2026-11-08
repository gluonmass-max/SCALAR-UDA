"""Extract the uploaded CAS zips and build a tile index.

Usage (inside conda env `cas`):
    python -m cas.build_index [--zips data/CAS] [--raw data/CAS_raw] [--out splits/index.csv] [--skip sid,sid]

For every subset: unzip (once) to <raw>/<sid>/{img,mask,label}, pair img/mask by file stem (case-insensitive
extension, Wenchuan uses .TIF for masks), sanity-check shapes/values, compute landslide pixel fraction per tile,
and write one CSV row per tile:  sid, region, platform, sensor, gsd_m, stem, img, mask, ls_frac, h, w
"""
import argparse, os, subprocess, sys, csv, glob
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.simplefilter("ignore", NotGeoreferencedWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cas.meta import SUBSETS  # noqa: E402


def extract(zip_path, dst):
    if os.path.isdir(os.path.join(dst, "img")) and os.path.isdir(os.path.join(dst, "mask")):
        return "cached"
    os.makedirs(dst, exist_ok=True)
    r = subprocess.run(["unzip", "-oq", zip_path, "-d", dst], capture_output=True, text=True)
    if r.returncode != 0:
        return f"unzip failed: {r.stderr.strip()[:200]}"
    # some zips may wrap content in one extra folder
    if not os.path.isdir(os.path.join(dst, "img")):
        inner = [d for d in glob.glob(os.path.join(dst, "*")) if os.path.isdir(os.path.join(d, "img"))]
        if inner:
            for sub in os.listdir(inner[0]):
                os.rename(os.path.join(inner[0], sub), os.path.join(dst, sub))
            os.rmdir(inner[0])
    return "extracted"


def index_subset(s, raw_root):
    d = os.path.join(raw_root, s.sid)
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(os.path.join(d, "img")) if f.lower().endswith((".tif", ".tiff"))}
    masks = {os.path.splitext(f)[0]: f for f in os.listdir(os.path.join(d, "mask")) if f.lower().endswith((".tif", ".tiff"))}
    rows, bad = [], []
    for stem in sorted(imgs):
        if stem not in masks:
            bad.append(f"no mask for {stem}"); continue
        ip, mp = os.path.join(d, "img", imgs[stem]), os.path.join(d, "mask", masks[stem])
        try:
            with rasterio.open(mp) as m:
                arr = m.read(1)
            with rasterio.open(ip) as im:
                c, h, w = im.count, im.height, im.width
        except Exception as e:
            bad.append(f"{stem}: {e!r}"); continue
        if c != 3 or arr.shape != (h, w):
            bad.append(f"{stem}: img bands={c} size={h}x{w} mask={arr.shape}"); continue
        u = np.unique(arr)
        if arr.max() > 1:  # tolerate 0/255 masks
            arr = (arr > 0).astype(np.uint8)
        if not set(u.tolist()) <= {0, 1, 255}:
            bad.append(f"{stem}: mask values {u[:6]}"); continue
        rows.append(dict(sid=s.sid, region=s.region, platform=s.platform, sensor=s.sensor, gsd_m=s.gsd_m, stem=stem,
                         img=os.path.relpath(ip, raw_root), mask=os.path.relpath(mp, raw_root),
                         ls_frac=round(float(arr.mean()), 5), h=h, w=w))
    extra_masks = sorted(set(masks) - set(imgs))
    return rows, bad, extra_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", default="data/CAS")
    ap.add_argument("--raw", default="data/CAS_raw")
    ap.add_argument("--out", default="splits/index.csv")
    ap.add_argument("--skip", default="", help="comma-separated sids to skip (e.g. still uploading)")
    a = ap.parse_args()
    skip = set(x for x in a.skip.split(",") if x)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    all_rows = []
    print(f"{'sid':20s} {'status':10s} {'tiles':>6s} {'expect':>6s} {'ls_frac':>8s}  problems")
    for s in SUBSETS:
        if s.sid in skip:
            print(f"{s.sid:20s} skipped"); continue
        zp = os.path.join(a.zips, s.zip_name)
        if not os.path.exists(zp):
            print(f"{s.sid:20s} MISSING zip {s.zip_name}"); continue
        st = extract(zp, os.path.join(a.raw, s.sid))
        if st.startswith("unzip failed"):
            print(f"{s.sid:20s} {st}"); continue
        rows, bad, extra = index_subset(s, a.raw)
        all_rows += rows
        fr = np.mean([r["ls_frac"] for r in rows]) if rows else float("nan")
        note = (f"{len(bad)} bad ({bad[0]})" if bad else "") + (f" +{len(extra)} masks w/o img" if extra else "")
        flag = "" if len(rows) == s.n_expected else f"  <-- expected {s.n_expected}"
        print(f"{s.sid:20s} {st:10s} {len(rows):6d} {s.n_expected:6d} {fr:8.4f}  {note}{flag}")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"\nwrote {len(all_rows)} rows -> {a.out}")
    uav = sum(r["platform"] == "UAV" for r in all_rows)
    print(f"UAV {uav}  SAT {len(all_rows) - uav}  empty-mask tiles {sum(r['ls_frac'] == 0 for r in all_rows)}")


if __name__ == "__main__":
    main()
