"""E9 (2026-10-21): tile the GDCLD test scenes and build the two external transfer splits.
Same tiling as cas/make_gdcld_scenes.py (1024 px, PNG, >= 10 % valid pixels, per-scene GSD from the geotransform), relative paths,
Palu scenes EXCLUDED (the Palu event is in CAS). Outputs under splits/:
  index_gdcld.csv          CAS index + GDCLD scene tiles (sid GDCLD_<scene>, region GDCLD_<event>, platform EXT)
  X_gd_uav2coarse.json     source = C_uav2sat train/val (CAS UAV); test = 3 m scenes (Lushan_3, Mesetas, Sumatra)
  X_gd_sat2fine.json       source = C_sat2uav train/val (CAS SAT); test = 0.5 m scenes (Lushan_1, Lushan_2)
  gdcld_meta.json          per-scene GSD, shape, tile counts, landslide fraction
"""
import csv, glob, json, math, os
import numpy as np, cv2, rasterio
from rasterio.windows import Window

HY = "."; RAW = f"{HY}/data/CAS_raw"; GD = f"{HY}/data/GDCLD"
OUT = f"{GD}/test_tiles"; T = 1024
EXCLUDE = ("Palu",)
os.makedirs(f"{OUT}/img", exist_ok=True); os.makedirs(f"{OUT}/mask", exist_ok=True)


def scene_gsd(ds):
    sx = abs(ds.transform.a)
    if ds.crs is not None and ds.crs.is_geographic:
        lat = ds.transform.f + ds.transform.e * ds.height / 2
        return sx * 111320.0 * math.cos(math.radians(lat))
    return sx


rows = list(csv.DictReader(open(f"{HY}/splits/index.csv")))
fields = list(rows[0].keys())
ext, meta = [], {}
imgs = sorted(glob.glob(f"{GD}/test_data/*/*.tif")) or sorted(glob.glob(f"{GD}/test_data/*.tif"))
assert imgs, "no scenes found under data/GDCLD/test_data"
for img in imgs:
    scene = os.path.basename(img)[:-4]
    if scene.startswith(EXCLUDE):
        print("skip (event overlaps CAS):", scene); continue
    lab = img.replace("/test_data/", "/test_label/")
    assert os.path.exists(lab), lab
    with rasterio.open(img) as di, rasterio.open(lab) as dl:
        g = round(scene_gsd(di), 3)
        H, W = di.height, di.width
        assert abs(dl.height - H) <= 2 and abs(dl.width - W) <= 2, (scene, dl.shape, (H, W))
        n = pos = kept = 0
        for y in range(0, H, T):
            for x in range(0, W, T):
                h, w = min(T, H - y), min(T, W - x)
                if h < T // 4 or w < T // 4:
                    continue
                a = di.read([1, 2, 3], window=Window(x, y, w, h)).transpose(1, 2, 0)
                valid = a.max(-1) > 0
                n += 1
                if valid.mean() < 0.10:
                    continue
                m = (dl.read(1, window=Window(x, y, w, h)) > 0).astype(np.uint8)
                if (h, w) != (T, T):
                    a = np.pad(a, ((0, T - h), (0, T - w), (0, 0)), mode="reflect")
                    m = np.pad(m, ((0, T - h), (0, T - w)))
                stem = f"{scene}_{y // T:03d}_{x // T:03d}"
                cv2.imwrite(f"{OUT}/img/{stem}.png", np.ascontiguousarray(a[..., ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 6])
                cv2.imwrite(f"{OUT}/mask/{stem}.png", m, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                event = scene.split("_")[0]
                ext.append({"sid": f"GDCLD_{scene}", "region": f"GDCLD_{event}", "platform": "EXT", "sensor": f"GDCLD_{g}m", "gsd_m": f"{g}",
                            "stem": stem, "img": os.path.relpath(f"{OUT}/img/{stem}.png", RAW), "mask": os.path.relpath(f"{OUT}/mask/{stem}.png", RAW),
                            "ls_frac": f"{m.mean():.5f}", "h": T, "w": T})
                kept += 1; pos += int(m.sum())
        meta[scene] = {"gsd_m": g, "shape": [H, W], "tiles_total": n, "tiles_kept": kept, "ls_frac_kept": pos / max(kept * T * T, 1)}
        print(scene, meta[scene], flush=True)

with open(f"{HY}/splits/index_gdcld.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader()
    for r in rows + ext:
        wr.writerow({k: r.get(k, "") for k in fields})
coarse = [f"{r['sid']}/{r['stem']}" for r in ext if float(r["gsd_m"]) >= 2.0]
fine = [f"{r['sid']}/{r['stem']}" for r in ext if float(r["gsd_m"]) < 2.0]
for name, src_split, test, note in (("X_gd_uav2coarse", "C_uav2sat", coarse, "E9: CAS UAV source -> GDCLD 3 m scenes (Lushan_3, Mesetas, Sumatra), transductive"),
                                    ("X_gd_sat2fine", "C_sat2uav", fine, "E9: CAS SAT source -> GDCLD 0.5 m scenes (Lushan_1, Lushan_2), transductive")):
    src = json.load(open(f"{HY}/splits/{src_split}.json"))
    json.dump({"note": note, "n": {"train": len(src["train"]), "val": len(src["val"]), "test": len(test)},
               "train": src["train"], "val": src["val"], "test": sorted(test)}, open(f"{HY}/splits/{name}.json", "w"))
    print(name, "test tiles", len(test))
json.dump(meta, open(f"{HY}/splits/gdcld_meta.json", "w"), indent=1)
print("tiles", len(ext), "landslide frac", float(np.mean([float(r["ls_frac"]) for r in ext])))
