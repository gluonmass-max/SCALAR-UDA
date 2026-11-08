"""Experiment B (external cross-sensor check): tile the seven GDCLD test scenes, whose GSD IS known from the GeoTIFF
geotransform (unlike the GDCLD val tiles), so that M1 can anchor them per scene.

  Lushan_1/2  0.5 m (UTM)    Palu_1/2  1.0729e-5 deg ~ 1.19 m (EPSG:4326, lat -1.1)    Lushan_3, Mesetas, Sumatra  3.0 m

Writes 1024x1024 non-overlapping tiles (PNG, lossless) for image and 0/1 mask; tiles that are >= 90 % nodata (all-zero
RGB) are skipped. Index rows get sid = GDCLD_<scene>, platform EXT, gsd_m = scene GSD.
Outputs
  splits/index_gdcld_scenes.csv            CAS index + GDCLD scene tiles
  splits/X_uav2gdcldscn.json               source = C_uav2sat train/val (CAS UAV), test = all scene tiles (transductive)
  splits/gdcld_scenes_meta.json            per-scene GSD, tile counts, landslide fraction
"""
import csv, glob, json, math, os
import numpy as np, cv2, rasterio
from rasterio.windows import Window

HY = "."; RAW = "data/CAS_raw"; GD = "data/GDCLD"
OUT = f"{GD}/test_tiles"; T = 1024
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
for img in sorted(glob.glob(f"{GD}/test_data/*/*.tif")):
    scene = os.path.basename(img)[:-4]
    lab = img.replace("/test_data/", "/test_label/")
    with rasterio.open(img) as di, rasterio.open(lab) as dl:
        g = round(scene_gsd(di), 3)
        H, W = di.height, di.width
        assert abs(dl.height - H) <= 2 and abs(dl.width - W) <= 2, (scene, dl.shape, (H, W))  # labels can be 1 px larger; read same pixel window
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
                if (h, w) != (T, T):  # pad edge tiles to a full tile (reflect image, zero mask)
                    a = np.pad(a, ((0, T - h), (0, T - w), (0, 0)), mode="reflect")
                    m = np.pad(m, ((0, T - h), (0, T - w)))
                stem = f"{scene}_{y // T:03d}_{x // T:03d}"
                cv2.imwrite(f"{OUT}/img/{stem}.png", np.ascontiguousarray(a[..., ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 6])
                cv2.imwrite(f"{OUT}/mask/{stem}.png", m, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                ext.append({"sid": f"GDCLD_{scene}", "region": "GDCLD", "platform": "EXT", "sensor": f"GDCLD_{g}m", "gsd_m": f"{g}",
                            "stem": stem, "img": os.path.relpath(f"{OUT}/img/{stem}.png", RAW), "mask": os.path.relpath(f"{OUT}/mask/{stem}.png", RAW),
                            "ls_frac": f"{m.mean():.5f}", "h": T, "w": T})
                kept += 1; pos += int(m.sum())
        meta[scene] = {"gsd_m": g, "shape": [H, W], "tiles_total": n, "tiles_kept": kept, "ls_frac_kept": pos / max(kept * T * T, 1)}
        print(scene, meta[scene], flush=True)

with open(f"{HY}/splits/index_gdcld_scenes.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader()
    for r in rows + ext:
        wr.writerow({k: r.get(k, "") for k in fields})
keys = [f"{r['sid']}/{r['stem']}" for r in ext]
src = json.load(open(f"{HY}/splits/C_uav2sat.json"))
json.dump({"note": "CAS UAV (C_uav2sat source train/val) -> GDCLD test scenes tiled 1024, per-scene GSD from geotransform",
           "n": {"train": len(src["train"]), "val": len(src["val"]), "test": len(keys)},
           "train": src["train"], "val": src["val"], "test": keys}, open(f"{HY}/splits/X_uav2gdcldscn.json", "w"))
json.dump(meta, open(f"{HY}/splits/gdcld_scenes_meta.json", "w"), indent=1)
print("tiles", len(ext), "landslide frac", np.mean([float(r["ls_frac"]) for r in ext]))
