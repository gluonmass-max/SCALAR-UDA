"""Per-tile predictions of several checkpoints on candidate target tiles (for the qualitative figure, Section 3.9).

Candidates: per target subset, tiles whose reference landslide fraction lies in [--fmin, --fmax]; at most --cand of them
(random.Random(0) order). Every checkpoint is run at its own anchored scale (GSA) and its probability map is restored to
the native grid, as in the test evaluation. Writes
  <out>/tiles.csv                    key, sid, ls_frac, IoU/P/R of every checkpoint at threshold 0.5
  <out>/prob/<name>/<sid>__<stem>.png   probability map * 255 (uint8, native grid)

  python -m cas.dump_tiles --split splits/C_sat2uav.json --index splits/index.csv --raw <CAS_raw> \
      --ckpts GSA=runs/C_sat2uav_M1_g0.5_s0/best.pt,SCALAR=runs/C_sat2uav_M1ST_asym_p05n15_s0_rerun/best.pt \
      --ckpt_scales GSA=0.5,SCALAR=0.5 --out runs/qual_sat2uav_s0
"""
import argparse, csv, json, os, random
from collections import defaultdict
import numpy as np, torch
from PIL import Image
import torch.nn.functional as F
from cas.train import load_index, build_model, CASTiles, predict_logits
from cas.render_qual import parse_kv


def predict(model, key, index, raw, gsd_target, device, wallis=False):
    """Native-grid probability map, as in the test evaluation (GSA resampling, sliding window, logits restored)."""
    ds = CASTiles([key], index, raw, False, gsd_target=gsd_target, gsd_clamp=(0.25, 2.0), wallis=wallis)
    x, y, _ = ds[0]
    with torch.no_grad():
        lg = predict_logits(model, x.unsqueeze(0).to(device), stride=256)
        if lg.shape[-2:] != y.shape[-2:]:
            lg = F.interpolate(lg, y.shape[-2:], mode="bilinear", align_corners=False)
    return torch.sigmoid(lg.float())[0, 0].cpu().numpy(), y.numpy()


def scores(prob, gt, thr=0.5):
    m = gt != 255; p = (prob >= thr) & m; g = (gt == 1) & m
    tp = float((p & g).sum()); fp = float((p & ~g).sum()); fn = float((~p & g).sum())
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    return iou, (tp / (tp + fp) if tp + fp else float("nan")), (tp / (tp + fn) if tp + fn else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--index", default="splits/index.csv")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--ckpts", required=True); ap.add_argument("--ckpt_scales", required=True)
    ap.add_argument("--fmin", type=float, default=0.02); ap.add_argument("--fmax", type=float, default=0.5)
    ap.add_argument("--cand", type=int, default=40); ap.add_argument("--out", required=True)
    ap.add_argument("--wallis", default="", help="comma-separated checkpoint names that use Wallis normalisation (LandsDANet-style)")
    ap.add_argument("--extra", default="", help="comma-separated keys always included (e.g. the Figure 1 / Figure 7 tiles)")
    a = ap.parse_args()
    index = load_index(a.index)
    test = json.load(open(a.split))["test"]
    by_sid = defaultdict(list)
    for k in test:
        r = index[k]
        if a.fmin <= float(r["ls_frac"]) <= a.fmax:
            by_sid[r["sid"]].append(k)
    keys = []
    for sid, ks in sorted(by_sid.items()):
        ks = sorted(ks); random.Random(0).shuffle(ks); keys += ks[:a.cand]
    keys += [k for k in a.extra.split(",") if k and k not in keys and k in index]
    print(f"{len(keys)} candidate tiles from {len(by_sid)} subsets", flush=True)
    ckpts, scales = parse_kv(a.ckpts), parse_kv(a.ckpt_scales, float); wal = set(filter(None, a.wallis.split(",")))
    dev = torch.device("cuda")
    rows = {k: dict(key=k, sid=index[k]["sid"], gsd_m=index[k]["gsd_m"], ls_frac=index[k]["ls_frac"]) for k in keys}
    for name, path in ckpts.items():
        model = build_model("segformer", "mit_b2", weights=None).to(dev)
        sd = torch.load(path, map_location="cpu"); model.load_state_dict(sd.get("model", sd)); model.eval()
        od = os.path.join(a.out, "prob", name); os.makedirs(od, exist_ok=True)
        for i, k in enumerate(keys):
            prob, gt = predict(model, k, index, a.raw, scales[name], dev, wallis=name in wal)   # bf16 autocast inside
            iou, p, r = scores(prob, gt)
            rows[k].update({f"{name}_IoU": round(iou, 4), f"{name}_P": round(p, 4), f"{name}_R": round(r, 4)})
            Image.fromarray((prob * 255).round().astype(np.uint8)).save(os.path.join(od, k.replace("/", "__") + ".png"))
            if i % 100 == 0:
                print(f"  {name} {i}/{len(keys)}", flush=True)
        del model; torch.cuda.empty_cache()
    with open(os.path.join(a.out, "tiles.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values())).keys())); w.writeheader(); w.writerows(rows.values())
    print("wrote", os.path.join(a.out, "tiles.csv"))


if __name__ == "__main__":
    main()
