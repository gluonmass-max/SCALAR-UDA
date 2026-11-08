"""F5 qualitative figure: image | ground truth | source-only | +M1 | SACS (| oracle) for selected target tiles.

Tile selection (fig_captions.md): per subset, tiles with landslide fraction in [0.05, 0.4] from the target test set of the
split; score them with the SACS checkpoint, take the tile at the median IoU and the tile at the 10th percentile (failure).
Predictions of every model are thresholded at 0.5 except SACS, which uses --sacs_thr (deployment θ_dep).

Usage (GPU, ~3 min):
  python -m cas.render_qual --split splits/C_uav2sat.json --subsets Hokkaido,Palu --gsd_target 1.0 \
      --ckpts source=runs/C_uav2sat_segformer_b2_s0/best.pt,M1=runs/C_uav2sat_M1_g1.0_s0/best.pt,SACS=runs/C_uav2sat_M1ST_otsuM_s0/best.pt \
      --ckpt_scales source=0,M1=1.0,SACS=1.0 --sacs_thr 0.40 --out paper/figs/F5_uav2sat.png --tiles_csv paper/collected/F5_tiles.csv
For GDCLD use --index splits/index_ext.csv --subsets GDCLD_val (sid as in index) and --tgt_scale if needed.
"""
import argparse, csv, json, os, random
import numpy as np, torch
import torch.nn.functional as F
from cas.train import load_index, CASTiles, build_model, predict_logits, MEAN, STD

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_kv(s, cast=str):
    return {k: cast(v) for k, v in (x.split("=") for x in s.split(","))} if s else {}


def predict(model, key, index, raw, gsd_target, tgt_scale, device):
    ds = CASTiles([key], index, raw, False, gsd_target=gsd_target, gsd_clamp=(0.25, 2.0), scale_mult=tgt_scale)
    x, y, _ = ds[0]
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        lg = predict_logits(model, x, stride=256)
        if lg.shape[-2:] != y.shape[-2:]:
            lg = F.interpolate(lg, y.shape[-2:], mode="bilinear", align_corners=False)
    return torch.sigmoid(lg.float())[0, 0].cpu().numpy(), y.numpy()


def iou(pred, gt):
    m = gt != 255
    p, g = pred[m], gt[m] == 1
    inter = (p & g).sum(); uni = (p | g).sum()
    return inter / uni if uni else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--index", default="splits/index.csv")
    ap.add_argument("--raw", default="data/CAS_raw")
    ap.add_argument("--subsets", required=True, help="comma-separated sid values")
    ap.add_argument("--ckpts", required=True, help="name=path,... (order = column order)")
    ap.add_argument("--ckpt_scales", required=True, help="name=gsd_target,... (0 = native)")
    ap.add_argument("--tgt_scale", type=float, default=1.0)
    ap.add_argument("--sacs_thr", type=float, default=0.5); ap.add_argument("--sacs_name", default="SACS")
    ap.add_argument("--per_subset", type=int, default=2); ap.add_argument("--cand", type=int, default=60)
    ap.add_argument("--out", required=True); ap.add_argument("--tiles_csv", default="")
    ap.add_argument("--only_keys", default="", help="skip selection; render these keys (comma-separated)")
    a = ap.parse_args()
    device = torch.device("cuda"); random.seed(0)
    index = load_index(a.index); sp = json.load(open(a.split))
    test = [k for k in sp["test"] if k in index]
    ckpts = parse_kv(a.ckpts); scales = parse_kv(a.ckpt_scales, float)
    models = {}
    for n, p in ckpts.items():
        m = build_model("segformer", "mit_b2").to(device).eval()
        m.load_state_dict(torch.load(p, map_location="cpu")["model"]); models[n] = m
    sacs = models[a.sacs_name]
    chosen = []
    if a.only_keys:
        chosen = [(k, float("nan")) for k in a.only_keys.split(",")]
    else:
        for sid in a.subsets.split(","):
            cands = [k for k in test if index[k]["sid"] == sid and 0.05 <= float(index[k]["ls_frac"]) <= 0.4]
            random.shuffle(cands); cands = cands[:a.cand]
            scored = []
            for k in cands:
                p, y = predict(sacs, k, index, a.raw, scales[a.sacs_name], a.tgt_scale, device)
                scored.append((iou(p > a.sacs_thr, y), k))
            scored = [s for s in scored if s[0] == s[0]]; scored.sort()
            if not scored:
                continue
            med = scored[len(scored) // 2]; low = scored[max(0, int(0.1 * len(scored)))]
            chosen += [(med[1], med[0]), (low[1], low[0])][: a.per_subset]
            print(f"{sid}: {len(scored)} candidates, median IoU {med[0]:.3f} ({med[1]}), p10 {low[0]:.3f} ({low[1]})", flush=True)
    if a.tiles_csv:
        os.makedirs(os.path.dirname(a.tiles_csv), exist_ok=True)
        with open(a.tiles_csv, "a", newline="") as f:
            w = csv.writer(f)
            for k, s in chosen:
                w.writerow([os.path.basename(a.split), index[k]["sid"], k, f"{s:.3f}"])
    # render
    cols = ["image", "ground truth"] + list(ckpts)
    rows = []
    for k, _ in chosen:
        r = index[k]
        ds = CASTiles([k], index, a.raw, False)
        x, y, _ = ds[0]
        img = (x.numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)).clip(0, 1)
        panels = [img, y.numpy()]
        for n in ckpts:
            p, _ = predict(models[n], k, index, a.raw, scales[n], a.tgt_scale, device)
            thr = a.sacs_thr if n == a.sacs_name else 0.5
            panels.append((p > thr).astype(np.uint8))
        rows.append((f"{r['sid']} {r['gsd_m']} m", panels))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if plt is None:
        print("matplotlib missing; cannot render"); return
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(1.6 * len(cols), 1.6 * len(rows)))
    ax = np.atleast_2d(ax)
    for i, (lab, panels) in enumerate(rows):
        for j, p in enumerate(panels):
            if j == 0:
                ax[i, j].imshow(p)
            else:
                m = np.asarray(p).astype(float); m[m == 255] = np.nan
                ax[i, j].imshow(panels[0]); ax[i, j].imshow(m, cmap="autumn", alpha=0.55, vmin=0, vmax=1)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=8)
        ax[i, 0].set_ylabel(lab, fontsize=7)
    fig.tight_layout(pad=0.2); fig.savefig(a.out, dpi=200); fig.savefig(a.out.replace(".png", ".pdf"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
