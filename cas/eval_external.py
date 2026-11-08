"""Zero-shot external evaluation of a CAS-trained checkpoint on large landslide rasters (GDCLD test, GVLM).

    python -m cas.eval_external --ckpt runs/A_segformer_b2_s0/best.pt --dataset gdcld_test --out runs/A_segformer_b2_s0/ext_gdcld_test.json
    python -m cas.eval_external --ckpt ... --dataset gvlm

Sliding-window inference (512 window, 256 stride, Gaussian-free mean blending, hflip TTA optional) with rasterio
windows so multi-GB images never sit fully in RAM as float. Metrics per scene and pooled: IoU/F1/P/R at 0.5 and at a
given threshold. GDCLD label rasters may be RGB or 0/255; anything >0 counts as landslide.
GVLM: post-event image img2_HR.tif with label.tif (0/255) for every site directory.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.simplefilter("ignore", NotGeoreferencedWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cas.train import build_model, MEAN, STD, Confusion  # noqa: E402


def scenes(dataset, root="data"):
    out = []
    if dataset == "gdcld_test":
        for img in sorted(glob.glob(f"{root}/GDCLD/test_data/*/*.tif")):
            lab = img.replace("/test_data/", "/test_label/")
            if os.path.exists(lab):
                out.append((os.path.basename(img)[:-4], img, lab, os.path.basename(os.path.dirname(img))))
    elif dataset == "gvlm":
        for d in sorted(glob.glob(f"{root}/GVLM/GVLM_1.1_8bit/*/")):
            img, lab = os.path.join(d, "img2_HR.tif"), os.path.join(d, "label.tif")
            if os.path.exists(img) and os.path.exists(lab):
                out.append((os.path.basename(d.rstrip("/")), img, lab, os.path.basename(d.rstrip("/"))))
    else:
        raise ValueError(dataset)
    return out


@torch.no_grad()
def predict_scene(model, img_path, device, win=512, stride=256, bs=16, tta=False, scale=1.0):
    """returns prob map (H, W) float16 at native resolution. scale != 1 resamples the image before inference
    (e.g. scale=0.5 halves resolution) and upsamples the probability back."""
    with rasterio.open(img_path) as ds:
        H, W = ds.height, ds.width
        h, w = (int(round(H * scale)), int(round(W * scale))) if scale != 1 else (H, W)
        prob = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.uint8)
        ys = list(range(0, max(h - win, 0) + 1, stride)); xs = list(range(0, max(w - win, 0) + 1, stride))
        if ys[-1] + win < h: ys.append(h - win)
        if xs[-1] + win < w: xs.append(w - win)
        tiles, coords = [], []

        def flush():
            if not tiles: return
            x = torch.from_numpy(np.stack(tiles)).to(device).permute(0, 3, 1, 2).float()
            x = (x / 255.0 - torch.tensor(MEAN, device=device)[:, None, None]) / torch.tensor(STD, device=device)[:, None, None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x)
                if tta:
                    lg = (lg + torch.flip(model(torch.flip(x, [-1])), [-1])) / 2
            p = torch.sigmoid(lg.float())[:, 0].cpu().numpy()
            for (yy, xx), pp in zip(coords, p):
                prob[yy:yy + win, xx:xx + win] += pp; cnt[yy:yy + win, xx:xx + win] += 1
            tiles.clear(); coords.clear()

        for y in ys:
            for x in xs:
                if scale == 1:
                    a = ds.read([1, 2, 3], window=Window(x, y, min(win, w - x), min(win, h - y)))
                else:  # read the corresponding native window, resample to win
                    Y, X = int(y / scale), int(x / scale); WH = int(win / scale)
                    a = ds.read([1, 2, 3], window=Window(X, Y, min(WH, W - X), min(WH, H - Y)), out_shape=(3, min(win, h - y), min(win, w - x)))
                a = a.transpose(1, 2, 0).astype(np.uint8)
                if a.shape[0] != win or a.shape[1] != win:
                    a = np.pad(a, ((0, win - a.shape[0]), (0, win - a.shape[1]), (0, 0)), mode="reflect")
                tiles.append(a); coords.append((y, x))
                if len(tiles) == bs: flush()
        flush()
        prob /= np.maximum(cnt, 1)
        if scale != 1:
            prob = torch.nn.functional.interpolate(torch.from_numpy(prob)[None, None], (H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
        return prob.astype(np.float16)


def read_label(lab_path, H, W):
    with rasterio.open(lab_path) as ds:
        a = ds.read(1, out_shape=(H, W)) if (ds.height, ds.width) != (H, W) else ds.read(1)
    return a > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, choices=["gdcld_test", "gvlm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default=None); ap.add_argument("--encoder", default=None)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--save_prob_dir", default="")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    arch = a.arch or ck["args"]["arch"]; enc = a.encoder or ck["args"]["encoder"]
    device = torch.device("cuda")
    model = build_model(arch, enc, weights=None); model.load_state_dict(ck["model"]); model.to(device).eval()
    conf = {t: Confusion() for t in sorted({0.5, a.thr})}
    per_scene = {}
    for name, img, lab, group in scenes(a.dataset):
        t0 = time.time()
        prob = predict_scene(model, img, device, stride=a.stride, tta=a.tta, scale=a.scale)
        gt = read_label(lab, *prob.shape)
        valid = np.ones_like(gt)  # could mask nodata here
        for t, c in conf.items():
            c.add((prob > t) & valid, gt & valid, (f"event:{group}",))
        sc = Confusion(); sc.add(prob > a.thr, gt, ())
        per_scene[name] = {**sc.report()["ALL"], "shape": list(prob.shape), "gt_frac": float(gt.mean()), "sec": round(time.time() - t0, 1)}
        print(f"{name:28s} {prob.shape} gt {gt.mean()*100:5.2f}%  IoU {per_scene[name]['IoU']:.4f} F1 {per_scene[name]['F1']:.4f} P {per_scene[name]['Precision']:.4f} R {per_scene[name]['Recall']:.4f}  {per_scene[name]['sec']}s", flush=True)
        if a.save_prob_dir:
            os.makedirs(a.save_prob_dir, exist_ok=True); np.save(os.path.join(a.save_prob_dir, name + ".npy"), prob)
        del prob, gt
    pooled = {str(t): c.report() for t, c in conf.items()}
    res = {"ckpt": a.ckpt, "dataset": a.dataset, "scale": a.scale, "stride": a.stride, "tta": a.tta, "thr": a.thr,
           "per_scene": per_scene, "pooled": pooled,
           "scene_mean": {k: float(np.mean([v[k] for v in per_scene.values()])) for k in ("IoU", "F1", "Precision", "Recall")}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    P = pooled[str(a.thr)]["ALL"]
    print(f"POOLED@{a.thr}  IoU {P['IoU']:.4f}  F1 {P['F1']:.4f}  P {P['Precision']:.4f}  R {P['Recall']:.4f}   scene-mean IoU {res['scene_mean']['IoU']:.4f}")


if __name__ == "__main__":
    main()
