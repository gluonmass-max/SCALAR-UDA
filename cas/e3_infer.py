"""E3 (review response, 2026-10-20): one inference pass per checkpoint that produces everything the threshold-free and
object-level analyses need. No training; target labels are used for SCORING ONLY.

  python -m cas.e3_infer --ckpt runs/C_sat2uav_M1_g0.5_s0/best.pt --split splits/C_sat2uav.json --gsd_target 0.5 \
      --out runs/E3/sat2uav_M1_s0 [--pool_split splits/C_xLMW_sat2uav.json] [--max_eval 50]

Per tile (native grid, GSA resampling + sliding window exactly as in the test evaluation):
  * pixel counts tp/fp/fn at three label-free thresholds: 0.5 (deployed), the logit-Otsu threshold of the unlabelled
    adaptation pool (inference-only LOC), and the EM prior-matched quantile threshold (Saerens et al.);
  * connected components of the reference (8-connectivity): for each object its area (px and m^2) and the fraction of
    its pixels predicted positive at each threshold (omission analysis, size stratification);
  * predicted components at each threshold and whether they touch / are half-covered by reference pixels (commission);
  * probability histograms (1024 bins) of reference-positive and reference-negative pixels, per subset (AP / PR curves).
Outputs: <out>/summary.json, tiles.csv, objects.npz (per reference object), hist.npz, pool.json (unlabelled statistics).
"""
import argparse, csv, json, os, random, sys, time
from collections import defaultdict
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cas.train import load_index, build_model, CASTiles, predict_logits, Confusion  # noqa: E402
from cas.train_uda import otsu_logit_threshold  # noqa: E402

NB = 1024
S8 = np.ones((3, 3), np.int32)
SIZE_EDGES = [0, 500, 2000, 10000, 50000, float("inf")]   # m^2 classes for the default summary


def ap_from_hist(hp, hn):
    """Average precision from histograms of positive / negative pixel scores (threshold descending)."""
    tp = np.cumsum(hp[::-1])[::-1]; fp = np.cumsum(hn[::-1])[::-1]       # pixels with score >= bin lower edge
    P = hp.sum()
    if P == 0:
        return float("nan"), float("nan")
    rec = tp / P; prec = tp / np.maximum(tp + fp, 1)
    order = np.argsort(rec); rec, prec = rec[order], prec[order]
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))               # step-wise AP
    env = np.maximum.accumulate(prec[::-1])[::-1]                                    # VOC-style envelope
    ap_env = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * env))
    return ap, ap_env


def components(mask):
    lab, n = ndimage.label(mask, structure=S8)
    return lab, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--index", default="splits/index.csv"); ap.add_argument("--raw", default="data/CAS_raw")
    ap.add_argument("--arch", default="segformer"); ap.add_argument("--encoder", default="mit_b2")
    ap.add_argument("--gsd_target", type=float, required=True); ap.add_argument("--gsd_clamp", default="0.25,2.0")
    ap.add_argument("--wallis", type=int, default=0)
    ap.add_argument("--pool_split", default="", help="split whose TEST keys form the unlabelled adaptation pool (default: --split)")
    ap.add_argument("--pool", type=int, default=512, help="pool tiles for the label-free thresholds (train_uda deployment calibration uses 512)")
    ap.add_argument("--max_eval", type=int, default=0); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    index = load_index(a.index)
    sp = json.load(open(a.split))
    keys = [k for k in sp["test"] if k in index]
    pool_keys = [k for k in json.load(open(a.pool_split))["test"] if k in index] if a.pool_split else list(keys)
    if a.max_eval:
        random.Random(0).shuffle(keys); keys = keys[:a.max_eval]
    tr = [k for k in sp["train"] if k in index]
    p_src = float(sum(float(index[k]["ls_frac"]) * int(index[k]["h"]) * int(index[k]["w"]) for k in tr) /
                  max(1, sum(int(index[k]["h"]) * int(index[k]["w"]) for k in tr)))
    clamp = tuple(float(x) for x in a.gsd_clamp.split(","))
    ds_kw = dict(gsd_target=a.gsd_target, gsd_clamp=clamp, wallis=bool(a.wallis))
    resampled = a.gsd_target > 0
    dev = torch.device("cuda")
    model = build_model(a.arch, a.encoder, weights=None).to(dev)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False); model.load_state_dict(sd.get("model", sd)); model.eval()

    @torch.no_grad()
    def run(dl):
        for x, y, idx in dl:
            x = x.to(dev, non_blocking=True)
            H, W = y.shape[-2:]
            if resampled:
                lg = predict_logits(model, x, stride=256)
                if lg.shape[-2:] != (H, W):
                    lg = F.interpolate(lg, (H, W), mode="bilinear", align_corners=False)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(x)
            yield int(idx[0]), torch.sigmoid(lg.float())[0, 0].cpu().numpy(), y[0].numpy()

    # ---- unlabelled pool statistics and label-free thresholds (same pool order as training: random.Random(1))
    sub = list(pool_keys); random.Random(1).shuffle(sub); sub = sub[:a.pool]
    pool_dl = DataLoader(CASTiles(sub, index, a.raw, False, **ds_kw), batch_size=1, shuffle=False, num_workers=a.workers)
    hist_pool = np.zeros(NB, np.int64); n_pool_px = 0; s1 = 0.0
    probs = []
    for _, prob, _ in run(pool_dl):
        hist_pool += np.bincount(np.minimum((prob * NB).astype(np.int32), NB - 1).ravel(), minlength=NB)
        n_pool_px += prob.size; s1 += float(prob.sum())
        probs.append(torch.from_numpy(prob.flatten()))
    p = torch.cat(probs)
    if p.numel() > 4_000_000:
        p = p[torch.randperm(p.numel())[:4_000_000]]
    p = p.clamp(1e-4, 1 - 1e-4)
    pmean = float(p.mean()); pi = pmean
    for _ in range(30):
        w1 = (pi / p_src) * p; w0 = ((1 - pi) / (1 - p_src)) * (1 - p)
        pi_new = float((w1 / (w1 + w0)).mean())
        if abs(pi_new - pi) < 1e-5:
            pi = pi_new; break
        pi = pi_new
    pi = float(min(max(pi, 1e-4), 0.9))
    thr_prior = float(min(max(float(torch.quantile(p, 1 - pi)), 0.05), 0.95))
    thr_otsu = otsu_logit_threshold(p)
    if thr_otsu is None:   # degenerate pool histogram: fall back to the uncalibrated boundary
        thr_otsu = 0.5
    lg = torch.log(p / (1 - p)); mu = lg.mean(); vt = lg.var()
    m = lg > float(torch.log(torch.tensor(thr_otsu / (1 - thr_otsu))))
    w = float(m.float().mean()); vb = w * (1 - w) * float((lg[m].mean() - lg[~m].mean()) ** 2) if 0 < w < 1 else 0.0
    pool_stats = dict(n_tiles=len(sub), n_px=n_pool_px, mean_prob=pmean, p_src=p_src, pi_em=pi, thr_prior=thr_prior,
                      thr_otsu=thr_otsu, frac_above_otsu=w, otsu_eta=float(vb / max(float(vt), 1e-12)),
                      hist=hist_pool.tolist())
    json.dump(pool_stats, open(os.path.join(a.out, "pool.json"), "w"))
    thrs = {"t05": 0.5, "otsu": thr_otsu, "prior": thr_prior}
    print(f"pool {len(sub)} tiles: mean prob {pmean:.4f}  p_src {p_src:.4f}  pi_em {pi:.4f}  thr_prior {thr_prior:.3f}  thr_otsu {thr_otsu:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- scored pass over the evaluation tiles
    dl = DataLoader(CASTiles(keys, index, a.raw, False, **ds_kw), batch_size=1, shuffle=False, num_workers=a.workers)
    pix = defaultdict(lambda: np.zeros(4, np.int64))   # per (thr, group): tp, fp, fn, tn
    hist_pos = defaultdict(lambda: np.zeros(NB, np.int64)); hist_neg = defaultdict(lambda: np.zeros(NB, np.int64))
    obj_rows = []      # per reference object: tile_i, obj_id, area_px, area_m2, cov_t05, cov_otsu, cov_prior
    tile_rows = []
    oc = defaultdict(lambda: np.zeros(6, np.int64))   # per (thr, group): n_gt, det_any, det_50, n_pred, comm_any, comm_50
    for n_done, (i, prob, gt) in enumerate(run(dl)):
        r = index[keys[i]]; g = float(r["gsd_m"]); grp = ("ALL", f"sid:{r['sid']}", f"region:{r['region']}", f"platform:{r['platform']}")
        valid = gt != 255; pos = (gt == 1) & valid
        bidx = np.minimum((prob * NB).astype(np.int32), NB - 1)
        hp = np.bincount(bidx[pos], minlength=NB); hn = np.bincount(bidx[valid & ~pos], minlength=NB)
        hist_pos[r["sid"]] += hp; hist_neg[r["sid"]] += hn
        glab, ng = components(pos)
        garea = np.bincount(glab.ravel(), minlength=ng + 1)[1:] if ng else np.zeros(0, np.int64)
        row = dict(key=keys[i], sid=r["sid"], region=r["region"], platform=r["platform"], gsd_m=g, n_valid=int(valid.sum()), gt_pos=int(pos.sum()), n_gt_obj=int(ng))
        covs = {}
        posv = pos[valid].astype(np.int8)
        for name, t in thrs.items():
            pred = (prob >= t) & valid
            code = np.bincount(pred[valid].astype(np.int8) * 2 + posv, minlength=4)   # 0 tn, 1 fn, 2 fp, 3 tp
            tp, fp, fn, tn = int(code[3]), int(code[2]), int(code[1]), int(code[0])
            v4 = np.array([tp, fp, fn, tn], np.int64)
            row.update({f"tp_{name}": tp, f"fp_{name}": fp, f"fn_{name}": fn})
            if ng:
                inter = np.bincount(glab[pred].ravel(), minlength=ng + 1)[1:]
                cov = inter / garea
            else:
                cov = np.zeros(0)
            covs[name] = cov
            plab, npred = components(pred)
            if npred:
                parea = np.bincount(plab.ravel(), minlength=npred + 1)[1:]
                pint = np.bincount(plab[pos].ravel(), minlength=npred + 1)[1:]
                comm_any = int((pint == 0).sum()); comm_50 = int((pint < 0.5 * parea).sum())
            else:
                comm_any = comm_50 = 0
            det_any = int((cov > 0).sum()); det_50 = int((cov >= 0.5).sum())
            row.update({f"det_any_{name}": det_any, f"det_50_{name}": det_50, f"n_pred_{name}": int(npred), f"comm_any_{name}": comm_any, f"comm_50_{name}": comm_50})
            v = np.array([ng, det_any, det_50, npred, comm_any, comm_50], np.int64)
            for gname in grp:
                oc[(name, gname)] += v; pix[(name, gname)] += v4
        for j in range(ng):
            obj_rows.append((i, j + 1, int(garea[j]), float(garea[j]) * g * g, float(covs["t05"][j]), float(covs["otsu"][j]), float(covs["prior"][j])))
        tile_rows.append(row)
        if n_done % 1000 == 0:
            print(f"  {n_done}/{len(keys)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- outputs
    with open(os.path.join(a.out, "tiles.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(tile_rows[0].keys())); wr.writeheader(); wr.writerows(tile_rows)
    ob = np.array(obj_rows, np.float64) if obj_rows else np.zeros((0, 7))
    np.savez_compressed(os.path.join(a.out, "objects.npz"), tile_i=ob[:, 0].astype(np.int32), obj_id=ob[:, 1].astype(np.int32),
                        area_px=ob[:, 2].astype(np.int32), area_m2=ob[:, 3].astype(np.float32),
                        cov_t05=ob[:, 4].astype(np.float16), cov_otsu=ob[:, 5].astype(np.float16), cov_prior=ob[:, 6].astype(np.float16),
                        keys=np.array(keys))
    sids = sorted(hist_pos)
    np.savez_compressed(os.path.join(a.out, "hist.npz"), sids=np.array(sids), pos=np.stack([hist_pos[s] for s in sids]) if sids else np.zeros((0, NB)),
                        neg=np.stack([hist_neg[s] for s in sids]) if sids else np.zeros((0, NB)), pool=hist_pool)
    # AP: ALL, per sid, per region, per platform
    groups = {"ALL": sids}
    for s in sids:
        r = next(index[k] for k in keys if index[k]["sid"] == s)
        groups.setdefault(f"region:{r['region']}", []).append(s); groups.setdefault(f"platform:{r['platform']}", []).append(s); groups[f"sid:{s}"] = [s]
    ap_out = {}
    for gname, ss in groups.items():
        hp = sum(hist_pos[s] for s in ss); hn = sum(hist_neg[s] for s in ss)
        apv, ape = ap_from_hist(hp, hn); ap_out[gname] = dict(AP=apv, AP_env=ape, pos_px=int(hp.sum()), neg_px=int(hn.sum()))
    # object summary per threshold / group, plus size classes (ALL only) from the object table
    obj_sum = {}
    for name in thrs:
        d = {}
        for (tn, gname), v in oc.items():
            if tn != name:
                continue
            n_gt, det_any, det_50, npred, comm_any, comm_50 = [int(x) for x in v]
            d[gname] = dict(n_gt=n_gt, det_any=det_any, det_50=det_50, omission_any=1 - det_any / max(n_gt, 1), omission_50=1 - det_50 / max(n_gt, 1),
                            n_pred=npred, comm_any=comm_any, comm_50=comm_50, commission_any=comm_any / max(npred, 1), commission_50=comm_50 / max(npred, 1))
        if len(ob):
            cov = ob[:, {"t05": 4, "otsu": 5, "prior": 6}[name]]; area = ob[:, 3]
            cls = {}
            for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:]):
                m = (area >= lo) & (area < hi)
                cls[f"{lo:g}-{hi:g}"] = dict(n=int(m.sum()), det_any=int((cov[m] > 0).sum()), det_50=int((cov[m] >= 0.5).sum()),
                                           omission_any=float(1 - (cov[m] > 0).mean()) if m.any() else float("nan"),
                                           omission_50=float(1 - (cov[m] >= 0.5).mean()) if m.any() else float("nan"))
            d["size_classes_m2"] = cls
        obj_sum[name] = d
    summary = dict(ckpt=a.ckpt, split=a.split, pool_split=a.pool_split or a.split, n_tiles=len(keys), gsd_target=a.gsd_target, gsd_clamp=a.gsd_clamp,
                   thresholds=thrs, pool={k: v for k, v in pool_stats.items() if k != "hist"},
                   pixel={name: {gname: Confusion.summarize(v) for (tn_, gname), v in pix.items() if tn_ == name} for name in thrs}, AP=ap_out, objects=obj_sum, seconds=round(time.time() - t0))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    A = summary["pixel"]
    print("E3  " + "  ".join(f"{n}@{thrs[n]:.3f}: IoU {A[n]['ALL']['IoU']:.4f} P {A[n]['ALL']['Precision']:.4f} R {A[n]['ALL']['Recall']:.4f}" for n in thrs))
    o = obj_sum["t05"]["ALL"]
    print(f"    AP {ap_out['ALL']['AP']:.4f} (env {ap_out['ALL']['AP_env']:.4f})  objects@0.5: n_gt {o['n_gt']} omission_any {o['omission_any']:.3f} omission_50 {o['omission_50']:.3f} commission_any {o['commission_any']:.3f}  ({summary['seconds']}s)")
    print("DONE")


if __name__ == "__main__":
    main()
