"""Unsupervised cross-sensor adaptation on CAS: scale anchoring (GSA) + label-free recalibration (LOC), optional RSR.

    python -m cas.train_uda --split splits/C_uav2sat.json --index splits/index.csv --raw data/CAS_raw --runs runs \
        --name C_uav2sat_otsuM_s0 --init runs/C_uav2sat_M1_g1.0_s0/best.pt --gsd_target 1.0 --gsd_clamp 0.25,2.0 \
        --pl_mode otsu --pl_gate margin --pl_margin 1.0 --prior_pool 256 --epochs 10 --seed 0

Reference loop (always on): EMA teacher pseudo-labels the target weak view, ClassMix pastes labelled source landslide
pixels into the target tile, the student is trained on the strongly augmented mix plus the plain source batch
(BCE + Dice). With --pl_mode fixed (threshold 0.5) and the absolute confidence gate this is the DACS-style row.

  GSA  --gsd_target g* --gsd_clamp 0.25,2.0   every source and target tile is resampled by clip(gsd_tile / g*, 1/4, 2)
                                              before the network (training, adaptation and inference; predictions are
                                              restored to the native grid before thresholding). g* is the source GSD
                                              nearest to the target median GSD: 1.0 m for UAV->SAT, 0.5 m for SAT->UAV.
  LOC  --pl_mode otsu --pl_gate margin        the pseudo-label boundary is the Otsu split of the teacher's target
       --pl_margin 1.0 --prior_pool 256       LOGIT histogram, re-estimated at every epoch start on a fixed pool of
                                              256 unlabelled target tiles; pixels are retained when their logit lies
                                              at least delta = 1 from that boundary (symmetric margins). If no valid
                                              two-group split exists, otsu_logit_threshold returns None and the
                                              target pseudo-label loss is skipped for that epoch.
  RSR  --pl_margin_pos 1.5 --pl_margin_neg 0.5  optional sign-asymmetric margins chosen by sign(log r), r = target
       (r > 1)  /  0.5, 1.5 (r < 1)           median GSD / g*; the sum of the two widths stays 2*delta. Not part of
                                              the default configuration; reported as an exploratory extension.

Comparators share the same loop and anchored initialisation: --adv_w/--ent_w (AdvEnt + MinEnt), --cbst_p0/--cbst_dp
(CBST-style class-wise quantiles, no ClassMix), --rcs_alpha (DAFormer-style rare-class sampling), --mic_ratio/--mic_w/
--mic_patch (MIC-style masked consistency), --wallis/--contrast_w/--contrast_tau (LandsDANet-style), --pl_mode
flex|free with --ada_m/--flex_warmup (FlexMatch-/FreeMatch-style gates), --otsu_axis prob (probability-axis control),
--pl_mode prior (EM prior-matched quantile control). FDA is trained in cas.train (--fam_*). Other flags are
development-stage variants that are not used in the paper's reported configurations.

--uda_mode transductive: adaptation pool = all target tiles of the split; inductive: per-subset 50/50 split into a
disjoint adaptation pool and test half. Checkpoints are selected on the SOURCE validation split; target labels are
used only to compute the metrics written to runs/<name>/metrics_test.json.
"""
import argparse, copy, json, math, os, random, sys, time
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cas.train import (load_index, CASTiles, build_model, BCEDice, evaluate, finer_view, predict_logits, Confusion,
                       IGNORE, MEAN, STD)  # noqa: E402


def strong_photometric(x):
    """x: normalised (B,3,H,W). Colour jitter + gaussian blur in normalised space, per sample."""
    B = x.shape[0]
    dev = x.device
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    img = x * std + mean
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
    s = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.4)
    img = img * b
    m = img.mean(dim=(1, 2, 3), keepdim=True); img = (img - m) * c + m
    g = img.mean(dim=1, keepdim=True); img = (img - g) * s + g
    hue = torch.empty(B, 3, 1, 1, device=dev).uniform_(-0.05, 0.05); img = img + hue
    blur = torch.rand(B, device=dev) < 0.5
    if blur.any():
        k = 5; sig = 1.0
        ax = torch.arange(k, device=dev, dtype=torch.float32) - k // 2
        g1 = torch.exp(-ax ** 2 / (2 * sig ** 2)); g1 = g1 / g1.sum()
        ker = (g1[:, None] * g1[None, :]).expand(3, 1, k, k)
        bl = F.conv2d(F.pad(img, (k // 2,) * 4, mode="reflect"), ker, groups=3)
        img = torch.where(blur.view(B, 1, 1, 1), bl, img)
    img = img.clamp(0, 1)
    return (img - mean) / std


@torch.no_grad()
def estimate_prior_and_threshold(teacher, raw, ds_kw, index, keys, device, n_pool, bs, workers, resampled, stride, p_src, iters=30, return_p=False, input_scale=1.0):
    """M3. (1) Saerens-Latinne-Decaestecker EM: the teacher's posteriors are calibrated for the source prior p_src;
    iterate the target prior pi until the re-weighted posteriors average to pi. (2) Threshold on the raw teacher
    probability whose positive fraction equals pi. Uses unlabelled target tiles only."""
    sub = list(keys); random.Random(1).shuffle(sub); sub = sub[:n_pool]
    dl = DataLoader(CASTiles(sub, index, raw, False, **ds_kw), batch_size=1 if resampled else bs, shuffle=False, num_workers=workers)
    probs = []
    teacher.eval()
    for x, y, _ in dl:
        x = x.to(device, non_blocking=True)
        if input_scale != 1.0:  # calibrate the teacher at the scale it will actually be run at (candidate 6)
            x = F.interpolate(x, scale_factor=input_scale, mode="bilinear", align_corners=False)
        if resampled:
            H, W = y.shape[-2:]
            lg = predict_logits(teacher, x, stride=stride)
            if lg.shape[-2:] != (H, W):
                lg = F.interpolate(lg, (H, W), mode="bilinear", align_corners=False)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = teacher(x)
        probs.append(torch.sigmoid(lg.float())[:, 0].flatten().cpu())
    p = torch.cat(probs)
    if p.numel() > 4_000_000:
        p = p[torch.randperm(p.numel())[:4_000_000]]
    p = p.clamp(1e-4, 1 - 1e-4)
    pmean = float(p.mean())
    pi = pmean
    for _ in range(iters):
        w1 = (pi / p_src) * p; w0 = ((1 - pi) / (1 - p_src)) * (1 - p)
        pi_new = float((w1 / (w1 + w0)).mean())
        if abs(pi_new - pi) < 1e-5:
            pi = pi_new; break
        pi = pi_new
    pi = float(min(max(pi, 1e-4), 0.9))
    thr = float(torch.quantile(p, 1 - pi))
    thr = float(min(max(thr, 0.05), 0.95))
    if return_p:
        return pi, thr, pmean, (otsu_prob_threshold(p) if OTSU_AXIS[0] == "prob" else otsu_logit_threshold(p)), p
    return pi, thr, pmean, (otsu_prob_threshold(p) if OTSU_AXIS[0] == "prob" else otsu_logit_threshold(p))


def stratum_of(r, mode):
    """M3 (stratified calibration): metadata key of a target tile. gsd_band: <=1 m / 1-3 m / >3 m; sid: subset id."""
    if mode == "sid":
        return str(r["sid"])
    g = float(r["gsd_m"])
    return "le1" if g <= 1.0 else ("1to3" if g <= 3.0 else "gt3")


def stratified_thresholds(teacher, raw, tg_kw, index, adapt_keys, device, mode, n_pool, bs, workers, resampled, stride, p_src,
                          thr_global, fallback_logit, min_pos, log):
    """Per-stratum logit-Otsu thresholds with a label-free fallback to the global threshold when the stratum estimate is
    implausible (too far from the global boundary or almost no positive mode)."""
    strata = {}
    for k in adapt_keys:
        strata.setdefault(stratum_of(index[k], mode), []).append(k)
    lg = lambda t: math.log(t / (1 - t))
    out = {}
    for sname, keys in sorted(strata.items()):
        _, _, pmean_s, thr_s, p_s = estimate_prior_and_threshold(teacher, raw, tg_kw, index, keys, device, min(n_pool, len(keys)), bs, workers, resampled, stride, p_src, return_p=True)
        if thr_s is None:
            thr_s = thr_global
        pih = float((p_s > thr_s).float().mean())
        used = thr_s; why = "ok"
        if abs(lg(thr_s) - lg(thr_global)) > fallback_logit:
            used = thr_global; why = "fallback:far"
        elif pih < min_pos:
            used = thr_global; why = "fallback:no-positive-mode"
        out[sname] = used
        log(f"    [strata:{mode}] {sname:16s} n {len(keys):5d} pool {min(n_pool, len(keys)):4d} mean {pmean_s:.4f} otsu {thr_s:.3f} pi_hat {pih:.4f} -> {used:.3f} ({why})")
    return out


def cbst_thresholds(p, frac):
    """CBST (Zou et al., ECCV 2018), binary form: per class keep the top `frac` most confident target pixels of that class
    (class decided by argmax, i.e. p > 0.5) as pseudo-labels. Returns (thr_pos on p, thr_neg on 1-p)."""
    pos = p[p > 0.5]; neg = 1.0 - p[p <= 0.5]
    tp = float(torch.quantile(pos, 1 - frac)) if pos.numel() > 100 else 0.5
    tn = float(torch.quantile(neg, 1 - frac)) if neg.numel() > 100 else 0.5
    return max(tp, 0.5), max(tn, 0.5)


class EntropyDiscriminator(nn.Module):
    """AdvEnt / AdaptSegNet-style FCN discriminator on the weighted self-information map (2 channels for binary)."""
    def __init__(self, ch=2, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 2, 1))

    def forward(self, x):
        return self.net(x)


def proto_contrast(model, x, y, tau=0.1, n_per_class=4096):
    """Supervised pixel-to-prototype InfoNCE on the decoder output (binary: two prototypes from the batch)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feats = model.decoder(model.encoder(x))
    f = F.normalize(feats.float(), dim=1)                       # B,C,h,w
    yl = F.interpolate(y[:, None].float(), f.shape[-2:], mode="nearest")[:, 0].long()
    f = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]); yl = yl.reshape(-1)
    valid = yl != IGNORE
    f, yl = f[valid], yl[valid]
    pos, neg = f[yl == 1], f[yl == 0]
    if pos.shape[0] < 16 or neg.shape[0] < 16:
        return torch.zeros((), device=x.device)
    protos = F.normalize(torch.stack([neg.mean(0), pos.mean(0)]), dim=1)   # 2,C
    idx_p = torch.randperm(pos.shape[0], device=x.device)[:n_per_class]; idx_n = torch.randperm(neg.shape[0], device=x.device)[:n_per_class]
    emb = torch.cat([pos[idx_p], neg[idx_n]]); lab = torch.cat([torch.ones(idx_p.numel(), dtype=torch.long, device=x.device), torch.zeros(idx_n.numel(), dtype=torch.long, device=x.device)])
    return F.cross_entropy(emb @ protos.t() / tau, lab)


def self_information(logit):
    """weighted self-information map I = -p log p over the two classes (AdvEnt), from the landslide logit."""
    p1 = torch.sigmoid(logit.float()).clamp(1e-6, 1 - 1e-6)
    p = torch.stack([p1, 1 - p1], 1)
    return -p * torch.log(p) / math.log(2)


OTSU_AXIS = ["logit"]   # set from --otsu_axis; module-level so the pool estimator needs no extra argument


def otsu_prob_threshold(p, bins=1000):
    """Ablation control for M2: the same Otsu criterion applied on the PROBABILITY axis instead of the logit axis."""
    p = p.clamp(1e-3, 1 - 1e-3)
    h = torch.histc(p, bins=bins, min=0.0, max=1.0).double()
    c = torch.linspace(0.0, 1.0, bins, dtype=torch.float64)
    tot = h.sum(); cw = torch.cumsum(h, 0); cm = torch.cumsum(h * c, 0)
    w0 = cw[:-1]; w1 = tot - w0
    ok = (w0 > 0) & (w1 > 0)
    m0 = cm[:-1] / w0.clamp(min=1); m1 = (cm[-1] - cm[:-1]) / w1.clamp(min=1)
    v = torch.where(ok, w0 * w1 * (m0 - m1) ** 2, torch.zeros_like(w0))
    return float(min(max(float(c[int(torch.argmax(v))]), 0.02), 0.98))


def otsu_logit_threshold(p, bins=1000):
    """Label-free bimodal split of the target score distribution on the LOGIT scale (M3). Otsu on probabilities
    fails because the landslide mode of an under-detecting model sits at low probabilities; on the logit axis the two
    modes separate. Returns the probability threshold."""
    p = p.clamp(1e-3, 1 - 1e-3)
    lg = torch.log(p / (1 - p))
    h = torch.histc(lg, bins=bins, min=-7.0, max=7.0).double()
    c = torch.linspace(-7.0, 7.0, bins, dtype=torch.float64)
    tot = h.sum(); cw = torch.cumsum(h, 0); cm = torch.cumsum(h * c, 0)
    w0 = cw[:-1]; w1 = tot - w0
    ok = (w0 > 0) & (w1 > 0)
    m0 = cm[:-1] / w0.clamp(min=1); m1 = (cm[-1] - cm[:-1]) / w1.clamp(min=1)
    v = torch.where(ok, w0 * w1 * (m0 - m1) ** 2, torch.zeros_like(w0))
    if not bool(ok.any()) or float(v.max()) <= 0.0:
        # Degenerate score distribution (all logits in one histogram bin, e.g. a constant teacher): no bimodal split
        # exists. Review response (R1-5, 2026-10-21): return None instead of the former 0.020 fallback, so the caller keeps
        # the previous threshold and skips the target loss for that epoch rather than labelling everything positive.
        return None
    i = int(torch.argmax(v))
    t = float(torch.sigmoid(c[i]))
    return float(min(max(t, 0.02), 0.98))


def otsu_logit_eta(lg, bins=256):
    """Logit-axis Otsu on one tile's teacher logits. Returns (boundary logit, separability eta = between-class /
    total variance in [0, 1]); eta is the shrinkage weight of the per-tile boundary towards the pool boundary."""
    lg = lg.float().clamp(-7.0, 7.0)
    h = torch.histc(lg, bins=bins, min=-7.0, max=7.0).double()
    c = torch.linspace(-7.0, 7.0, bins, dtype=torch.float64, device=h.device)
    tot = h.sum().clamp(min=1); cw = torch.cumsum(h, 0); cm = torch.cumsum(h * c, 0)
    w0 = cw[:-1]; w1 = tot - w0; ok = (w0 > 0) & (w1 > 0)
    m0 = cm[:-1] / w0.clamp(min=1); m1 = (cm[-1] - cm[:-1]) / w1.clamp(min=1)
    vb = torch.where(ok, w0 * w1 * (m0 - m1) ** 2, torch.zeros_like(w0)) / tot ** 2
    i = int(torch.argmax(vb)); mu = cm[-1] / tot; vt = (h * (c - mu) ** 2).sum() / tot
    return float(c[i]), float((vb[i] / vt.clamp(min=1e-12)).clamp(0, 1))


def patch_shuffle(x, G, perm):
    """DeYO-style object-destroying transform: split into a GxG grid of patches and permute them (same perm for the
    batch). The same call with argsort(perm) undoes it. H, W must be multiples of G."""
    B, C, H, W = x.shape; ph, pw = H // G, W // G
    p = x.reshape(B, C, G, ph, G, pw).permute(0, 2, 4, 1, 3, 5).reshape(B, G * G, C, ph, pw)[:, perm]
    return p.reshape(B, G, G, C, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


class RelHead(nn.Module):
    """M3 reliability attention (training-time only). Input: detached decoder features (C), the teacher's signed logit
    distance to the calibrated boundary (scaled to [-1, 1]) and a pasted-source indicator. Output: s = log of the expected
    per-pixel pseudo-label loss. Depthwise 3x3 for local context, then a 16-channel bottleneck."""
    def __init__(self, c):
        super().__init__()
        c2 = c + 2
        self.net = nn.Sequential(nn.Conv2d(c2, c2, 3, padding=1, groups=c2), nn.Conv2d(c2, 16, 1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)   # start at w = 1 everywhere

    def forward(self, f, extra):
        return self.net(torch.cat([f, extra], 1))[:, 0]


def split_target(keys, index, mode, seed=0):
    if mode == "transductive":
        return list(keys), list(keys)
    by = {}
    for k in keys:
        by.setdefault(index[k]["sid"], []).append(k)
    adapt, test = [], []
    for sid in sorted(by):
        ks = sorted(by[sid]); random.Random(seed).shuffle(ks)
        adapt += ks[::2]; test += ks[1::2]
    return adapt, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--index", default="splits/index.csv")
    ap.add_argument("--raw", default="data/CAS_raw")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="segformer"); ap.add_argument("--encoder", default="mit_b2")
    ap.add_argument("--init", default="", help="source-trained checkpoint to start from (recommended)")
    ap.add_argument("--epochs", type=int, default=10, help="adaptation epochs (one epoch = one pass over the source set)")
    ap.add_argument("--bs", type=int, default=8, help="per-domain batch size")
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--lr_head_mult", type=float, default=10.0)
    ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gsd_target", type=float, default=0.0); ap.add_argument("--gsd_clamp", default="0.25,2.0")
    ap.add_argument("--uda_mode", default="transductive", choices=["transductive", "inductive"])
    ap.add_argument("--pl_tau", type=float, default=0.9, help="pseudo-label confidence gate on max(p,1-p) (pl_gate abs)")
    ap.add_argument("--pl_gate", default="abs", choices=["abs", "margin", "component"], help="abs: keep max(p,1-p)>tau (ignores the calibrated threshold!); margin: keep |logit(p)-logit(thr)| > pl_margin")
    ap.add_argument("--pl_margin", type=float, default=1.0, help="logit margin around the pseudo-label threshold (pl_gate margin)")
    ap.add_argument("--pl_thresh", type=float, default=0.5, help="probability threshold defining the pseudo-label (pl_mode fixed)")
    ap.add_argument("--rev_classmix", type=float, default=0.0, help="M3 candidate 8: weight of the reverse mix, pasting gated target pseudo-label components into SOURCE tiles (0 = off)")
    ap.add_argument("--pl_margin_pos", type=float, default=-1.0, help="M3 candidate 7: margin on the positive side of the calibrated boundary (<0 = use --pl_margin)")
    ap.add_argument("--pl_margin_rule", default="fixed", choices=["fixed", "cont"], help="M3: fixed = one (delta+, delta-) for the whole target; cont = per-tile margins 1 +/- m3_amp*rho_i, rho_i = clip(log r_i / log m3_sat, -1, 1), r_i = max(g_i, g*) / g_ref (information-limiting GSD of the tile over the median one of the source)")
    ap.add_argument("--m3_amp", type=float, default=0.5, help="M3 cont: half-width of the asymmetry (0.5 reproduces the fixed rule (1.5, 0.5) at saturation)")
    ap.add_argument("--m3_sat", type=float, default=4.0, help="M3 cont: GSD ratio at which the asymmetry saturates (4 = the widest factor M1 resamples by)")
    ap.add_argument("--rel_att", type=int, default=0, help="M3 reliability attention (training only): a ~7k-parameter head on the detached decoder features + the teacher's signed logit distance to the calibrated boundary predicts the per-pixel pseudo-label loss (heteroscedastic, s = log E[l]); the target loss is re-weighted by exp(-s), normalised to mean 1. Discarded after training.")
    ap.add_argument("--rel_warm", type=int, default=300, help="M3 reliability attention: head-training steps before its weights are applied")
    ap.add_argument("--pl_margin_neg", type=float, default=-1.0, help="M3 candidate 7: margin on the negative side (<0 = use --pl_margin)")
    ap.add_argument("--tea_calib", type=int, default=0, help="M3 candidate 6: calibrate each teacher scale separately (logit-Otsu per scale) and average the aligned logits instead of the raw probabilities")
    ap.add_argument("--mosaic_tgt", type=int, default=-1, help="mosaic packing for the TARGET stream only (-1 = same as --mosaic); lets the source and target sides be separated")
    ap.add_argument("--mosaic", type=int, default=0, help="M3 candidate 5: pack crops of down-sampled tiles with this many distinct real tiles instead of reflect-padding (0 = off, 1 = repeat the tile itself under flips)")
    ap.add_argument("--src_mismatch_w", type=float, default=0.0, help="M3 candidate 4: down-weight source tiles whose scale was clipped, w = exp(-lam*|log(g_i/s_i) - log(g*)|); 0 = off")
    ap.add_argument("--gradual", type=int, default=0, help="M3 candidate 3: K stages of gradual self-training along the GSD axis (target strata ordered by |log g - log g*|; 0 = off)")
    ap.add_argument("--otsu_axis", default="logit", choices=["logit", "prob"], help="M2 ablation: axis on which Otsu splits the target score distribution")
    ap.add_argument("--calib_strata", default="none", choices=["none", "gsd_band", "sid"], help="M3: stratified calibration key (metadata)")
    ap.add_argument("--strata_pool", type=int, default=256, help="M3: adaptation tiles per stratum for the threshold estimate")
    ap.add_argument("--strata_fallback_logit", type=float, default=1.0, help="M3: fall back to the global threshold if |logit diff| exceeds this")
    ap.add_argument("--strata_min_pos", type=float, default=0.005, help="M3: fall back if the stratum positive fraction above its Otsu threshold is below this")
    ap.add_argument("--pl_mode", default="fixed", choices=["fixed", "prior", "otsu", "cbst", "flexmatch", "freematch"], help="M3: fixed / EM-prior-matched / logit-Otsu pseudo-label threshold; flexmatch / freematch = adaptive per-class confidence gates around the argmax (E5a baselines)")
    ap.add_argument("--ada_m", type=float, default=0.999, help="E5a: EMA momentum of the adaptive-threshold statistics (FreeMatch lambda; FlexMatch learning-status counts)")
    ap.add_argument("--flex_warmup", type=int, default=1, help="E5a FlexMatch: threshold warm-up (unused unlabelled pixels count in the normaliser)")
    ap.add_argument("--tgt_scale", type=float, default=1.0, help="dataset-level resampling factor for TARGET tiles only (external targets without per-tile GSD)")
    ap.add_argument("--eval_bs", type=int, default=16, help="eval batch size for un-resampled tiles (lower for 1024-px external tiles)")
    ap.add_argument("--prior_pool", type=int, default=256, help="M3: adaptation tiles used to estimate the target prior each epoch")
    ap.add_argument("--consist_w", type=float, default=0.0, help="(closed) target-side student-vs-student cross-scale consistency")
    ap.add_argument("--mic_ratio", type=float, default=0.0, help="MIC: fraction of 64-px patches masked on the target view; student on the masked view is supervised by the teacher's pseudo-label from the full view (0 = off)")
    ap.add_argument("--mic_w", type=float, default=1.0)
    ap.add_argument("--wallis", type=int, default=0, help="LandsDANet-style Wallis radiometric normalisation on all tiles (train + eval)")
    ap.add_argument("--contrast_w", type=float, default=0.0, help="supervised pixel-prototype contrastive loss on source decoder features (LandsDANet-style, lambda .5)")
    ap.add_argument("--contrast_tau", type=float, default=0.1)
    ap.add_argument("--cbst_p0", type=float, default=0.2, help="CBST: initial per-class keep fraction"); ap.add_argument("--cbst_dp", type=float, default=0.05, help="CBST: increase per epoch (cap 0.5)")
    ap.add_argument("--adv_w", type=float, default=0.0, help="AdvEnt: weight of the adversarial loss on entropy maps (paper 1e-3); 0 = off")
    ap.add_argument("--adv_lr", type=float, default=1e-4)
    ap.add_argument("--ent_calib", type=int, default=0, help="MinEnt about the CALIBRATED boundary: entropy of sigmoid(logit - logit(theta)), theta = current pseudo-label threshold (M3)")
    ap.add_argument("--frac_w", type=float, default=0.0, help="positive-fraction regulariser: KL(mean target prob || pi_hat), pi_hat = fraction of the adaptation pool above the Otsu threshold (M3 statistic)")
    ap.add_argument("--ent_w", type=float, default=0.0, help="MinEnt: weight of direct target entropy minimisation (AdvEnt paper 1e-3); 0 = off")
    ap.add_argument("--tea_scales", default="1.0", help="comma list of relative scales for the teacher's pseudo-label ensemble, e.g. 1.0,0.5")
    ap.add_argument("--mic_patch", type=int, default=64, help="MIC mask patch size in px at the anchored scale; scale-adaptive rule: ~1/4 of the median landslide size in px after anchoring")
    ap.add_argument("--rcs_alpha", type=float, default=0.0, help="RCS: sample source tiles with probability ~ (landslide fraction)^alpha (0 = uniform)")
    # ---- M3 search, round 3 (2026-10-01): zero-parameter, training-only candidates (defaults = off)
    ap.add_argument("--la_tau", type=float, default=0.0, help="C1 prior logit adjustment: source loss on logits + tau*(logit(p_src) - logit(pi_hat)); pi_hat = pool fraction above the Otsu boundary (Menon 2021 / ACR 2023 / CPG 2025)")
    ap.add_argument("--tile_otsu", type=int, default=0, help="C2 per-tile boundary (CSL-style, ICCV 2025): each tile's own logit-Otsu boundary, shrunk to the pool boundary by its separability eta")
    ap.add_argument("--deyo_thr", type=float, default=0.0, help="C3 shape reliance (DeYO PLPD, ICLR 2024): positive pseudo-labels weighted by clip(rel. prob. drop under a GxG patch shuffle / thr, floor, 1); 0 = off")
    ap.add_argument("--deyo_grid", type=int, default=4); ap.add_argument("--deyo_floor", type=float, default=0.1)
    ap.add_argument("--twins_w", type=float, default=0.0, help="C4 complementary-mask twins consistency (MaskTwins, ICML 2025): student on x*M and x*(1-M), symmetric BCE to each other's detached probabilities")
    ap.add_argument("--twins_patch", type=int, default=32)
    ap.add_argument("--fdrop_w", type=float, default=0.0, help="C5 complementary channel dropout (UniMatch V2, TPAMI 2025): two decoder passes on encoder features masked by M and 1-M, each supervised by the gated pseudo-label")
    ap.add_argument("--recap_k", type=float, default=0.0, help="C6 regional instability (ReCAP, ICML 2025): target pixel weight exp(-k * local variance of the teacher probability)")
    ap.add_argument("--recap_win", type=int, default=7)
    ap.add_argument("--soft_gate", type=float, default=0.0, help="C7 soft margin gate (SoftMatch, ICLR 2023): pixels inside the margin get weight exp(-(m-|d|)^2 / 2 s^2) instead of being ignored; value = s")
    ap.add_argument("--comp_denoise", type=int, default=0, help="C8 loss-mode component denoising (SeCo correction step, NeurIPS 2024, without SAM): positive pseudo-label components in the high-loss Otsu mode of the batch are ignored")
    # ---- round-3 improvement 1 (v2): data-driven regime reg = +1 if pi_hat < p_src (teacher over-predicts), -1 otherwise
    ap.add_argument("--la_both", type=int, default=0, help="C1v2: apply the prior logit offset to the target pseudo-label loss too (the deployed raw logit is shifted consistently)")
    ap.add_argument("--tile_clip", type=float, default=0.0, help="C2v2: clamp each tile's boundary to within +/- this many logits of the pool boundary (0 = off)")
    ap.add_argument("--tile_side", default="all", choices=["all", "prior"], help="C2v3: prior = per-tile boundary may only be stricter (reg=+1) or looser (reg=-1) than the pool boundary")
    ap.add_argument("--tile_eta_pow", type=float, default=1.0, help="C2v2: shrinkage weight eta^pow (larger = trust single tiles less)")
    ap.add_argument("--recap_side", default="all", choices=["all", "prior"], help="C6v2: prior = down-weight only the side of the dominant error (positives when reg=+1, negatives when reg=-1)")
    ap.add_argument("--soft_side", default="all", choices=["all", "prior"], help="C7v2: prior = soft-include only the side opposite to the dominant error (negatives when reg=+1, positives when reg=-1)")
    ap.add_argument("--comp_cap", type=float, default=1.0, help="C8v2: at most this fraction of the batch's positive components can be dropped")
    ap.add_argument("--comp_start", type=int, default=0, help="C8v2: first epoch (0-based) at which component denoising is active")
    ap.add_argument("--comp_prior", type=int, default=0, help="C8v2: denoise only in the over-predicting regime (reg=+1)")
    ap.add_argument("--deyo_prior", type=int, default=0, help="C3v2: shape-reliance weighting only in the over-predicting regime (reg=+1)")
    ap.add_argument("--twins_T", type=float, default=0.0, help="C4v2: sharpen the twin targets, sigmoid(logit / T) (0 = raw probabilities)")
    ap.add_argument("--consist_rho", type=float, default=2.0)
    ap.add_argument("--lam_t", type=float, default=1.0); ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--classmix", type=int, default=1)
    ap.add_argument("--warm_steps", type=int, default=200, help="steps of source-only training before pseudo-labels")
    ap.add_argument("--eval_stride", type=int, default=256)
    ap.add_argument("--max_train", type=int, default=0); ap.add_argument("--max_eval", type=int, default=0)
    a = ap.parse_args()

    OTSU_AXIS[0] = a.otsu_axis
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    out = os.path.join(a.runs, a.name); os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.txt"), "a")

    def P(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); log.write(msg + "\n"); log.flush()

    P("args", json.dumps(vars(a)))
    index = load_index(a.index); sp = json.load(open(a.split))
    tr_keys = [k for k in sp["train"] if k in index]; va_keys = [k for k in sp["val"] if k in index]
    tgt_all = [k for k in sp["test"] if k in index]
    adapt_keys, te_keys = split_target(tgt_all, index, a.uda_mode, a.seed)
    if a.max_train:
        random.Random(0).shuffle(tr_keys); tr_keys = tr_keys[:a.max_train]
        random.Random(0).shuffle(adapt_keys); adapt_keys = adapt_keys[:a.max_train]
    if a.max_eval:
        random.Random(0).shuffle(va_keys); va_keys = va_keys[:a.max_eval]
        random.Random(0).shuffle(te_keys); te_keys = te_keys[:a.max_eval]
    clamp = tuple(float(x) for x in a.gsd_clamp.split(","))
    resampled = a.gsd_target > 0 or a.tgt_scale != 1.0
    tea_scales = [float(v) for v in a.tea_scales.split(",")]
    P(f"split {os.path.basename(a.split)} [{a.uda_mode}]: source train {len(tr_keys)} val {len(va_keys)} | target adapt {len(adapt_keys)} test {len(te_keys)}")
    if resampled:
        P(f"M1 canonical resampling: target GSD {a.gsd_target} m, clamp {clamp} (applied to both domains)")

    dl_kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    ds_kw = dict(gsd_target=a.gsd_target, gsd_clamp=clamp, wallis=bool(a.wallis), mosaic=a.mosaic)
    tg_kw = dict(ds_kw, scale_mult=a.tgt_scale, mosaic=(a.mosaic if a.mosaic_tgt < 0 else a.mosaic_tgt))   # target-side datasets only
    w = torch.ones(len(tr_keys), dtype=torch.double); use_sampler = False
    if a.rcs_alpha > 0:  # rare-class sampling on the SOURCE (DAFormer RCS, binary form): landslide-rich tiles drawn more often
        w *= torch.tensor([(float(index[k]["ls_frac"]) + 1e-3) ** a.rcs_alpha for k in tr_keys], dtype=torch.double)
        use_sampler = True
        fr = torch.tensor([float(index[k]["ls_frac"]) for k in tr_keys], dtype=torch.double)
        P(f"RCS source sampling: alpha {a.rcs_alpha}; mean landslide fraction of a sampled tile {float((w / w.sum() * fr).sum()):.4f} vs uniform {float(fr.mean()):.4f}")
    if a.src_mismatch_w > 0 and a.gsd_target > 0:
        # M1's clip leaves residual scale mismatch for subsets more than 2x (or 4x) from the anchor: after resampling by
        # s = clip(g/g*, 1/4, 2) the tile sits at g/s, not at g*. Down-weight those tiles instead of trusting them equally.
        d = torch.tensor([abs(math.log((float(index[k]["gsd_m"]) / min(max(float(index[k]["gsd_m"]) / a.gsd_target, clamp[0]), clamp[1])) / a.gsd_target)) for k in tr_keys], dtype=torch.double)
        w *= torch.exp(-a.src_mismatch_w * d); use_sampler = True
        by_g = {}
        for k, di, wi in zip(tr_keys, d.tolist(), torch.exp(-a.src_mismatch_w * d).tolist()):
            by_g.setdefault(float(index[k]["gsd_m"]), (di, wi, 0))
            by_g[float(index[k]["gsd_m"])] = (di, wi, by_g[float(index[k]["gsd_m"])][2] + 1)
        P("source residual-mismatch weighting (lam %.2f): " % a.src_mismatch_w + "; ".join(f"{g} m: n {n}, |dlog| {di:.3f}, w {wi:.3f}" for g, (di, wi, n) in sorted(by_g.items())))
    if use_sampler:
        sampler = WeightedRandomSampler(w, num_samples=len(tr_keys), replacement=True)
        src_dl = DataLoader(CASTiles(tr_keys, index, a.raw, True, **ds_kw), batch_size=a.bs, sampler=sampler, drop_last=True, **dl_kw)
    else:
        src_dl = DataLoader(CASTiles(tr_keys, index, a.raw, True, **ds_kw), batch_size=a.bs, shuffle=True, drop_last=True, **dl_kw)
    tgt_dl = DataLoader(CASTiles(adapt_keys, index, a.raw, True, **tg_kw), batch_size=a.bs, shuffle=True, drop_last=True, **dl_kw)
    act_keys = list(adapt_keys); stage_of = None; stage_starts = []
    if a.gradual > 0 and a.gsd_target > 0:  # gradual self-training: distinct target GSDs ordered by log-distance to the anchor, K cumulative stages
        gs = sorted({float(index[k]["gsd_m"]) for k in adapt_keys}, key=lambda g: abs(math.log(g) - math.log(a.gsd_target)))
        groups = [list(x) for x in np.array_split(np.array(gs), min(a.gradual, len(gs)))]
        gstage = {g: si for si, grp in enumerate(groups) for g in grp}
        stage_of = {k: gstage[float(index[k]["gsd_m"])] for k in adapt_keys}
        K = len(groups); stage_starts = [int(round(a.epochs * si / (2 * max(K - 1, 1)))) for si in range(K)]
        P(f"gradual self-training: {K} stages along the GSD axis (anchor {a.gsd_target} m): " + "; ".join(f"stage {si} from epoch {stage_starts[si]}: GSD {grp} ({sum(1 for k in adapt_keys if stage_of[k] <= si)} tiles cumulative)" for si, grp in enumerate(groups)))
    ebs = 1 if resampled else a.eval_bs
    va_dl = DataLoader(CASTiles(va_keys, index, a.raw, False, **ds_kw), batch_size=ebs, shuffle=False, **dl_kw)
    te_dl = DataLoader(CASTiles(te_keys, index, a.raw, False, **tg_kw), batch_size=ebs, shuffle=False, **dl_kw)

    model = build_model(a.arch, a.encoder).to(device).to(memory_format=torch.channels_last)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu")["model"]); P(f"init from {a.init}")
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    ev = lambda m, dl, keys, ths=(0.5,), stride=a.eval_stride: evaluate(m, dl, index, keys, device, ths, resampled=resampled, stride=stride)

    enc_params = list(model.encoder.parameters()); enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": head_params, "lr": a.lr * a.lr_head_mult}], weight_decay=a.wd)
    rel_head = None; feat_buf = []; cap = [False]   # decoder features are captured only for the student's mixed-view forward
    if a.rel_att:
        model.decoder.register_forward_hook(lambda m_, i_, o_: feat_buf.append(o_) if cap[0] else None)
        cap[0] = True
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.eval(); model(torch.zeros(1, 3, 256, 256, device=device)); model.train()
        cap[0] = False
        c_dec = feat_buf[-1].shape[1]; feat_buf.clear()
        rel_head = RelHead(c_dec).to(device)
        opt.add_param_group({"params": list(rel_head.parameters()), "lr": a.lr * a.lr_head_mult, "weight_decay": 0.0})
        P(f"M3 reliability attention: {sum(p.numel() for p in rel_head.parameters())} params on {c_dec}-ch decoder features (training only)")
    m3_rho = None
    if a.pl_margin_rule == "cont":
        g_star = a.gsd_target if a.gsd_target > 0 else 0.0
        eff = lambda g: max(g, g_star)
        g_ref = float(np.median([eff(float(index[k]["gsd_m"])) for k in tr_keys]))
        m3_rho = {k: float(min(max(math.log(eff(float(index[k]["gsd_m"])) / g_ref) / math.log(a.m3_sat), -1.0), 1.0)) for k in adapt_keys}
        hist = {}
        for k in adapt_keys:
            hist.setdefault(round(m3_rho[k], 2), 0); hist[round(m3_rho[k], 2)] += 1
        P(f"M3 cont: g_ref {g_ref} m (median information-limiting source GSD), amp {a.m3_amp}, sat {a.m3_sat}; rho histogram " + json.dumps(dict(sorted(hist.items()))))
    total = a.epochs * len(src_dl); warm = min(200, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (it + 1) / warm if it < warm else (1 - (it - warm) / max(1, total - warm)))
    crit = BCEDice()
    disc = opt_d = None
    if a.adv_w > 0:
        disc = EntropyDiscriminator().to(device)
        opt_d = torch.optim.Adam(disc.parameters(), lr=a.adv_lr, betas=(0.9, 0.99))
        P(f"AdvEnt discriminator on ({sum(p.numel() for p in disc.parameters())/1e6:.2f} M params, training-time only)")
    num = sum(float(index[k]["ls_frac"]) * float(index[k]["h"]) * float(index[k]["w"]) for k in tr_keys)
    den = sum(float(index[k]["h"]) * float(index[k]["w"]) for k in tr_keys)
    p_src = num / den
    P(f"source prior (landslide pixel fraction) {p_src:.4f}; pseudo-label mode {a.pl_mode}; consist_w {a.consist_w}")
    pl_thresh = a.pl_thresh
    ada = None
    if a.pl_mode in ("flexmatch", "freematch"):
        # E5a adaptive-threshold baselines (review response): the decision boundary stays at 0.5 (argmax pseudo-label);
        # what adapts is a class-wise confidence gate on max(p, 1-p).
        #   freematch (Wang et al., ICLR 2023): global tau_t = EMA of the mean confidence, local p~_t(c) = EMA of the mean
        #     class probability; tau_t(c) = p~_t(c) / max_c' p~_t(c') * tau_t. Init tau_0 = p~_0 = 1/C. Without the SAF term.
        #   flexmatch (Zhang et al., NeurIPS 2021): tau(c) = tau * beta(c), beta(c) = sigma(c) / max(max_c' sigma(c'), N - sum sigma)
        #     with sigma(c) = (EMA) count of unlabelled pixels predicted c with confidence > tau (learning status, linear mapping,
        #     warm-up variant); tau = --pl_tau.
        ada = dict(tau=0.5, ptil=[0.5, 0.5], sigma=[0.0, 0.0], N=0.0, last=[0.5, 0.5])
        pl_thresh = 0.5
        P(f"E5a adaptive confidence gate: {a.pl_mode}, EMA momentum {a.ada_m}, base tau {a.pl_tau} (boundary fixed at 0.5)")
    best, best_ep, it = -1, -1, 0
    for ep in range(a.epochs):
        cbst = None; pi_hat = None; thr_strata = None; thr_scale = None; degenerate_epoch = False
        if stage_of is not None:
            cur = max(si for si, st in enumerate(stage_starts) if ep >= st)
            new_keys = [k for k in adapt_keys if stage_of[k] <= cur]
            if len(new_keys) != len(act_keys) or ep == 0:
                act_keys = new_keys
                tgt_dl = DataLoader(CASTiles(act_keys, index, a.raw, True, **tg_kw), batch_size=a.bs, shuffle=True, drop_last=True, **dl_kw)
                P(f"  [gradual] epoch {ep+1}: stage {cur}, {len(act_keys)} active target tiles")
        if a.pl_mode in ("prior", "otsu", "cbst") and a.lam_t > 0:
            pi, thr_em, pmean, thr_otsu, ppool = estimate_prior_and_threshold(teacher, a.raw, tg_kw, index, act_keys, device, a.prior_pool, a.eval_bs, a.workers, resampled, a.eval_stride, p_src, return_p=True)
            pi_hat = float((ppool > (thr_otsu if thr_otsu is not None else 0.5)).float().mean())
            if a.pl_mode == "cbst":
                frac = min(0.5, a.cbst_p0 + a.cbst_dp * ep); cbst = cbst_thresholds(ppool, frac); pl_thresh = 0.5
                P(f"  [cbst] epoch {ep+1}: teacher mean prob {pmean:.4f} | keep frac {frac:.2f} -> thr_pos {cbst[0]:.3f} thr_neg(on 1-p) {cbst[1]:.3f}")
            else:
                degenerate_epoch = a.pl_mode == "otsu" and thr_otsu is None
                if degenerate_epoch:   # R1-5 fallback: keep last epoch's threshold, no target supervision this epoch
                    thr_otsu = pl_thresh if ep > 0 else 0.5
                    P(f"  [otsu] epoch {ep+1}: DEGENERATE score histogram -> threshold kept at {thr_otsu:.3f}, target loss disabled for this epoch")
                pl_thresh = thr_em if a.pl_mode == "prior" else thr_otsu
                P(f"  [{a.pl_mode}] epoch {ep+1}: teacher mean prob {pmean:.4f} | EM prior {pi:.4f} -> thr {thr_em:.3f} | logit-Otsu thr {thr_otsu:.3f} -> using {pl_thresh:.3f} | pi_hat {pi_hat:.4f}")
                if a.tea_calib and len(tea_scales) > 1:
                    thr_scale = {}
                    for sc_ in tea_scales:
                        if sc_ == 1.0:
                            thr_scale[sc_] = thr_otsu
                        else:
                            _, _, pm_, th_ = estimate_prior_and_threshold(teacher, a.raw, tg_kw, index, act_keys, device, min(128, a.prior_pool), a.eval_bs, a.workers, resampled, a.eval_stride, p_src, input_scale=sc_)
                            thr_scale[sc_] = th_
                    P("  [tea_calib] per-scale logit-Otsu thresholds: " + ", ".join(f"x{k}: {v:.3f}" for k, v in sorted(thr_scale.items())))
                if a.calib_strata != "none" and a.pl_mode == "otsu":
                    thr_strata = stratified_thresholds(teacher, a.raw, tg_kw, index, act_keys, device, a.calib_strata, a.strata_pool, a.eval_bs, a.workers,
                                                       resampled, a.eval_stride, p_src, pl_thresh, a.strata_fallback_logit, a.strata_min_pos, P)
        la_b = 0.0
        reg = 0 if pi_hat is None else (1 if pi_hat < p_src else -1)   # v2 regime: +1 over-predicting, -1 under-detecting
        if pi_hat is not None and (a.recap_side == "prior" or a.soft_side == "prior" or a.comp_prior):
            P(f"  [v2 regime] epoch {ep+1}: pi_hat {pi_hat:.4f} vs p_src {p_src:.4f} -> reg {reg:+d} ({'over-predicting' if reg > 0 else 'under-detecting'})")
        if a.la_tau > 0 and pi_hat is not None:
            lgt = lambda q: math.log(min(max(q, 1e-3), 1 - 1e-3) / (1 - min(max(q, 1e-3), 1 - 1e-3)))
            la_b = float(min(max(a.la_tau * (lgt(p_src) - lgt(pi_hat)), -3.0), 3.0))
            P(f"  [C1 prior-LA] epoch {ep+1}: p_src {p_src:.4f} pi_hat {pi_hat:.4f} -> source logit offset {la_b:+.3f}")
        m3s = {"tile_eta": 0.0, "deyo_wpos": 0.0, "recap_w": 0.0, "soft_w": 0.0, "comp_drop": 0.0, "twins": 0.0, "fdrop": 0.0, "n": 0}
        model.train(); t0 = time.time(); run_s = run_t = run_c = 0.0; kept = 0.0; nb = 0
        if rel_head is not None:
            rel_head.train()
        tgt_iter = iter(tgt_dl)
        for xs, ys, ws, _ in src_dl:
            try:
                xt, _, _, ti = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_dl); xt, _, _, ti = next(tgt_iter)
            xs = xs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            ys = ys.to(device, non_blocking=True); ws = ws.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls = model(xs)[:, 0]
            if la_b != 0.0:  # C1: fit the source labels with prior-shifted logits, so the raw logit tracks the target prior
                loss = crit(ls.float() + la_b, ys, ws)
            else:
                loss = crit(ls.float(), ys, ws)
            lt = torch.zeros((), device=device); lc = torch.zeros((), device=device)
            if a.contrast_w > 0:
                loss = loss + a.contrast_w * proto_contrast(model, xs, ys, a.contrast_tau)
            if it >= a.warm_steps and (a.lam_t > 0 or a.mic_ratio > 0):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if tea_scales == [1.0]:
                        pt = torch.sigmoid(teacher(xt)[:, 0].float())
                    else:  # scale-ensemble teacher. Averaging raw probabilities collapses (Table S4): the same network is
                        # differently calibrated at different scales. With --tea_calib each scale's logits are first shifted
                        # to its own logit-Otsu boundary, averaged there, then re-centred on the current pseudo-label
                        # threshold so every downstream gate is unchanged.
                        acc = 0.0
                        for sc_ in tea_scales:
                            xv = xt if sc_ == 1.0 else F.interpolate(xt, scale_factor=sc_, mode="bilinear", align_corners=False)
                            lv = teacher(xv)[:, 0:1].float()
                            if sc_ != 1.0:
                                lv = F.interpolate(lv, size=xt.shape[-2:], mode="bilinear", align_corners=False)
                            if a.tea_calib and thr_scale is not None:
                                t_ = min(max(thr_scale[sc_], 1e-3), 1 - 1e-3)
                                acc = acc + (lv[:, 0] - math.log(t_ / (1 - t_)))
                            else:
                                acc = acc + torch.sigmoid(lv[:, 0])
                        acc = acc / len(tea_scales)
                        if a.tea_calib and thr_scale is not None:
                            t0 = min(max(pl_thresh, 1e-3), 1 - 1e-3)
                            pt = torch.sigmoid(acc + math.log(t0 / (1 - t0)))
                        else:
                            pt = acc
                if thr_strata is not None:  # M3 stratified: one calibrated threshold per tile, from its metadata stratum
                    thr_b = torch.tensor([thr_strata[stratum_of(index[act_keys[int(i_)]], a.calib_strata)] for i_ in ti], device=device, dtype=pt.dtype).view(-1, 1, 1)
                elif a.tile_otsu and a.pl_mode == "otsu":  # C2: per-tile boundary, shrunk to the pool boundary by separability
                    pg = min(max(float(pl_thresh), 1e-3), 1 - 1e-3); lg0 = math.log(pg / (1 - pg))
                    ptc = pt.clamp(1e-4, 1 - 1e-4); lgt_ = torch.log(ptc / (1 - ptc))
                    tl = []
                    for b_ in range(pt.shape[0]):
                        tb, eta = otsu_logit_eta(lgt_[b_].flatten())
                        eta = eta ** a.tile_eta_pow
                        tv = eta * tb + (1 - eta) * lg0
                        if a.tile_clip > 0:
                            tv = min(max(tv, lg0 - a.tile_clip), lg0 + a.tile_clip)
                        if a.tile_side == "prior" and reg != 0:  # C2v3: a tile may only move against the dominant error
                            tv = max(tv, lg0) if reg > 0 else min(tv, lg0)
                        tl.append(tv); m3s["tile_eta"] += eta / pt.shape[0]
                    thr_b = torch.sigmoid(torch.tensor(tl, device=device, dtype=pt.dtype)).view(-1, 1, 1)
                else:
                    thr_b = torch.full((pt.shape[0], 1, 1), float(pl_thresh), device=device, dtype=pt.dtype)
                yt = (pt > thr_b).long()
                w_tgt = torch.ones_like(pt, dtype=torch.float32)   # per-pixel weight of the target pseudo-label (C3/C6/C7)
                if ada is not None:  # E5a: FlexMatch / FreeMatch class-wise adaptive confidence gate around the argmax
                    conf = torch.maximum(pt, 1 - pt); cls = pt > 0.5
                    if a.pl_mode == "freematch":
                        ada["tau"] = a.ada_m * ada["tau"] + (1 - a.ada_m) * float(conf.float().mean())
                        pm_ = float(pt.float().mean())
                        ada["ptil"] = [a.ada_m * ada["ptil"][0] + (1 - a.ada_m) * (1 - pm_), a.ada_m * ada["ptil"][1] + (1 - a.ada_m) * pm_]
                        mx_ = max(ada["ptil"]); tau_c = [ada["tau"] * ada["ptil"][0] / mx_, ada["tau"] * ada["ptil"][1] / mx_]
                    else:
                        hi = conf > a.pl_tau
                        n0_ = float((hi & ~cls).sum()); n1_ = float((hi & cls).sum()); nt_ = float(conf.numel())
                        ada["sigma"] = [a.ada_m * ada["sigma"][0] + (1 - a.ada_m) * n0_, a.ada_m * ada["sigma"][1] + (1 - a.ada_m) * n1_]
                        ada["N"] = a.ada_m * ada["N"] + (1 - a.ada_m) * nt_
                        den_ = max(max(ada["sigma"]), (ada["N"] - sum(ada["sigma"])) if a.flex_warmup else 0.0, 1e-6)
                        tau_c = [a.pl_tau * ada["sigma"][0] / den_, a.pl_tau * ada["sigma"][1] / den_]
                    ada["last"] = tau_c
                    thr_pix = torch.where(cls, torch.full_like(conf, tau_c[1]), torch.full_like(conf, tau_c[0]))
                    keepm = conf >= thr_pix
                elif cbst is not None:  # CBST: class-wise top-fraction confidence selection around the argmax
                    keepm = ((pt > 0.5) & (pt >= cbst[0])) | ((pt <= 0.5) & ((1 - pt) >= cbst[1]))
                elif a.pl_gate == "margin":  # gate relative to the (calibrated) threshold on the logit axis
                    pc = pt.clamp(1e-4, 1 - 1e-4); lt_ = torch.log(pc / (1 - pc)); l0 = torch.log(thr_b / (1 - thr_b))
                    dl_ = lt_ - l0
                    # The two error types are not symmetric: an under-detecting teacher needs a permissive positive side,
                    # an over-predicting one a strict positive side (candidate 7). Defaults keep the symmetric gate.
                    mp = a.pl_margin if a.pl_margin_pos < 0 else a.pl_margin_pos
                    mn = a.pl_margin if a.pl_margin_neg < 0 else a.pl_margin_neg
                    if m3_rho is not None:  # continuous M3: each tile's own GSD ratio sets its asymmetry
                        rho_b = torch.tensor([m3_rho[act_keys[int(i_)]] for i_ in ti], device=device, dtype=dl_.dtype).view(-1, 1, 1)
                        mp = a.pl_margin + a.m3_amp * rho_b; mn = a.pl_margin - a.m3_amp * rho_b
                    keepm = ((dl_ > 0) & (dl_ >= mp)) | ((dl_ <= 0) & (-dl_ >= mn))
                    if a.soft_gate > 0:  # C7: truncated-Gaussian weight inside the margin instead of ignoring
                        side = torch.where(dl_ > 0, torch.as_tensor(mp, device=device, dtype=dl_.dtype), torch.as_tensor(mn, device=device, dtype=dl_.dtype))
                        gap = (side - dl_.abs()).clamp(min=0)
                        ws_ = torch.exp(-gap.float() ** 2 / (2 * a.soft_gate ** 2))
                        if a.soft_side == "prior" and reg != 0:  # soften only the side opposite to the dominant error
                            safe = (dl_ <= 0) if reg > 0 else (dl_ > 0)
                            ws_ = torch.where(safe, ws_, keepm.float())   # dominant-error side keeps the hard gate
                            keepm = keepm | safe
                        else:
                            keepm = torch.ones_like(keepm)
                        w_tgt = w_tgt * ws_
                        m3s["soft_w"] += float(ws_.mean())
                elif a.pl_gate == "component":  # M3 candidate 2: component-level gate for the positive class
                    # negatives: pixel margin as above; positives: each connected component of the teacher's positive
                    # prediction is kept or ignored as a whole, by its mean logit distance from the calibrated boundary
                    pc = pt.clamp(1e-4, 1 - 1e-4); lt_ = torch.log(pc / (1 - pc)); l0 = torch.log(thr_b / (1 - thr_b))
                    dlog = lt_ - l0
                    keep_np = (dlog.abs() >= a.pl_margin).cpu().numpy()
                    pos_np = (dlog > 0).cpu().numpy(); d_np = dlog.float().cpu().numpy()
                    for b_ in range(pos_np.shape[0]):
                        lab, ncomp = ndi.label(pos_np[b_])
                        if ncomp == 0:
                            continue
                        means = ndi.mean(d_np[b_], labels=lab, index=np.arange(1, ncomp + 1))
                        good = np.zeros(ncomp + 1, bool); good[1:] = np.asarray(means) >= a.pl_margin
                        m_ = pos_np[b_]; keep_np[b_][m_] = good[lab[m_]]
                    keepm = torch.from_numpy(keep_np).to(device)
                else:
                    conf = torch.maximum(pt, 1 - pt); keepm = conf >= a.pl_tau
                if degenerate_epoch:
                    keepm = torch.zeros_like(keepm)
                yt[~keepm] = IGNORE
                kept += keepm.float().mean().item(); nb += 1
                m3s["n"] += 1
                if a.deyo_thr > 0 and not (a.deyo_prior and reg <= 0):  # C3: shape reliance of the teacher's positive pseudo-labels (PLPD under patch shuffle)
                    G = a.deyo_grid; Hc, Wc = (xt.shape[-2] // G) * G, (xt.shape[-1] // G) * G
                    perm = torch.randperm(G * G, device=device)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        ps_ = torch.sigmoid(teacher(patch_shuffle(xt[..., :Hc, :Wc].contiguous(), G, perm))[:, 0:1].float())
                    ps_ = patch_shuffle(ps_, G, torch.argsort(perm))[:, 0]
                    drop = torch.zeros_like(pt)
                    drop[:, :Hc, :Wc] = (pt[:, :Hc, :Wc] - ps_) / pt[:, :Hc, :Wc].clamp(min=0.05)
                    drop = F.avg_pool2d(drop[:, None], 9, 1, 4, count_include_pad=False)[:, 0]   # region-level signal
                    wpos = (drop / a.deyo_thr).clamp(a.deyo_floor, 1.0)
                    posm = yt == 1
                    w_tgt = torch.where(posm, w_tgt * wpos, w_tgt)
                    if posm.any():
                        m3s["deyo_wpos"] += float(wpos[posm].mean())
                if a.recap_k > 0:  # C6: regional instability of the teacher probability
                    k_ = a.recap_win
                    mu_ = F.avg_pool2d(pt[:, None].float(), k_, 1, k_ // 2, count_include_pad=False)
                    va_ = (F.avg_pool2d(pt[:, None].float() ** 2, k_, 1, k_ // 2, count_include_pad=False) - mu_ ** 2).clamp(min=0)[:, 0]
                    wr_ = torch.exp(-a.recap_k * va_)
                    if a.recap_side == "prior" and reg != 0:  # down-weight only the side of the dominant error
                        side_m = (yt == 1) if reg > 0 else (yt == 0)
                        wr_ = torch.where(side_m, wr_, torch.ones_like(wr_))
                    w_tgt = w_tgt * wr_; m3s["recap_w"] += float(wr_.mean())
                xm, ym = xt.clone(), yt.clone()
                src_px = torch.zeros_like(yt, dtype=torch.bool)
                if a.classmix:
                    perm = torch.randperm(xs.shape[0], device=device)
                    m = (ys[perm] == 1)
                    src_px = m
                    if m.any():
                        m3 = m[:, None].expand_as(xm)
                        xm = torch.where(m3, xs[perm], xm)
                        ym = torch.where(m, ys[perm], ym)
                xm = strong_photometric(xm).to(memory_format=torch.channels_last)
                feat_buf.clear(); cap[0] = rel_head is not None
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lm = model(xm)[:, 0]
                cap[0] = False
                wt = torch.ones_like(ym, dtype=torch.float32)
                if a.deyo_thr > 0 or a.recap_k > 0 or a.soft_gate > 0:  # pasted source pixels keep weight 1
                    wt = torch.where(src_px, wt, w_tgt)
                    if a.recap_k > 0:  # ReCAP re-weights all target pixels: renormalise to mean 1 over valid pixels
                        v_ = (ym != IGNORE).float()
                        wt = wt / ((wt * v_).sum() / v_.sum().clamp(min=1)).clamp(min=1e-6)
                if a.comp_denoise and ep >= a.comp_start and not (a.comp_prior and reg <= 0):  # C8: drop positive pseudo-label components in the high-loss mode of the batch
                    with torch.no_grad():
                        lpx = F.binary_cross_entropy_with_logits(lm.detach().float(), (ym == 1).float(), reduction="none").cpu().numpy()
                        posm_np = ((yt == 1) & ~src_px).cpu().numpy()
                        labs, comps = [], []
                        for b_ in range(posm_np.shape[0]):
                            lab_, n_ = ndi.label(posm_np[b_]); labs.append(lab_)
                            if n_:
                                mm = ndi.mean(lpx[b_], labels=lab_, index=np.arange(1, n_ + 1))
                                comps += [(b_, j + 1, float(v)) for j, v in enumerate(np.atleast_1d(mm))]
                        if len(comps) >= 4:
                            lv = torch.tensor([math.log(c_[2] + 1e-6) for c_ in comps])
                            t_, eta_ = otsu_logit_eta(lv.clamp(-7, 7), bins=128)
                            hi = [c_ for c_, l_ in zip(comps, lv.tolist()) if l_ > t_]
                            lo = [c_[2] for c_, l_ in zip(comps, lv.tolist()) if l_ <= t_]
                            if a.comp_cap < 1.0:  # C8v2: keep only the worst components, at most comp_cap of the batch
                                hi = sorted(hi, key=lambda c_: -c_[2])[:max(1, int(a.comp_cap * len(comps)))]
                            if hi and lo and np.mean([c_[2] for c_ in hi]) >= 2 * np.mean(lo) and eta_ > 0.5:
                                drop_np = np.zeros_like(posm_np)
                                for b_, j_, _ in hi:
                                    drop_np[b_] |= labs[b_] == j_
                                dm = torch.from_numpy(drop_np).to(device)
                                ym = torch.where(dm, torch.full_like(ym, IGNORE), ym)
                                m3s["comp_drop"] += len(hi) / len(comps)
                if rel_head is not None:
                    f_ = feat_buf[-1].detach().float(); feat_buf.clear()
                    pc_ = pt.clamp(1e-4, 1 - 1e-4)
                    dist = ((torch.log(pc_ / (1 - pc_)) - torch.log(thr_b / (1 - thr_b))) / 6.0).clamp(-1, 1)
                    dist = torch.where(src_px, torch.ones_like(dist), dist)
                    ex = torch.stack([dist, src_px.float()], 1)
                    ex = F.interpolate(ex, f_.shape[-2:], mode="bilinear", align_corners=False)
                    s_ = F.interpolate(rel_head(f_, ex)[:, None], ym.shape[-2:], mode="bilinear", align_corners=False)[:, 0].clamp(-4, 4)
                    valid_ = (ym != IGNORE).float()
                    lpix = F.binary_cross_entropy_with_logits(lm.detach().float(), (ym == 1).float(), reduction="none")
                    l_rel = ((torch.exp(-s_) * lpix + s_) * valid_).sum() / valid_.sum().clamp(min=1)
                    loss = loss + l_rel
                    if it - a.warm_steps >= a.rel_warm:
                        w_ = torch.exp(-s_.detach()) * valid_
                        wt = w_ / (w_.sum() / valid_.sum().clamp(min=1)).clamp(min=1e-6)
                lt = crit(lm.float() + (la_b if a.la_both else 0.0), ym, wt)
                loss = loss + a.lam_t * lt
                if a.twins_w > 0:  # C4: complementary-mask twins, symmetric consistency between the two masked views
                    B_, _, H_, W_ = xt.shape
                    Mk = (torch.rand(B_, 1, max(1, H_ // a.twins_patch), max(1, W_ // a.twins_patch), device=device) < 0.5).float()
                    Mk = F.interpolate(Mk, (H_, W_), mode="nearest")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        la_t = model((xt * Mk).to(memory_format=torch.channels_last))[:, 0].float()
                        lb_t = model((xt * (1 - Mk)).to(memory_format=torch.channels_last))[:, 0].float()
                    Tt = a.twins_T if a.twins_T > 0 else 1.0
                    ltw = 0.5 * (F.binary_cross_entropy_with_logits(la_t, torch.sigmoid(lb_t / Tt).detach()) +
                                 F.binary_cross_entropy_with_logits(lb_t, torch.sigmoid(la_t / Tt).detach()))
                    loss = loss + a.twins_w * ltw; m3s["twins"] += float(ltw)
                if a.fdrop_w > 0:  # C5: complementary channel dropout on the encoder features, both halves supervised
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        feats = model.encoder(xt.to(memory_format=torch.channels_last))
                        Ms = [(torch.rand(f.shape[0], f.shape[1], 1, 1, device=device) < 0.5).to(f.dtype) if f.dim() == 4 and f.shape[1] > 3 else None for f in feats]
                        outs = []
                        for flip in (0, 1):
                            ff = [f if Mc is None else f * ((Mc if flip == 0 else 1 - Mc) * 2) for f, Mc in zip(feats, Ms)]
                            outs.append(model.segmentation_head(model.decoder(ff))[:, 0])
                    lfd = 0.0
                    for lo_ in outs:
                        lo_ = lo_.float()
                        if lo_.shape[-2:] != yt.shape[-2:]:
                            lo_ = F.interpolate(lo_[:, None], yt.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
                        lfd = lfd + 0.5 * crit(lo_, yt, w_tgt)
                    loss = loss + a.fdrop_w * lfd; m3s["fdrop"] += float(lfd)
                if a.rev_classmix > 0:
                    # Candidate 8 (A+B composition): ClassMix pastes SOURCE landslides into target tiles, which teaches
                    # target context but never target appearance under reliable labels. The reverse view pastes the target
                    # components that survive the calibrated gate into SOURCE tiles, so the pasted pixels carry target
                    # appearance while every surrounding pixel keeps a ground-truth label.
                    mt = (yt == 1) & keepm
                    if mt.any():
                        pr = torch.randperm(xt.shape[0], device=device)
                        mt_p = mt[pr]
                        xr = torch.where(mt_p[:, None].expand_as(xs), xt[pr], xs)
                        yr = torch.where(mt_p, torch.ones_like(ys), ys)
                        xr = strong_photometric(xr).to(memory_format=torch.channels_last)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            lr_ = model(xr)[:, 0]
                        loss = loss + a.rev_classmix * crit(lr_.float(), yr, torch.ones_like(yr, dtype=torch.float32))
                if a.mic_ratio > 0:  # MIC: masked target view, supervised by the teacher's full-view pseudo-label
                    B_, _, H_, W_ = xt.shape
                    ph = max(1, H_ // a.mic_patch); pw = max(1, W_ // a.mic_patch)
                    keep = (torch.rand(B_, 1, ph, pw, device=device) >= a.mic_ratio).float()
                    keep = F.interpolate(keep, (H_, W_), mode="nearest")
                    xmic = (xt * keep).to(memory_format=torch.channels_last)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lmic = model(xmic)[:, 0]
                    lm_ = crit(lmic.float(), yt, torch.ones_like(yt, dtype=torch.float32))
                    loss = loss + a.mic_w * lm_
                if a.consist_w > 0:  # (closed) cross-scale consistency on the target tile, student vs student
                    xb, _, _, (o, c) = finer_view(xt, yt, wt, a.consist_rho)
                    xb = xb.to(memory_format=torch.channels_last)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        la = model(xt)[:, 0]; lb = model(xb)[:, 0]
                    pa = torch.sigmoid(F.interpolate(la.float()[:, None, o:o + c, o:o + c], lb.shape[-2:], mode="bilinear", align_corners=False)[:, 0])
                    pb = torch.sigmoid(lb.float())
                    lc = ((pa - pb) ** 2).mean()
                    loss = loss + a.consist_w * lc
            if a.ent_w > 0 or a.adv_w > 0 or a.frac_w > 0:  # AdvEnt family / calibrated variants (target view, student)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lt_logit = model(xt)[:, 0]
                if a.ent_calib and a.pl_mode in ("otsu", "prior"):  # shift the decision boundary to the calibrated threshold
                    lt_shift = lt_logit.float() - math.log(pl_thresh / (1 - pl_thresh))
                else:
                    lt_shift = lt_logit.float()
                it_map = self_information(lt_shift)
                if a.ent_w > 0:
                    loss = loss + a.ent_w * it_map.sum(1).mean()
                if a.frac_w > 0 and pi_hat is not None:  # KL(mean prob || pi_hat) over the batch, at the calibrated boundary
                    q = torch.sigmoid(lt_shift).mean().clamp(1e-4, 1 - 1e-4); pih = min(max(pi_hat, 1e-3), 0.5)
                    loss = loss + a.frac_w * (q * torch.log(q / pih) + (1 - q) * torch.log((1 - q) / (1 - pih)))
                if a.adv_w > 0:
                    for p_ in disc.parameters():
                        p_.requires_grad_(False)
                    d_t = disc(it_map)
                    loss = loss + a.adv_w * F.binary_cross_entropy_with_logits(d_t, torch.ones_like(d_t))  # fool D: target looks like source
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); it += 1
            if a.adv_w > 0:  # discriminator step
                for p_ in disc.parameters():
                    p_.requires_grad_(True)
                d_s = disc(self_information(ls.detach())); d_t = disc(it_map.detach())
                ld = 0.5 * (F.binary_cross_entropy_with_logits(d_s, torch.ones_like(d_s)) + F.binary_cross_entropy_with_logits(d_t, torch.zeros_like(d_t)))
                opt_d.zero_grad(set_to_none=True); ld.backward(); opt_d.step()
            with torch.no_grad():  # EMA teacher
                for pt_, ps_ in zip(teacher.parameters(), model.parameters()):
                    pt_.mul_(a.ema).add_(ps_.detach(), alpha=1 - a.ema)
                for bt_, bs_ in zip(teacher.buffers(), model.buffers()):
                    bt_.copy_(bs_)
            run_s += (loss - a.lam_t * lt - a.consist_w * lc).item(); run_t += lt.item(); run_c += lc.item()
        msg = (f"ep {ep+1}/{a.epochs} Ls {run_s/len(src_dl):.4f} Lt {run_t/len(src_dl):.4f} Lc {run_c/len(src_dl):.4f} "
               f"thr {pl_thresh:.2f} pl_kept {kept/max(nb,1):.3f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s"
               + (f" | ada tau_neg {ada['last'][0]:.3f} tau_pos {ada['last'][1]:.3f}" if ada is not None else ""))
        if m3s["n"]:
            act = {k: round(v / m3s["n"], 4) for k, v in m3s.items() if k != "n" and v != 0.0}
            if act:
                msg += " | M3 " + json.dumps(act)
        rep = ev(model, va_dl, va_keys, stride=512)[0.5]["ALL"]
        msg += f" | src-val IoU {rep['IoU']:.4f}"
        if rep["IoU"] > best:
            best, best_ep = rep["IoU"], ep + 1
            torch.save({"model": model.state_dict(), "teacher": teacher.state_dict(), "epoch": ep + 1, "val_IoU": best, "args": vars(a)}, os.path.join(out, "best.pt"))
            msg += " *"
        torch.save({"model": model.state_dict(), "teacher": teacher.state_dict(), "epoch": ep + 1, "args": vars(a)}, os.path.join(out, "last.pt"))
        P(msg)
    P(f"best src-val IoU {best:.4f} at epoch {best_ep}")
    res = {}
    for tag, path in (("best", "best.pt"), ("last", "last.pt")):
        ck = torch.load(os.path.join(out, path), map_location="cpu")
        for who in ("model", "teacher"):
            model.load_state_dict(ck[who])
            rt = ev(model, te_dl, te_keys, (0.5,))
            A = rt[0.5]["ALL"]; res[f"{tag}_{who}"] = rt[0.5]
            P(f"TEST[{tag}/{who}]@0.5  IoU {A['IoU']:.4f}  F1 {A['F1']:.4f}  P {A['Precision']:.4f}  R {A['Recall']:.4f}")
    # deployment-time calibration (M3): logit-Otsu threshold from the unlabelled adaptation pool, applied to the test set
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location="cpu")["model"])
    _, _, _, t_dep = estimate_prior_and_threshold(model, a.raw, tg_kw, index, adapt_keys, device, max(a.prior_pool, 512), a.eval_bs, a.workers, resampled, a.eval_stride, p_src)
    if t_dep is None:   # degenerate pool histogram at deployment: fall back to the uncalibrated boundary
        t_dep = 0.5; P("  [deploy] DEGENERATE score histogram -> deployment threshold 0.5")
    rt = ev(model, te_dl, te_keys, (t_dep,))
    res["best_model_otsu"] = rt[t_dep]
    A = rt[t_dep]["ALL"]
    P(f"TEST[best/model]@otsu({t_dep:.3f})  IoU {A['IoU']:.4f}  F1 {A['F1']:.4f}  P {A['Precision']:.4f}  R {A['Recall']:.4f}")
    if a.calib_strata != "none":
        t_dep_s = stratified_thresholds(model, a.raw, tg_kw, index, adapt_keys, device, a.calib_strata, max(a.strata_pool, 512), a.eval_bs, a.workers,
                                        resampled, a.eval_stride, p_src, t_dep, a.strata_fallback_logit, a.strata_min_pos, P)
        conf = Confusion(); model.eval()
        with torch.no_grad():
            for x, y, idx in te_dl:
                x = x.to(device, non_blocking=True)
                if resampled:
                    lg_ = predict_logits(model, x, stride=a.eval_stride)
                    if lg_.shape[-2:] != y.shape[-2:]:
                        lg_ = F.interpolate(lg_, y.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg_ = model(x)
                pr = torch.sigmoid(lg_.float())[:, 0].cpu().numpy(); yn = y.numpy()
                for b in range(len(idx)):
                    r_ = index[te_keys[int(idx[b])]]
                    conf.add(pr[b] > t_dep_s[stratum_of(r_, a.calib_strata)], yn[b], (f"region:{r_['region']}", f"sid:{r_['sid']}"))
        rep_s = conf.report(); res["best_model_otsu_strata"] = rep_s
        A = rep_s["ALL"]
        P(f"TEST[best/model]@otsu_strata({a.calib_strata})  IoU {A['IoU']:.4f}  F1 {A['F1']:.4f}  P {A['Precision']:.4f}  R {A['Recall']:.4f}  thr {json.dumps({k_: round(v_, 3) for k_, v_ in t_dep_s.items()})}")
    json.dump({"at_0.5": res["best_model"], "at_otsu": {"t": t_dep, **rt[t_dep]}, "variants": res, "uda_mode": a.uda_mode,
               "n_adapt": len(adapt_keys), "n_test": len(te_keys), "final_pl_thresh": pl_thresh, "p_src": p_src, "ada_gates": (ada["last"] if ada is not None else None)},
              open(os.path.join(out, "metrics_test.json"), "w"), indent=1)
    A = res["best_model"]["ALL"]
    P(f"TEST@0.5  IoU {A['IoU']:.4f}  F1 {A['F1']:.4f}  P {A['Precision']:.4f}  R {A['Recall']:.4f}  mIoU {A['mIoU']:.4f}   (best student)")
    for k in sorted(res["best_model"]):
        if k.startswith("region:") or k.startswith("platform:"):
            v = res["best_model"][k]; P(f"   {k:22s} IoU {v['IoU']:.4f} F1 {v['F1']:.4f}")
    P("DONE")


if __name__ == "__main__":
    main()
