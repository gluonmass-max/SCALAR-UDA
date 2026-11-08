"""Binary landslide segmentation on CAS with segmentation-models-pytorch.

    python -m cas.train --split splits/A_712.json --arch segformer --encoder mit_b2 --name A712_segformer_b2_s0 --seed 0

Method switches (all default off -> plain baseline):
  M1 level-1  --gsd_target 1.0 --gsd_clamp 0.25,2.0
      every tile is resampled by s = clamp(gsd_tile / gsd_target) before the network, in training (then random 512
      crop, padded areas carry ignore label 255) and at inference (sliding window over the resampled tile,
      probabilities resampled back to native resolution before thresholding).
  M2 level-1  --fam_prob 0.5 --fam_beta 0.01,0.09 --fam_alpha 1.0
      cross-subset Fourier amplitude mixing: with prob fam_prob the low-frequency amplitude of the training tile is
      replaced (alpha=1) / mixed with that of a random tile from a DIFFERENT subset; phase untouched (cas.aug).
  M2 level-2  --ain_stages 1,2 --ain_alpha 0.5 --ain_beta 0.05
      feature-level amplitude normalisation after MiT stages: low-frequency log-amplitude of each feature map is
      pulled toward the running source-domain mean spectrum, phase kept (cas.modules).
  M3 level-2  --consist_w 0.5 --consist_rho 2.0
      cross-GSD consistency: each training crop is also seen as a rho-times finer view (centre window of side
      512/rho upsampled back to 512, i.e. exactly the operation M1 performs on a coarse test tile). Both views are
      supervised, and the predictions must agree on the shared ground area (masked MSE on probabilities).
  M3 level-1  --boundary_w 3.0 --boundary_sigma 8
      per-pixel BCE weight 1 + w*exp(-d^2/2sigma^2), d = distance to nearest landslide boundary (cas.aug).
  negative control  --gsd_aug 0.5,2.0   symmetric random scale jitter.

Protocol JSON: {"train": ["sid/stem", ...], "val": [...], "test": [...]} (from cas.splits).
Outputs under runs/<name>/: log.txt, best.pt, metrics_val.json, metrics_test.json (overall + per subset/region/platform).
Metrics: IoU / F1 / Precision / Recall of the landslide class over all pixels of the split (pixel-accumulated),
plus mIoU over the two classes, at threshold 0.5 (val also reports the best threshold).
"""
import argparse, csv, json, math, os, random, sys, time
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cas.meta import BY_ID  # noqa: E402
from cas.aug import fourier_amplitude_mix, boundary_weight  # noqa: E402
from cas.modules import attach_amplitude_norm  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IGNORE = 255


# ----------------------------------------------------------------------------- data
def load_index(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {f"{r['sid']}/{r['stem']}": r for r in rows}


def _imread_any(p):
    """tifffile first; fall back to cv2 for PNG/JPEG payloads with a .tif extension (GDCLD val tiles)."""
    try:
        return tifffile.imread(p)
    except Exception:
        a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise
        if a.ndim == 3 and a.shape[-1] >= 3:
            a = np.ascontiguousarray(a[..., 2::-1]) if a.shape[-1] == 3 else np.ascontiguousarray(a[..., :3][..., ::-1])
        return a


def read_img(p):
    a = _imread_any(p)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = a.transpose(1, 2, 0)
    return np.ascontiguousarray(a[..., :3])


def read_mask(p):
    m = _imread_any(p)
    if m.ndim == 3:
        m = m[0] if m.shape[0] < m.shape[-1] else m[..., 0]
    return (m > 0).astype(np.uint8)


def wallis_filter(img, sigma=16.0, mu_t=0.5, sd_t=0.2, c=0.8, b=0.6):
    """Wallis adaptive contrast normalisation (per channel, Gaussian window): pulls local mean/std toward (mu_t, sd_t).
    Used by LandsDANet (RS 2026) as radiometric pre-processing for cross-sensor landslide mapping."""
    m = cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101)
    v = cv2.GaussianBlur(img * img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101) - m * m
    sd = np.sqrt(np.maximum(v, 1e-6))
    out = (img - m) * (c * sd_t / (c * sd + (1 - c) * sd_t)) + b * mu_t + (1 - b) * m
    return np.clip(out, 0, 1).astype(np.float32)


def resample(img, msk, s, psf_k=0.0):
    """img float32 HxWx3 in [0,1], msk uint8 HxW (0/1/255). Bilinear for image, nearest for mask.
    psf_k > 0: sensor-aware (PSF/MTF) downsampling -- blur with a Gaussian of sigma = psf_k * 0.5 / s pixels before
    decimating, i.e. simulate what a coarser sensor's optics actually integrate, instead of a purely geometric resize.
    Only applies when s < 1 (going coarser); the built-in antialias filter is then switched off to avoid double
    filtering. Costs nothing at inference when the test tiles are the coarse ones (training-side operation)."""
    if abs(s - 1) < 0.02:
        return img, msk
    h, w = msk.shape
    nh, nw = max(32, int(round(h * s))), max(32, int(round(w * s)))
    if psf_k > 0 and s < 1:
        sig = psf_k * 0.5 / s
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=sig, sigmaY=sig, borderType=cv2.BORDER_REFLECT101)
    ti = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None]
    img = F.interpolate(ti, (nh, nw), mode="bilinear", align_corners=False, antialias=(s < 1 and psf_k <= 0))[0].permute(1, 2, 0).numpy()
    tm = torch.from_numpy(np.ascontiguousarray(msk))[None, None].float()
    msk = F.interpolate(tm, (nh, nw), mode="nearest")[0, 0].numpy().astype(np.uint8)
    return np.ascontiguousarray(img), msk


class CASTiles(Dataset):
    def __init__(self, keys, index, raw_root, train, crop=512, gsd_aug=None, gsd_target=0.0, gsd_clamp=(0.25, 2.0),
                 fam_prob=0.0, fam_beta=(0.01, 0.09), fam_alpha=1.0, boundary_w=0.0, boundary_sigma=8.0, psf_k=0.0, deg_prob=0.0, deg_ratio=(1.5, 5.0), mosaic=0, gsd_lookup=None, gsd_anchors=None, fam_ref_keys=None, scale_mult=1.0, wallis=False):
        self.keys, self.index, self.raw, self.train, self.crop = keys, index, raw_root, train, crop
        self.gsd_aug = gsd_aug          # (lo, hi): symmetric random scale jitter (negative control)
        self.gsd_target = gsd_target    # M1: canonical target GSD in metres (0 = off)
        self.gsd_clamp = gsd_clamp
        self.fam_prob, self.fam_beta, self.fam_alpha = fam_prob, fam_beta, fam_alpha   # M2
        self.boundary_w, self.boundary_sigma = boundary_w, boundary_sigma               # M3
        self.psf_k = psf_k                                                              # M1b: sensor-aware kernel
        self.deg_prob, self.deg_ratio = deg_prob, deg_ratio                             # M2-new: degradation sim
        self.gsd_lookup, self.gsd_anchors = gsd_lookup, (list(gsd_anchors) if gsd_anchors else None)  # M1 level-2
        self.fam_target = fam_ref_keys is not None
        self.scale_mult = float(scale_mult)   # dataset-level scale factor (external targets without per-tile GSD)
        self.wallis = bool(wallis)            # LandsDANet-style radiometric normalisation (Wallis filter), train + eval
        self.mosaic = int(mosaic)             # M3 candidate 5: tiles smaller than the crop after M1 are packed, not reflect-padded
        if fam_prob > 0:  # reference pools per subset: own keys (cross-subset mixing) or target keys (FDA-style UDA)
            self.by_sid = defaultdict(list)
            for k in (fam_ref_keys if fam_ref_keys is not None else keys):
                self.by_sid[index[k]["sid"]].append(k)
            self.sids = sorted(self.by_sid)

    def __len__(self):
        return len(self.keys)

    def tile_gsd(self, r):
        """nominal GSD from metadata, or the effective GSD estimated from the spectrum (M1 level-2)."""
        if self.gsd_lookup is not None:
            v = self.gsd_lookup.get(f"{r['sid']}/{r['stem']}")
            if v is not None:
                return v
        return float(r["gsd_m"])

    def scale_for(self, r):
        g = self.tile_gsd(r)
        if self.gsd_anchors:  # anchor set: snap to the nearest source GSD (log distance); source tiles stay native
            tgt = min(self.gsd_anchors, key=lambda a_: abs(math.log(g) - math.log(a_)))
            s = g / tgt
        elif self.gsd_target:
            s = g / self.gsd_target
        else:
            s = 1.0
        s *= self.scale_mult
        return float(min(max(s, self.gsd_clamp[0]), self.gsd_clamp[1]))

    def _ref_tile(self, sid):
        others = self.sids if self.fam_target else ([s for s in self.sids if s != sid] or self.sids)
        rs = random.choice(others)
        k = random.choice(self.by_sid[rs])
        return read_img(os.path.join(self.raw, self.index[k]["img"])).astype(np.float32) / 255.0

    def _load_scaled(self, key):
        r = self.index[key]
        img = read_img(os.path.join(self.raw, r["img"])).astype(np.float32) / 255.0
        msk = read_mask(os.path.join(self.raw, r["mask"]))
        if self.wallis:
            img = wallis_filter(img)
        return resample(img, msk, self.scale_for(r), self.psf_k)

    def _mosaic_fill(self, img, msk, i):
        """M1's clip leaves fine-GSD tiles smaller than the training crop (a 0.2 m tile at a 1 m anchor is 128 px), and
        reflect-padding them wastes 15/16 of the crop on synthetic content whose mask is ignored. Instead pack the crop
        with real tiles: read up to `mosaic` further tiles, then tile the canvas with the loaded ones under random flips."""
        h, w = msk.shape
        pool = [(img, msk)]
        for _ in range(max(0, self.mosaic - 1)):
            j = random.randrange(len(self.keys))
            if j == i:
                continue
            im2, mk2 = self._load_scaled(self.keys[j])
            if im2.shape[0] >= h and im2.shape[1] >= w:  # same or larger: take a random window of the same size
                y0, x0 = random.randint(0, im2.shape[0] - h), random.randint(0, im2.shape[1] - w)
                pool.append((im2[y0:y0 + h, x0:x0 + w], mk2[y0:y0 + h, x0:x0 + w]))
        cimg = np.zeros((self.crop, self.crop, 3), np.float32)
        cmsk = np.full((self.crop, self.crop), IGNORE, msk.dtype)
        for y in range(0, self.crop, h):
            for x in range(0, self.crop, w):
                a_, b_ = pool[random.randrange(len(pool))]
                if random.random() < 0.5:
                    a_, b_ = a_[:, ::-1], b_[:, ::-1]
                if random.random() < 0.5:
                    a_, b_ = a_[::-1], b_[::-1]
                hh, ww = min(h, self.crop - y), min(w, self.crop - x)
                cimg[y:y + hh, x:x + ww] = a_[:hh, :ww]
                cmsk[y:y + hh, x:x + ww] = b_[:hh, :ww]
        return cimg, cmsk

    def __getitem__(self, i):
        r = self.index[self.keys[i]]
        img = read_img(os.path.join(self.raw, r["img"])).astype(np.float32) / 255.0
        msk = read_mask(os.path.join(self.raw, r["mask"]))
        native = msk.copy()
        if self.wallis:
            img = wallis_filter(img)
        s = self.scale_for(r)
        if self.train and self.gsd_aug is not None:
            s *= math.exp(random.uniform(math.log(self.gsd_aug[0]), math.log(self.gsd_aug[1])))
        if self.train and self.deg_prob > 0 and random.random() < self.deg_prob:
            # M2-new: sensor-degradation simulation AT THE CANONICAL SCALE. A coarse-sensor tile upsampled by M1 to
            # the canonical GSD is a blurred, aliased version of a fine tile; simulate exactly that on the source
            # imagery (blur with the coarser sensor's PSF, decimate by r, put back on the canonical grid) so the
            # model learns the appearance of landslides that were only a few pixels wide in the raw image.
            r = math.exp(random.uniform(math.log(self.deg_ratio[0]), math.log(self.deg_ratio[1])))
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=0.5 * r, sigmaY=0.5 * r, borderType=cv2.BORDER_REFLECT101)
            hh, ww = img.shape[:2]
            small = cv2.resize(img, (max(8, int(round(ww / r))), max(8, int(round(hh / r)))), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (ww, hh), interpolation=cv2.INTER_LINEAR)
        if self.train and self.fam_prob > 0 and random.random() < self.fam_prob:  # M2 at native resolution
            beta = random.uniform(*self.fam_beta)
            img = fourier_amplitude_mix(img, self._ref_tile(r["sid"]), beta, self.fam_alpha)
        img, msk = resample(img, msk, s, self.psf_k)
        if self.train:
            h, w = msk.shape
            ph, pw = max(0, self.crop - h), max(0, self.crop - w)
            if (ph or pw) and self.mosaic >= 1:  # pack the crop with real tiles instead of padding it (1 = tile the crop with the tile itself under random flips, no extra reads)
                img, msk = self._mosaic_fill(img, msk, i)
                h, w = msk.shape
            elif ph or pw:  # small resampled tiles: reflect-pad image, ignore-pad mask
                img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
                msk = np.pad(msk, ((0, ph), (0, pw)), mode="constant", constant_values=IGNORE)
                h, w = msk.shape
            if h > self.crop or w > self.crop:
                y, x = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
                img, msk = img[y:y + self.crop, x:x + self.crop], msk[y:y + self.crop, x:x + self.crop]
            if random.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if random.random() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = random.randint(0, 3)
            if k:
                img, msk = np.rot90(img, k), np.rot90(msk, k)
            if random.random() < 0.8:
                img = np.clip(img * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08), 0, 1)
            msk = np.ascontiguousarray(msk)
            wgt = boundary_weight(msk, self.boundary_sigma, self.boundary_w) if self.boundary_w > 0 else np.ones(msk.shape, np.float32)
            img = (img - MEAN) / STD
            return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float(),
                    torch.from_numpy(msk).long(), torch.from_numpy(wgt).float(), i)
        # eval: resampled image, NATIVE mask; metrics are computed at native resolution
        img = (img - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float(),
                torch.from_numpy(native).long(), i)


# ----------------------------------------------------------------------------- model / loss
def build_model(arch, encoder, weights="imagenet"):
    kw = dict(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1)
    if arch == "segformer":
        return smp.Segformer(**kw)
    if arch == "unet":
        return smp.Unet(**kw)
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(**kw)
    if arch == "upernet":
        return smp.UPerNet(**kw)
    raise ValueError(arch)


def finer_view(x, y, wgt, rho):
    """M3 level-2: centre window of side S/rho upsampled back to S (a rho-times finer look at the same ground).
    Returns the image view, its labels/weights, and the matching centre crop of the coarse tensors."""
    S = x.shape[-1]
    c = max(32, int(round(S / rho)))
    o = (S - c) // 2
    sl = (slice(None), Ellipsis, slice(o, o + c), slice(o, o + c))
    xb = F.interpolate(x[:, :, o:o + c, o:o + c], (S, S), mode="bilinear", align_corners=False)
    yb = F.interpolate(y[:, None, o:o + c, o:o + c].float(), (S, S), mode="nearest")[:, 0].long()
    wb = F.interpolate(wgt[:, None, o:o + c, o:o + c], (S, S), mode="nearest")[:, 0]
    return xb, yb, wb, (o, c)


class BCEDice(nn.Module):
    def __init__(self, w_dice=1.0):
        super().__init__()
        self.w = w_dice

    def forward(self, logit, y, wgt=None):
        valid = (y != IGNORE).float()
        if wgt is None:
            wgt = valid
        else:
            wgt = wgt * valid
        yf = (y == 1).float()
        bce = (F.binary_cross_entropy_with_logits(logit, yf, reduction="none") * wgt).sum() / wgt.sum().clamp(min=1)
        p = torch.sigmoid(logit) * valid
        inter = (p * yf).sum((1, 2))
        dice = 1 - (2 * inter + 1) / (p.sum((1, 2)) + (yf * valid).sum((1, 2)) + 1)
        return bce + self.w * dice.mean()


# ----------------------------------------------------------------------------- inference helpers
@torch.no_grad()
def predict_logits(model, x, win=512, stride=256):
    """x: 1x3xHxW (any size). Sliding-window logits at the same size (mean over overlaps)."""
    _, _, H, W = x.shape
    if H <= win and W <= win:  # small (down-sampled) tiles: whole-image forward, pad only to a multiple of 32
        ph, pw = (-H) % 32, (-W) % 32
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(xp)
        return lg.float()[:, :, :H, :W]
    ph, pw = (-H) % 32, (-W) % 32
    xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
    Hp, Wp = xp.shape[-2:]
    out = torch.zeros(1, 1, Hp, Wp, device=x.device); cnt = torch.zeros(1, 1, Hp, Wp, device=x.device)
    ys = list(range(0, max(Hp - win, 0) + 1, stride)); xs = list(range(0, max(Wp - win, 0) + 1, stride))
    if ys[-1] + win < Hp: ys.append(Hp - win)
    if xs[-1] + win < Wp: xs.append(Wp - win)
    tiles, coords = [], []
    for yy in ys:
        for xx in xs:
            tiles.append(xp[:, :, yy:yy + win, xx:xx + win]); coords.append((yy, xx))
    for i in range(0, len(tiles), 16):
        b = torch.cat(tiles[i:i + 16])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b).float()
        for (yy, xx), l in zip(coords[i:i + 16], lg):
            out[:, :, yy:yy + win, xx:xx + win] += l; cnt[:, :, yy:yy + win, xx:xx + win] += 1
    return (out / cnt)[:, :, :H, :W]


# ----------------------------------------------------------------------------- metrics
class Confusion:
    """pixel-accumulated confusion for the landslide class, per group."""
    def __init__(self):
        self.c = defaultdict(lambda: np.zeros(4, np.int64))  # tp, fp, fn, tn

    def add(self, pred, gt, group):
        pred, gt = pred.astype(bool), gt.astype(bool)
        tp = np.logical_and(pred, gt).sum(); fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum(); tn = pred.size - tp - fp - fn
        v = np.array([tp, fp, fn, tn], np.int64)
        self.c["ALL"] += v
        for g in group:
            self.c[g] += v

    @staticmethod
    def summarize(v):
        tp, fp, fn, tn = [float(x) for x in v]
        iou = tp / max(tp + fp + fn, 1); iou_bg = tn / max(tn + fp + fn, 1)
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        return dict(IoU=iou, F1=f1, Precision=p, Recall=r, mIoU=(iou + iou_bg) / 2, OA=(tp + tn) / max(tp + fp + fn + tn, 1))

    def report(self):
        return {k: self.summarize(v) for k, v in self.c.items()}


@torch.no_grad()
def evaluate(model, loader, index, keys, device, thresholds=(0.5,), tta=False, resampled=False, stride=256):
    model.eval()
    confs = {t: Confusion() for t in thresholds}
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        if resampled:  # batch of 1, variable size: sliding window then resample probabilities back to native
            H, W = y.shape[-2:]
            logit = predict_logits(model, x, stride=stride)
            if tta:
                logit = (logit + torch.flip(predict_logits(model, torch.flip(x, [-1]), stride=stride), [-1])) / 2
            if logit.shape[-2:] != (H, W):
                logit = F.interpolate(logit, (H, W), mode="bilinear", align_corners=False)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x)
                if tta:
                    logit = (logit + torch.flip(model(torch.flip(x, [-1])), [-1])) / 2
        prob = torch.sigmoid(logit.float())[:, 0].cpu().numpy()
        y = y.numpy()
        for b in range(len(idx)):
            r = index[keys[int(idx[b])]]
            grp = (f"sid:{r['sid']}", f"region:{r['region']}", f"platform:{r['platform']}")
            for t, c in confs.items():
                c.add(prob[b] > t, y[b] == 1, grp)
    return {t: c.report() for t, c in confs.items()}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--index", default="splits/index.csv")
    ap.add_argument("--raw", default="data/CAS_raw")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="segformer")
    ap.add_argument("--encoder", default="mit_b2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--lr_head_mult", type=float, default=10.0)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gsd_aug", default="", help="negative control: e.g. 0.5,2.0 symmetric random scale jitter")
    ap.add_argument("--scale_mult", type=float, default=1.0, help="dataset-level resampling factor applied to every tile (eval-only probes on external data)")
    ap.add_argument("--mosaic", type=int, default=0, help="M3 candidate 5: pack crops of down-sampled tiles with this many distinct real tiles instead of reflect-padding (0 = off, 1 = repeat the tile itself under flips)")
    ap.add_argument("--gsd_target", type=float, default=0.0, help="M1: canonical target GSD in metres (0 = off)")
    ap.add_argument("--gsd_clamp", default="0.25,2.0", help="M1: clamp of the resampling factor")
    ap.add_argument("--gsd_source", default="nominal", choices=["nominal", "eff_tile", "eff_subset"], help="M1 level-2: which GSD drives the resampling")
    ap.add_argument("--effgsd", default="splits/effgsd.csv")
    ap.add_argument("--gsd_anchors", default="", help="M1 anchor set: comma list of GSDs, or 'auto' = distinct GSDs of the training split; each tile is resampled to the nearest anchor")
    ap.add_argument("--fam_prob", type=float, default=0.0, help="M2: prob of cross-subset Fourier amplitude mixing")
    ap.add_argument("--fam_beta", default="0.01,0.09", help="M2: low-frequency band half-width range")
    ap.add_argument("--fam_ref", default="source", choices=["source", "target"], help="M2/FDA: amplitude reference pool; target = unlabelled test-split tiles (UDA baseline FDA)")
    ap.add_argument("--fam_alpha", type=float, default=1.0, help="M2: mixing weight (1 = full replacement)")
    ap.add_argument("--ain_stages", default="", help="M2 level-2: MiT stages to attach AmplitudeNorm to, e.g. 1,2")
    ap.add_argument("--ain_alpha", type=float, default=0.5, help="M2 level-2: initial mixing weight toward the source mean spectrum")
    ap.add_argument("--ain_beta", type=float, default=0.05, help="M2 level-2: low-frequency band radius as a fraction of feature height")
    ap.add_argument("--boundary_w", type=float, default=0.0, help="M3: boundary weight amplitude (0 = off)")
    ap.add_argument("--boundary_sigma", type=float, default=8.0, help="M3: boundary band width in px")
    ap.add_argument("--deg_prob", type=float, default=0.0, help="M2-new: prob of simulating a coarser sensor at the canonical scale")
    ap.add_argument("--deg_ratio", default="1.5,5.0", help="M2-new: range of the simulated resolution ratio")
    ap.add_argument("--psf_k", type=float, default=0.0, help="M1b: sensor-aware (PSF) blur before downsampling, sigma = psf_k*0.5/s px (0 = plain antialiased resize)")
    ap.add_argument("--consist_w", type=float, default=0.0, help="M3 level-2: weight of the cross-GSD consistency term (0 = off)")
    ap.add_argument("--consist_rho", type=float, default=2.0, help="M3 level-2: scale ratio of the finer view")
    ap.add_argument("--eval_stride", type=int, default=256)
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--max_train", type=int, default=0, help="debug: cap train tiles")
    ap.add_argument("--max_eval", type=int, default=0, help="debug: cap val/test tiles")
    ap.add_argument("--eval_only", default="", help="path to checkpoint: skip training, evaluate on val+test")
    ap.add_argument("--init", default="", help="checkpoint to initialise the weights from before training (fine-tuning / oracle rows)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    out = os.path.join(a.runs, a.name); os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.txt"), "a")

    def P(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    P("args", json.dumps(vars(a)))
    index = load_index(a.index)
    sp = json.load(open(a.split))
    tr_keys = [k for k in sp["train"] if k in index]
    va_keys = [k for k in sp["val"] if k in index]
    te_keys = [k for k in sp["test"] if k in index]
    if a.max_train:
        random.Random(0).shuffle(tr_keys); tr_keys = tr_keys[:a.max_train]
    if a.max_eval:
        random.Random(0).shuffle(va_keys); va_keys = va_keys[:a.max_eval]
        random.Random(0).shuffle(te_keys); te_keys = te_keys[:a.max_eval]
    P(f"split {os.path.basename(a.split)}: train {len(tr_keys)} val {len(va_keys)} test {len(te_keys)}  ({sp.get('note','')})")
    gsd = tuple(float(x) for x in a.gsd_aug.split(",")) if a.gsd_aug else None
    clamp = tuple(float(x) for x in a.gsd_clamp.split(","))
    fam_beta = tuple(float(x) for x in a.fam_beta.split(","))
    deg_ratio = tuple(float(x) for x in a.deg_ratio.split(","))
    resampled = a.gsd_target > 0
    if resampled:
        P(f"M1 canonical resampling: target GSD {a.gsd_target} m, clamp {clamp}; per-subset factors: " +
          ", ".join(f"{sid}={min(max(s.gsd_m / a.gsd_target, clamp[0]), clamp[1]):.2f}" for sid, s in sorted(BY_ID.items())))
    if a.fam_prob > 0:
        P(f"M2 Fourier amplitude mixing: prob {a.fam_prob}, beta {fam_beta}, alpha {a.fam_alpha}")
    if a.deg_prob > 0:
        P(f"M2-new sensor-degradation simulation: prob {a.deg_prob}, ratio {deg_ratio}")
    if a.psf_k > 0:
        P(f"M1b sensor-aware PSF downsampling: k {a.psf_k}")
    if a.consist_w > 0:
        P(f"M3 level-2 cross-GSD consistency: w {a.consist_w}, rho {a.consist_rho}")
    if a.boundary_w > 0:
        P(f"M3 boundary weighting: w {a.boundary_w}, sigma {a.boundary_sigma}px")

    dl_kw = dict(num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    gsd_lookup = None
    if a.gsd_source != "nominal":
        col = "gsd_eff_tile" if a.gsd_source == "eff_tile" else "gsd_eff_subset"
        with open(a.effgsd) as f:
            gsd_lookup = {row["key"]: float(row[col]) for row in csv.DictReader(f)}
        P(f"M1 level-2: resampling driven by {a.gsd_source} GSD from {a.effgsd} ({len(gsd_lookup)} tiles)")
    anchors = None
    if a.gsd_anchors:
        if a.gsd_anchors == "auto":
            src = sorted({(gsd_lookup.get(k) if gsd_lookup else None) or float(index[k]["gsd_m"]) for k in tr_keys})
            anchors = sorted({round(v, 2) for v in src})
        else:
            anchors = [float(v) for v in a.gsd_anchors.split(",")]
        P(f"M1 anchor set: {anchors} (each tile snaps to the nearest anchor in log-GSD)")
    resampled = resampled or bool(anchors) or a.scale_mult != 1.0
    ds_kw = dict(gsd_target=a.gsd_target, gsd_clamp=clamp, psf_k=a.psf_k, gsd_lookup=gsd_lookup, gsd_anchors=anchors, scale_mult=a.scale_mult, mosaic=a.mosaic)
    fam_ref = te_keys if (a.fam_prob > 0 and a.fam_ref == "target") else None
    if fam_ref is not None:
        P(f"FDA-style UDA: amplitude references drawn from {len(fam_ref)} unlabelled target tiles")
    tr_ds = CASTiles(tr_keys, index, a.raw, True, gsd_aug=gsd, deg_prob=a.deg_prob, deg_ratio=deg_ratio, fam_ref_keys=fam_ref, fam_prob=a.fam_prob, fam_beta=fam_beta, fam_alpha=a.fam_alpha,
                     boundary_w=a.boundary_w, boundary_sigma=a.boundary_sigma, **ds_kw)
    tr_dl = DataLoader(tr_ds, batch_size=a.bs, shuffle=True, drop_last=True, **dl_kw)
    ebs = 1 if resampled else a.bs
    va_dl = DataLoader(CASTiles(va_keys, index, a.raw, False, **ds_kw), batch_size=ebs, shuffle=False, **dl_kw)
    te_dl = DataLoader(CASTiles(te_keys, index, a.raw, False, **ds_kw), batch_size=ebs, shuffle=False, **dl_kw)

    model = build_model(a.arch, a.encoder).to(device).to(memory_format=torch.channels_last)
    if a.ain_stages:
        stg = [int(v) for v in a.ain_stages.split(",")]
        attach_amplitude_norm(model, stg, a.ain_alpha, a.ain_beta)
        model.to(device)
        P(f"M2 level-2 AmplitudeNorm on MiT stages {stg}: alpha_init {a.ain_alpha}, band {a.ain_beta}")
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    P(f"model {a.arch}/{a.encoder} params {n_par:.2f}M")
    ev = lambda dl, keys, ths=(0.5,), stride=a.eval_stride: evaluate(model, dl, index, keys, device, ths, resampled=resampled, stride=stride)

    if a.init and not a.eval_only:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"]); P(f"init from {a.init}")
    if a.eval_only:
        model.load_state_dict(torch.load(a.eval_only, map_location="cpu")["model"])
    else:
        enc_params = list(model.encoder.parameters()); enc_ids = {id(p) for p in enc_params}
        head_params = [p for p in model.parameters() if id(p) not in enc_ids]
        opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": head_params, "lr": a.lr * a.lr_head_mult}], weight_decay=a.wd)
        total = a.epochs * len(tr_dl); warm = min(500, total // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (it + 1) / warm if it < warm else (1 - (it - warm) / max(1, total - warm)) ** 1.0)
        crit = BCEDice()
        best, best_ep, it = -1, -1, 0
        for ep in range(a.epochs):
            model.train(); t0 = time.time(); run = 0.0
            for x, y, wgt, _ in tr_dl:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True); wgt = wgt.to(device, non_blocking=True)
                if a.consist_w > 0:  # M3 level-2: coarse view + rho-times finer view of the same ground
                    xb, yb, wb, (o, c) = finer_view(x, y, wgt, a.consist_rho)
                    xb = xb.to(memory_format=torch.channels_last)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logit = model(torch.cat([x, xb], 0))[:, 0]
                    la, lb = logit[: x.shape[0]].float(), logit[x.shape[0]:].float()
                    loss = crit(la, y, wgt) + crit(lb, yb, wb)
                    pa = torch.sigmoid(F.interpolate(la[:, None, o:o + c, o:o + c], lb.shape[-2:], mode="bilinear", align_corners=False)[:, 0])
                    pb = torch.sigmoid(lb)
                    valid = (yb != IGNORE).float()
                    loss = loss + a.consist_w * (((pa - pb) ** 2) * valid).sum() / valid.sum().clamp(min=1)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logit = model(x)[:, 0]
                    loss = crit(logit.float(), y, wgt)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); it += 1; run += loss.item()
            msg = f"ep {ep+1}/{a.epochs} loss {run/len(tr_dl):.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s"
            if (ep + 1) % a.val_every == 0 or ep + 1 == a.epochs:
                t1 = time.time()
                rep = ev(va_dl, va_keys, stride=512)[0.5]["ALL"]
                msg += f" | val IoU {rep['IoU']:.4f} F1 {rep['F1']:.4f} P {rep['Precision']:.4f} R {rep['Recall']:.4f} ({time.time()-t1:.0f}s)"
                if rep["IoU"] > best:
                    best, best_ep = rep["IoU"], ep + 1
                    torch.save({"model": model.state_dict(), "epoch": ep + 1, "val_IoU": best, "args": vars(a)}, os.path.join(out, "best.pt"))
                    msg += " *"
            P(msg)
        P(f"best val IoU {best:.4f} at epoch {best_ep}")
        model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location="cpu")["model"])

    ths = (0.3, 0.4, 0.5, 0.6, 0.7)
    rv = ev(va_dl, va_keys, ths)
    best_t = max(ths, key=lambda t: rv[t]["ALL"]["IoU"])
    json.dump({"per_threshold_ALL": {str(t): rv[t]["ALL"] for t in ths}, "best_threshold": best_t, "at_0.5": rv[0.5]}, open(os.path.join(out, "metrics_val.json"), "w"), indent=1)
    rt = ev(te_dl, te_keys, (0.5, best_t))
    json.dump({"at_0.5": rt[0.5], "at_val_best_threshold": {"t": best_t, **rt[best_t]}}, open(os.path.join(out, "metrics_test.json"), "w"), indent=1)
    A = rt[0.5]["ALL"]
    P(f"TEST@0.5  IoU {A['IoU']:.4f}  F1 {A['F1']:.4f}  P {A['Precision']:.4f}  R {A['Recall']:.4f}  mIoU {A['mIoU']:.4f}   (val-best t={best_t}: IoU {rt[best_t]['ALL']['IoU']:.4f})")
    for k in sorted(rt[0.5]):
        if k.startswith("platform:") or k.startswith("region:"):
            v = rt[0.5][k]; P(f"   {k:22s} IoU {v['IoU']:.4f} F1 {v['F1']:.4f}")
    P("DONE")


if __name__ == "__main__":
    main()
