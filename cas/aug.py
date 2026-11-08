"""Method modules that live outside the network.

M2 level-1 -- cross-subset Fourier amplitude mixing (multi-source FDA-style appearance randomisation):
    fourier_amplitude_mix(img, ref, beta, alpha) : swap/mix the low-frequency amplitude spectrum of `img` with that of a
    reference tile from another subset, phase untouched. img/ref float32 HxWx3 in [0,1]; ref is resized to img's size.
    beta = half-width of the low-frequency square as a fraction of the spectrum (FDA uses 0.01-0.09);
    alpha in [0,1] = mixing weight (1 -> full replacement like FDA, <1 -> FOSMix-style interpolation).

M3 level-1 -- boundary distance weighting for the pixel loss:
    boundary_weight(mask, sigma_px, w_max) : per-pixel weight 1 + w_max*exp(-d^2/(2 sigma^2)) where d is the distance (px)
    to the nearest landslide boundary (both sides). Ignore pixels (255) get weight 0.
"""
import numpy as np
import cv2


def fourier_amplitude_mix(img, ref, beta=0.05, alpha=1.0):
    h, w = img.shape[:2]
    if ref.shape[:2] != (h, w):
        ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA if ref.shape[0] > h else cv2.INTER_LINEAR)
    out = np.empty_like(img)
    b = max(1, int(np.floor(min(h, w) * beta)))
    cy, cx = h // 2, w // 2
    for c in range(3):
        fi = np.fft.fftshift(np.fft.fft2(img[..., c]))
        fr = np.fft.fftshift(np.fft.fft2(ref[..., c]))
        amp_i, pha_i = np.abs(fi), np.angle(fi)
        amp_r = np.abs(fr)
        amp_m = amp_i.copy()
        y0, y1, x0, x1 = cy - b, cy + b + 1, cx - b, cx + b + 1
        amp_m[y0:y1, x0:x1] = (1 - alpha) * amp_i[y0:y1, x0:x1] + alpha * amp_r[y0:y1, x0:x1]
        rec = np.fft.ifft2(np.fft.ifftshift(amp_m * np.exp(1j * pha_i)))
        out[..., c] = np.real(rec)
    return np.clip(out, 0, 1).astype(np.float32)


def boundary_weight(mask, sigma_px=8.0, w_max=3.0):
    """mask uint8 HxW with 0/1/255. Returns float32 HxW weights (0 on ignore)."""
    valid = mask != 255
    fg = (mask == 1).astype(np.uint8)
    if fg.sum() == 0 or fg.sum() == fg.size:
        w = np.ones(mask.shape, np.float32)
    else:
        # distance to boundary = min(distance inside fg to bg, distance inside bg to fg)
        d_in = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
        d_out = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 3)
        d = np.where(fg == 1, d_in, d_out).astype(np.float32)
        w = 1.0 + w_max * np.exp(-(d ** 2) / (2 * sigma_px ** 2))
    w[~valid] = 0.0
    return w.astype(np.float32)


if __name__ == "__main__":  # tiny self-test
    rng = np.random.default_rng(0)
    a = rng.random((64, 64, 3), dtype=np.float32); r = rng.random((32, 32, 3), dtype=np.float32)
    m = fourier_amplitude_mix(a, r, 0.05, 1.0)
    assert m.shape == a.shape and m.dtype == np.float32
    msk = np.zeros((64, 64), np.uint8); msk[20:40, 20:40] = 1; msk[:4] = 255
    w = boundary_weight(msk)
    assert w[:4].max() == 0 and w[20, 20] > w[30, 30] and w[50, 50] < w[20, 20]
    print("aug self-test OK", float(w.max()), float(w[30, 30]))
