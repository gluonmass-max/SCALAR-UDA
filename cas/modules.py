"""M2 level-2: feature-level amplitude normalisation for the MiT encoder (segmentation-models-pytorch Segformer).

AmplitudeNorm re-styles a feature map toward the source-domain average amplitude spectrum:
  F = rfft2(f) per (sample, channel); phase kept untouched;
  in the low-frequency band (|ky| <= r, kx <= r, r = beta * H) the log-amplitude is pulled toward a running mean
  spectrum (per channel and frequency bin, accumulated over training batches):
      la' = (1 - alpha_c) * la + alpha_c * mu_run        (alpha_c learnable per channel, sigmoid-parameterised)
  outside the band the amplitude is unchanged; f' = irfft2(exp(la') * exp(i * phase)).
At initialisation (running mean = first batch) a typical source image passes almost unchanged; an image whose
low-frequency amplitude deviates from the source statistics (other sensor, other tone) is normalised toward them.
No reference images are needed, so it acts on unseen sensors at inference. Attached with forward hooks on
encoder.norm1 / norm2, the stage outputs of shape (B, C, H, W). The stored band is resampled bilinearly when the
feature size at inference differs from training (sliding windows, resampled tiles, whole-image forward).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AmplitudeNorm(nn.Module):
    def __init__(self, channels, alpha_init=0.5, beta=0.05, momentum=0.01, eps=1e-6):
        super().__init__()
        a = min(max(alpha_init, 1e-3), 1 - 1e-3)
        self.alpha_logit = nn.Parameter(torch.full((1, channels, 1, 1), math.log(a / (1 - a))))
        self.beta, self.momentum, self.eps = beta, momentum, eps
        self.register_buffer("mu_run", torch.zeros(0))       # (1, C, 2r+1, r+1), ky ordered -r..r; lazily sized
        self.register_buffer("initialised", torch.tensor(0))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "mu_run"
        if k in state_dict and state_dict[k].numel() != self.mu_run.numel():
            self.mu_run = torch.zeros_like(state_dict[k], device=self.alpha_logit.device)  # resize before strict load
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _band(self, H, W2):
        r = max(1, int(round(self.beta * H)))
        r = min(r, (H - 1) // 2, W2 - 1)
        return r, r + 1

    def forward(self, f):
        dtype = f.dtype
        x = f.float()
        with torch.autocast("cuda", enabled=False):
            Fx = torch.fft.rfft2(x, norm="ortho")                        # (B, C, H, W//2+1)
            B, C, H, W2 = Fx.shape
            if H < 4 or W2 < 3:
                return f
            amp = Fx.abs()
            pha = torch.angle(Fx)
            la = torch.log(amp + self.eps)
            r, w = self._band(H, W2)
            top = la[:, :, : r + 1, :w]                                  # ky 0..r
            bot = la[:, :, H - r :, :w]                                  # ky -r..-1
            band = torch.cat([bot, top], dim=2)                          # ky -r..r  -> (B, C, 2r+1, w)
            if self.training:
                with torch.no_grad():
                    m = band.mean(dim=0, keepdim=True)
                    if self.initialised.item() == 0 or self.mu_run.shape != m.shape:
                        self.mu_run = m.detach().clone(); self.initialised.fill_(1)
                    else:
                        self.mu_run.mul_(1 - self.momentum).add_(m.detach(), alpha=self.momentum)
            if self.mu_run.numel() == 0:                                 # eval before any training step
                return f
            mu = self.mu_run
            if mu.shape[-2:] != band.shape[-2:]:
                mu = F.interpolate(mu, size=band.shape[-2:], mode="bilinear", align_corners=True)
            alpha = torch.sigmoid(self.alpha_logit)
            band_n = (1 - alpha) * band + alpha * mu
            la = la.clone()
            la[:, :, H - r :, :w] = band_n[:, :, :r]
            la[:, :, : r + 1, :w] = band_n[:, :, r:]
            Fy = torch.polar(torch.clamp(torch.exp(la) - self.eps, min=0.0), pha)
            y = torch.fft.irfft2(Fy, s=(H, x.shape[-1]), norm="ortho")
        y = y.to(dtype)
        return y.contiguous(memory_format=torch.channels_last) if f.is_contiguous(memory_format=torch.channels_last) else y.contiguous()


def attach_amplitude_norm(model, stages=(1, 2), alpha_init=0.5, beta=0.05):
    """Register AmplitudeNorm after the given MiT stages (1..4) via forward hooks on encoder.normX."""
    enc = model.encoder
    for s in stages:
        ln = getattr(enc, f"norm{s}")
        mod = AmplitudeNorm(ln.weight.numel(), alpha_init, beta)
        setattr(enc, f"ain{s}", mod)
        ln.register_forward_hook(lambda m, i, o, mod=mod: mod(o))
    return model


if __name__ == "__main__":  # self-test
    import segmentation_models_pytorch as smp
    m = smp.Segformer(encoder_name="mit_b2", encoder_weights=None, classes=1)
    attach_amplitude_norm(m, (1, 2), 0.5, 0.05)
    m.train()
    x = torch.randn(2, 3, 256, 256)
    y = m(x); y.sum().backward()
    print("train ok", y.shape, tuple(m.encoder.ain1.mu_run.shape), m.encoder.ain1.alpha_logit.grad.abs().sum().item() > 0)
    m.eval()
    with torch.no_grad():
        y2 = m(x); y3 = m(torch.randn(1, 3, 512, 160))   # different feature size -> band interpolation
    print("eval ok", y2.shape, y3.shape, "params", sum(p.numel() for n, p in m.named_parameters() if "ain" in n))
    sd = m.state_dict()
    m2 = smp.Segformer(encoder_name="mit_b2", encoder_weights=None, classes=1); attach_amplitude_norm(m2, (1, 2))
    m2.load_state_dict(sd); print("reload ok", [k for k in sd if "ain" in k])
