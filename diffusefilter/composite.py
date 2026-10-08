"""Operators that transfer a filter image onto a training image (both in [0, 1])."""
import math

import torch
import torch.fft as fft
import torch.nn.functional as F


def lowfreq_replace(x, style, alpha=0.5, radius=4):
    """Low-frequency replacement (LFR) with a circular mask and amplitude normalization."""
    h, w = x.shape[-2:]
    rw = w // 2 + 1

    X = fft.rfft2(x, dim=(-2, -1))
    S = fft.rfft2(style, dim=(-2, -1))

    ys = torch.arange(h, device=x.device)
    xs = torch.arange(rw, device=x.device)
    ys = torch.min(ys, h - ys)
    dist = (ys[:, None] ** 2 + xs[None, :] ** 2).float().sqrt()
    mask = dist < radius

    X_mixed = X.clone()
    amp_x = X[..., mask].abs().mean(dim=-1, keepdim=True) + 1e-5
    amp_s = S[..., mask].abs().mean(dim=-1, keepdim=True) + 1e-5
    S_normalized = S[..., mask] * (amp_x / amp_s)
    X_mixed[..., mask] = (1.0 - alpha) * X[..., mask] + alpha * S_normalized

    result = fft.irfft2(X_mixed, s=(h, w), dim=(-2, -1))
    return result.clamp(0.0, 1.0)


def lowfreq_weight_map(h, w, radius, mask_shape="circle", device="cpu"):
    """Low-frequency weight map in rfft2 layout [h, w // 2 + 1]."""
    rw = w // 2 + 1
    ys = torch.arange(h, device=device)
    xs = torch.arange(rw, device=device)
    ys = torch.min(ys, h - ys).float()
    xs = xs.float()
    if mask_shape == "circle":
        d = (ys[:, None] ** 2 + xs[None, :] ** 2).sqrt()
        W = (d < radius).float()
    elif mask_shape == "square":
        d = torch.max(ys[:, None].expand(h, rw), xs[None, :].expand(h, rw))
        W = (d < radius).float()
    elif mask_shape == "gaussian":
        # sigma is chosen so that the weight is 0.5 at d = radius.
        d2 = ys[:, None] ** 2 + xs[None, :] ** 2
        sigma = max(float(radius), 1e-6) / math.sqrt(2.0 * math.log(2.0))
        W = torch.exp(-d2 / (2.0 * sigma ** 2))
    else:
        raise ValueError(f"unknown mask_shape: {mask_shape}")
    return W


def lowfreq_replace_general(x, style, alpha=0.5, radius=4, mask_shape="circle", amp_norm=True):
    """LFR with a configurable mask and optional amplitude normalization."""
    h, w = x.shape[-2:]
    X = fft.rfft2(x.float(), dim=(-2, -1))
    S = fft.rfft2(style.float(), dim=(-2, -1))
    W = lowfreq_weight_map(h, w, radius, mask_shape, device=x.device)
    if amp_norm:
        wsum = W.sum().clamp_min(1e-8)
        amp_x = (X.abs() * W).sum(dim=(-2, -1), keepdim=True) / wsum + 1e-5
        amp_s = (S.abs() * W).sum(dim=(-2, -1), keepdim=True) / wsum + 1e-5
        S_hat = S * (amp_x / amp_s)
    else:
        S_hat = S
    X_mixed = X + (alpha * W) * (S_hat - X)
    result = fft.irfft2(X_mixed, s=(h, w), dim=(-2, -1))
    return result.clamp(0.0, 1.0)


def fda_amplitude_mix(x, style, alpha=1.0, radius=4, mask_shape="square", amp_norm=False):
    """FDA-style amplitude swap (Yang & Soatto, CVPR 2020); a very large radius gives APR."""
    h, w = x.shape[-2:]
    X = fft.rfft2(x.float(), dim=(-2, -1))
    S = fft.rfft2(style.float(), dim=(-2, -1))
    W = lowfreq_weight_map(h, w, radius, mask_shape, device=x.device)
    amp_x, pha_x = X.abs(), torch.angle(X)
    amp_s = S.abs()
    if amp_norm:
        wsum = W.sum().clamp_min(1e-8)
        mx = (amp_x * W).sum(dim=(-2, -1), keepdim=True) / wsum + 1e-5
        ms = (amp_s * W).sum(dim=(-2, -1), keepdim=True) / wsum + 1e-5
        amp_s = amp_s * (mx / ms)
    amp_new = amp_x + (alpha * W) * (amp_s - amp_x)
    X_new = torch.polar(amp_new, pha_x)
    result = fft.irfft2(X_new, s=(h, w), dim=(-2, -1))
    return result.clamp(0.0, 1.0)


def pixel_alpha_blend(x, style, alpha=0.3):
    return ((1.0 - alpha) * x + alpha * style).clamp(0.0, 1.0)


def reinhard_color_transfer(content, style):
    import kornia
    content_lab = kornia.color.rgb_to_lab(content)
    style_lab = kornia.color.rgb_to_lab(style)
    c_mean = content_lab.mean(dim=(2, 3), keepdim=True)
    c_std = content_lab.std(dim=(2, 3), keepdim=True) + 1e-5
    s_mean = style_lab.mean(dim=(2, 3), keepdim=True)
    s_std = style_lab.std(dim=(2, 3), keepdim=True)
    out_lab = (content_lab - c_mean) / c_std * s_std + s_mean
    out_rgb = kornia.color.lab_to_rgb(out_lab.clamp(0, 100))
    return out_rgb.clamp(0, 1)


COMPOSITES = ("lfr", "fda", "pixel", "reinhard", "colorjitter")


def composite(name, x, style, alpha=0.5, radius=4, mask_shape="circle", amp_norm=True):
    if style.shape[-2:] != x.shape[-2:]:
        style = F.interpolate(style, size=x.shape[-2:], mode="bilinear", align_corners=False)

    if name == "lfr":
        if mask_shape == "circle" and amp_norm:
            out = lowfreq_replace(x, style, alpha=alpha, radius=radius)
        else:
            out = lowfreq_replace_general(x, style, alpha=alpha, radius=radius,
                                          mask_shape=mask_shape, amp_norm=amp_norm)
    elif name == "fda":
        out = fda_amplitude_mix(x, style, alpha=alpha, radius=radius,
                                mask_shape=mask_shape, amp_norm=amp_norm)
    elif name == "pixel":
        out = pixel_alpha_blend(x, style, alpha=alpha)
    elif name == "reinhard":
        out = reinhard_color_transfer(x, style)
    elif name == "colorjitter":
        # Control: ColorJitter in place of the filter, inside the same pipeline.
        from .baselines import color_jitter01
        out = color_jitter01(x)
    else:
        raise ValueError(f"unknown composite: {name}")
    return out.clamp(0.0, 1.0)
