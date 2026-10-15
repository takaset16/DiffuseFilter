"""Comparison augmentations, applied to normalized GPU batches (simplified re-implementations)."""
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from . import saliency
from .augment import IMAGENET_MEAN, IMAGENET_STD


def soft_labels(labels, perm, lam, num_classes):
    y = F.one_hot(labels.long(), num_classes).float() if labels.dim() == 1 else labels.float()
    lam = lam.float().view(-1, 1).to(y.device)
    return lam * y + (1.0 - lam) * y[perm.to(y.device)]


def _norm_stats(device, dtype=torch.float32):
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


# ---------------------------------------------------------------------------
# Geometric / erasing / noise
# ---------------------------------------------------------------------------

def vertical_flip(image, prob):
    image2 = image.clone()
    rand = np.random.rand(image.shape[0])
    reverse = torch.arange(image.shape[2] - 1, -1, -1)
    for i in range(image.shape[0]):
        if rand[i] <= prob:
            image2[i] = image[i, :, reverse, :]
    return image2


def random_crop(image, prob):
    n, c, h, w = image.shape
    padded = torch.zeros(n, c, h + 4, w + 4, device=image.device)
    rand = np.random.rand(n)
    pad = torch.nn.ZeroPad2d(2)
    for i in range(n):
        if rand[i] <= prob:
            for j in range(c):
                padded[i][j] = pad(image[i][j])
            top = np.random.randint(0, 4)
            left = np.random.randint(0, 4)
            image[i] = padded[i, :, top:top + h, left:left + w].clone()
    return image


def cutout(image, prob):
    image2 = image.clone()
    n, _, h, w = image.shape
    mask_size = int(h // 2.0)
    rand = np.random.rand(n)
    for i in range(n):
        if rand[i] <= prob:
            top = np.random.randint(-(mask_size // 2), h - mask_size // 2)
            left = np.random.randint(-(mask_size // 2), w - mask_size // 2)
            bottom, right = min(top + mask_size, h), min(left + mask_size, w)
            image2[i][:, max(top, 0):bottom, max(left, 0):right] = 0
    return image2


def _offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    return offset @ matrix @ reset


def _shear(img, level, prob, fixed, axis):
    level = int(level)
    device = img.device
    n, c, h, w = img.shape
    arr = img.to("cpu").detach().numpy().copy().transpose((0, 2, 3, 1))
    rand = np.random.rand(n)
    magnitudes = np.linspace(-0.3, 0.3, 7 if axis == "x" else 11)
    for i in range(n):
        if rand[i] <= prob:
            s = magnitudes[level] if fixed else random.uniform(0, magnitudes[level])
            m = np.array([[1, s, 0], [0, 1, 0], [0, 0, 1]]) if axis == "x" else np.array([[1, 0, 0], [s, 1, 0], [0, 0, 1]])
            m = _offset_center(m, h, w)
            arr[i] = np.stack([ndimage.affine_transform(arr[i][:, :, ch], m[:2, :2], m[:2, 2])
                               for ch in range(c)], axis=2)
    return torch.from_numpy(arr.transpose((0, 3, 1, 2)).astype(np.float32)).to(device)


def shear_x(img, level, prob, fixed):
    return _shear(img, level, prob, fixed, "x")


def shear_y(img, level, prob, fixed):
    return _shear(img, level, prob, fixed, "y")


def rotate(img, max_angle, prob):
    """Rotates each image by an angle drawn from U(-max_angle, max_angle), keeping the image size."""
    import kornia
    n = img.shape[0]
    angle = (torch.rand(n, device=img.device) * 2 - 1) * max_angle
    angle = torch.where(torch.rand(n, device=img.device) < prob, angle, torch.zeros_like(angle))
    return kornia.geometry.transform.rotate(img, angle, mode="bilinear", padding_mode="zeros")


def patch_gaussian(x, patch_size, max_scale, prob):
    """Patch Gaussian (Lopes et al., 2019) in [0, 1] space with sigma ~ U(0, max_scale) per image."""
    mean, std = _norm_stats(x.device, x.dtype)
    n, _, h, w = x.shape
    img = x * std + mean
    apply = (torch.rand(n, device=x.device) < prob).view(n, 1, 1)
    sigma = torch.rand(n, 1, 1, 1, device=x.device) * max_scale
    noisy = torch.clamp(img + torch.randn_like(img) * sigma, 0.0, 1.0)
    cy = torch.randint(0, h, (n, 1, 1), device=x.device)
    cx = torch.randint(0, w, (n, 1, 1), device=x.device)
    ys = torch.arange(h, device=x.device).view(1, h, 1)
    xs = torch.arange(w, device=x.device).view(1, 1, w)
    mask = ((ys - cy).abs() <= patch_size // 2) & ((xs - cx).abs() <= patch_size // 2) & apply
    img = torch.where(mask.unsqueeze(1), noisy, img)
    return (img - mean) / std


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------

def mixup(x, y, num_classes, alpha=1.0, prob=1.0):
    b = x.size(0)
    if b < 2 or alpha <= 0 or torch.rand(()) > prob:
        return x, y
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(b, device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    return x_mix, soft_labels(y, perm, torch.full((b,), lam, device=x.device), num_classes)


def cutmix(x, y, num_classes, alpha=1.0, prob=1.0):
    b, _, h, w = x.shape
    if b < 2 or alpha <= 0 or torch.rand(()) > prob:
        return x, y
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(b, device=x.device)
    cut_rat = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_rat), int(w * cut_rat)
    cy = int(torch.randint(0, h, (1,)))
    cx = int(torch.randint(0, w, (1,)))
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam_adj = 1.0 - (y2 - y1) * (x2 - x1) / float(h * w)
    return x_mix, soft_labels(y, perm, torch.full((b,), lam_adj, device=x.device), num_classes)


# ---------------------------------------------------------------------------
# ColorJitter with independent parameters per sample
# ---------------------------------------------------------------------------

def _rgb_to_gray(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def _rgb2hsv(x, eps=1e-8):
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(dim=1)
    minc, _ = x.min(dim=1)
    delta = maxc - minc
    s = delta / (maxc + eps)
    rc = (maxc - r) / (delta + eps)
    gc = (maxc - g) / (delta + eps)
    bc = (maxc - b) / (delta + eps)
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = torch.where(delta > 0, h, torch.zeros_like(h))
    return torch.stack([h, s, maxc], dim=1)


def _hsv2rgb(x):
    h, s, v = x[:, 0], x[:, 1], x[:, 2]
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i.long() % 6
    r = torch.stack([v, q, p, p, t, v], dim=1).gather(1, i.unsqueeze(1)).squeeze(1)
    g = torch.stack([t, v, v, q, p, p], dim=1).gather(1, i.unsqueeze(1)).squeeze(1)
    b = torch.stack([p, p, t, v, v, q], dim=1).gather(1, i.unsqueeze(1)).squeeze(1)
    return torch.stack([r, g, b], dim=1)


def color_jitter01(x, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, prob=1.0):
    b = x.size(0)
    dev = x.device
    apply = (torch.rand(b, device=dev) < prob).float().view(b, 1, 1, 1)
    out = x.float()

    def u(lo, hi):
        return (torch.rand(b, device=dev) * (hi - lo) + lo).view(b, 1, 1, 1)

    if brightness > 0:
        out = (out * u(1 - brightness, 1 + brightness)).clamp(0, 1)
    if contrast > 0:
        mean = _rgb_to_gray(out).mean(dim=(1, 2, 3), keepdim=True)
        out = (mean + (out - mean) * u(1 - contrast, 1 + contrast)).clamp(0, 1)
    if saturation > 0:
        gray = _rgb_to_gray(out)
        out = (gray + (out - gray) * u(1 - saturation, 1 + saturation)).clamp(0, 1)
    if hue > 0:
        hsv = _rgb2hsv(out)
        hsv[:, 0] = (hsv[:, 0] + u(-hue, hue)[:, 0]) % 1.0
        out = _hsv2rgb(hsv).clamp(0, 1)
    return apply * out + (1 - apply) * x.float()


def color_jitter(x, strength=0.4, hue=0.1, prob=1.0):
    mean, std = _norm_stats(x.device)
    x01 = (x.float() * std + mean).clamp(0, 1)
    x01 = color_jitter01(x01, strength, strength, strength, hue, prob)
    return (x01 - mean) / std


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Default (da_param1, da_param2, da_param3) when not given on the command line.
DEFAULT_PARAMS = {
    "patch_gaussian": (112, 1.0, 1.0),  # patch size, max sigma, prob
    "vertical_flip": (0.5, 1.0, 1.0),   # prob
    "random_crop": (1.0, 1.0, 1.0),     # prob
    "cutout": (1.0, 1.0, 1.0),          # prob
    "rotate": (30.0, 1.0, 1.0),         # max angle (degrees), prob
    "shear_x": (1.0, 1.0, 1.0),         # magnitude index, prob
    "shear_y": (1.0, 1.0, 1.0),         # magnitude index, prob
    "guided_mixup": (1.0, 1.0, 0.7),    # alpha, prob, guidance strength
    "saliency_mix": (1.0, 0.5, 1.0),    # beta, prob
    "puzzle_mix": (1.0, 1.0, 4.0),      # alpha, prob, grid size
    "co_mixup": (0.5, 2.0, 1.0),        # prob, grid size
    "snapmix": (1.0, 1.0, 1.0),         # beta, prob
    "mixup": (1.0, 1.0, 1.0),           # alpha, prob
    "cutmix": (1.0, 1.0, 1.0),          # alpha, prob
    "color_jitter": (0.4, 0.1, 1.0),    # strength (brightness = contrast = saturation), hue, prob
}
BASELINES = ("none",) + tuple(DEFAULT_PARAMS)


def apply_baseline(name, x, y, num_classes, p1=None, p2=None, p3=None, fixed=False):
    """Returns (images, labels); labels become soft labels for mixing methods."""
    if name == "none":
        return x, y
    d1, d2, d3 = DEFAULT_PARAMS[name]
    p1 = d1 if p1 is None else p1
    p2 = d2 if p2 is None else p2
    p3 = d3 if p3 is None else p3

    if name == "patch_gaussian":
        x = patch_gaussian(x, int(p1), float(p2), float(p3))
    elif name == "vertical_flip":
        x = vertical_flip(x, p1)
    elif name == "random_crop":
        x = random_crop(x, p1)
    elif name == "cutout":
        x = cutout(x, p1)
    elif name == "rotate":
        x = rotate(x, float(p1), float(p2))
    elif name == "shear_x":
        x = shear_x(x, p1, p2, fixed)
    elif name == "shear_y":
        x = shear_y(x, p1, p2, fixed)
    elif name in ("guided_mixup", "saliency_mix", "puzzle_mix"):
        if name == "guided_mixup":
            x, perm, lam = saliency.guided_mixup(x, alpha=float(p1), mix_prob=float(p2), guidance_strength=float(p3))
        elif name == "saliency_mix":
            x, perm, lam = saliency.saliency_mix(x, beta=float(p1), salmix_prob=float(p2))
        else:
            x, perm, lam = saliency.puzzle_mix(x, alpha=float(p1), mix_prob=float(p2), grid_size=float(p3))
        if perm is not None:
            y = soft_labels(y, perm, lam, num_classes)
    elif name == "co_mixup":
        x, y = saliency.co_mixup(x, y, mix_prob=float(p1), grid_size=int(p2), num_classes=num_classes)
    elif name == "snapmix":
        # SnapMix works in [0, 1] space.
        mean, std = _norm_stats(x.device, x.dtype)
        x01 = (x * std + mean).clamp(0.0, 1.0)
        x01, y = saliency.snapmix(x01, y, beta=float(p1), prob=float(p2), num_classes=num_classes)
        x = (x01 - mean) / std
    elif name == "mixup":
        x, y = mixup(x, y, num_classes, alpha=float(p1), prob=float(p2))
    elif name == "cutmix":
        x, y = cutmix(x, y, num_classes, alpha=float(p1), prob=float(p2))
    elif name == "color_jitter":
        x = color_jitter(x, strength=float(p1), hue=float(p2), prob=float(p3))
    return x, y
