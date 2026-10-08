"""Saliency-guided mixing baselines (simplified: spectral-residual saliency replaces CAM/gradients)."""
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _clamp(value, v_min, v_max):
    return max(v_min, min(v_max, value))


def _to_int(value, v_min, v_max):
    return int(_clamp(round(value), v_min, v_max))


def _validate_images(images: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(images):
        raise TypeError(f"images must be torch.Tensor, got {type(images)!r}")
    if images.dim() != 4:
        raise ValueError(f"Expected (N,C,H,W), got {tuple(images.shape)}")
    return images


def _normalize_saliency(maps: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if maps.dim() == 3:
        maps = maps.unsqueeze(1)
    b = maps.shape[0]
    flat = maps.view(b, -1)
    minv = flat.min(dim=1, keepdim=True).values
    maxv = flat.max(dim=1, keepdim=True).values
    denom = (maxv - minv).clamp_min(eps)
    return ((flat - minv) / denom).view_as(maps)


def _spectral_residual_saliency(
    images: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    with torch.no_grad():
        images = _validate_images(images)
        b, _, h, w = images.shape
        gray = images.mean(dim=1, keepdim=True).to(torch.float32)
        flat = gray.view(b, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True, unbiased=False)
        std = torch.where(std < eps, torch.full_like(std, 1.0), std)
        gray = ((flat - mean) / std).view(b, 1, h, w)

        spec = torch.fft.fft2(gray)
        amp = torch.abs(spec)
        log_amp = torch.log(amp + eps)
        smooth = F.avg_pool2d(log_amp, kernel_size=3, stride=1, padding=1)
        residual = log_amp - smooth
        phase = torch.angle(spec)

        spec_new = torch.complex(
            torch.exp(residual) * torch.cos(phase), torch.exp(residual) * torch.sin(phase)
        )
        sal = torch.fft.ifft2(spec_new).abs().pow(2)
        sal = F.avg_pool2d(sal, kernel_size=3, stride=1, padding=1)
    return _normalize_saliency(sal).to(dtype=images.dtype, device=images.device)


def _get_saliency(
    images: torch.Tensor, mode: str = "spectral_residual"
) -> torch.Tensor:
    mode = mode.lower()
    b, _, h, w = images.shape
    if mode == "none":
        return torch.ones((b, 1, h, w), dtype=images.dtype, device=images.device)
    if mode == "spectral_residual":
        return _spectral_residual_saliency(images)
    raise ValueError(f"Unsupported saliency mode: {mode}")


def _sample_beta_batch(
    batch_size: int, alpha: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    concentration = torch.full((2,), alpha, device=device, dtype=torch.float32)
    dist = torch.distributions.Beta(concentration[0], concentration[1])
    return dist.sample((batch_size,)).to(dtype=dtype)


def _compute_box_and_area(
    height: int, width: int, center_y: int, center_x: int, area_ratio: float
) -> tuple[int, int, int, int, int]:
    area_ratio = float(max(0.0, min(1.0, area_ratio)))
    if area_ratio <= 0.0:
        return 0, 0, 0, 0, 0

    cut_ratio = math.sqrt(area_ratio)
    box_w = max(1, int(width * cut_ratio))
    box_h = max(1, int(height * cut_ratio))
    x1 = max(0, min(int(center_x) - box_w // 2, width - box_w))
    y1 = max(0, min(int(center_y) - box_h // 2, height - box_h))
    x2 = x1 + box_w
    y2 = y1 + box_h
    return y1, y2, x1, x2, (y2 - y1) * (x2 - x1)


def guided_mixup(
    images: torch.Tensor,
    alpha: float = 1.0,
    mix_prob: float = 1.0,
    guidance_strength: float = 0.7,
    pair_guided_flag: float = 1.0,
    downsample_factor: float = 16.0,
):
    images = _validate_images(images)
    b, _, h, w = images.shape
    if b < 2 or mix_prob <= 0.0 or alpha <= 0.0:
        return images, None, None

    sal = _get_saliency(images, "spectral_residual")
    if pair_guided_flag >= 0.5:
        stride_h = max(1, h // _to_int(downsample_factor, 1, 64))
        stride_w = max(1, w // _to_int(downsample_factor, 1, 64))
        small = F.avg_pool2d(sal, kernel_size=(stride_h, stride_w), stride=(stride_h, stride_w))
        flat = small.view(b, -1)
        flat = flat / (flat.norm(p=2, dim=1, keepdim=True) + 1e-6)
        sim = torch.matmul(flat, flat.t())
        sim.fill_diagonal_(1e6)
        perm = torch.arange(b, device=images.device)
        used = torch.zeros(b, dtype=torch.bool, device=images.device)
        for i in range(b):
            if used[i]:
                continue
            mask = ~used
            mask[i] = False
            cand = torch.nonzero(mask, as_tuple=False).flatten()
            if cand.numel() == 0:
                perm[i] = i
                used[i] = True
                break
            j = cand[torch.argmin(sim[i, cand])].item()
            perm[i] = j
            perm[j] = i
            used[i] = True
            used[j] = True
    else:
        perm = torch.randperm(b, device=images.device)

    x_perm = images[perm]
    s_perm = sal[perm]
    lam = _sample_beta_batch(b, alpha, device=images.device, dtype=images.dtype)
    mix_mask = (torch.rand(b, device=images.device) < mix_prob).to(images.dtype)
    lam = lam * mix_mask + (1.0 - mix_mask)

    s_sum = sal + s_perm + 1e-6
    lam_sal = sal / s_sum
    lam_map = (
        lam.view(b, 1, 1, 1) * (1.0 - guidance_strength)
        + lam_sal * guidance_strength
    )
    mixed = lam_map * images + (1.0 - lam_map) * x_perm
    return mixed, perm, lam_map.mean(dim=(1, 2, 3))


def saliency_mix(
    images: torch.Tensor,
    beta: float = 1.0,
    salmix_prob: float = 1.0,
    min_lambda: float = 0.2,
    max_lambda: float = 0.8,
):
    images = _validate_images(images)
    b, _, h, w = images.shape
    if b < 2 or salmix_prob <= 0.0 or beta <= 0.0:
        return images, None, None

    if min_lambda > max_lambda:
        min_lambda, max_lambda = max_lambda, min_lambda

    sal = _get_saliency(images, "spectral_residual")
    perm = torch.randperm(b, device=images.device)
    source = images[perm]
    sal_source = sal[perm]
    lam = _sample_beta_batch(b, beta, device=images.device, dtype=images.dtype).clamp(min_lambda, max_lambda)
    mixed = images.clone()
    lam_label = torch.ones(b, dtype=images.dtype, device=images.device)

    for i in range(b):
        if torch.rand((), device=images.device) > salmix_prob:
            continue
        s_map = sal_source[i, 0]
        max_idx = int(s_map.view(-1).argmax().item())
        cy = max_idx // w
        cx = max_idx % w
        area_ratio = max(0.0, min(1.0, 1.0 - float(lam[i].item())))
        y1, y2, x1, x2, area = _compute_box_and_area(h, w, cy, cx, area_ratio)
        if area == 0:
            continue
        mixed[i, :, y1:y2, x1:x2] = source[i, :, y1:y2, x1:x2]
        lam_label[i] = 1.0 - (y2 - y1) * (x2 - x1) / float(h * w)

    return mixed, perm, lam_label


def puzzle_mix(
    images: torch.Tensor,
    alpha: float = 1.0,
    mix_prob: float = 1.0,
    grid_size: float = 4.0,
):
    images = _validate_images(images)
    b, _, h, w = images.shape
    if b < 2 or mix_prob <= 0.0 or alpha <= 0.0:
        return images, None, None

    sal = _get_saliency(images, "spectral_residual")
    perm = torch.randperm(b, device=images.device)
    partner = images[perm]
    sal_partner = sal[perm]

    gh = _to_int(grid_size, 1, 32)
    gw = gh
    h_trim = (h // gh) * gh
    w_trim = (w // gw) * gw
    if h_trim == 0 or w_trim == 0:
        lam = _sample_beta_batch(b, alpha, device=images.device, dtype=images.dtype)
        out = lam.view(b, 1, 1, 1) * images + (1.0 - lam).view(b, 1, 1, 1) * partner
        return out, perm, lam

    patch_h = h_trim // gh
    patch_w = w_trim // gw
    n_patches = gh * gw
    lam = _sample_beta_batch(b, alpha, device=images.device, dtype=images.dtype)
    mixed = images.clone()
    lam_label = torch.ones(b, dtype=images.dtype, device=images.device)

    for i in range(b):
        if torch.rand((), device=images.device) > mix_prob:
            continue
        s1 = sal[i, :, :h_trim, :w_trim]
        s2 = sal_partner[i, :, :h_trim, :w_trim]
        s1_patches = s1.view(1, gh, patch_h, gw, patch_w).permute(0, 1, 3, 2, 4).contiguous().view(gh * gw, -1).sum(dim=1)
        s2_patches = s2.view(1, gh, patch_h, gw, patch_w).permute(0, 1, 3, 2, 4).contiguous().view(gh * gw, -1).sum(dim=1)
        diff = s1_patches - s2_patches

        k_from_anchor = int(round(float(lam[i].item()) * n_patches))
        k_from_anchor = max(0, min(n_patches, k_from_anchor))
        mask_flat = torch.zeros(n_patches, dtype=torch.bool, device=images.device)
        if k_from_anchor > 0:
            order = torch.argsort(diff, descending=True)
            mask_flat[order[:k_from_anchor]] = True

        for idx in range(n_patches):
            if mask_flat[idx]:
                continue
            py = idx // gw
            px = idx % gw
            y1 = py * patch_h
            y2 = y1 + patch_h
            x1 = px * patch_w
            x2 = x1 + patch_w
            mixed[i, :, y1:y2, x1:x2] = partner[i, :, y1:y2, x1:x2]
        n_swapped = int((~mask_flat).sum().item())
        lam_label[i] = 1.0 - n_swapped * patch_h * patch_w / float(h * w)

    return mixed, perm, lam_label


def co_mixup(
    images: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    mix_prob: float = 0.5,
    grid_size: int = 2,
    num_classes: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    images = _validate_images(images)
    b, c, h, w = images.shape

    if b < 2 or mix_prob <= 0.0 or torch.rand(()) > mix_prob:
        return images, labels

    gh = gw = _to_int(grid_size, 1, 32)
    h_trim = (h // gh) * gh
    w_trim = (w // gw) * gw

    if h_trim == 0 or w_trim == 0:
        return images, labels

    patch_h = h_trim // gh
    patch_w = w_trim // gw
    patch_count = gh * gw

    sal = _get_saliency(images, "spectral_residual")
    s_crop = sal[:, :, :h_trim, :w_trim]
    s_patch = (
        s_crop.view(b, 1, gh, patch_h, gw, patch_w)
        .permute(0, 1, 2, 4, 3, 5).contiguous()
        .view(b, patch_count, -1)
        .sum(dim=2)
        .clamp(min=0.0)
    )

    mixed = images.clone()
    src_all = torch.zeros(b, patch_count, dtype=torch.int64, device=images.device)

    for pidx in range(patch_count):
        sal_p = s_patch[:, pidx]
        total = float(sal_p.sum())
        probs = (
            sal_p / total if total > 0.0
            else torch.full((b,), 1.0 / b, device=images.device, dtype=images.dtype)
        )
        src_for_dest = torch.multinomial(
            probs.unsqueeze(0).expand(b, -1),
            num_samples=1
        ).squeeze(1)

        src_all[:, pidx] = src_for_dest

        py, px = divmod(pidx, gw)
        y1, y2 = py * patch_h, py * patch_h + patch_h
        x1, x2 = px * patch_w, px * patch_w + patch_w
        mixed[:, :, y1:y2, x1:x2] = images[src_for_dest, :, y1:y2, x1:x2]

    if h_trim < h:
        mixed[:, :, h_trim:, :] = images[:, :, h_trim:, :]
    if w_trim < w:
        mixed[:, :, :h_trim, w_trim:] = images[:, :, :h_trim, w_trim:]

    if labels is None:
        return mixed, None

    if labels.dim() == 2:
        soft_labels = torch.zeros_like(labels, dtype=torch.float32)
        for pidx in range(patch_count):
            soft_labels += labels[src_all[:, pidx]].float()
        mixed_labels = soft_labels / patch_count
    else:
        nc = num_classes if num_classes > 0 else int(labels.max().item()) + 1
        eye = torch.eye(nc, dtype=torch.float32, device=images.device)
        soft_labels = torch.zeros(b, nc, dtype=torch.float32, device=images.device)
        for pidx in range(patch_count):
            soft_labels += eye[labels[src_all[:, pidx]]]
        mixed_labels = soft_labels / patch_count

    return mixed, mixed_labels


def snapmix(
    images: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    beta: float = 5.0,
    prob: float = 1.0,
    num_classes: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    images = _validate_images(images)
    b, _, h, w = images.shape
    if b < 2 or prob <= 0.0 or beta <= 0.0:
        return images, labels

    cam = _get_saliency(images, "spectral_residual")
    perm = torch.randperm(b, device=images.device)
    x2 = images[perm]
    cam2 = cam[perm]
    lam = _sample_beta_batch(b, beta, device=images.device, dtype=images.dtype)
    mixed = images.clone()

    lam_actual = torch.ones(b, dtype=images.dtype, device=images.device)

    for i in range(b):
        if torch.rand((), device=images.device) > prob:
            continue
        b_img = x2[i]
        cam_a = cam[i, 0]
        cam_b = cam2[i, 0]
        area1 = max(0.0, min(1.0, 1.0 - float(lam[i].item())))
        area2 = max(0.0, min(1.0, float(lam[i].item())))

        flat_a = cam_a.view(-1)
        idx_a = (
            int(torch.multinomial(flat_a / flat_a.sum(), 1).item())
            if float(flat_a.sum()) > 0.0
            else int(torch.randint(0, h * w, (), device=images.device).item())
        )
        flat_b = cam_b.view(-1)
        idx_b = (
            int(torch.multinomial(flat_b / flat_b.sum(), 1).item())
            if float(flat_b.sum()) > 0.0
            else int(torch.randint(0, h * w, (), device=images.device).item())
        )
        cy1, cx1 = idx_a // w, idx_a % w
        cy2, cx2 = idx_b // w, idx_b % w

        y1_1, y1_2, x1_1, x1_2, area_box1 = _compute_box_and_area(h, w, cy1, cx1, area1)
        y2_1, y2_2, x2_1, x2_2, area_box2 = _compute_box_and_area(h, w, cy2, cx2, area2)
        if area_box1 == 0 or area_box2 == 0:
            continue

        patch_b = b_img[:, y2_1:y2_2, x2_1:x2_2].unsqueeze(0)
        patch_b_resized = F.interpolate(
            patch_b, size=(y1_2 - y1_1, x1_2 - x1_1), mode="bilinear", align_corners=False
        )[0]
        mixed[i, :, y1_1:y1_2, x1_1:x1_2] = patch_b_resized

        lam_actual[i] = 1.0 - area_box1 / float(h * w)

    if labels is None:
        return mixed, None

    if labels.dim() == 2:
        mixed_labels = (
            lam_actual.view(b, 1) * labels.float()
            + (1.0 - lam_actual).view(b, 1) * labels[perm].float()
        )
    else:
        nc = num_classes if num_classes > 0 else int(labels.max().item()) + 1
        eye = torch.eye(nc, dtype=images.dtype, device=images.device)
        mixed_labels = (
            lam_actual.view(b, 1) * eye[labels]
            + (1.0 - lam_actual).view(b, 1) * eye[labels[perm]]
        )

    return mixed, mixed_labels
