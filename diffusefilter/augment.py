import torch
import torch.nn.functional as F

from .composite import composite

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def concat_with_mask(original_imgs, augmented_imgs, blend_width):
    """Join the left/top half of the original with the other half of the augmented image."""
    B, C, H, W = original_imgs.shape
    device = original_imgs.device

    directions = torch.randint(0, 2, (B,), device=device).bool()

    y0 = max(0, H // 2 - blend_width // 2)
    y1 = min(H, y0 + blend_width)
    mask_v = torch.ones(H, W, device=device)
    mask_v[:y0, :] = 0.0
    if y1 > y0:
        mask_v[y0:y1, :] = torch.linspace(0.0, 1.0, steps=y1 - y0, device=device).view(-1, 1).expand(y1 - y0, W)
    mask_v[y1:, :] = 1.0
    mask_v = mask_v.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

    x0 = max(0, W // 2 - blend_width // 2)
    x1 = min(W, x0 + blend_width)
    mask_h = torch.ones(H, W, device=device)
    mask_h[:, :x0] = 0.0
    if x1 > x0:
        mask_h[:, x0:x1] = torch.linspace(0.0, 1.0, steps=x1 - x0, device=device).view(1, -1).expand(H, x1 - x0)
    mask_h[:, x1:] = 1.0
    mask_h = mask_h.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)

    mask = torch.where(directions.view(B, 1, 1, 1).expand(B, C, H, W), mask_v, mask_h)
    return original_imgs * (1.0 - mask) + augmented_imgs * mask


def _resize_to(imgs, ref):
    if imgs.shape[-2:] != ref.shape[-2:]:
        imgs = F.interpolate(imgs, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return imgs


class DiffuseFilter:
    """On-the-fly DiffuseFilter augmentation for normalized image batches."""

    def __init__(self, filter_imgs, fractal_imgs, composite="lfr", alpha=0.4, radius=16,
                 mask_shape="circle", amp_norm=True, blend_width=32, fractal_alpha=0.05,
                 apply_ratio=0.5, use_filter=True, use_concat=True, use_fractal=True,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.filter_imgs = filter_imgs
        self.fractal_imgs = fractal_imgs
        self.composite = composite
        self.alpha = alpha
        self.radius = radius
        self.mask_shape = mask_shape
        self.amp_norm = amp_norm
        self.blend_width = blend_width
        self.fractal_alpha = fractal_alpha
        self.apply_ratio = apply_ratio
        self.use_filter = use_filter
        self.use_concat = use_concat
        self.use_fractal = use_fractal
        self.mean = mean
        self.std = std

    def _stats(self, device):
        mean = torch.tensor(self.mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        return mean, std

    def __call__(self, images):
        batch_size = images.shape[0]
        device = images.device
        mean, std = self._stats(device)
        work = (images.float() * std + mean).clamp(0.0, 1.0)

        if self.use_filter:
            idx = torch.randint(0, self.filter_imgs.size(0), (batch_size,))
            filters = _resize_to(self.filter_imgs[idx].to(device, non_blocking=True), work)
            transferred = composite(self.composite, work, filters, alpha=self.alpha, radius=self.radius,
                                    mask_shape=self.mask_shape, amp_norm=self.amp_norm)
            combined = concat_with_mask(work, transferred, self.blend_width) if self.use_concat else transferred
        else:
            combined = work

        idx = torch.randint(0, self.fractal_imgs.size(0), (batch_size,))
        fractals = _resize_to(self.fractal_imgs[idx].to(device, dtype=torch.float32, non_blocking=True), combined)
        if self.use_fractal:
            blended = (1 - self.fractal_alpha) * combined + self.fractal_alpha * fractals
        else:
            blended = combined

        # Replace a random subset (apply_ratio of the batch) with augmented images.
        n = int(batch_size * self.apply_ratio)
        chosen = torch.randperm(batch_size, device=device)[:n]
        out = work.clone()
        out[chosen] = blended[chosen]

        return (out - mean) / std
