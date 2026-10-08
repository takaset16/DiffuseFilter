import os
import random
import re

import torch
from PIL import Image
from torchvision import transforms

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def _load(paths, size):
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    return torch.stack([transform(Image.open(p).convert("RGB")) for p in paths])


def load_images(img_dir, size=(224, 224)):
    paths = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS))
    if not paths:
        raise FileNotFoundError(f"no images found in {img_dir}")
    return _load(paths, size)


def group_by_style(img_dir):
    """Group "<style>_<index>.<ext>" files by style, each sorted by index."""
    groups = {}
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(IMAGE_EXTS):
            continue
        stem = os.path.splitext(fname)[0]
        m = re.match(r"^(.*)_(\d+)$", stem)
        style, idx = (m.group(1), int(m.group(2))) if m else (stem, 0)
        groups.setdefault(style, []).append((idx, fname))
    return {k: [os.path.join(img_dir, f) for _, f in sorted(v)] for k, v in sorted(groups.items())}


def load_filters(img_dir, size=(224, 224), per_style=0, styles=None, seed=0):
    """Load filter images, optionally keeping `per_style` images of the styles matching `styles`."""
    groups = group_by_style(img_dir)
    if styles:
        groups = {k: v for k, v in groups.items() if any(s in k for s in styles)}
    rng = random.Random(seed)
    chosen, info = [], {}
    for style, paths in groups.items():
        sel = list(paths)
        if per_style and per_style < len(sel):
            sel = sorted(rng.sample(sel, per_style))
        chosen.extend(sel)
        info[style] = len(sel)
    if not per_style and not styles:
        chosen = sorted(chosen)
    if not chosen:
        raise FileNotFoundError(f"no filter images selected from {img_dir} (styles={styles})")
    print(f"[filters] {len(chosen)} images from {img_dir}: {info}")
    return _load(chosen, size), {"total": len(chosen), "per_style": info}
