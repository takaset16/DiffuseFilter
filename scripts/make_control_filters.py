"""Control filter sets for testing whether diffusion-generated filters are needed.

Modes:
  random_smooth  smooth random color fields with a linear gradient
  powerlaw       1/f^beta color noise
  solid          uniform colors
  natural        random crops from another image dataset (--src_dir)
  phase_rand     filters from --src_dir with randomized phase (amplitude spectrum kept)
  stats          mean color, saturation and low-frequency energy ratio of image folders

Example:
  python scripts/make_control_filters.py powerlaw --out filters_powerlaw --per_style 20
  python scripts/make_control_filters.py phase_rand --out filters_phase_rand --src_dir filters
"""
import argparse
import colorsys
import csv
import os
import random
import re

import numpy as np
from PIL import Image

SIZE = 512
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def _rand_color(rng):
    h = rng.random()
    s = rng.uniform(0.2, 1.0)
    v = rng.uniform(0.3, 1.0)
    return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32)


def make_random_smooth(rng):
    g = rng.choice([2, 3, 4, 6])
    grid = np.stack([[_rand_color(rng) for _ in range(g)] for _ in range(g)])
    img = Image.fromarray((grid * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    theta = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] / (SIZE - 1)
    t = (np.cos(theta) * xx + np.sin(theta) * yy)
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    c0, c1 = _rand_color(rng), _rand_color(rng)
    grad = (1 - t)[..., None] * c0 + t[..., None] * c1
    w = rng.uniform(0.2, 0.6)
    out = np.clip((1 - w) * arr + w * grad, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


def make_powerlaw(rng, nrng, beta):
    fy = np.fft.fftfreq(SIZE)[:, None]
    fx = np.fft.rfftfreq(SIZE)[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2)
    f[0, 0] = 1.0
    amp = 1.0 / f ** beta
    amp[0, 0] = 0.0

    def field():
        ph = np.angle(np.fft.rfft2(nrng.standard_normal((SIZE, SIZE))))
        y = np.fft.irfft2(amp * np.exp(1j * ph), s=(SIZE, SIZE))
        return (y - y.mean()) / (y.std() + 1e-8)

    common = field()
    base = _rand_color(rng)
    out = np.empty((SIZE, SIZE, 3), np.float32)
    c = rng.uniform(0.4, 0.8)
    for ch in range(3):
        z = c * common + (1 - c) * field()
        out[..., ch] = base[ch] + rng.uniform(0.08, 0.2) * z
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def make_solid(rng):
    c = _rand_color(rng)
    return Image.fromarray((np.ones((SIZE, SIZE, 3), np.float32) * c * 255).astype(np.uint8))


def _list_images(src_dir):
    out = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(root, f))
    return sorted(out)


def make_natural(rng, paths):
    img = Image.open(rng.choice(paths)).convert('RGB')
    w, h = img.size
    s = int(min(w, h) * rng.uniform(0.5, 1.0))
    x0 = rng.randint(0, w - s)
    y0 = rng.randint(0, h - s)
    return img.crop((x0, y0, x0 + s, y0 + s)).resize((SIZE, SIZE), Image.BICUBIC)


def phase_randomize(img, nrng):
    x = np.asarray(img.convert('RGB').resize((SIZE, SIZE), Image.BICUBIC)).astype(np.float32) / 255.0
    noise_phase = np.angle(np.fft.rfft2(nrng.standard_normal((SIZE, SIZE))))
    out = np.empty_like(x)
    for c in range(3):
        X = np.fft.rfft2(x[..., c])
        ph = noise_phase.copy()
        ph[0, 0] = np.angle(X[0, 0])
        y = np.fft.irfft2(np.abs(X) * np.exp(1j * ph), s=(SIZE, SIZE))
        y = (y - y.mean()) / (y.std() + 1e-8) * x[..., c].std() + x[..., c].mean()
        out[..., c] = y
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def generate(args):
    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    if args.mode in ('natural', 'phase_rand'):
        if not args.src_dir:
            raise SystemExit('--src_dir is required')
        paths = _list_images(args.src_dir)
        if not paths:
            raise SystemExit(f'no images found in {args.src_dir}')
    n = 0
    if args.mode == 'phase_rand':
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            m = re.match(r'^(.*)_(\d+)$', stem)
            style, idx = (m.group(1), int(m.group(2))) if m else (stem, 0)
            phase_randomize(Image.open(p), nrng).save(os.path.join(args.out, f'phaserand-{style}_{idx}.png'))
            n += 1
    else:
        for s in range(args.n_styles):
            for i in range(args.per_style):
                if args.mode == 'random_smooth':
                    img = make_random_smooth(rng)
                elif args.mode == 'solid':
                    img = make_solid(rng)
                elif args.mode == 'powerlaw':
                    img = make_powerlaw(rng, nrng, args.beta)
                else:
                    img = make_natural(rng, paths)
                img.save(os.path.join(args.out, f'{args.mode}-g{s:02d}_{i + 1}.png'))
                n += 1
    print(f'{n} images -> {args.out}')


def lowfreq_energy_ratio(x, radius):
    X = np.fft.fft2(x - x.mean())
    P = np.abs(X) ** 2
    h, w = x.shape
    fy = np.minimum(np.arange(h), h - np.arange(h))[:, None]
    fx = np.minimum(np.arange(w), w - np.arange(w))[None, :]
    d = np.sqrt(fy ** 2 + fx ** 2)
    tot = P.sum()
    if tot < 1e-6 * x.size:
        return float('nan')
    return float(P[d < radius].sum() / tot)


def stats(args):
    rows = []
    for d in args.dirs:
        paths = _list_images(d)
        if args.max_images:
            paths = paths[:args.max_images]
        acc = {f'lowE_r{r}': [] for r in args.radius}
        means, sats = [], []
        for p in paths:
            img = Image.open(p).convert('RGB').resize((224, 224), Image.BILINEAR)
            a = np.asarray(img).astype(np.float32) / 255.0
            means.append(a.mean(axis=(0, 1)))
            mx, mn = a.max(axis=2), a.min(axis=2)
            sats.append(float(((mx - mn) / (mx + 1e-8)).mean()))
            gray = a @ np.array([0.299, 0.587, 0.114], np.float32)
            for r in args.radius:
                acc[f'lowE_r{r}'].append(lowfreq_energy_ratio(gray, r))
        m = np.mean(means, axis=0) if means else [np.nan] * 3
        row = {'dir': d, 'n': len(paths), 'mean_R': float(m[0]), 'mean_G': float(m[1]), 'mean_B': float(m[2]),
               'saturation': float(np.mean(sats)) if sats else np.nan}
        for k, v in acc.items():
            row[k] = float(np.mean(v)) if v else np.nan
        rows.append(row)
        print(row)
    with open(args.csv, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f'-> {args.csv}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['random_smooth', 'powerlaw', 'solid', 'natural', 'phase_rand', 'stats'])
    ap.add_argument('--beta', type=float, default=1.15, help='powerlaw: slope of the amplitude spectrum')
    ap.add_argument('--out', default='')
    ap.add_argument('--src_dir', default='')
    ap.add_argument('--n_styles', type=int, default=9)
    ap.add_argument('--per_style', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dirs', nargs='*', default=[])
    ap.add_argument('--radius', nargs='*', type=int, default=[4, 8, 16, 32])
    ap.add_argument('--max_images', type=int, default=0)
    ap.add_argument('--csv', default='filter_stats.csv')
    args = ap.parse_args()
    if args.mode == 'stats':
        stats(args)
    else:
        if not args.out:
            raise SystemExit('--out is required')
        generate(args)


if __name__ == '__main__':
    main()
