"""Figure 4 (revision 2): what LFR changes in the frequency domain.

Each row: original | filter | LFR output (clipped, as used in training) |
          log spectrum of the original | |F(LFR before clipping) - F(original)| | |F(LFR after clipping) - F(original)|
Spectra: 2-D DFT of each RGB channel at 224x224 in [0,1], centred (DC in the middle),
         log10(1 + |.|) averaged over the three channels. The white circle is the LFR mask (radius r).
The two difference panels share one colour scale.

usage:
  python scripts/make_fig4.py --images A.jpg B.jpg --filters F1.png F2.png --out fig4.png
  (radius 16, alpha 0.4 by default = the setting used in the paper)
"""
import argparse, os, sys
import numpy as np, torch, torch.fft as fft
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusefilter.composite import lowfreq_replace as lowfreq_style_mix


def load(p, size):
    im = Image.open(p).convert('RGB').resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)[None]


def lfr_unclipped(x, style, alpha, radius):
    """lowfreq_style_mix without the final clamp (same code path)"""
    h, w = x.shape[-2:]; rw = w // 2 + 1
    X = fft.rfft2(x, dim=(-2, -1)); S = fft.rfft2(style, dim=(-2, -1))
    ys = torch.arange(h); xs = torch.arange(rw); ys = torch.min(ys, h - ys)
    mask = (ys[:, None] ** 2 + xs[None, :] ** 2).float().sqrt() < radius
    Xm = X.clone()
    amp_x = X[..., mask].abs().mean(dim=-1, keepdim=True) + 1e-5
    amp_s = S[..., mask].abs().mean(dim=-1, keepdim=True) + 1e-5
    Xm[..., mask] = (1 - alpha) * X[..., mask] + alpha * S[..., mask] * (amp_x / amp_s)
    return fft.irfft2(Xm, s=(h, w), dim=(-2, -1))


def spec(x):
    return fft.fftshift(fft.fft2(x[0].double(), dim=(-2, -1)), dim=(-2, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', nargs='+', required=True)
    ap.add_argument('--filters', nargs='+', required=True)
    ap.add_argument('--radius', type=float, default=16)
    ap.add_argument('--alpha', type=float, default=0.4)
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--out', default='fig4_lfr_spectrum.png')
    a = ap.parse_args()
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rows = []
    for ip, fp in zip(a.images, a.filters):
        x, s = load(ip, a.size), load(fp, a.size)
        raw = lfr_unclipped(x, s, a.alpha, a.radius)
        used = lowfreq_style_mix(x, s, alpha=a.alpha, radius=a.radius)
        assert torch.allclose(raw.clamp(0, 1), used, atol=1e-5)
        X, Rb, Ra = spec(x), spec(raw), spec(used)
        h = a.size; yy, xx = np.mgrid[:h, :h]; d = np.hypot(yy - h // 2, xx - h // 2)
        out = d >= a.radius + 1  # strictly outside the mask
        db = (Rb - X).abs().mean(0).numpy(); da = (Ra - X).abs().mean(0).numpy()
        print(f'{os.path.basename(ip)}: max |dF| outside mask before clipping = {db[out].max():.2e}, '
              f'after clipping = {da[out].max():.2e} (inside: {db[~out].max():.2e}); '
              f'clipped pixels = {((raw < 0) | (raw > 1)).float().mean().item() * 100:.1f}%')
        rows.append((x, s, used, np.log10(1 + X.abs().mean(0).numpy()), np.log10(1 + db), np.log10(1 + da)))
    vmax = max(max(r[4].max(), r[5].max()) for r in rows)
    titles = ['Original', 'Filter', 'LFR output', 'log spectrum (original)',
              '|ΔF|, LFR before clipping', '|ΔF|, LFR after clipping']
    fig, axs = plt.subplots(len(rows), 6, figsize=(13, 2.3 * len(rows)), squeeze=False,
                            gridspec_kw=dict(wspace=0.05, hspace=0.08, right=0.92))
    for i, r in enumerate(rows):
        for j in range(6):
            ax = axs[i, j]; ax.axis('off')
            if j < 3:
                ax.imshow(r[j][0].permute(1, 2, 0).clamp(0, 1).numpy())
            else:
                im = ax.imshow(r[j], cmap='magma', vmin=0, vmax=(None if j == 3 else vmax))
                ax.add_patch(plt.Circle((a.size // 2, a.size // 2), a.radius, fill=False, ec='w', lw=0.8))
                if j == 5: last = im
            if i == 0: ax.set_title(titles[j], fontsize=9)
    cax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
    fig.colorbar(last, cax=cax, label='log10(1 + |ΔF|)')
    fig.savefig(a.out, dpi=300, bbox_inches='tight')
    print('saved', a.out)


if __name__ == '__main__':
    main()
