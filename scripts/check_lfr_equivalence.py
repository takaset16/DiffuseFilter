"""Numerical check that LFR equals x + alpha * (L s_hat - L x) in the spatial domain (L: ideal low-pass)."""
import numpy as np


def mask_rfft(h, w, r):
    ys = np.arange(h)
    ys = np.minimum(ys, h - ys)
    xs = np.arange(w // 2 + 1)
    return (np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2) < r).astype(float)


def lfr(x, s, alpha, r, amp_norm=True):
    h, w = x.shape[-2:]
    X = np.fft.rfft2(x)
    S = np.fft.rfft2(s)
    M = mask_rfft(h, w, r).astype(bool)
    Xm = X.copy()
    for c in range(x.shape[0]):
        if amp_norm:
            ax = np.abs(X[c][M]).mean() + 1e-5
            as_ = np.abs(S[c][M]).mean() + 1e-5
            sn = S[c][M] * (ax / as_)
        else:
            sn = S[c][M]
        Xm[c][M] = (1 - alpha) * X[c][M] + alpha * sn
    return np.fft.irfft2(Xm, s=(h, w))


def lowpass(x, r):
    h, w = x.shape[-2:]
    return np.fft.irfft2(np.fft.rfft2(x) * mask_rfft(h, w, r), s=(h, w))


def fda(x, s, alpha, r):
    h, w = x.shape[-2:]
    X, S = np.fft.rfft2(x), np.fft.rfft2(s)
    M = mask_rfft(h, w, r)
    amp = np.abs(X) + alpha * M * (np.abs(S) - np.abs(X))
    return np.fft.irfft2(amp * np.exp(1j * np.angle(X)), s=(h, w)), X, S


def main():
    rng = np.random.default_rng(0)
    C, H, W = 3, 224, 224
    x = rng.random((C, H, W))
    s = rng.random((C, H, W))
    r = 16

    y1 = lfr(x, s, 1.0, r, amp_norm=False)
    y2 = lowpass(s, r) + x - lowpass(x, r)
    print(f'(1) LFR(alpha=1,no-norm) vs spatial hybrid  max|diff| = {np.abs(y1 - y2).max():.2e}')

    a = 0.6
    y1 = lfr(x, s, a, r, amp_norm=False)
    y2 = x + a * (lowpass(s, r) - lowpass(x, r))
    print(f'    LFR(alpha={a},no-norm) vs x + a(Ls - Lx)  max|diff| = {np.abs(y1 - y2).max():.2e}')

    hp = lambda z: z - lowpass(z, r)
    y = lfr(x, s, 1.0, r, amp_norm=True)
    print(f'    high-pass(x\') vs high-pass(x)             max|diff| = {np.abs(hp(y) - hp(x)).max():.2e}')

    yf, X, S = fda(x, s, 1.0, r)
    M = mask_rfft(H, W, r).astype(bool)
    Yf = np.fft.rfft2(yf)
    Yl = np.fft.rfft2(lfr(x, s, 1.0, r, amp_norm=False))
    dphase_fda = np.abs(np.angle(Yf[0][M] * np.conj(X[0][M]))).mean()
    dphase_lfr_to_s = np.abs(np.angle(Yl[0][M] * np.conj(S[0][M]))).mean()
    print(f'(2) FDA: phase change inside the mask       mean|dphi| = {dphase_fda:.2e} rad (kept)')
    print(f'    LFR: phase difference from the filter     mean|dphi| = {dphase_lfr_to_s:.2e} rad (replaced)')

    for rr in [4, 8, 16, 32, 64]:
        n = int(mask_rfft(H, W, rr).sum())
        print(f'(3) radius={rr:3d}: {n:5d} of {H * (W // 2 + 1)} rfft coefficients in the mask ({100 * n / (H * (W // 2 + 1)):.2f}%)')


if __name__ == '__main__':
    main()
