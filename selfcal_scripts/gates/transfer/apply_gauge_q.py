"""Post-solve gauge, extended to the absorbed part of a sky QUADRATIC (no x^2 dof in the model needed).

A sky quadratic + plane  P(p) = g.(p - p0) + (p - p0)^T H (p - p0) / 2  seen through frame k (p - p0 = D_k + J_k d, d the
detector offset from the centre, J_k the detector -> grid Jacobian) is
    [g.D_k + D_k^T H D_k / 2]  +  d . J_k^T (g + H D_k)  +  d^T (J_k^T H J_k) d / 2 .
The per-frame model (scalar, band constants, x-slope) absorbs the first two (the y-parts as band steps) and the xy term
along the bands (an x-slope that varies with the band's dy); the x^2 term has no counterpart (the in-frame curvature).
So the converged per-frame x-slopes carry the mirror of any sky plane AND of the absorbed part of any sky quadratic:
    b[k,s] / 1020 = x_k.(g + H D_k) + (x_k^T H y_k) dy_s + h_s + h_D(D_k)          (x_k, y_k = the columns of J_k)
with h_s a detector-fixed gradient per subchannel and h_D a smooth function of the frame's position (degree HDEG, default
1: the zodi gradient's amplitude varies with pointing) -- neither rotates with the frame, g and H do (the rolls span all
angles), so they separate. Fitted over every (frame, subchannel) by coverage-weighted least squares; then g and H move
into the sky and their absorbed image leaves the offsets (slopes, band constants, scalar). The x^2 part cannot leave the
offsets (it is not in them): the move changes the model prediction by -d_x^2 (J^T H J)_xx / 2 per frame, reported.
    DET=d [HDEG=1] python apply_gauge_q.py CAL.h5 --frames-dir DIR [--out OUT.h5] [--plane-only]"""
import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apply_gauge import X_CENTRE, X_HALF, geometry, geometry_kind, header_of  # noqa: E402


def main():
    import h5py
    import hdf5plugin  # noqa: F401
    ap = argparse.ArgumentParser(); ap.add_argument('cal'); ap.add_argument('--out', default=None); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--frames-dir', default=None); ap.add_argument('--plane-only', action='store_true')
    a = ap.parse_args()
    out = a.out or a.cal.replace('.h5', '_gaugeq.h5')
    det = os.environ.get('DET', '3'); hdeg = int(os.environ.get('HDEG', '1'))
    with h5py.File(a.cal, 'r') as f:
        sky = f['sky/continuum'][:].astype(np.float64); cov = f['sky_coverage/continuum'][:]
        c = f['offsets/map_0'][:].astype(np.float64); b = f['offsets/map_1'][:].astype(np.float64)
        ocov = f['offset_coverage/map_1'][:].astype(np.float64)
        scalar = f['frame_scalar'][:].astype(np.float64)
        cmap = f['chunk_maps/map_0'][:]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f['reproj_list'][:]]
    K, S = b.shape
    paths = [os.path.join(a.frames_dir, os.path.basename(n)) if a.frames_dir else n for n in names]
    with ProcessPoolExecutor(a.workers) as ex:
        hdr = np.array(list(ex.map(header_of, paths, chunksize=64)))
    ys = np.full(S, np.nan)
    yy = np.arange(cmap.shape[0])[:, None] * np.ones(cmap.shape, dtype=np.float64)
    for cid in np.unique(cmap[cmap >= 0]):
        if cid < S:
            ys[cid] = yy[cmap == cid].mean()
    dy = np.where(np.isfinite(ys), ys - X_CENTRE, 0.0)
    J, centres, _ = geometry(hdr, det, sky.shape)
    h, w = sky.shape
    m = cov > 0
    pyy, pxx = np.nonzero(m)
    p0 = np.array([pxx.mean(), pyy.mean()])
    D = centres - p0
    xk, yk = J[:, :, 0], J[:, :, 1]
    good = (ocov > 0) & np.isfinite(b)
    kk, ss = np.nonzero(good)
    v = b[kk, ss] / X_HALF
    wts = np.sqrt(ocov[kk, ss])
    x0, x1, y0, y1 = xk[kk, 0], xk[kk, 1], yk[kk, 0], yk[kk, 1]
    D0, D1 = D[kk, 0], D[kk, 1]
    dys = dy[ss]
    cols, names_c = [x0, x1], ['g0', 'g1']
    if not a.plane_only:
        cols += [x0 * D0 + x0 * y0 * dys, x0 * D1 + x1 * D0 + (x0 * y1 + x1 * y0) * dys, x1 * D1 + x1 * y1 * dys]
        names_c += ['H00', 'H01', 'H11']
    for s_ in range(S):                                                 # h_s: detector-fixed, per subchannel
        if (ss == s_).any():
            cols.append((ss == s_).astype(np.float64)); names_c.append(f'h{s_}')
    scale_D = max(w, h) / 2.0
    for p in range(1, hdeg + 1):                                        # h_D: smooth in the frame's position (no constant: h_s has it)
        for q in range(p + 1):
            cols.append((D0 / scale_D) ** (p - q) * (D1 / scale_D) ** q); names_c.append(f'hD{p - q}{q}')
    A = np.stack(cols, axis=1)
    sol, *_ = np.linalg.lstsq(A * wts[:, None], v * wts, rcond=None)
    res = v - A @ sol
    par = dict(zip(names_c, sol))
    g = np.array([par['g0'], par['g1']])
    H = np.zeros((2, 2)) if a.plane_only else np.array([[par['H00'], par['H01']], [par['H01'], par['H11']]])
    wrms = lambda z: float(np.sqrt(np.sum(wts ** 2 * z ** 2) / np.sum(wts ** 2)))   # noqa: E731
    print(f"{K} frames x {S} subchannels, {kk.size} slopes; fit: {len(names_c)} parameters (h_D degree {hdeg}); slope rms {wrms(v):.3e} "
          f"per det px, residual after the fit {wrms(res):.3e}")
    print(f"plane g = ({g[0] * w / 2:+.3e}, {g[1] * h / 2:+.3e}) MJy/sr per half-field")
    if not a.plane_only:
        hw = np.array([w / 2, h / 2])
        print(f"quadratic H * half-field^2 = [[{H[0, 0] * hw[0] ** 2:+.3e}, {H[0, 1] * hw[0] * hw[1]:+.3e}], [., {H[1, 1] * hw[1] ** 2:+.3e}]] MJy/sr")
    # ---- the move
    HJ = np.einsum('kai,ab,kbj->kij', J, H, J)                                       # J^T H J (per det px^2)
    lin = np.einsum('kai,ka->ki', J, (D @ H.T) + g[None, :])                         # J^T (H D_k + g) (per det px)
    b_new = b - X_HALF * (lin[:, 0][:, None] + HJ[:, 0, 1][:, None] * dy[None, :])
    adj = lin[:, 1][:, None] * dy[None, :] + 0.5 * HJ[:, 1, 1][:, None] * dy[None, :] ** 2
    gm = good.any(axis=1)
    adj_mean = np.array([adj[k, good[k]].mean() if gm[k] else 0.0 for k in range(K)])
    c_new = c - (adj - adj_mean[:, None])
    scalar_new = scalar - adj_mean - (D @ g) - 0.5 * np.einsum('ka,ab,kb->k', D, H, D)
    px_, py_ = pxx - p0[0], pyy - p0[1]
    sky_add = g[0] * px_ + g[1] * py_ + 0.5 * (H[0, 0] * px_ ** 2 + 2 * H[0, 1] * px_ * py_ + H[1, 1] * py_ ** 2)
    sky[m] += sky_add - sky_add.mean()
    if not a.plane_only:
        # what the offsets cannot take: the in-frame x^2 curvature, d_x^2 (J^T H J)_xx / 2, rms over the detector width
        curv = 0.5 * np.abs(HJ[:, 0, 0]) * X_HALF ** 2 * np.sqrt(np.mean((np.linspace(-1, 1, 201) ** 2 - 1 / 3) ** 2))
        print(f"left in the residual (in-frame x^2 curvature of the moved quadratic): rms over frames {np.sqrt(np.mean(curv ** 2)):.3e} MJy/sr")
    shutil.copyfile(a.cal, out)
    with h5py.File(out, 'r+') as f:
        f['sky/continuum'][...] = sky.astype(np.float32)
        f['offsets/map_0'][...] = c_new.astype(f['offsets/map_0'].dtype)
        f['offsets/map_1'][...] = b_new.astype(f['offsets/map_1'].dtype)
        f['frame_scalar'][...] = scalar_new.astype(f['frame_scalar'].dtype)
        f.attrs['gauge'] = f"post-solve gauge: plane g = {g.tolist()} per grid px; quadratic H = {H.tolist()} per grid px^2; h_D degree {hdeg}; geometry {geometry_kind()}"
    print('wrote', out)


if __name__ == '__main__':
    main()
