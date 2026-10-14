"""Post-solve gauge: the survey mean of the per-frame offset modes, in sky coordinates, moved into the sky.

The model's per-frame offsets per subchannel s of frame k are  c[k,s] + b[k,s] xt + q[k,s] (xt^2 - 1/3)  with
xt = (x - xc) / 1020 (the x^2 term optional: a cal with offsets/map_2).  A sky quadratic + plane,
P(p) = g.(p - p0) + (p - p0)^T H (p - p0) / 2, restricted to frame k (p - p0 = J_k d + D_k, d the detector offset from
the centre, J_k the detector -> reference-grid Jacobian from the WCS, D_k the frame centre's offset) is
  d^T (J^T H J) d / 2 + d^T J^T (H D_k + g) + (D_k^T H D_k / 2 + g.D_k),
i.e. per band at detector y_s (dy_s = y_s - yc): x^2 coefficient (J^T H J)_xx 1020^2 / 2; x-slope
1020 [(J^T H J)_xy dy_s + (J^T (H D_k + g))_x]; constant (J^T H J)_yy dy_s^2 / 2 + (J^T (H D_k + g))_y dy_s;
scalar D_k^T H D_k / 2 + g.D_k.  The gauge: H (3 numbers) from the per-frame x^2 coefficients (the survey mean of
q[k] over the covered bands = (x_k^T H x_k) 1020^2 / 2 + a free detector-fixed h2), then g (2 numbers) from the per-frame
x-slopes after H's part is removed (= 1020 (x_k . g) + a free detector-fixed h); each moved into the sky with its
representation removed from the offsets. The data residual is unchanged up to the band discretisation.
    DET=d python apply_gauge.py CAL.h5 [--out OUT.h5] [--frames-dir DIR] [--no-quadratic] [--workers N]"""
import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np  # noqa: E402

X_CENTRE, X_HALF = 1019.5, 1020.0
REF = {str(d): f'/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D{d}_6p2arcsec/ref.fits' for d in range(1, 7)}


def header_of(path):
    import h5py
    import hdf5plugin  # noqa: F401
    from astropy.io import fits
    with h5py.File(path, 'r') as f:
        h = f.attrs['det_header']
    h = fits.Header.fromstring(h.decode() if isinstance(h, bytes) else str(h))
    return (float(h['CRVAL1']), float(h['CRVAL2']), float(h['PC1_1']), float(h['PC1_2']), float(h['PC2_1']), float(h['PC2_2']))


def geometry_kind():
    """'exact' (default) or 'approximate' (GAUGE_EXACT_J=0): how the frames' detector-to-grid Jacobians are built."""
    return 'approximate' if os.environ.get('GAUGE_EXACT_J', '1') == '0' else 'exact'


def geometry(hdr, det, shape):
    """Per frame: J (grid px per detector px, columns = the detector x and y axes) and the centre on the grid."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra, dec, pc11, pc12, pc21, pc22 = hdr.T
    K = ra.size
    wref = WCS(fits.getheader(os.environ.get('GAUGE_REF') or REF[det]))     # GAUGE_REF: the field's ref.fits when not NEP W17
    h, w = shape
    wrap = lambda d: (d + 180) % 360 - 180      # noqa: E731
    c0 = wref.pixel_to_world(w / 2, h / 2); cx = wref.pixel_to_world(w / 2 + 1, h / 2); cy = wref.pixel_to_world(w / 2, h / 2 + 1)
    cosd = np.cos(np.radians(c0.dec.deg))
    ex_ = np.array([wrap(cx.ra.deg - c0.ra.deg) * cosd, cx.dec.deg - c0.dec.deg]); ey_ = np.array([wrap(cy.ra.deg - c0.ra.deg) * cosd, cy.dec.deg - c0.dec.deg])
    ref_scale = np.hypot(*ex_); ex_ /= np.hypot(*ex_); ey_ /= np.hypot(*ey_)
    xk = np.zeros((K, 2)); yk = np.zeros((K, 2))
    for k in range(K):
        wk = WCS(naxis=2); wk.wcs.ctype = ['RA---TAN', 'DEC--TAN']; wk.wcs.crval = [ra[k], dec[k]]; wk.wcs.crpix = [X_CENTRE + 1, X_CENTRE + 1]
        wk.wcs.pc = [[pc11[k], pc12[k]], [pc21[k], pc22[k]]]; wk.wcs.cdelt = [1.0, 1.0]
        o = wk.pixel_to_world(X_CENTRE, X_CENTRE); ox = wk.pixel_to_world(X_CENTRE + 1, X_CENTRE); oy = wk.pixel_to_world(X_CENTRE, X_CENTRE + 1)
        cd = np.cos(np.radians(dec[k]))
        xk[k] = [wrap(ox.ra.deg - o.ra.deg) * cd, ox.dec.deg - o.dec.deg]; yk[k] = [wrap(oy.ra.deg - o.ra.deg) * cd, oy.dec.deg - o.dec.deg]
    J = np.zeros((K, 2, 2))
    J[:, 0, 0] = xk @ ex_ / ref_scale; J[:, 1, 0] = xk @ ey_ / ref_scale
    J[:, 0, 1] = yk @ ex_ / ref_scale; J[:, 1, 1] = yk @ ey_ / ref_scale
    if geometry_kind() == 'exact':
        # exact (the default since 2026-10-13, user): the frame's detector axes mapped through the reference grid's WCS
        # at the frame. The approximation above (GAUGE_EXACT_J=0, kept only to reproduce earlier products) projects them
        # on the grid's axes at the grid CENTRE, which mis-rotates off-centre frames: up to ~19 deg on D6.
        for k in range(K):
            wk = WCS(naxis=2); wk.wcs.ctype = ['RA---TAN', 'DEC--TAN']; wk.wcs.crval = [ra[k], dec[k]]; wk.wcs.crpix = [X_CENTRE + 1, X_CENTRE + 1]
            wk.wcs.pc = [[pc11[k], pc12[k]], [pc21[k], pc22[k]]]; wk.wcs.cdelt = [1.0, 1.0]
            q = np.array([wref.world_to_pixel(wk.pixel_to_world(X_CENTRE + a_, X_CENTRE + b_)) for a_, b_ in ((0, 0), (1, 0), (0, 1))], dtype=np.float64)
            J[k, :, 0] = q[1] - q[0]; J[k, :, 1] = q[2] - q[0]
    px, py = wref.world_to_pixel(SkyCoord(ra=ra * u.deg, dec=dec * u.deg))
    return J, np.stack([px, py], axis=1), ref_scale


def main():
    import h5py
    import hdf5plugin  # noqa: F401
    ap = argparse.ArgumentParser(); ap.add_argument('cal'); ap.add_argument('--out', default=None); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--frames-dir', default=None); ap.add_argument('--no-quadratic', action='store_true')
    a = ap.parse_args()
    out = a.out or a.cal.replace('.h5', '_gauge.h5')
    det = os.environ.get('DET', '3')
    with h5py.File(a.cal, 'r') as f:
        sky = f['sky/continuum'][:].astype(np.float64); cov = f['sky_coverage/continuum'][:]
        n_maps = int(f.attrs.get('num_maps', len(f['offsets'])))
        c = f['offsets/map_0'][:].astype(np.float64); b = f['offsets/map_1'][:].astype(np.float64)
        q = f['offsets/map_2'][:].astype(np.float64) if n_maps >= 3 and not a.no_quadratic else None
        ocov = f['offset_coverage/map_1'][:]
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
    dy = np.where(np.isfinite(ys), ys - X_CENTRE, 0.0)                     # band centres from the detector centre (px)
    J, centres, ref_scale = geometry(hdr, det, sky.shape)
    h, w = sky.shape
    m = cov > 0
    pyy, pxx = np.nonzero(m)
    p0 = np.array([pxx.mean(), pyy.mean()])
    D = centres - p0                                                        # frame centres from p0 (grid px)
    good = ocov > 0
    frame_ok = good.any(axis=1)
    def bandmean(arr):
        return np.array([arr[k, good[k]].mean() if frame_ok[k] else np.nan for k in range(K)])
    xk, ykv = J[:, :, 0], J[:, :, 1]                                        # the detector axes in grid units
    print(f"{frame_ok.sum()} frames; ref {ref_scale * 3600:.3f}\"/px; roll (det x on grid) spans {np.degrees(np.arctan2(J[:, 1, 0], J[:, 0, 0])).std():.0f} deg rms")
    sky_add = np.zeros(m.sum())
    px_, py_ = pxx - p0[0], pyy - p0[1]
    # ---------------------------------------------------------------- the quadratic: H from the per-frame x^2 coefficients
    if q is not None:
        qk = bandmean(q) / (X_HALF ** 2)                                     # per det px^2 (of 2 * the x^2 coefficient's half ... see below)
        ok = np.isfinite(qk)
        # q[k] (xt^2 - 1/3) with xt = dx / 1020:  the x^2 coefficient per det px^2 is q / 1020^2 = (x_k^T H x_k) / 2
        A = np.column_stack([xk[ok, 0] ** 2, 2 * xk[ok, 0] * xk[ok, 1], xk[ok, 1] ** 2, np.ones(ok.sum())]) / 2.0
        A[:, 3] *= 2.0                                                      # the free constant h2 (per det px^2)
        sol = np.linalg.lstsq(A, qk[ok], rcond=None)[0]
        H = np.array([[sol[0], sol[1]], [sol[1], sol[2]]]); h2 = sol[3]
        print(f"quadratic H = [[{H[0, 0]:+.3e}, {H[0, 1]:+.3e}], [., {H[1, 1]:+.3e}]] MJy/sr per grid px^2 "
              f"(per half-field^2: {H[0, 0] * (w / 2) ** 2:+.3e}, {H[0, 1] * (w / 2) * (h / 2):+.3e}, {H[1, 1] * (h / 2) ** 2:+.3e}); "
              f"detector-fixed x^2 h2 = {h2:+.3e} per det px^2 kept; per-frame x^2 rms {np.nanstd(qk):.3e}, after H and h2 {np.nanstd(qk[ok] - A @ sol):.3e}")
        # the representation of the quadratic in the offsets, per frame
        HJ = np.einsum('kai,ab,kbj->kij', J, H, J)                          # J^T H J, (K, 2, 2) in det coords (per det px^2)
        lin = np.einsum('kai,ab,kb->ki', J, H, D)                           # J^T H D_k, (K, 2) per det px
        q -= (HJ[:, 0, 0] / 2 * X_HALF ** 2)[:, None]
        b -= X_HALF * (HJ[:, 0, 1][:, None] * dy[None, :] + lin[:, 0][:, None])
        adj = HJ[:, 1, 1][:, None] / 2 * dy[None, :] ** 2 + lin[:, 1][:, None] * dy[None, :]
        adj_mean = bandmean(adj); adj_mean = np.where(np.isfinite(adj_mean), adj_mean, 0.0)
        c -= adj - adj_mean[:, None]
        scalar -= adj_mean + 0.5 * np.einsum('ka,ab,kb->k', D, H, D)
        sky_add += 0.5 * (H[0, 0] * px_ ** 2 + 2 * H[0, 1] * px_ * py_ + H[1, 1] * py_ ** 2)
    # ---------------------------------------------------------------- the plane: g from the per-frame x-slopes
    bx = bandmean(b) / X_HALF
    ok = np.isfinite(bx)
    A = np.column_stack([xk[ok], np.ones(ok.sum())])
    sol = np.linalg.lstsq(A, bx[ok], rcond=None)[0]
    g, hfix = sol[:2], sol[2]
    print(f"plane g = ({g[0]:+.3e}, {g[1]:+.3e}) MJy/sr per grid px = ({g[0] * w / 2:+.3e}, {g[1] * h / 2:+.3e}) per half-field; "
          f"detector-fixed x-gradient h = {hfix:+.3e} per det px kept; per-frame x-gradients rms {np.nanstd(bx):.3e}, after g and h {np.nanstd(bx[ok] - A @ sol):.3e}")
    b -= X_HALF * (xk @ g)[:, None]
    adj = (ykv @ g)[:, None] * dy[None, :]
    adj_mean = bandmean(adj); adj_mean = np.where(np.isfinite(adj_mean), adj_mean, 0.0)
    c -= adj - adj_mean[:, None]
    scalar -= adj_mean + D @ g
    sky_add += g[0] * px_ + g[1] * py_
    sky[m] += sky_add - sky_add.mean()
    shutil.copyfile(a.cal, out)
    with h5py.File(out, 'r+') as f:
        f['sky/continuum'][...] = sky.astype(np.float32)
        f['offsets/map_0'][...] = c.astype(f['offsets/map_0'].dtype)
        f['offsets/map_1'][...] = b.astype(f['offsets/map_1'].dtype)
        if q is not None:
            f['offsets/map_2'][...] = q.astype(f['offsets/map_2'].dtype)
        f['frame_scalar'][...] = scalar.astype(f['frame_scalar'].dtype)
        f.attrs['gauge'] = f"survey-mean plane g = {g.tolist()} per grid px" + (f"; quadratic H = {H.tolist()} per grid px^2" if q is not None else "") + f"; geometry {geometry_kind()}"
    print('wrote', out)


if __name__ == '__main__':
    main()
