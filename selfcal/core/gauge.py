"""The post-solve gauge of a per-frame gradient model.

A model whose offsets hold, per frame, a constant per chunk (a band of detector rows) and a slope
along detector x leaves a family of solutions the data cannot tell apart: a sky plane is
indistinguishable from its image in those offsets -- per frame, the plane's value at the frame's
centre, its steps from band to band and its slope along x (exactly when a band is one detector
row; with bands of a few rows, up to the plane's ramp across a band). The solve lands anywhere in
that family. The gauge picks one member: the survey-mean plane the offsets absorbed, found in the
solved slopes and moved into the sky. A sky quadratic is absorbed the same way apart from its
curvature along x within a frame, which the offsets have no term for: the data pin it, weakly, so
it is not a freedom of the model, and moving it (``degree=2``) can take real sky curvature out of
the map along with what the solve left there. The plane is the default.

The plane (``degree=1``) is fitted to each frame's mean slope over its covered chunks, per detector
pixel after dividing by the basis's half-width ``h``::

    mean_s b[k, s] / h = x_k . g + c

``x_k`` is the frame's detector x axis on the reference grid (its Jacobian's first column), ``g`` the
sky plane and ``c`` a slope fixed on the detector, the same in every frame. The frames roll through
many angles, so the two separate; least squares over the frames, unweighted.

With ``degree=2`` the plane and the quadratic are fitted to every solved slope ``b[k, s]`` (frame
``k``, chunk ``s``), separating what turns with the frame from what does not::

    b[k, s] / h = x_k . (g + H D_k) + (x_k^T H y_k) dy_s + h_s + h_D(D_k)

``y_k`` is the frame's detector y axis on the grid, ``D_k`` the frame's centre from ``p0`` (the mean
covered pixel), ``dy_s`` the chunk's row offset from the detector centre, ``H`` the sky quadratic,
``h_s`` a detector-fixed slope per chunk and ``h_D`` a polynomial in the frame's position of such a
slope (degree ``position_degree``; the amplitude of a detector-fixed gradient may vary with
pointing); a coverage-weighted least squares over every (frame, chunk).

The polynomial ``P(p) = g . (p - p0) + (p - p0)^T H (p - p0) / 2`` (``H = 0`` for the plane), minus its
mean over the covered pixels, is then added to the sky term and recorded on the cal (``gauge/``);
with ``apply='map'`` every mosaic made from the cal adds it to its maps (the offsets stay as solved),
with ``apply='offsets'`` its image is removed from the offsets instead (the slopes, the chunk
constants, the per-frame scalar), which is exact only up to the band discretisation (and, for a
quadratic, its curvature within a frame).

The geometry: each frame's detector axes and centre on the reference grid, from a tangent plane
through the frame's stored header (``geometry='tangent'``: its pointing, reference pixel, PC matrix
and scale, no distortion terms) or from its full stored WCS (``'header'``). These are the
computations of the transfer-function campaign's post-solve gauges, the plane and the quadratic,
kept bit for bit (the ``transfer`` and ``transfer_quad`` gates of ``selfcal_scripts/gates``).
"""
from __future__ import annotations

import numpy as np

__all__ = ['GAUGE_KEYS', 'frame_geometry', 'chunk_row_offsets', 'fit_gauge', 'move_into_offsets', 'polynomial',
           'gauge_cal', 'read_gauge', 'apply_to_maps']

GAUGE_KEYS = ('CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CDELT1', 'CDELT2',
              'CTYPE1', 'CTYPE2')


# ---------------------------------------------------------------------------------------------- geometry
def frame_geometry(headers, ref_wcs, centre, geometry='tangent'):
    """Per frame: ``J`` (2, 2) -- the detector x and y axes on the reference grid, in grid pixels per
    detector pixel, at the detector ``centre`` ``(cx, cy)`` -- and the centre's grid position.

    ``headers``: ``{keyword: (n_frames,) values}`` of :data:`GAUGE_KEYS` (``'tangent'``) or the
    frames' ``astropy.io.fits.Header`` objects (``'header'``, the full stored WCS)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS

    cx, cy = float(centre[0]), float(centre[1])
    if geometry == 'tangent':
        ra, dec = np.asarray(headers['CRVAL1'], np.float64), np.asarray(headers['CRVAL2'], np.float64)
        K = ra.size
        for k in GAUGE_KEYS:
            if k not in headers or len(headers[k]) != K:
                raise ValueError(f"the frames' headers lack {k} (the gauge's tangent geometry needs {GAUGE_KEYS})")
        bad = [k for k in GAUGE_KEYS[:10] if not np.all(np.isfinite(np.asarray(headers[k], np.float64)))]
        if bad:
            raise ValueError(f"a frame's header has no finite {bad}: the gauge cannot place it on the grid")
        wcs_of = []
        for k in range(K):
            w = WCS(naxis=2)
            w.wcs.ctype = [str(headers['CTYPE1'][k]).replace('-SIP', ''), str(headers['CTYPE2'][k]).replace('-SIP', '')]
            w.wcs.crval = [ra[k], dec[k]]
            w.wcs.crpix = [float(headers['CRPIX1'][k]), float(headers['CRPIX2'][k])]
            w.wcs.pc = [[float(headers['PC1_1'][k]), float(headers['PC1_2'][k])],
                        [float(headers['PC2_1'][k]), float(headers['PC2_2'][k])]]
            w.wcs.cdelt = [float(headers['CDELT1'][k]), float(headers['CDELT2'][k])]
            wcs_of.append(w)
    elif geometry == 'header':
        wcs_of = [WCS(h) for h in headers]
        K = len(wcs_of)
        c = [w.pixel_to_world(cx, cy) for w in wcs_of]
        ra = np.array([s.ra.deg for s in c], np.float64)
        dec = np.array([s.dec.deg for s in c], np.float64)
    else:
        raise ValueError(f"geometry {geometry!r}: 'tangent' or 'header'")
    J = np.zeros((K, 2, 2))
    for k, w in enumerate(wcs_of):
        q = np.array([ref_wcs.world_to_pixel(w.pixel_to_world(cx + a, cy + b)) for a, b in ((0, 0), (1, 0), (0, 1))],
                     dtype=np.float64)
        J[k, :, 0] = q[1] - q[0]
        J[k, :, 1] = q[2] - q[0]
    px, py = ref_wcs.world_to_pixel(SkyCoord(ra=ra * u.deg, dec=dec * u.deg))
    return J, np.stack([np.asarray(px, np.float64), np.asarray(py, np.float64)], axis=1)


def chunk_row_offsets(chunk_map, n_chunks, row_centre):
    """Each chunk's mean detector row minus ``row_centre`` (0 for a chunk absent from the map)."""
    ys = np.full(n_chunks, np.nan)
    yy = np.arange(chunk_map.shape[0])[:, None] * np.ones(chunk_map.shape, dtype=np.float64)
    for cid in np.unique(chunk_map[chunk_map >= 0]):
        if cid < n_chunks:
            ys[cid] = yy[chunk_map == cid].mean()
    return np.where(np.isfinite(ys), ys - float(row_centre), 0.0)


# ---------------------------------------------------------------------------------------------- the fit
def fit_gauge(b, ocov, J, D, dy, half, *, degree=1, position_degree=2, scale_D=1.0):
    """``g`` (2,), ``H`` (2, 2) and the fit's statistics from the solved slopes ``b`` (n_frames,
    n_chunks; coverage ``ocov``), the frames' Jacobians ``J`` and the basis's half-width ``half``.
    ``degree`` 1: the plane, from each frame's mean slope (``H`` is zero). 2: the plane and the
    quadratic from every slope, with the frames' centres ``D`` (from ``p0``), the chunks' row offsets
    ``dy`` and a position polynomial of degree ``position_degree`` whose positions are scaled by
    ``scale_D`` (half the grid); the plane ignores these four."""
    if degree == 1:
        return _fit_plane(b, ocov, J[:, :, 0], half)
    K, S = b.shape
    xk, yk = J[:, :, 0], J[:, :, 1]
    good = (ocov > 0) & np.isfinite(b)
    kk, ss = np.nonzero(good)
    if kk.size == 0:
        raise ValueError('the gauge has no solved slopes to fit (no frame covers the slope term)')
    v = b[kk, ss] / half
    wts = np.sqrt(ocov[kk, ss])
    x0, x1, y0, y1 = xk[kk, 0], xk[kk, 1], yk[kk, 0], yk[kk, 1]
    D0, D1 = D[kk, 0], D[kk, 1]
    dys = dy[ss]
    cols, names = [x0, x1], ['g0', 'g1']
    if degree == 2:
        cols += [x0 * D0 + x0 * y0 * dys, x0 * D1 + x1 * D0 + (x0 * y1 + x1 * y0) * dys, x1 * D1 + x1 * y1 * dys]
        names += ['H00', 'H01', 'H11']
    for s_ in range(S):                                                 # h_s: detector-fixed, per chunk
        if (ss == s_).any():
            cols.append((ss == s_).astype(np.float64))
            names.append(f'h{s_}')
    for p in range(1, int(position_degree) + 1):                        # h_D: smooth in the frame's position
        for q in range(p + 1):
            cols.append((D0 / scale_D) ** (p - q) * (D1 / scale_D) ** q)
            names.append(f'hD{p - q}{q}')
    A = np.stack(cols, axis=1)
    sol = np.linalg.lstsq(A * wts[:, None], v * wts, rcond=None)[0]
    res = v - A @ sol
    par = dict(zip(names, sol))
    g = np.array([par['g0'], par['g1']])
    H = (np.array([[par['H00'], par['H01']], [par['H01'], par['H11']]]) if degree == 2 else np.zeros((2, 2)))

    def wrms(z):
        return float(np.sqrt(np.sum(wts ** 2 * z ** 2) / np.sum(wts ** 2)))

    stats = {'n_slopes': int(kk.size), 'n_parameters': len(names), 'slope_rms': wrms(v), 'residual_rms': wrms(res)}
    if degree == 2:
        HJ = np.einsum('kai,ab,kbj->kij', J, H, J)
        curv = 0.5 * np.abs(HJ[:, 0, 0]) * half ** 2 * np.sqrt(np.mean((np.linspace(-1, 1, 201) ** 2 - 1 / 3) ** 2))
        stats['curvature_left_rms'] = float(np.sqrt(np.mean(curv ** 2)))   # what the offsets cannot take
    return g, H, stats


def _fit_plane(b, ocov, xk, half):
    """The plane alone: each frame's mean slope over its covered chunks, per detector pixel, is
    ``x_k . g + c`` with ``c`` a slope fixed on the detector; unweighted least squares over the frames."""
    K = b.shape[0]
    good = ocov > 0
    frame_ok = good.any(axis=1)
    bx = np.array([b[k, good[k]].mean() if frame_ok[k] else np.nan for k in range(K)]) / half
    ok = np.isfinite(bx)
    if not ok.any():
        raise ValueError('the gauge has no solved slopes to fit (no frame covers the slope term)')
    A = np.column_stack([xk[ok], np.ones(ok.sum())])
    sol = np.linalg.lstsq(A, bx[ok], rcond=None)[0]
    stats = {'n_slopes': int(good[ok].sum()), 'n_frames': int(ok.sum()), 'n_parameters': 3,
             'slope_rms': float(np.nanstd(bx)), 'residual_rms': float(np.std(bx[ok] - A @ sol)),
             'detector_slope': float(sol[2])}
    return sol[:2].copy(), np.zeros((2, 2)), stats


def move_into_offsets(c, b, scalar, good, J, D, dy, g, H, half):
    """The image of the polynomial removed from the offsets (``apply='offsets'``): the slopes ``b``,
    the chunk constants ``c`` (per frame, mean-zero over its covered chunks) and the per-frame
    ``scalar``; returns the three new arrays."""
    K = b.shape[0]
    HJ = np.einsum('kai,ab,kbj->kij', J, H, J)                             # J^T H J (per det px^2)
    lin = np.einsum('kai,ka->ki', J, (D @ H.T) + g[None, :])               # J^T (H D_k + g) (per det px)
    b_new = b - half * (lin[:, 0][:, None] + HJ[:, 0, 1][:, None] * dy[None, :])
    adj = lin[:, 1][:, None] * dy[None, :] + 0.5 * HJ[:, 1, 1][:, None] * dy[None, :] ** 2
    gm = good.any(axis=1)
    adj_mean = np.array([adj[k, good[k]].mean() if gm[k] else 0.0 for k in range(K)])
    c_new = c - (adj - adj_mean[:, None])
    scalar_new = scalar - adj_mean - (D @ g) - 0.5 * np.einsum('ka,ab,kb->k', D, H, D)
    return c_new, b_new, scalar_new


def polynomial(shape, g, H, p0, mean=0.0):
    """``P(p) - mean`` over a grid of ``shape`` (float64): ``g . (p - p0) + (p - p0)^T H (p - p0) / 2``."""
    H_, W_ = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:H_, 0:W_]
    px, py = xx.astype(np.float64) - p0[0], yy.astype(np.float64) - p0[1]
    return (g[0] * px + g[1] * py + 0.5 * (H[0, 0] * px ** 2 + 2 * H[0, 1] * px * py + H[1, 1] * py ** 2)) - mean


def _polynomial_at(pxx, pyy, g, H, p0):
    px, py = pxx - p0[0], pyy - p0[1]
    return g[0] * px + g[1] * py + 0.5 * (H[0, 0] * px ** 2 + 2 * H[0, 1] * px * py + H[1, 1] * py ** 2)


# ---------------------------------------------------------------------------------------------- on a cal
def gauge_cal(cal_path, frames, ref_wcs, *, slope_map, constant_map, centre, half, degree=1, position_degree=2,
              apply='map', geometry='tangent', sky_name=None, log=print, max_workers=8):
    """Gauge the cal at ``cal_path`` in place: fit the sky plane (``degree=1``) or plane and quadratic
    (``degree=2``, with a position polynomial of degree ``position_degree`` in the nuisance) to the
    solved slopes of offset map ``slope_map`` (the chunk constants in map ``constant_map``), add it
    (minus its mean over the covered pixels) to the sky term ``sky_name`` (default: the first), record
    it under ``gauge/`` and, with ``apply='offsets'``, remove its image from the offsets. ``frames``:
    the cal's frames (their headers give the geometry); ``centre`` ``(cx, cy)`` and ``half``: the slope
    basis's detector centre and half-width, ``b = (det_x - cx) / half``. Returns a summary for the record."""
    import h5py

    from ..io.frames import frame_header_values

    with h5py.File(cal_path, 'r') as f:
        names = f.attrs.get('sky_components')
        if sky_name is None:
            if names is not None and len(names):
                n0 = names[0]
                sky_name = n0.decode() if isinstance(n0, bytes) else str(n0)
            else:
                sky_name = 'continuum'
        sky = f[f'sky/{sky_name}'][:].astype(np.float64)
        cov = f[f'sky_coverage/{sky_name}'][:]
        c = f[f'offsets/map_{constant_map}'][:].astype(np.float64)
        b = f[f'offsets/map_{slope_map}'][:].astype(np.float64)
        ocov = f[f'offset_coverage/map_{slope_map}'][:].astype(np.float64)
        scalar = f['frame_scalar'][:].astype(np.float64) if 'frame_scalar' in f else None
        cmap = f[f'chunk_maps/map_{constant_map}'][:]
        n_frames = len(f['reproj_list'])
    if len(frames) != n_frames:
        raise ValueError(f"the gauge was given {len(frames)} frames for a cal of {n_frames}")
    K, S = b.shape
    if c.shape != (K, S):
        raise ValueError(f"the gauge's constant map {constant_map} has {c.shape[1]} chunks, the slope map {slope_map} "
                         f"{S}: the two terms must share their chunk map")
    if apply == 'offsets' and scalar is None:
        raise ValueError("apply='offsets' moves part of the polynomial into the per-frame scalar, which this cal has none of")
    if geometry == 'tangent':
        headers = frame_header_values(frames, GAUGE_KEYS, max_workers=max_workers)
    else:
        headers = [_stored_header(p) for p in frames]
    J, centres = frame_geometry(headers, ref_wcs, centre, geometry=geometry)
    dy = chunk_row_offsets(cmap, S, centre[1])
    h, w = sky.shape
    m = cov > 0
    pyy, pxx = np.nonzero(m)
    p0 = np.array([pxx.mean(), pyy.mean()])
    D = centres - p0
    g, H, stats = fit_gauge(b, ocov, J, D, dy, half, degree=degree, position_degree=position_degree,
                            scale_D=max(w, h) / 2.0)
    sky_add = _polynomial_at(pxx.astype(np.float64), pyy.astype(np.float64), g, H, p0)
    mean = float(sky_add.mean())
    sky[m] += sky_add - mean
    if degree == 1:
        fitted = (f"the plane, from {stats['n_frames']} frames' mean slopes (of {K} frames x {S} chunks): mean slope rms "
                  f"{stats['slope_rms']:.3e} -> {stats['residual_rms']:.3e} per det px after the fit, the detector-fixed "
                  f"slope {stats['detector_slope']:+.3e} per det px stays in the offsets")
    else:
        fitted = (f"the plane and the quadratic, from {stats['n_slopes']} slopes of {K} frames x {S} chunks, "
                  f"{stats['n_parameters']} parameters (position degree {position_degree}): slope rms "
                  f"{stats['slope_rms']:.3e} -> {stats['residual_rms']:.3e} per det px after the fit")
    log(f"[gauge] {fitted} ({geometry} geometry); plane g = ({g[0] * w / 2:+.3e}, {g[1] * h / 2:+.3e}) per half-field"
        + (f"; quadratic H (half-field^2) = [[{H[0, 0] * (w / 2) ** 2:+.3e}, {H[0, 1] * (w / 2) * (h / 2):+.3e}], "
           f"[., {H[1, 1] * (h / 2) ** 2:+.3e}]], curvature left in the frames rms "
           f"{stats['curvature_left_rms']:.3e}" if degree == 2 else '') + f"; applied to the {apply}")
    with h5py.File(cal_path, 'r+') as f:
        f[f'sky/{sky_name}'][...] = sky.astype(np.float32)
        if apply == 'offsets':
            good = (ocov > 0) & np.isfinite(b)
            c_new, b_new, scalar_new = move_into_offsets(c, b, scalar, good, J, D, dy, g, H, half)
            f[f'offsets/map_{constant_map}'][...] = c_new.astype(f[f'offsets/map_{constant_map}'].dtype)
            f[f'offsets/map_{slope_map}'][...] = b_new.astype(f[f'offsets/map_{slope_map}'].dtype)
            f['frame_scalar'][...] = scalar_new.astype(f['frame_scalar'].dtype)
        if 'gauge' in f:
            del f['gauge']
        grp = f.create_group('gauge')
        grp.create_dataset('g', data=np.asarray(g, np.float64))
        grp.create_dataset('H', data=np.asarray(H, np.float64))
        grp.create_dataset('p0', data=np.asarray(p0, np.float64))
        grp.attrs['mean'] = mean
        grp.attrs['sky'] = sky_name
        grp.attrs['degree'] = int(degree)
        if degree == 2:
            grp.attrs['position_degree'] = int(position_degree)
        grp.attrs['applied_to'] = apply
        grp.attrs['geometry'] = geometry
        grp.attrs['slope_map'] = int(slope_map)
        grp.attrs['constant_map'] = int(constant_map)
        grp.attrs['centre'] = np.asarray(centre, np.float64)
        grp.attrs['half'] = float(half)
        for k_, v_ in stats.items():
            grp.attrs[k_] = v_
    return {'g': [float(x) for x in g], 'H': [[float(x) for x in row] for row in H], 'p0': [float(x) for x in p0],
            'mean': mean, 'degree': int(degree), **({'position_degree': int(position_degree)} if degree == 2 else {}),
            'applied_to': apply, 'geometry': geometry, 'sky': sky_name, **stats}


def _stored_header(path):
    """The frame's stored detector header (``det_header``) as an ``astropy.io.fits.Header``."""
    import h5py
    from astropy.io import fits
    with h5py.File(path, 'r') as f:
        h = f.attrs.get('det_header', b'')
    return fits.Header.fromstring(h.decode() if isinstance(h, bytes) else str(h))


def read_gauge(f):
    """The gauge recorded on an open cal (an ``h5py.File``), or None."""
    if 'gauge' not in f:
        return None
    grp = f['gauge']
    out = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in grp.attrs.items()}
    out['g'] = grp['g'][:]
    out['H'] = grp['H'][:]
    out['p0'] = grp['p0'][:]
    return out


def apply_to_maps(maps, gauge, log=print):
    """Add the recorded polynomial (``gauge``, from :func:`read_gauge`, applied to the map) to the mean
    and clipped-mean maps of a mosaic's ``maps`` (``{name: {'data', 'weight', ...}}``) where they have
    weight; the std map is unchanged."""
    if gauge is None or gauge.get('applied_to') != 'map':
        return []
    done = []
    add = None
    for name in ('mean_map', 'sc_mean_map'):
        entry = maps.get(name)
        if entry is None or entry.get('data') is None:
            continue
        d = np.asarray(entry['data'], dtype=np.float32)
        w = np.asarray(entry['weight'])
        if add is None:
            add = polynomial(d.shape, gauge['g'], gauge['H'], gauge['p0'], float(gauge['mean'])).astype(np.float32)
        entry['data'] = np.where(w > 0, d + add, d).astype(np.float32)
        done.append(name)
    if done:
        log(f"[gauge] the cal's post-solve gauge (plane g = {np.asarray(gauge['g']).tolist()} per grid px, degree "
            f"{gauge.get('degree')}) added to {', '.join(done)}")
    return done
