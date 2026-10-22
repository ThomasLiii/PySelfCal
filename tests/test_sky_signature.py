"""``Offsets(exclude=SkySignature(k))``: an offset term may not carry its response to the sky polynomials of degree 1 to
k. Toy: frames of 8 x 8 detector pixels at random places and quarter-turn rotations on a sky grid, each frame with one
constant per detector row (the per-subchannel constants of a strip instrument) and an x-slope, one per frame or one
per row. A sky plane is then an exact gauge (in every frame the row constants take its part across the rows and the
slope its part along them); with the slope's plane signature excluded (shapes over every frame's slope, which follow
the frames' rotations) the plane can only be fitted by the sky. Comparisons over the covered pixels, the mean (the
gauge left) removed. Enough frames (600 on a 28 x 28 grid) that no frame row is covered only by frames of one
orientation (that would add local gauges of the toy, a pattern along the rows' other direction)."""
import json

import numpy as np
import pytest
import scipy.sparse as sp

import selfcal as sc
from selfcal.core.solve import apply_lsqr

S = 8                                                     # detector side (rows x columns)


def _toy(h=28, w=28, frames=600, plane=(0.6, -0.35), quad=0.0, shared=False, slope_per_row=True, slope_scale=0.0,
         noise=1e-6, seed=5, rotations=(0, 1, 2, 3), fixed_slope=0.0, weak_frames=0.0, weak_weight=1e-6):
    """A sky (a plane + quad * a centred bowl + a random field), random row constants, slopes of ``slope_scale``,
    ``noise``; ``shared``: a pattern along the detector columns shared by every frame (data and model);
    ``rotations``: the quarter turns frames take; ``fixed_slope``: a slope the same in every frame (per row: a
    detector-fixed pattern) added to the data; ``weak_frames``: the fraction of frames whose rows carry the weight
    ``weak_weight`` (rows and data scaled by its square root). Returns a dict: A, b, npix, grid, slope_cols,
    slope_chunks, sky (the true sky)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx / (w - 1)).ravel(), (yy / (h - 1)).ravel()
    sky = plane[0] * u + plane[1] * v + quad * ((u - 0.5) ** 2 + (v - 0.5) ** 2) + 0.05 * rng.normal(size=h * w)
    npix = h * w
    ns = S if slope_per_row else 1
    per_frame = S + ns
    shared0 = npix + frames * per_frame
    pattern = 0.1 * np.cos(np.arange(S) * 0.8)
    fixed = fixed_slope * (1.0 + 0.5 * np.sin(np.arange(ns) * 1.3))
    rows, cols, vals, b, slope_cols, slope_chunks = [], [], [], [], [], []
    r = 0
    for f in range(frames):
        rot = int(rng.choice(rotations))
        y0, x0 = rng.integers(0, h - S + 1), rng.integers(0, w - S + 1)
        sw = np.sqrt(weak_weight) if rng.random() < weak_frames else 1.0
        c0 = npix + f * per_frame
        slope_cols.extend(range(c0 + S, c0 + S + ns))
        slope_chunks.extend(range(ns))
        const = rng.normal(0.0, 0.3, S)
        slope = (rng.normal(0.0, slope_scale, ns) if slope_scale else np.zeros(ns)) + fixed
        for i in range(S):
            for j in range(S):
                gy, gx = {0: (i, j), 1: (j, S - 1 - i), 2: (S - 1 - i, S - 1 - j), 3: (S - 1 - j, i)}[rot]
                p = (y0 + gy) * w + (x0 + gx)
                t = (j - (S - 1) / 2) / ((S - 1) / 2)
                k = i if slope_per_row else 0
                rows += [r, r, r]
                cols += [p, c0 + i, c0 + S + k]
                vals += [sw, sw, sw * t]
                val = sky[p] + const[i] + slope[k] * t + noise * rng.normal()
                if shared:
                    rows.append(r); cols.append(shared0 + j); vals.append(sw)
                    val += pattern[j]
                b.append(sw * val)
                r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, shared0 + (S if shared else 0)))
    return {'A': A, 'b': np.asarray(b), 'npix': npix, 'grid': (h, w), 'slope_cols': np.asarray(slope_cols),
            'slope_chunks': np.asarray(slope_chunks), 'sky': sky}


def _sig(t, k):
    return [(t['slope_cols'], k, 'slopes', t['slope_chunks'])]


def _solve(t, deflate, tmp, signatures=None, iter_lim=800):
    A, b = t['A'], t['b']
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    return apply_lsqr(A[:, active].tocsr(), b.copy(), ref_shape=t['grid'], atol=0.0, btol=0.0, conlim=0.0, damp=0.0,
                      solver='cgls', n_threads=1, iter_lim=iter_lim, active_mask=active, num_cols_full=A.shape[1],
                      return_record=True, scratch_dir=tmp, sky_names=('sky',), deflate=deflate, signatures=signatures)


def _error(t, x):
    """The solved sky minus the true one over the covered pixels, the mean (a gauge) removed."""
    cov = np.asarray((t['A'][:, :t['npix']] != 0).sum(axis=0)).ravel() > 0
    d = (x[:t['npix']] - t['sky'])[cov]
    return d - d.mean(), cov


def _fields(t, cov, quad=False):
    h, w = t['grid']
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx / (w - 1)).ravel(), (yy / (h - 1)).ravel()
    cols = [np.ones(h * w), u, v] + ([u * u, v * v, u * v] if quad else [])
    return np.stack(cols, 1)[cov]


def _seen(rec):
    """The deflation's seen fractions, sqrt(lambda / lambda_max), ascending."""
    ev = np.sort(np.asarray(json.loads(rec.deflation)['eigenvalues'], dtype=np.float64))
    return np.sqrt(np.maximum(ev, 0.0) / ev.max())


def test_the_plane_is_an_exact_gauge_without_and_recovered_with_the_exclusion(tmp_path):
    t = _toy(slope_per_row=False)
    x_free, rec_free = _solve(t, sc.Deflate(2), str(tmp_path))
    x_sig, rec_sig = _solve(t, sc.Deflate(2), str(tmp_path), signatures=_sig(t, 1))
    d_free, d_sig = json.loads(rec_free.deflation), json.loads(rec_sig.deflation)
    e_free, cov = _error(t, x_free)
    e_sig, _ = _error(t, x_sig)
    print(f'exact gauges free {d_free["gauges"]}, with the exclusion {d_sig["gauges"]}; |solved - true sky| rms: free '
          f'{e_free.std():.2e}, with {e_sig.std():.2e}')
    assert 'signature_exclusions' not in d_free
    [info] = d_sig['signature_exclusions']
    assert info['directions'] == 2 and info['degree'] == 1 and info['constraint_held'] < 1e-10
    assert d_free['gauges'] == 3 and d_sig['gauges'] == 1          # the mean + two planes, then the mean alone
    assert e_free.std() > 0.05                                     # free: the start sets the plane
    assert e_sig.std() < 1e-5                                      # the true sky, plane included
    A, b = t['A'], t['b']
    rf, rs = np.linalg.norm(A @ x_free - b), np.linalg.norm(A @ x_sig - b)
    assert abs(rs - rf) < 1e-6 * max(rf, 1e-12) + 1e-9             # a gauge: the fit is the same


def test_what_the_true_offsets_hold_along_the_signature_goes_to_the_sky_as_that_field(tmp_path):
    """The exclusion's assumption: the term carries nothing along the signature (beyond its detector-fixed part).
    Slopes that do (here random ones) put exactly that part into the sky, as the field: the error is a plane and
    nothing else."""
    t = _toy(slope_per_row=False, slope_scale=0.1)
    x, _ = _solve(t, sc.Deflate(2), str(tmp_path), signatures=_sig(t, 1))
    e, cov = _error(t, x)
    M = _fields(t, cov)
    rest = e - M @ np.linalg.lstsq(M, e, rcond=None)[0]
    print(f'|solved - true sky| rms {e.std():.2e}; after a plane {rest.std():.2e}')
    assert e.std() > 1e-3 and rest.std() < 1e-5


def test_a_detector_fixed_slope_stays_in_the_offsets_whatever_the_orientations(tmp_path):
    """Frames at three of the four quarter turns, so the plane's slope signature does not average out over the
    frames, and a slope the same in every frame (fixed on the detector) in the data: the detector-fixed part of the
    signature is left free, so that slope stays in the offsets and the true sky comes back. (One slope per frame:
    with one per row a saddle u v is an exact gauge of this toy as well, which degree 1 leaves.)"""
    t = _toy(rotations=(0, 1, 2), fixed_slope=0.3, slope_per_row=False, seed=9)
    x, rec = _solve(t, sc.Deflate(2), str(tmp_path), signatures=_sig(t, 1))
    [info] = json.loads(rec.deflation)['signature_exclusions']
    e, _ = _error(t, x)
    print(f'|solved - true sky| rms {e.std():.2e}; share of the signature not detector-fixed {info["share_constrained"]}')
    assert min(info['share_constrained']) < 0.99                    # the unbalanced orientations: a detector-fixed part
    assert e.std() < 1e-5


def test_poorly_measured_frames_do_not_take_the_constraint_up(tmp_path):
    """A tenth of the frames at a millionth of the weight: their slopes cost the fit almost nothing, so a constraint
    counting every coefficient alike could be met by them alone while the other frames' slopes went on absorbing
    the plane. In the data's metric they count by what the data say about them: the planes stay as well seen as
    without those frames."""
    seen = {}
    for weak in (0.0, 0.1):
        t = _toy(weak_frames=weak, slope_per_row=False, seed=13)
        _, rec = _solve(t, sc.Deflate(2), str(tmp_path), signatures=_sig(t, 1), iter_lim=50)
        seen[weak] = _seen(rec)[1]                                  # the least-seen direction after the mean gauge
        assert json.loads(rec.deflation)['gauges'] == 1
    print(f'least-seen direction after the mean: {seen}')
    assert seen[0.1] > 0.5 * seen[0.0]


def test_unset_or_with_signatures_none_nothing_changes(tmp_path):
    t = _toy(frames=120, seed=7)
    x1, _ = _solve(t, sc.Deflate(2), str(tmp_path))
    x2, _ = _solve(t, sc.Deflate(2), str(tmp_path), signatures=None)
    assert np.array_equal(x1, x2)


def test_with_a_shared_pattern_the_quadratics_come_back_with_degree_2(tmp_path):
    """Slopes per row and a pattern along the detector columns shared by every frame: the bilinear fields (the planes,
    the saddle) are gauges, and a centred bowl nearly one (the shared pattern takes its quadratic along the columns,
    the row constants the rest, the slopes its gradient). Degree 1 leaves the saddle; degree 2 makes all of them the
    sky's: the true sky comes back."""
    t = _toy(quad=0.8, shared=True, seed=11)
    x1, rec1 = _solve(t, sc.Deflate(2), str(tmp_path), signatures=_sig(t, 1))
    x2, rec2 = _solve(t, sc.Deflate(2), str(tmp_path), signatures=_sig(t, 2))
    d2 = json.loads(rec2.deflation)
    e1, _ = _error(t, x1)
    e2, _ = _error(t, x2)
    print(f'|solved - true sky| rms: degree 1 {e1.std():.2e}, degree 2 {e2.std():.2e}; exact gauges with degree 2 {d2["gauges"]}')
    assert d2['signature_exclusions'][0]['directions'] == 5 and d2['gauges'] == 1
    assert e1.std() > 1e-3 and e2.std() < 1e-5


def test_every_pattern_but_the_constant(tmp_path):
    """SkySignature('all'): the term may not carry its response to any pattern of the deflation (cosines included) but
    the constant. The toy's true slopes are zero, so the constraint holds for the truth: the true sky comes back as
    with degree 2, the constant stays the one exact gauge, and the record names every other pattern."""
    t = _toy(quad=0.8, shared=True, seed=11)
    x, rec = _solve(t, sc.Deflate(2), str(tmp_path), signatures=_sig(t, 'all'))
    d = json.loads(rec.deflation)
    [info] = d['signature_exclusions']
    e, _ = _error(t, x)
    print(f"'all': {len(info['fields'])} fields, {info['directions']} directions; |solved - true sky| rms {e.std():.2e}")
    assert info['degree'] == 'all' and len(info['fields']) == len(d['eigenvalues']) - 1
    assert not any(f.split(':', 1)[1] in ('0,0', 'P0,0') for f in info['fields'])
    assert d['gauges'] == 1 and e.std() < 1e-5 and info['constraint_held'] < 1e-10


def test_the_setting_lowers_and_needs_the_deflation(tmp_path):
    o = sc.Offsets('xs', on='subchannel', exclude=sc.SkySignature(2))
    assert o.lower()['signature_degree'] == 2 and 'exclude_degree' not in o.lower()
    assert sc.Offsets('xs', on='subchannel', exclude=sc.SkySignature('all')).lower()['signature_degree'] == 'all'
    for bad in (0, 3, 'any', 1.5):
        with pytest.raises(sc.ConfigError):
            sc.SkySignature(bad)
    with pytest.raises(sc.ConfigError, match='per frame'):
        sc.Offsets('xs', on='subchannel', per='all', exclude=sc.SkySignature(1))
    t = _toy(frames=40, seed=3)
    with pytest.raises(ValueError, match='deflate'):
        _solve(t, None, str(tmp_path), signatures=_sig(t, 1))
