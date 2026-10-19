"""The deflation's freeze rules. ``Deflate(gauge_below=)``: directions of the pattern space the data see below a
fraction of the best-seen one are gauges (zero sky amplitude), whatever footprint or model made them near-null.
``Deflate(absorbed_by_shared=)``: directions the data would see much better if the frame-shared offsets (whatever
their shape or basis) could not absorb them are what the shared pattern mimics, and are frozen likewise. A frozen
direction stays in the deflation space at zero amplitude; the rest of the solution is the least-squares one. Unset,
nothing changes."""
import json

import numpy as np
import pytest
import scipy.sparse as sp

import selfcal as sc
from selfcal.core import deflation
from selfcal.core.solve import apply_lsqr


def _shared_pattern_system(h=24, w=24, frames=300, carry=0.95, seed=3):
    """A sky of h x w pixels (a 0.7 y ramp, a 0.4 (x/w)^2 bowl, noise) seen by frames of 8 rows x 16 columns at
    random places, each with a constant and a linear slope in x; a
    fraction ``carry`` of the frames also carry a frame-shared pattern: 16 offsets, one per frame column, the
    same in every carrying frame (a detector-fixed pattern, as a shared term of any basis is). In a carrying frame
    the bowl is the frame's constant + slope + the shared pattern's x^2, exactly; only the other frames see it.
    Exact gauges: the sky's mean against the constants and its x-plane against the slopes. Returns
    (A, b, npix, grid, bowl pattern)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    bowl = ((xx / w) ** 2).ravel()
    sky = 0.7 * (yy / h).ravel() + 0.4 * bowl + 0.05 * rng.normal(size=h * w)
    pattern = 0.1 * np.sin(np.arange(16) * 0.9)
    carrier = rng.uniform(size=frames) < carry
    npix = h * w
    n_shared0 = npix + 2 * frames
    rows, cols, vals, b = [], [], [], []
    r = 0
    for f in range(frames):
        y0, x0 = rng.integers(0, h - 8 + 1), rng.integers(0, w - 16 + 1)
        for yy_ in range(8):
            for xx_ in range(16):
                p = (y0 + yy_) * w + (x0 + xx_)
                rows += [r, r, r]
                cols += [p, npix + 2 * f, npix + 2 * f + 1]
                vals += [1.0, 1.0, (xx_ - 7.5) / 7.5]
                v = sky[p] + 0.2 * np.sin(f) + 0.01 * rng.normal()
                if carrier[f]:
                    rows.append(r); cols.append(n_shared0 + xx_); vals.append(1.0)
                    v += pattern[xx_]
                b.append(v)
                r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, n_shared0 + 16))
    return A, np.asarray(b), npix, (h, w), bowl


def _solve(system, deflate, tmp, iter_lim=600, float32=False):
    A, b, npix, grid = system[:4]
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    return apply_lsqr(A[:, active].tocsr(), b.copy(), ref_shape=grid, atol=0.0, btol=0.0, conlim=0.0, damp=0.0,
                      solver='cgls', n_threads=1, iter_lim=iter_lim, active_mask=active, num_cols_full=A.shape[1],
                      return_record=True, scratch_dir=tmp, sky_names=('sky',), deflate=deflate, use_float32=float32)


def _bowl_amplitude(sky, grid, bowl):
    """The coefficient of the (x/w)^2 pattern in ``sky`` (with a constant, x, y removed)."""
    h, w = grid
    yy, xx = np.mgrid[0:h, 0:w]
    M = np.stack([np.ones(h * w), (xx / w).ravel(), (yy / h).ravel(), bowl], 1)
    return float(np.linalg.lstsq(M, sky, rcond=None)[0][3])


def _common_amplitudes(d_free, d_rule):
    """The kept directions' solved amplitudes of the two records, over the directions both kept."""
    kf, kr = np.array(d_free['kept_directions']), np.array(d_rule['kept_directions'])
    cf = np.full(kf.size, np.nan); cf[kf] = d_free['coefficients']
    cr = np.full(kr.size, np.nan); cr[kr] = d_rule['coefficients']
    both = kf & kr
    return cf[both], cr[both]


def test_unset_the_rules_change_nothing_and_the_record_shows_the_spectrum(tmp_path):
    system = _shared_pattern_system()
    x1, r1 = _solve(system, sc.Deflate(2), str(tmp_path))
    x2, r2 = _solve(system, sc.Deflate(2, gauge_below=None, absorbed_by_shared=None), str(tmp_path))
    assert np.array_equal(x1, x2)
    d = json.loads(r1.deflation)
    assert d['gauge_below'] is None and d['absorbed_by_shared'] is None
    assert len(d['seen']) == d['kept'] + d['gauges'] and d['seen'] == sorted(d['seen'])
    assert d['seen'][-1] == pytest.approx(1.0)
    assert d['seen'][0] == pytest.approx(0.0, abs=1e-5) and d['exact_gauge_directions'].count(True) == d['gauges']
    assert d['frozen_directions'].count(True) == 0 and d['frozen_amplitudes'] == []
    assert len(d['eigenvectors']) == len(d['seen']) and len(d['eigenvectors'][0]) == len(d['pattern_labels'])
    assert len(d['coefficients']) == d['kept']


@pytest.mark.parametrize('float32', [False, True])
def test_a_near_null_sky_mode_is_frozen_by_the_rule_and_solved_without_it(tmp_path, float32):
    """The bowl is seen through the 5 % of frames without the shared pattern: a near-null direction of the pattern
    space. Free, the solve recovers it; under the rule it is frozen at zero, the directions both solves keep get the
    same amplitudes (they are orthogonal to the frozen one in the data's metric), the residual rises by what the
    frozen direction fitted, and the iterate stays put (no drift along the system's gauges)."""
    system = _shared_pattern_system(carry=0.95)
    A, b, npix, grid, bowl = system
    x_free, rec_free = _solve(system, sc.Deflate(2), str(tmp_path), float32=float32)
    x_rule, rec_rule = _solve(system, sc.Deflate(2, gauge_below=0.3), str(tmp_path), float32=float32)
    d_free, d_rule = json.loads(rec_free.deflation), json.loads(rec_rule.deflation)
    a_free, a_rule = _bowl_amplitude(x_free[:npix], grid, bowl), _bowl_amplitude(x_rule[:npix], grid, bowl)
    seen = np.array(d_free['seen_by_direction'])
    print('seen spectrum', np.round(np.sort(seen), 3)[:6], f'bowl amplitude free {a_free:+.3f}, rule {a_rule:+.3f}')
    assert a_free == pytest.approx(0.4, abs=0.05)
    near = (seen > 1e-5) & (seen < 0.3)
    assert near.sum() == 1 and d_rule['frozen_directions'] == [bool(v) for v in near]
    assert d_rule['gauges'] == d_free['gauges'] + 1 and d_rule['kept'] == d_free['kept'] - 1
    assert abs(a_rule) < 0.05, a_rule
    cf, cr = _common_amplitudes(d_free, d_rule)
    assert np.allclose(cf, cr, rtol=1e-6, atol=1e-9 * np.abs(cf).max())
    [frozen_j] = np.flatnonzero(near)
    free_c = np.full(seen.size, np.nan); free_c[d_free['kept_directions']] = d_free['coefficients']
    assert d_rule['frozen_amplitudes'] == pytest.approx([free_c[frozen_j]], rel=1e-6)
    rf, rr = np.linalg.norm(A @ x_free - b), np.linalg.norm(A @ x_rule - b)
    assert rr > rf and rr < 3.0 * rf, (rf, rr)
    assert np.linalg.norm(x_rule) < 2.0 * np.linalg.norm(x_free)
    x_long, _ = _solve(system, sc.Deflate(2, gauge_below=0.3), str(tmp_path), iter_lim=3000, float32=float32)
    assert np.linalg.norm(x_long - x_rule) < 1e-3 * np.linalg.norm(x_rule)                 # no drift once converged


def test_the_shared_absorption_rule_freezes_what_the_shared_offsets_mimic(tmp_path):
    """``Deflate(absorbed_by_shared=)``: the setup measures each direction's visibility with and without the
    frame-shared offsets' absorption (found in the system itself: the columns coupled to far more columns than
    the others, whatever term made them); the bowl, which the shared pattern mimics in the carrying frames, has a
    ratio about 1/sqrt(1 - carry); the others near 1; with the rule the bowl is frozen; the record says so."""
    system = _shared_pattern_system(carry=0.95)
    A, b, npix, grid, bowl = system
    x_free, rec_free = _solve(system, sc.Deflate(2), str(tmp_path))
    x_rule, rec_rule = _solve(system, sc.Deflate(2, absorbed_by_shared=3.0), str(tmp_path))
    d_free, d_rule = json.loads(rec_free.deflation), json.loads(rec_rule.deflation)
    assert d_rule['shared_offsets'] == 16                                  # the pattern's columns, found by their coupling
    ratio = np.array(d_rule['visibility_ratio_without_shared'], dtype=float)
    seen = np.array(d_rule['seen_by_direction'], dtype=float)
    assert ratio.shape == seen.shape and np.allclose(seen, d_free['seen_by_direction'], atol=1e-6)   # the Schur path vs the dense blocks
    exact = seen > 1e-5
    print('visibility ratios without / with the shared offsets:', np.round(np.sort(ratio[exact])[::-1][:6], 2))
    big = exact & (ratio > 3.0)
    assert big.sum() == 1 and ratio[big][0] == pytest.approx(1 / np.sqrt(0.05), rel=0.3)
    assert np.median(ratio[exact]) < 1.3                                   # the rest the per-frame terms absorb as well
    assert d_rule['frozen_directions'] == [bool(v) for v in big] and d_rule['gauges'] == d_free['gauges'] + 1
    a_free, a_rule = _bowl_amplitude(x_free[:npix], grid, bowl), _bowl_amplitude(x_rule[:npix], grid, bowl)
    print(f'bowl amplitude free {a_free:+.3f}, with the shared-absorption rule {a_rule:+.3f}')
    assert a_free == pytest.approx(0.4, abs=0.05) and abs(a_rule) < 0.05
    cf, cr = _common_amplitudes(d_free, d_rule)
    assert np.allclose(cf, cr, rtol=1e-6, atol=1e-9 * np.abs(cf).max())
    x_again, _ = _solve(system, sc.Deflate(2, absorbed_by_shared=None), str(tmp_path))
    assert np.array_equal(x_again, x_free)


def test_the_shared_absorption_rule_without_shared_offsets_freezes_nothing(tmp_path):
    """A model with per-frame terms only: no frame-shared column is found, every ratio is 1, and the rule's solve
    is the free one."""
    system = _shared_pattern_system(carry=0.0)
    x_free, rec_free = _solve(system, sc.Deflate(2), str(tmp_path))
    x_rule, rec_rule = _solve(system, sc.Deflate(2, absorbed_by_shared=2.0), str(tmp_path))
    d = json.loads(rec_rule.deflation)
    assert d['shared_offsets'] == 0 and d['frozen_directions'].count(True) == 0
    ratio = np.array(d['visibility_ratio_without_shared'], dtype=float)
    assert np.allclose(ratio[np.array(d['seen_by_direction']) > 1e-5], 1.0)
    assert np.array_equal(x_rule, x_free)


def test_the_shared_absorption_rule_finds_the_shared_offsets_in_a_small_system(tmp_path, monkeypatch):
    """The frame-shared offsets are looked for whenever the rule asks, not only when the offsets' block is too large
    for a dense solve: the same freeze with the dense-block threshold far above the toy's size."""
    monkeypatch.setattr(deflation, '_DENSE_BLOCK', 1 << 20)
    system = _shared_pattern_system(carry=0.95)
    _, rec = _solve(system, sc.Deflate(2, absorbed_by_shared=3.0), str(tmp_path))
    d = json.loads(rec.deflation)
    assert d['shared_offsets'] == 16 and d['frozen_directions'].count(True) == 1
