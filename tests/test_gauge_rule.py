"""``Deflate(gauge_below=)``: directions of the pattern space the data see below a fraction of the best-seen one
are gauges (zero sky amplitude), whatever footprint or model made them near-null; unset, nothing changes."""
import numpy as np
import pytest
import scipy.sparse as sp

import selfcal as sc
from selfcal.core.solve import apply_lsqr

from test_deflation import _chunk_system  # noqa: E402


def _partly_shared_system(frames=300, shared=16, carry=0.9, seed=3, chunks=1):
    """The chunk system (one constant + slope per frame: ``chunks=1``, so that a frame alone does not absorb a
    quadratic) plus ``shared`` detector-column offsets carried by a fraction ``carry`` of the frames (one unknown
    per frame-local column, the same in every carrying frame). The truth sky has a 0.4 (x/w)^2 bowl: in a
    carrying frame it is the frame's constant + slope + the shared columns' x^2, exactly; the frames without
    the term see it. Returns (A, b, npix, grid, shared, the sky's bowl pattern)."""
    A, b, npix, grid = _chunk_system(frames=frames, seed=seed, chunks=chunks)
    rng = np.random.default_rng(seed + 1)
    carrier = rng.uniform(size=frames) < carry
    pattern = 0.1 * np.sin(np.arange(shared) * 0.9)
    n = A.shape[1]
    coo = A.tocoo()
    rows, cols, vals = list(coo.row), list(coo.col), list(coo.data)
    b = b.copy()
    r = 0
    for f in range(frames):
        for yy_ in range(8):
            for xx_ in range(16):
                if carrier[f]:
                    rows.append(r); cols.append(n + xx_ % shared); vals.append(1.0)
                    b[r] += pattern[xx_ % shared]
                r += 1
        r += 1
    A2 = sp.csr_matrix((vals, (rows, cols)), shape=(A.shape[0], n + shared))
    h, w = grid
    yy, xx = np.mgrid[0:h, 0:w]
    bowl = ((xx / w) ** 2).ravel()
    return A2, b, npix, grid, shared, bowl


def _solve(system, deflate, tmp, iter_lim=600):
    A, b, npix, grid = system[:4]
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    return apply_lsqr(A[:, active].tocsr(), b.copy(), ref_shape=grid, atol=0.0, btol=0.0, conlim=0.0, damp=0.0,
                      solver='cgls', n_threads=1, iter_lim=iter_lim, active_mask=active, num_cols_full=A.shape[1],
                      return_record=True, scratch_dir=tmp, sky_names=('sky',), deflate=deflate)


def _bowl_amplitude(sky, grid, bowl):
    """The coefficient of the (x/w)^2 pattern in ``sky`` (with a constant, x, y removed)."""
    h, w = grid
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.stack([np.ones(h * w), (xx / w).ravel(), (yy / h).ravel(), bowl], 1)
    return float(np.linalg.lstsq(A, sky, rcond=None)[0][3])


def test_unset_the_rule_changes_nothing_and_the_record_shows_the_spectrum(tmp_path):
    system = _partly_shared_system()
    x1, r1 = _solve(system, sc.Deflate(2), str(tmp_path))
    x2, r2 = _solve(system, sc.Deflate(2, gauge_below=None), str(tmp_path))
    assert np.array_equal(x1, x2)
    d = __import__('json').loads(r1.deflation)
    assert d['gauge_below'] is None and len(d['seen']) == d['kept'] + d['gauges'] and d['seen'] == sorted(d['seen'])
    assert d['seen'][-1] == pytest.approx(1.0)


def test_a_near_null_sky_mode_is_zeroed_by_the_rule_and_solved_without_it(tmp_path):
    system = _partly_shared_system(carry=0.9)
    A, b, npix, grid, shared, bowl = system
    x_free, rec_free = _solve(system, sc.Deflate(2), str(tmp_path))
    x_rule, rec_rule = _solve(system, sc.Deflate(2, gauge_below=0.3), str(tmp_path))
    a_free = _bowl_amplitude(x_free[:npix], grid, bowl)
    a_rule = _bowl_amplitude(x_rule[:npix], grid, bowl)
    import json
    d_free, d_rule = json.loads(rec_free.deflation), json.loads(rec_rule.deflation)
    print(f"bowl amplitude: free {a_free:+.3f}, with the rule {a_rule:+.3f} (truth 0.4); gauges {d_free['gauges']} -> {d_rule['gauges']}; "
          f"seen spectrum {np.round(d_free['seen'][:6], 3)}")
    # solved freely, the near-null bowl direction takes what little the data say (noise and the other slow
    # modes included): a value of either sign, far from the truth; under the rule it is frozen at zero
    assert abs(a_free) > 0.2, a_free
    # the rule leaves that direction out (its seen fraction is below 0.3): the sky's bowl amplitude is ~0
    assert d_rule['gauges'] > d_free['gauges']
    assert abs(a_rule) < 0.08, a_rule
    # the price of the convention: the frozen directions' little visibility is no longer fitted, so the residual
    # cannot be lower; the toy's data see them at a few tens of per cent, so it rises, but not out of proportion
    rf, rr = np.linalg.norm(A @ x_free - b), np.linalg.norm(A @ x_rule - b)
    print(f'residual free {rf:.3f}, with the rule {rr:.3f}')
    assert rr >= rf - 1e-9 and rr < 3.0 * rf, (rf, rr)
