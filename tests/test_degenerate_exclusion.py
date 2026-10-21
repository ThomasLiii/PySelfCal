"""Hard constraints on the sky's shape through the deflation: ``Sky(exclude=)`` (shapes given by hand, a constraint
in map space) and ``Deflate(exclude=Degenerate(below=))`` (the directions of the pattern space the data see below a
fraction of the best-seen one, found by the decomposition), on a toy whose bowl a frame-shared pattern mimics; and
the decomposition the deflation's record keeps."""
import json

import numpy as np
import pytest
import scipy.sparse as sp

import selfcal as sc
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


def test_the_record_shows_the_decomposition(tmp_path):
    """The record keeps the pattern space's decomposition: every direction's seen fraction (sorted, the best-seen at
    1, the exact gauges at 0 and counted), its pattern weights, and the kept directions' amplitudes."""
    system = _shared_pattern_system()
    _, r1 = _solve(system, sc.Deflate(2), str(tmp_path))
    d = json.loads(r1.deflation)
    assert len(d['seen']) == d['kept'] + d['gauges'] and d['seen'] == sorted(d['seen'])
    assert d['seen'][-1] == pytest.approx(1.0)
    assert d['seen'][0] == pytest.approx(0.0, abs=1e-5) and d['exact_gauge_directions'].count(True) == d['gauges']
    assert len(d['eigenvectors']) == len(d['seen']) and len(d['eigenvectors'][0]) == len(d['pattern_labels'])
    assert len(d['coefficients']) == d['kept'] and d['excluded_degenerate'] is None


def test_a_sky_shape_excluded_in_map_space_holds_the_bowl_at_zero(tmp_path):
    """A hard constraint on the sky's shape (``Sky(exclude=)``, one group over the sky columns in ``exclusions=``).
    With the bowl's subspace excluded the sky has no bowl at all, the shared pattern takes it, the fit is the
    constrained least squares, and the deflation (its patterns projected into the allowed subspace) changes nothing
    in that. (A matrix handed to apply_lsqr is scaled in place: one per solve.)"""
    from test_exclusions import _constrained_least_squares
    system = _shared_pattern_system(carry=0.95)
    A, b, npix, grid, bowl = system
    n = A.shape[1]
    h, w = grid
    yy, xx = np.mgrid[0:h, 0:w]
    # the bowl with the lower orders a gauge could trade against it (the toy's: the mean against the frame constants,
    # the x-plane against the slopes; its y-plane is seen, and a constraint on a seen shape would cost the fit)
    shapes = np.stack([np.ones(npix), (xx / w).ravel(), bowl], 1)
    P = np.linalg.qr(shapes)[0]
    excl = [(np.arange(npix)[None, :], P)]
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0

    def solve(deflate):
        return apply_lsqr(A[:, active].tocsr(), b.copy(), ref_shape=grid, atol=0.0, btol=0.0, conlim=0.0, damp=0.0,
                          solver='cgls', n_threads=1, iter_lim=600, active_mask=active, num_cols_full=n, return_record=True,
                          scratch_dir=str(tmp_path), sky_names=('sky',), deflate=deflate, exclusions=excl)

    # the reference on the active system (an uncovered pixel would be a free variable of the constraint otherwise)
    act_sky = active[:npix]
    x_ref = _constrained_least_squares(A[:, active].toarray(), b, np.arange(int(act_sky.sum())), np.linalg.qr(P[act_sky])[0])
    fit_ref = A[:, active] @ x_ref
    for deflate in (None, sc.Deflate(2)):
        x, rec = solve(deflate)
        a = _bowl_amplitude(x[:npix], grid, bowl)
        quad = np.polyfit(np.linspace(-1, 1, 16), x[n - 16:], 2)[0]
        print(f'{deflate}: bowl {a:+.2e}, the shared pattern quadratic {quad:+.3f}, |fit - constrained LS fit| / |b| {np.linalg.norm(A @ x - fit_ref) / np.linalg.norm(b):.1e}')
        assert np.abs(P.T @ x[:npix]).max() < 1e-6 * np.linalg.norm(x[:npix])          # exactly no component along the shapes
        assert abs(a) < 0.02 and quad > 0.05
        assert np.linalg.norm(A @ x - fit_ref) < 1e-4 * np.linalg.norm(b)
        if deflate is not None:
            d = json.loads(rec.deflation)
            assert d['gauges'] == 0 and min(d['seen']) > 0.1                            # no near-null direction is left in the space: the degenerate shapes are excluded


def test_the_automatic_form_finds_the_degenerate_shapes_and_excludes_them(tmp_path):
    """``Deflate(exclude=sc.Degenerate(below=))``: nothing is named by hand. The decomposition runs, the directions seen
    below the level (here the mean, the x-plane and the bowl, at 0, 0 and 0.097 of the best-seen one) are selected,
    their sky shapes become a sky constraint, the pattern space is re-based in the same setup, and the deflated solve
    runs on the constrained system: the bowl is gone (the shared pattern takes it, its own constraint still holding
    exactly: the operator applies the column constraint before the deflation's projection, which do not commute), the
    fit is that of the explicit constrained least squares, the record says what was excluded and why."""
    from test_exclusions import _poly_basis, _constrained_least_squares
    system = _shared_pattern_system(carry=0.95)
    A, b, npix, grid, bowl = system
    n = A.shape[1]
    h, w = grid
    yy, xx = np.mgrid[0:h, 0:w]
    excl = [(np.arange(n - 16, n)[None, :], _poly_basis(16, 1))]
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0

    def solve(deflate, nt=1):
        return apply_lsqr(A[:, active].tocsr(), b.copy(), ref_shape=grid, atol=0.0, btol=0.0, conlim=0.0, damp=0.0,
                          solver='cgls', n_threads=nt, iter_lim=400, active_mask=active, num_cols_full=n, return_record=True,
                          scratch_dir=str(tmp_path), sky_names=('sky',), deflate=deflate, exclusions=excl)

    x_free, _ = solve(sc.Deflate(2))
    x, rec = solve(sc.Deflate(2, exclude=sc.Degenerate(below=0.15)))
    d = json.loads(rec.deflation)
    ex = d['excluded_degenerate']
    assert ex['count'] == 3 and ex['below'] == 0.15 and max(ex['seen']) < 0.15
    assert d['gauges'] == 0 and min(d['seen_constrained']) > 0.1                    # nothing near-null is left
    a_free, a = _bowl_amplitude(x_free[:npix], grid, bowl), _bowl_amplitude(x[:npix], grid, bowl)
    print(f'bowl free {a_free:+.3f}, automatic {a:+.3f}; selected: ' + ' | '.join(f"seen {s:.2g}: {c}" for s, c in zip(ex['seen'], ex['composition'])))
    assert a_free == pytest.approx(0.4, abs=0.05) and abs(a) < 0.03
    t = np.linspace(-1, 1, 16)
    assert np.polyfit(t, x[n - 16:], 2)[0] > 0.05                                   # the bowl is the shared pattern's
    assert abs(np.polyfit(t, x[n - 16:], 1)[0]) < 1e-6                                # its constraint still holds exactly
    # the fit equals the explicit constrained least squares with [1, x, bowl] (the span the selection finds)
    P = np.linalg.qr(np.stack([np.ones(npix), (xx / w).ravel(), bowl], 1))[0]
    act_sky = active[:npix]
    x_ref = _constrained_least_squares(A[:, active].toarray(), b, np.arange(int(act_sky.sum())), np.linalg.qr(P[act_sky])[0])
    assert np.linalg.norm(A @ x - A[:, active] @ x_ref) < 1e-3 * np.linalg.norm(b)
    assert np.linalg.norm(A @ x_free - A[:, active] @ x_ref) > 3e-3 * np.linalg.norm(b)
    x4, rec4 = solve(sc.Deflate(2, exclude=sc.Degenerate(below=0.15)), nt=4)
    assert json.loads(rec4.deflation)['excluded_degenerate']['directions'] == ex['directions']
    assert np.linalg.norm(x4 - x) < 1e-12 * np.linalg.norm(x)
