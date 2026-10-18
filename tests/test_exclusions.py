"""Hard column constraints by reparametrisation (:mod:`selfcal.core.projector`, ``apply_lsqr(exclusions=)``,
``sc.Offsets(exclude=sc.Poly(k))``): the solution carries no polynomial part along the excluded
directions, exactly, with and without deflation, and equals the constrained least-squares solution."""
import numpy as np
import pytest
import scipy.sparse as sp

import selfcal as sc
from selfcal.core.projector import ColumnProjector
from selfcal.core.solve import apply_lsqr

from test_deflation import _shared_system  # noqa: E402  (the toy: per-frame chunk offsets + columns shared by every frame)


def _poly_basis(n, degree):
    t = np.linspace(-1, 1, n)
    return np.linalg.qr(np.vander(t, degree + 1, increasing=True))[0]


def _solve(system, exclusions, deflate=None, iter_lim=600, solver='cgls', tmp=None):
    A, b, npix, grid, shared = system
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    return apply_lsqr(A[:, active].tocsr(), b.copy(), ref_shape=grid, atol=0.0, btol=0.0, conlim=0.0, damp=0.0,
                      solver=solver, n_threads=1, iter_lim=iter_lim, active_mask=active, num_cols_full=A.shape[1],
                      return_record=True, scratch_dir=tmp, sky_names=('sky',), deflate=deflate, exclusions=exclusions)


def _constrained_least_squares(A, b, cols, P):
    """The least-squares solution with ``P^T x[cols] = 0``, by the explicit null-space basis (dense)."""
    n = A.shape[1]
    N = np.eye(n)
    Pc = P @ P.T
    N[np.ix_(cols, cols)] -= Pc                 # the projector on the constrained block
    AN = (A @ N)
    AtA = AN.T @ AN; Atb = AN.T @ np.asarray(b)
    y = np.linalg.pinv(AtA, rcond=1e-12) @ Atb
    return N @ y


def test_projector_removes_the_directions_in_physical_units_and_is_a_projector():
    rng = np.random.default_rng(0)
    n, P = 12, _poly_basis(12, 2)
    cols = np.array([np.arange(20, 32), np.arange(40, 52)])
    M = rng.uniform(0.5, 2, 60)
    pr = ColumnProjector.build([(cols, P)], None, M, 60)
    y = rng.normal(size=60); z = pr.apply(y.copy())
    for row in cols:
        assert np.abs(P.T @ (M[row] * z[row])).max() < 1e-12          # no polynomial part in x = M y
    assert np.abs(pr.apply(z.copy()) - z).max() < 1e-13                # idempotent
    u, v = rng.normal(size=60), rng.normal(size=60)
    assert abs(pr.apply(u.copy()) @ v - u @ pr.apply(v.copy())) < 1e-12   # symmetric
    act = np.ones(60, bool); act[45:52] = False                        # compacted columns: the basis follows the active ones
    pr2 = ColumnProjector.build([(cols, P)], act, M[act], int(act.sum()))
    cm = np.full(60, -1); cm[act] = np.arange(act.sum())
    yc = pr2.apply(rng.normal(size=int(act.sum())))
    a = cols[1][act[cols[1]]]
    assert np.abs(P[act[cols[1]]].T @ (M[act][cm[a]] * yc[cm[a]])).max() < 1e-12


@pytest.mark.parametrize('deflate', [None, sc.Deflate(2)])
def test_the_solve_honours_the_constraint_exactly_and_is_the_constrained_least_squares(tmp_path, deflate):
    system = _shared_system(frames=40, shared=16, seed=5)
    A, b, npix, grid, shared = system
    n = A.shape[1]
    cols = np.arange(n - shared, n)
    P = _poly_basis(shared, 1)                                          # no constant, no slope across the shared columns
    x, rec = _solve(system, [(cols[None, :], P)], deflate=deflate, tmp=str(tmp_path))
    moments = P.T @ x[cols]
    assert np.abs(moments).max() < 1e-9 * max(np.abs(x[cols]).max(), 1e-300), moments
    ref = _constrained_least_squares(A.toarray(), b, cols, P)
    # the two agree on the fit and, up to the system's gauges, on the unknowns: compare the residuals
    ra, rb = np.linalg.norm(A @ x - b), np.linalg.norm(A @ ref - b)
    assert abs(ra - rb) < 1e-6 * rb, (ra, rb)
    # a least-squares fit's prediction A x is unique: the two solutions predict the same data (their unknowns may
    # differ along the system's gauges, e.g. the shared columns' chunk-wise constants against the per-frame ones)
    assert np.linalg.norm(A @ (x - ref)) < 1e-6 * np.linalg.norm(A @ ref)


def test_without_exclusions_nothing_changes(tmp_path):
    system = _shared_system(frames=30, shared=8, seed=2)
    x1, _ = _solve(system, None, tmp=str(tmp_path), iter_lim=300)
    x2, _ = _solve(system, [], tmp=str(tmp_path), iter_lim=300)
    assert np.array_equal(x1, x2)


def test_the_option_lowers_into_the_term_and_is_validated():
    t = sc.Offsets('columns', on='detector_column', per='all', exclude=sc.Poly(2))
    low = t.lower()
    assert low['exclude_degree'] == 2 and low['exclude_axis'] is None
    with pytest.raises(Exception):
        sc.Offsets(per='all', exclude=sc.Poly(2, weight=1.0))
    with pytest.raises(Exception):
        sc.Offsets(per='all', exclude=sc.Poly(1), polynomial=sc.Poly(1, window=range(0, 3)))


def test_a_shared_pattern_without_its_linear_part_end_to_end(tmp_path):
    """The camera toy of :mod:`test_any_telescope` (a detector-fixed pattern shared by every frame, times a
    function of the exposure's temperature, plus a free gradient per frame): with ``exclude=Poly(1, along=<the
    chunk columns>)`` the solved pattern has no constant and no slope along the chunk columns in any chunk row,
    exactly, while the pattern's remaining structure is still recovered."""
    from test_any_telescope import (CHUNKS, COADD, FIT, H, NUMERICS, PIX, REF, TWO_WORKERS, W, agreement, centred, plane,
                                    true_sky, write_exposure)
    from selfcal.instruments.grid import rect_grid_chunk_map
    rng = np.random.default_rng(11)
    sky = true_sky()
    cm = rect_grid_chunk_map((H, W), *CHUNKS)
    ny, nx = CHUNKS
    pattern = rng.normal(0, 0.05, ny * nx)
    pattern += np.tile(np.linspace(-0.08, 0.08, nx), ny)             # plus a slope along the chunk columns
    n_exp = 48
    temp = 80.0 + rng.uniform(-4, 4, n_exp)
    grad = rng.normal(0, 0.3, (n_exp, 2))
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    gx, gy = plane(x, y)
    for k in range(n_exp):
        oy, ox = rng.integers(0, REF - H), rng.integers(0, REF - W)
        img = (sky[oy:oy + H, ox:ox + W] + centred(temp[k]) * pattern[cm]
               + grad[k, 0] * gx + grad[k, 1] * gy + rng.normal(0, 1.0) + rng.normal(0, 0.01, (H, W)))
        write_exposure(tmp_path / 'exposures' / f'toy_exp_{k:03d}.fits', img, ox, oy, {'TEMP': temp[k]})
    cam = sc.Camera((H, W), chunks=CHUNKS, tag='Toy')
    axes = cam.geometry(1).chunk_map.axes
    along = axes.names[1]                                                 # the chunk columns
    field = sc.Field(tmp_path / 'toy_run', cam, PIX, compute=TWO_WORKERS)
    field.reproject(tmp_path / 'exposures' / 'toy_exp_*.fits', method='interp', padding=8)
    model = sc.Model(sky=[sc.Sky(damping=1e-4)],
                     offsets=[sc.Offsets('thermal', per='all', times=centred, exclude=sc.Poly(1, along=along)),
                              sc.Offsets('gradient', on='detector', basis=plane, n=2)],
                     variables={'temperature': sc.Header('TEMP')})
    result = field.calibrate(sc.Recipe(model, fit=FIT, coadd=COADD, numerics=NUMERICS))
    with result.cal() as cal:
        got = np.asarray(cal.offsets[0][0], dtype=np.float64).reshape(ny, nx)
    t = np.linspace(-1, 1, nx)
    P = np.linalg.qr(np.vander(t, 2, increasing=True))[0]
    moments = got @ P                                                      # (ny, 2): constant and slope per chunk row
    assert np.abs(moments).max() < 1e-6 * max(np.abs(got).max(), 1e-300), moments
    truth = pattern.reshape(ny, nx)
    truth_rest = truth - (truth @ P) @ P.T                                 # what the term may hold
    r, slope = agreement(got.ravel(), truth_rest.ravel(), demean=False)
    print(f"pattern without its linear part: r = {r:.4f}, slope {slope:.3f}")
    assert r > 0.99 and abs(slope - 1) < 0.05, (r, slope)
