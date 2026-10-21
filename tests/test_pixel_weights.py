"""The observation weight from the data: the built-in variables ``value`` (the stored value) and
``flags`` (the stored flag bits), ``sc.Variance`` (a noise variance: the solve's rows weighted by
``1/sqrt`` of it, the coadd by ``1/`` it) and the shipped shot-noise model ``sc.shot_noise()``,
whose floor is the frame's median. A toy camera with a flag plane; the solves take seconds."""
import os
import sys

import numpy as np
import pytest
from astropy.io import fits

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TESTS)
for _p in (_REPO, _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from test_any_telescope import (  # noqa: E402
    CHUNKS,
    NUMERICS,
    PIX,
    REF,
    TWO_WORKERS,
    H,
    W,
    against_truth,
    agreement,
    pointing_wcs,
    true_sky,
)

import selfcal as sc  # noqa: E402
from selfcal.models.noise import shot_noise_variance  # noqa: E402

FIT = sc.Fit(300, clip=None, tolerance=1e-12)                      # the flag plane masks (use_mask=True)
COADD = sc.Coadd(clip=None)
VALUE_RANGE = (0.5, 20.0)                                          # the toy images lie here (the bias keeps them positive)


def write_exposure_with_flags(path, image, flags, ox, oy):
    """A FITS exposure: the image (SCI, extension 1) and its flag plane (DQ, extension 2)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = pointing_wcs(ox, oy).to_header()
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.asarray(image, np.float32), header=hdr, name='SCI'),
                  fits.ImageHDU(np.asarray(flags, np.int16), name='DQ')]).writeto(path, overwrite=True)


def toy_field(tmp_path, n_exp=12, seed=7):
    """``n_exp`` pointings of the toy camera on the true sky, every image offset per chunk and biased so its
    values stay positive; 1 % of the pixels flagged with bit 1 (value 2). Returns the field (frames
    reprojected), the images, the flag planes and the pointings."""
    rng = np.random.default_rng(seed)
    sky = true_sky() + 5.0
    from selfcal.instruments.grid import rect_grid_chunk_map
    cm = rect_grid_chunk_map((H, W), *CHUNKS)
    images, flags, pointings = [], [], []
    for k in range(n_exp):
        oy, ox = int(rng.integers(0, REF - H)), int(rng.integers(0, REF - W))
        off = rng.normal(0, 0.2, cm.max() + 1)
        img = sky[oy:oy + H, ox:ox + W] + (off - off.mean())[cm] + rng.normal(0, 0.02, (H, W))
        fl = np.zeros((H, W), np.int16)
        fl[rng.random((H, W)) < 0.01] = 2
        write_exposure_with_flags(tmp_path / 'exposures' / f'toy_exp_{k:03d}.fits', img, fl, ox, oy)
        images.append(img)
        flags.append(fl)
        pointings.append((oy, ox))
    field = sc.Field(tmp_path / 'toy_run', sc.Camera((H, W), chunks=CHUNKS, dq_ext=2, tag='Toy'), PIX,
                     compute=TWO_WORKERS)
    field.reproject(tmp_path / 'exposures' / 'toy_exp_*.fits', method='interp', padding=8)
    assert len(field.frames) == n_exp
    return field, images, flags, pointings


def model(weight):
    return sc.Model(sky=[sc.Sky(damping=1e-4)], offsets=[sc.Offsets(mean_zero=True)], weight=weight)


# ---- functions of the data variables (importable by the worker processes) --------------------------
def checked_variance(value, flags, lo, hi):
    """A unit variance that checks what the weight function receives: the stored values (float64,
    finite, in the images' range) and their flag bits (0, or 2 where the frame was flagged)."""
    assert value.dtype == np.float64 and flags.dtype.kind in 'iu', (value.dtype, flags.dtype)
    assert np.all(np.isfinite(value)) and value.min() >= lo and value.max() <= hi, (value.min(), value.max())
    assert set(np.unique(flags)) <= {0, 2}, np.unique(flags)
    return np.ones(value.shape)


def constant_variance(value, v=4.0):
    return np.full(np.shape(value), v)


def constant_weight(value, w=0.5):
    return np.full(np.shape(value), w)


# ---- tests ---------------------------------------------------------------------------------------------
def test_value_and_flags_reach_the_weight_function(tmp_path):
    """``value`` / ``flags`` are the frame's stored values and flag bits, at the observations that
    carry weight, in the solve and in the coadd (the flagged pixels enter only where the coadd
    ignores their bit, so the coadd's weight function sees the bit 2 there)."""
    field, images, flags, _ = toy_field(tmp_path)
    weight = sc.Variance(checked_variance, of=('value', 'flags'), lo=VALUE_RANGE[0], hi=VALUE_RANGE[1])
    coadd = sc.Coadd(clip=None, ignore_flags=(1,))
    result = field.calibrate(sc.Recipe(model(weight), fit=FIT, coadd=coadd, numerics=NUMERICS))
    with result.cal() as cal:
        got, cov = cal.sky(0), cal.sky_coverage(0)
    g, t = against_truth(field, got, (cov >= 3) & np.isfinite(got), true_sky() + 5.0)
    r, _ = agreement(g, t)
    assert r > 0.99, r


def test_variance_is_the_squared_row_weight(tmp_path):
    """``Variance(4)`` weights the rows by ``1/sqrt(4)`` and the coadd by ``1/4``: the same solution
    as ``Function(0.5)``, and a mosaic weight map of ``0.25`` per covering frame."""
    field, _, flags, pointings = toy_field(tmp_path)
    a = field.calibrate(sc.Recipe(model(sc.Variance(constant_variance, of='value', v=4.0)), fit=FIT, coadd=COADD,
                                  numerics=NUMERICS, name='var'))
    b = field.calibrate(sc.Recipe(model(sc.Function(constant_weight, of='value', w=0.5)), fit=FIT, coadd=COADD,
                                  numerics=NUMERICS, name='fn'))
    with a.cal() as ca, b.cal() as cb:
        assert np.allclose(ca.sky(0), cb.sky(0), rtol=1e-5, atol=1e-5, equal_nan=True)
        assert np.allclose(ca.offsets[0], cb.offsets[0], rtol=1e-5, atol=1e-6)
    with a.mosaic() as ma, b.mosaic() as mb:
        wa, wb = np.asarray(ma.weight('MEAN_MAP'), np.float64), np.asarray(mb.weight('MEAN_MAP'), np.float64)
        # a noise-variance weight gives the mosaic its variance map, 1 / MEAN_MAP_WEIGHT; a row weight does not
        assert 'VAR_MAP' in ma.names and 'VAR_MAP' not in mb.names
        var = np.asarray(ma['VAR_MAP'], np.float64)
        assert np.allclose(var[wa > 0], 1.0 / wa[wa > 0], rtol=1e-6) and np.all(var[wa == 0] == 0)
    assert np.allclose(wa, wb, rtol=1e-5, atol=1e-6)
    expected = np.zeros((REF, REF))
    for (oy, ox), fl in zip(pointings, flags):
        expected[oy:oy + H, ox:ox + W] += 0.25 * (fl == 0)
    m, e = against_truth(field, wa, wa > 0, expected)
    ratio = m[e > 0] / e[e > 0]
    assert abs(np.median(ratio) - 1) < 0.01, np.median(ratio)


def test_shot_noise_floors_at_the_frame_median(tmp_path):
    """``sc.shot_noise()``: the coadd weights every observation by ``1/max(|value|, m)``, ``m`` the
    median of the frame's unflagged values, so the mosaic's weight map is that sum over the frames."""
    field, images, flags, pointings = toy_field(tmp_path)
    result = field.calibrate(sc.Recipe(model(sc.shot_noise()), fit=FIT, coadd=COADD, numerics=NUMERICS))
    with result.mosaic() as mos:
        wmap = np.asarray(mos.weight('MEAN_MAP'), np.float64)
    expected = np.zeros((REF, REF))
    for (oy, ox), img, fl in zip(pointings, images, flags):
        img = np.asarray(img, np.float32).astype(np.float64)        # the stored (float32) values
        m = float(np.median(img[fl == 0]))
        expected[oy:oy + H, ox:ox + W] += np.where(fl == 0, 1.0 / np.maximum(np.abs(img), m), 0.0)
    got, e = against_truth(field, wmap, wmap > 0, expected)
    ratio = got[e > 0] / e[e > 0]
    print(f"shot-noise weights / expected: median {np.median(ratio):.5f}, 5-95 % "
          f"{np.percentile(ratio, 5):.5f}..{np.percentile(ratio, 95):.5f}")
    assert abs(np.median(ratio) - 1) < 0.01, np.median(ratio)
    with result.cal() as cal:
        got_sky, cov = cal.sky(0), cal.sky_coverage(0)
    g, t = against_truth(field, got_sky, (cov >= 3) & np.isfinite(got_sky), true_sky() + 5.0)
    assert agreement(g, t)[0] > 0.99


def test_shot_noise_variance_function():
    value, flags = np.array([0.1, -0.5, 2.0, np.nan, 0.3]), np.array([0, 0, 21, 0, 0])
    # the floor: the median of the finite values with no flag set (0.1, -0.5, 0.3 -> 0.1)
    assert np.allclose(shot_noise_variance(value, flags), [0.1, 0.5, 2.0, np.nan, 0.3], equal_nan=True)
    assert np.allclose(shot_noise_variance(value, flags, floor=1.0), [1.0, 1.0, 2.0, np.nan, 1.0], equal_nan=True)
    assert np.all(np.isnan(shot_noise_variance(value, np.full(5, 4))))        # nothing unflagged: no information
    with pytest.raises(ValueError):
        shot_noise_variance(value, flags, floor='mean')


def test_variance_is_encoded_as_a_variance():
    """A Variance and a Function of the same function are different settings (different fingerprints)."""
    v = sc.shot_noise()
    f = sc.Function(shot_noise_variance, of=('value', 'flags'), floor='median')
    assert isinstance(v, sc.Variance) and isinstance(v, sc.Function)
    assert v.lower()['kind'] == 'variance' and 'kind' not in f.lower()
    assert sc.Model(weight=v).to_dict() != sc.Model(weight=f).to_dict()
    assert repr(v).startswith('Variance(')


def test_a_frame_without_weighted_pixels_gets_zero_coadd_weight():
    """A frame with no observation carrying weight (in a large field, a frame whose job window is all
    masked) crashed the coadd's weight hook: the bilinear matrix of zero pixels came out 0x0 instead of
    0 x (detector pixels), so a detector-map variable could not be read. It now gets weight 0."""
    from selfcal.core.subframe import FrameContext
    from selfcal.geometry.map_helper import make_linear_interp_matrix
    from selfcal.models.variables import VariableSet
    from selfcal.pipeline.model_eval import ObservationWeight, frame_observations
    H, W = 6, 5
    m = make_linear_interp_matrix(np.full((2, 4), np.nan), input_shape=(H, W))
    assert m.shape == (4, H * W) and m.nnz == 0 and m.dtype == np.float32
    assert make_linear_interp_matrix(np.zeros((2, 0)), input_shape=(H, W)).shape == (0, H * W)
    yy, xx = np.mgrid[0:3, 0:3].astype(np.float64)
    variables = VariableSet(detector={'gain': np.tile(np.arange(W, dtype=np.float64), (H, 1))})

    def ctx(weight):
        return FrameContext(stage='post', file='f.h5', exp_idx=0, det_idx=0, ref_coords=(0, 3, 0, 3),
                            sub_data=np.ones((3, 3), np.float32), sub_weight=weight, sub_mapping=np.stack([xx, yy]))

    none = (np.array([], dtype=int), np.array([], dtype=int))
    obs, interp = frame_observations(ctx(np.zeros((3, 3), np.float32)), 0, variables, pixels=none, det_shape=(H, W))
    assert obs.n == 0 and interp.shape == (0, H * W)

    class OnePlusGain:                # a row weight read off a detector map
        variables = ('gain',)

        def evaluate(self, obs):
            return 1.0 + obs['gain']

    hook = ObservationWeight(OnePlusGain(), variables, det_shape=(H, W))
    _, w0 = hook(ctx(np.zeros((3, 3), np.float32)))
    assert np.all(w0 == 0)
    _, w1 = hook(ctx(np.ones((3, 3), np.float32)))
    assert np.allclose(w1, (1.0 + xx) ** 2)


def _x_and_variance(det_x, variance):
    """Two basis functions of a built-in variable and a layer."""
    return [np.asarray(det_x, dtype=np.float64) - 3.0, np.sqrt(np.asarray(variance, dtype=np.float64))]


def test_offset_hooks_evaluate_the_weighted_observations_only(tmp_path, monkeypatch):
    """The mosaic's and the N-pass's offset hooks evaluate the offset terms at the observations that carry
    weight: there the data come out bit for bit as from the evaluation at every pixel with a detector
    position (each observation's value depends on itself alone), the other pixels keep their values, and
    the hooks composed on one frame read its layers once."""
    from selfcal.core.subframe import FrameContext
    from selfcal.geometry.map_helper import compute_chunk_contrib
    from selfcal.io import reproj
    from selfcal.io.frames import standard_frame_path, write_frame
    from selfcal.models.offset_model import Basis
    from selfcal.models.sky_model import Coefficient
    from selfcal.models.variables import VariableSet
    from selfcal.npass.sky import OffsetModelSubtractor
    from selfcal.pipeline import model_eval
    from selfcal.pipeline.model_eval import (
        BasisOffsetSubtractor,
        ComposedHook,
        ObservationWeight,
        frame_observations,
    )

    rng = np.random.default_rng(3)
    sh, sw, dh, dw = 9, 11, 6, 8                                             # the subframe, the detector
    x, y = np.meshgrid(np.linspace(-0.6, dw - 0.4, sw), np.linspace(-0.7, dh - 0.3, sh))
    mapping = np.stack([x, y]).astype(np.float32)
    mapping[:, 0, :3] = np.nan                                               # pixels without a detector position
    data = rng.standard_normal((sh, sw)).astype(np.float32)
    weight = ((rng.random((sh, sw)) < 0.3) * 2.0).astype(np.float32)
    variance = (rng.random((sh, sw)) + 0.5).astype(np.float32)
    path = write_frame(standard_frame_path(str(tmp_path), 0, 0), data, [0, sh, 0, sw], mapping,
                       layers={'variance': variance})
    name = os.path.basename(path)
    cm = np.arange(dh * dw).reshape(dh, dw) % 3                              # three chunks
    basis = Basis(Coefficient(('det_x', 'variance'), _x_and_variance), 2)
    a = rng.standard_normal((1, 3 * 2))                                      # one frame: 3 chunks x 2 coefficients
    variables = VariableSet(layers=('variance',))

    def ctx():
        return FrameContext(stage='post', file=path, exp_idx=0, det_idx=0, ref_coords=[0, sh, 0, sw],
                            sub_data=data.copy(), sub_weight=weight.copy(), sub_mapping=mapping)

    def at_every_mapped_pixel(scalar):
        # the hooks' evaluation before 2026-10-21: every pixel with a detector position
        obs, interp = frame_observations(ctx(), 0, variables, det_shape=(dh, dw))
        phi = np.asarray(basis.evaluate(obs), dtype=np.float64)
        off = np.asarray(compute_chunk_contrib(cm, interp).T @ a[0].reshape(3, 2))
        out = data.copy()
        out[obs.pixels] -= np.nan_to_num((off * phi).sum(axis=1)).astype(np.float32)
        if scalar is not None:
            out[obs.pixels] -= np.float32(scalar)
        return out

    weighted = (weight > 0) & np.isfinite(mapping[0]) & np.isfinite(mapping[1])
    assert 0 < weighted.sum() < np.isfinite(mapping[0]).sum()
    mosaic_hook = object.__new__(BasisOffsetSubtractor)                     # the solved offsets set directly
    mosaic_hook.terms, mosaic_hook.variables, mosaic_hook.oversample_factor = [(0, basis, cm)], variables, 1
    mosaic_hook.rows, mosaic_hook.index, mosaic_hook.var_index = {0: a}, {name: 0}, {name: 0}
    npass_hook = object.__new__(OffsetModelSubtractor)
    npass_hook.terms, npass_hook.variables, npass_hook.oversample_factor = [(0, basis, cm)], variables, 1
    npass_hook.rows, npass_hook.index, npass_hook.var_index = {0: a}, {name: 0}, {name: 0}
    npass_hook.scalar = np.array([0.25])
    for hook, scalar in ((mosaic_hook, None), (npass_hook, 0.25)):
        ref = at_every_mapped_pixel(scalar)
        out = hook(ctx())
        assert out[weighted].tobytes() == ref[weighted].tobytes()
        assert out[~weighted].tobytes() == data[~weighted].tobytes()

    # one read of the layers for the hooks of one frame
    calls = []
    real = reproj.load_reproj_file

    def counting(file, fields):
        calls.append(tuple(fields))
        return real(file, fields)

    weight_hook = ObservationWeight(Coefficient('variance', np.reciprocal), variables, det_shape=(dh, dw))
    monkeypatch.setattr(model_eval, 'load_reproj_file', counting)
    ComposedHook([mosaic_hook, weight_hook])(ctx())
    assert calls == [('layers/variance',)]
