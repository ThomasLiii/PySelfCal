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
