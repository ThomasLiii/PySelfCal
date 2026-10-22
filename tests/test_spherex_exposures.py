"""The SPHEREx L2b reader, the variance layer and the dilated mask.

``read_l2b`` stores the L2b ``VARIANCE`` with every frame as the layer ``variance``, finite
wherever the frame has a detector position (the bad pixels are filled before the interpolation),
and ``add_layers`` / ``add_reader_layers`` give frames made without the layer the same one. It
also sets bit 30 (``DILATED_DEAD_BIT``) on the one-pixel dilation of the SUR_ERROR and NONFUNC bits,
keeping those bits as read, so that the mask covers every output pixel the exact reprojection
spreads a dead pixel's value into; ``refresh_mask`` / ``refresh_reader_mask`` give frames made
before that the same mask. The toy L2b files here: a small detector with a TAN WCS
rotated against the reference grid, and the ``IMAGE`` / ``FLAGS`` / ``VARIANCE`` extensions, with a
dead pixel of 600 MJy/sr flagged like the real ones.
"""
import os
import sys

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, map_coordinates

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal.instruments.spherex.exposures import (  # noqa: E402
    DILATED_BITS,
    DILATED_DEAD_BIT,
    l2b_flags,
    l2b_variance,
    read_l2b,
)
from selfcal.io.frames import dilate_bits, fill_missing  # noqa: E402
from selfcal.io.reproj import load_reproj_file  # noqa: E402
from selfcal.io.reprojection import (  # noqa: E402
    add_layers,
    add_reader_layers,
    refresh_mask,
    refresh_reader_mask,
)

H, W = 40, 48                   # the toy detector (rows, cols)
PIX = 20.0                      # arcsec per pixel, detector and reference grid alike
BAD = [(5, 7), (20, 30), (21, 30), (21, 31), (35, 2)]     # bad pixels: flagged, NaN data, NaN variance
NO_VARIANCE = [(10, 40), (30, 12)]                       # flagged, data kept, NaN variance
ZERO_VARIANCE = (25, 25)                                 # flagged, a zero variance
DEAD = (12, 30)                                          # a dead pixel: 600 MJy/sr, TRANSIENT | SUR_ERROR | NONFUNC, NaN variance
DEAD_BITS = (1 << 0) | (1 << 2) | (1 << 6)
ROTATION = 7.0                                           # degrees between the detectors and the reference grid

sc.set_progress(False)


def toy_wcs(ox, oy):
    """The WCS of a detector whose pixel (0, 0) sees the pixel (oy, ox) of a common sky grid, its
    rows rotated by ``ROTATION`` degrees against that grid."""
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [W / 2 + 0.5 - ox, H / 2 + 0.5 - oy]
    w.wcs.crval = [180.0, 30.0]
    s, c, sn = PIX / 3600.0, np.cos(np.radians(ROTATION)), np.sin(np.radians(ROTATION))
    w.wcs.cd = [[-s * c, s * sn], [s * sn, s * c]]
    return w


def write_l2b(path, rng, ox, oy, variance=True):
    """A toy L2b file: ``IMAGE`` (a smooth sky plus noise, NaN at the bad pixels), ``FLAGS`` (bit 0
    on the bad pixels and the zero-variance one, bit 2 on the pixels without a variance, bit 21 on
    a source) and ``VARIANCE`` (sigma squared rising across the detector; NaN at the bad pixels
    and at ``NO_VARIANCE``, zero at ``ZERO_VARIANCE``). Returns the three planes as written."""
    yy, xx = np.mgrid[0:H, 0:W]
    var = (0.01 * (1.0 + xx / W)) ** 2
    img = 1.0 + 0.1 * np.sin(xx / 7.0) + rng.normal(0.0, 1.0, (H, W)) * np.sqrt(var)
    flags = np.zeros((H, W), dtype=np.int32)
    for y, x in BAD:
        flags[y, x] |= 1
        img[y, x] = np.nan
        var[y, x] = np.nan
    for y, x in NO_VARIANCE:
        flags[y, x] |= 1 << 2
        var[y, x] = np.nan
    flags[15:18, 20:23] |= 1 << 21
    flags[ZERO_VARIANCE] |= 1
    var[ZERO_VARIANCE] = 0.0
    flags[DEAD] |= DEAD_BITS
    img[DEAD] = 600.0
    var[DEAD] = np.nan
    hdr = toy_wcs(ox, oy).to_header()
    hdr['DETECTOR'], hdr['FINAST'] = 3, 0
    hdus = [fits.PrimaryHDU(), fits.ImageHDU(img.astype(np.float32), header=hdr, name='IMAGE'),
            fits.ImageHDU(flags, header=hdr, name='FLAGS')]
    if variance:
        hdus.append(fits.ImageHDU(var.astype(np.float32), header=hdr, name='VARIANCE'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return img.astype(np.float32), flags, var.astype(np.float32)


def _exposures(tmp_path, rng, n=6):
    for k in range(n):
        write_l2b(str(tmp_path / 'exp' / f'l2b_{k:03d}.fits'), rng, int(rng.integers(0, 60)), int(rng.integers(0, 60)))


def read_l2b_undilated(path, sci_ext, dq_ext=None, header_only=False):
    """The L2b reader without the dilation: the frames of before 2026-10-15."""
    return read_l2b(path, sci_ext, dq_ext, header_only=header_only, dilate=())


def _reproject(tmp_path, name, reader):
    camera = sc.Camera((H, W), chunks=(2, 2), sci_ext=1, dq_ext=2, reader=reader)
    field = sc.Field(str(tmp_path / name), camera, PIX, compute=sc.Compute(workers=2))
    field.reproject(str(tmp_path / 'exp' / 'l2b_*.fits'), method='exact', padding=8)
    return field


def test_fill_missing_fills_in_rounds():
    plane = np.arange(49, dtype=np.float64).reshape(7, 7)
    missing = np.zeros((7, 7), dtype=bool)
    missing[3, 3] = True                                      # isolated: the mean of its eight neighbours
    out = fill_missing(plane, missing)
    assert out[3, 3] == pytest.approx((plane[2:5, 2:5].sum() - plane[3, 3]) / 8)
    assert np.array_equal(out[~missing], plane[~missing]) and plane[3, 3] == 24.0      # the input untouched
    missing[2:5, 2:5] = True                                  # a 3x3 blob: the ring first, then its centre
    out = fill_missing(plane, missing)
    assert np.isfinite(out).all()
    ring = out[2:5, 2:5][[0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2]]
    assert out[3, 3] == pytest.approx(ring.mean())
    assert np.isnan(fill_missing(plane, np.ones((7, 7), dtype=bool))).all()            # nothing known: NaN
    with pytest.raises(ValueError, match='mask'):
        fill_missing(plane, np.ones((3, 3), dtype=bool))


def test_read_l2b_reads_the_variance_as_a_finite_layer(tmp_path):
    rng = np.random.default_rng(1)
    p = str(tmp_path / 'exp' / 'l2b_000.fits')
    img, flags, var = write_l2b(p, rng, 3, 5)
    exp = read_l2b(p, 1, 2)
    assert np.array_equal(exp.data, img, equal_nan=True)
    dead = 1 << DILATED_DEAD_BIT
    assert np.array_equal(exp.mask, dilate_bits(flags, DILATED_BITS, into=DILATED_DEAD_BIT))
    assert np.array_equal(exp.mask & ~dead, flags)                     # the bits as the file has them
    grown = binary_dilation((flags & sum(1 << b for b in DILATED_BITS)) != 0, structure=np.ones((3, 3), dtype=bool))
    assert np.array_equal((exp.mask & dead) != 0, grown) and grown[DEAD] and grown.sum() > 9
    assert np.array_equal(l2b_flags(p), exp.mask)
    assert np.array_equal(read_l2b(p, 1, 2, dilate=()).mask, flags)
    assert np.array_equal(read_l2b(p, 1, 2, into=None).mask, dilate_bits(flags, DILATED_BITS))   # 2026-10-15 to 10-22
    layer = exp.layers['variance']
    assert layer.dtype == np.float64 and np.isfinite(layer).all() and (layer > 0).all()
    usable = np.isfinite(var) & (var > 0)
    assert np.array_equal(layer[usable], var[usable].astype(np.float64))
    for y, x in [BAD[0], BAD[4], *NO_VARIANCE, ZERO_VARIANCE, DEAD]:  # isolated: the mean of the eight neighbours
        nb = var[y - 1:y + 2, x - 1:x + 2].astype(np.float64)
        known = np.isfinite(nb) & (nb > 0)
        known[1, 1] = False
        assert layer[y, x] == pytest.approx(nb[known].mean())
    for y, x in BAD[1:4]:                                             # the blob: within its surroundings
        nb = var[y - 2:y + 3, x - 2:x + 3]
        ok = np.isfinite(nb) & (nb > 0)
        assert nb[ok].min() <= layer[y, x] <= nb[ok].max()
    assert np.array_equal(l2b_variance(p), layer)
    only = read_l2b(p, 1, 2, header_only=True)
    assert only.data is None and only.frame_shape == (H, W)
    q = str(tmp_path / 'exp' / 'novar.fits')
    write_l2b(q, rng, 0, 0, variance=False)
    assert read_l2b(q, 1, 2).layers == {} and l2b_variance(q) is None
    assert sc.SPHEREx(3).layout().reader is read_l2b


def test_the_layer_is_the_variance_at_every_detector_position(tmp_path):
    rng = np.random.default_rng(2)
    _exposures(tmp_path, rng)
    field = _reproject(tmp_path, 'with_layer', read_l2b)
    assert len(field.frames) == 6
    for f in field.frames:
        got = load_reproj_file(f, ['sub_data', 'sub_mapping', 'layers/variance', 'file_path'])
        layer, mapping = got['layers/variance'], got['sub_mapping']
        assert layer is not None and layer.dtype == np.float32 and layer.shape == got['sub_data'].shape
        has_position = np.isfinite(mapping[0]) & np.isfinite(mapping[1])
        assert has_position.sum() > 0.5 * H * W
        assert np.array_equal(np.isfinite(layer), has_position)       # finite exactly where the frame has a position
        assert (layer[has_position] > 0).all()
        filled = l2b_variance(got['file_path'])                        # the bilinear sample of the filled plane
        x, y = (mapping[i][has_position].astype(np.float64) for i in (0, 1))
        inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
        expect = map_coordinates(filled, [y[inside], x[inside]], order=1, mode='nearest')
        assert np.allclose(layer[has_position][inside], expect, rtol=1e-4, atol=0.0)


def test_a_pass_over_existing_frames_writes_what_the_reprojection_writes(tmp_path):
    rng = np.random.default_rng(3)
    _exposures(tmp_path, rng)
    with_layer = _reproject(tmp_path, 'with_layer', read_l2b)
    without = _reproject(tmp_path, 'without', None)           # the default FITS reader: IMAGE and FLAGS, no layer
    keys = ['sub_data', 'sub_foot', 'sub_bitmask', 'sub_mapping', 'ref_coords', 'det_header', 'sub_header', 'file_path']
    for fa, fb in zip(with_layer.frames, without.frames):
        assert os.path.basename(fa) == os.path.basename(fb)
        before = load_reproj_file(fb, keys + ['layers/variance'])
        assert before['layers/variance'] is None
        assert add_reader_layers(fb, read_l2b, 1, 2) == ['variance']
        after = load_reproj_file(fb, keys + ['layers/variance'])
        for k in keys:                                                      # everything else unchanged
            if isinstance(before[k], np.ndarray):
                assert np.array_equal(before[k], after[k], equal_nan=True), k
            else:
                assert before[k] == after[k], k
        fresh = load_reproj_file(fa, ['layers/variance'])['layers/variance']
        assert np.array_equal(fresh, after['layers/variance'], equal_nan=True)
        raw = open(fb, 'rb').read()
        assert add_reader_layers(fb, read_l2b, 1, 2) == [] and open(fb, 'rb').read() == raw    # present: kept
        assert add_layers(fb, {'variance': np.full((H, W), 2.0)}, replace=True) == ['variance']
        new = load_reproj_file(fb, ['layers/variance'])['layers/variance']
        assert np.array_equal(np.isfinite(new), np.isfinite(fresh)) and (new[np.isfinite(new)] == 2.0).all()
        with pytest.raises(ValueError, match='detector'):
            add_layers(fb, {'variance': np.ones((H + 1, W))}, replace=True)
        with pytest.raises(ValueError, match='no layer'):
            add_reader_layers(fb, read_l2b, 1, 2, names=['nope'])


def test_dilate_bits_grows_only_the_given_bits():
    m = np.zeros((9, 9), dtype=np.int32)
    m[4, 4] = (1 << 2) | (1 << 0)
    m[1, 7] = 1 << 2
    out = dilate_bits(m, (2,))
    assert out.dtype == m.dtype and m[4, 4] == 5                    # the input untouched
    bit2 = (out >> 2) & 1
    expect = np.zeros((9, 9), dtype=np.int32)
    expect[3:6, 3:6] = 1
    expect[0:3, 6:9] = 1
    assert np.array_equal(bit2, expect)
    assert np.array_equal((out >> 0) & 1, (m >> 0) & 1)              # bit 0 not dilated
    assert np.array_equal(dilate_bits(m, ()), m)


def test_dilate_bits_into_another_bit_keeps_the_bits():
    m = np.zeros((9, 9), dtype=np.int32)
    m[4, 4] = (1 << 2) | (1 << 0)
    m[1, 7] = 1 << 6
    out = dilate_bits(m, (2, 6), into=30)
    assert out.dtype == m.dtype and np.array_equal(out & ~(1 << 30), m)          # every bit as it was
    expect = np.zeros((9, 9), dtype=bool)
    expect[3:6, 3:6] = True
    expect[0:3, 6:9] = True
    assert np.array_equal((out & (1 << 30)) != 0, expect)                        # the union's dilation, in bit 30
    assert np.array_equal(dilate_bits(m, (), into=30), m)
    for bad in (2, 31, 32):                                                       # a dilated bit; past a positive int32
        with pytest.raises(ValueError):
            dilate_bits(m, (2, 6), into=bad)
    assert (dilate_bits(m.astype(np.uint32), (2,), into=31) >> 31).sum() == 9    # unsigned: bit 31 fits


def test_the_dilated_mask_covers_the_dead_pixel_s_spread(tmp_path):
    """The exact reprojection spreads the dead pixel (600 MJy/sr) into the output pixels its area touches,
    beyond the bilinear footprint of its mask: with the frames rotated against the grid, the undilated
    mask leaves some of them unflagged; the reader's bit 30 (the one-pixel dilation of SUR_ERROR and
    NONFUNC) flags every one of them, every other bit as the undilated reprojection has it."""
    rng = np.random.default_rng(5)
    _exposures(tmp_path, rng, n=8)
    old = _reproject(tmp_path, 'undilated', read_l2b_undilated)
    new = _reproject(tmp_path, 'dilated', read_l2b)
    leaked_unflagged_old, leaked, flagged_old, flagged_new = 0, 0, 0, 0
    for fa, fb in zip(old.frames, new.frames):
        a = load_reproj_file(fa, ['sub_data', 'sub_bitmask'])
        b = load_reproj_file(fb, ['sub_data', 'sub_bitmask'])
        assert np.array_equal(a['sub_data'], b['sub_data'], equal_nan=True)          # the data do not change
        leak = np.isfinite(a['sub_data']) & (np.abs(a['sub_data'] - 1.0) > 2.0)      # the sky is 1 +- 0.03
        leaked += int(leak.sum())
        dil = sum(1 << b for b in DILATED_BITS) | (1 << DILATED_DEAD_BIT)
        leaked_unflagged_old += int((leak & ((a['sub_bitmask'] & dil) == 0)).sum())
        assert not (leak & ((b['sub_bitmask'] & dil) == 0)).any()                     # every leak flagged now
        flagged_old += int(((a['sub_bitmask'] & dil) != 0).sum())
        flagged_new += int(((b['sub_bitmask'] & dil) != 0).sum())
        dead = 1 << DILATED_DEAD_BIT
        assert np.array_equal(b['sub_bitmask'] & ~dead, a['sub_bitmask'])            # only the new bit differs
    assert leaked > 8 and leaked_unflagged_old > 0                                    # the flaw the dilation fixes
    assert flagged_new > flagged_old


def test_refresh_mask_writes_what_the_reprojection_writes(tmp_path):
    rng = np.random.default_rng(6)
    _exposures(tmp_path, rng, n=5)
    old = _reproject(tmp_path, 'undilated', read_l2b_undilated)
    new = _reproject(tmp_path, 'dilated', read_l2b)
    keys = ['sub_data', 'sub_foot', 'sub_mapping', 'ref_coords', 'det_header', 'sub_header', 'file_path', 'layers/variance']
    for fa, fb in zip(old.frames, new.frames):
        before = load_reproj_file(fa, keys + ['sub_bitmask'])
        changed = refresh_reader_mask(fa, read_l2b, 1, 2, bits=(DILATED_DEAD_BIT,))  # the new bit's plane only
        after = load_reproj_file(fa, keys + ['sub_bitmask'])
        fresh = load_reproj_file(fb, ['sub_bitmask'])['sub_bitmask']
        assert np.array_equal(after['sub_bitmask'], fresh)                            # exactly the new reprojection's
        assert refresh_reader_mask(fa, read_l2b, 1, 2) == 0                           # every plane: the same mask
        assert changed == int((before['sub_bitmask'] != fresh).sum())
        for k in keys:                                                                # everything else unchanged
            if isinstance(before[k], np.ndarray):
                assert np.array_equal(before[k], after[k], equal_nan=True), k
            else:
                assert before[k] == after[k], k
        raw = open(fa, 'rb').read()
        assert refresh_reader_mask(fa, read_l2b, 1, 2) == 0 and open(fa, 'rb').read() == raw   # already current
        assert refresh_mask(fa, l2b_flags(after['file_path'])) == 0
        with pytest.raises(ValueError, match='detector'):
            refresh_mask(fa, np.zeros((H + 1, W), dtype=np.int32))
