"""``Fit(sky_mask=sc.SkyMask(flags, above=...))``: sky pixels the frames flag too often with given bits are dropped
from every frame's rows (:mod:`selfcal.core.skymask`).

Synthetic frames written with :func:`selfcal.io.frames.write_frame` (no instrument): a source fixed on the sky whose
flagged area moves from frame to frame, so the pixels at its edge are flagged by some frames only.
"""
import numpy as np
import pytest

from selfcal import _state
from selfcal.core.skymask import flagged_fraction_mask, grid_window
from selfcal.core.system import setup_lsqr
from selfcal.io.frames import standard_frame_path, write_frame

SRC = 21                       # the flag bit (any bit: nothing here is instrument-specific)
OTHER = 3                      # another masking bit, set at random
H, W, F = 8, 8, 6
GRID = (H + F, W)


def _frames(tmp_path):
    """F frames, frame k on the grid rows k..k+H. The source covers grid rows 6-7, columns 3-4 in every frame;
    the even frames also flag column 5 (the mask's edge moves); frame 1 has one non-finite value and frame 2 one
    pixel with another bit set."""
    _state.set_progress(False)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    paths, bitmasks, datas = [], [], []
    for k in range(F):
        vals = (1.0 + 0.01 * k + 0.001 * x).astype(np.float32)
        bm = np.zeros((H, W), np.int32)
        gy = y + k                                              # grid row of each frame pixel
        src = (gy >= 6) & (gy <= 7) & (x >= 3) & (x <= 4 + (k % 2 == 0))
        bm[src] |= 1 << SRC
        if k == 1:
            vals[0, 0] = np.nan
        if k == 2:
            bm[1, 1] |= 1 << OTHER
        paths.append(write_frame(standard_frame_path(str(tmp_path), k, 0), vals, [k, k + H, 0, W],
                                 np.stack([x, y]), bitmask=bm))
        bitmasks.append(bm)
        datas.append(vals)
    return paths, bitmasks, datas


def _reference_counts(bitmasks, datas, rule_bits=((SRC,),), ignore=(), det_valid=None):
    hits = np.zeros(GRID, int)
    flagged = np.zeros((len(rule_bits),) + GRID, int)
    not_masking = set(ignore) | {b for bits in rule_bits for b in bits}
    for k, (bm, v) in enumerate(zip(bitmasks, datas)):
        other = bm.copy()
        for b in not_masking:
            other &= ~(1 << b)
        valid = np.isfinite(v) & (other == 0)
        if det_valid is not None:                       # the frames' pixels sit at detector (x, y) = (column, row)
            valid &= det_valid
        hits[k:k + H] += valid
        for r, bits in enumerate(rule_bits):
            flagged[r, k:k + H] += valid & (sum((bm >> b) & 1 for b in bits) > 0)
    return hits, flagged


def test_the_counts_and_the_mask_match_a_direct_count(tmp_path):
    paths, bitmasks, datas = _frames(tmp_path)
    hits_ref, flagged_ref = _reference_counts(bitmasks, datas)
    for above in (0.0, 0.5):
        drop, hits, flagged = flagged_fraction_mask(paths, GRID, [([SRC], above)])
        np.testing.assert_array_equal(hits, hits_ref)
        np.testing.assert_array_equal(flagged, flagged_ref)
        np.testing.assert_array_equal(drop, (hits_ref > 0) & (flagged_ref[0] > above * hits_ref))
    # the moving edge: column 5 of rows 6-7 is flagged by the even frames only, about half its covering frames
    drop0, _, _ = flagged_fraction_mask(paths, GRID, [([SRC], 0.0)])
    drop5, _, _ = flagged_fraction_mask(paths, GRID, [([SRC], 0.5)])
    assert drop0[6:8, 5].all() and not drop5[6:8, 5].any()       # flagged once -> union; half -> not above 0.5
    assert drop5[6:8, 3:5].all()                                 # flagged by every frame
    # an ignored bit does not mask, and the worker pool gives the same counts
    _, hits_ign, _ = flagged_fraction_mask(paths, GRID, [([SRC], 0.5)], ignore_list=[OTHER])
    np.testing.assert_array_equal(hits_ign, _reference_counts(bitmasks, datas, ignore=[OTHER])[0])
    drop_p, hits_p, flagged_p = flagged_fraction_mask(paths, GRID, [([SRC], 0.5)], max_workers=2)
    np.testing.assert_array_equal(hits_p, hits_ref)
    np.testing.assert_array_equal(drop_p, drop5)


def test_several_rules_each_with_its_bits_and_threshold(tmp_path):
    paths, bitmasks, datas = _frames(tmp_path)
    rules = [([SRC], 0.5), ([OTHER, 7], 0.0)]                   # the source bit at half, the other bits at once
    drop, hits, flagged = flagged_fraction_mask(paths, GRID, rules)
    hits_ref, flagged_ref = _reference_counts(bitmasks, datas, rule_bits=((SRC,), (OTHER, 7)))
    np.testing.assert_array_equal(hits, hits_ref)               # covered apart from every rule's bits
    np.testing.assert_array_equal(flagged, flagged_ref)
    np.testing.assert_array_equal(drop, (hits_ref > 0) & ((flagged_ref[0] > 0.5 * hits_ref) | (flagged_ref[1] > 0)))
    assert drop[3, 1] and not flagged_fraction_mask(paths, GRID, [([SRC], 0.5)])[0][3, 1]   # frame 2's other bit
    with pytest.raises(ValueError):
        flagged_fraction_mask(paths, GRID, [([SRC], 1.0)])


def test_only_the_detector_region_the_solve_uses_counts(tmp_path):
    """With the setup's detector-grid valid weight (a job using part of the detector), only the frames' pixels in
    that region count, as in the solve; a partial region changes which edge pixels go."""
    paths, bitmasks, datas = _frames(tmp_path)
    det_valid = np.zeros((H, W), np.float32)
    det_valid[:, :5] = 1.0                                      # the job's detector columns 0..4
    drop, hits, flagged = flagged_fraction_mask(paths, GRID, [([SRC], 0.0)], grid_valid_weight=det_valid)
    hits_ref, flagged_ref = _reference_counts(bitmasks, datas, det_valid=det_valid > 0)
    np.testing.assert_array_equal(hits, hits_ref)
    np.testing.assert_array_equal(flagged, flagged_ref)
    assert hits[:, 5:].sum() == 0 and not drop[6:8, 5].any()    # column 5 is outside the job: never counted


def test_grid_window_clips_to_the_grid():
    m = np.arange(12).reshape(3, 4) % 2 == 0
    np.testing.assert_array_equal(grid_window(m, [-1, 2, 2, 5], (3, 3)),
                                  [[False, False, False], [True, False, False], [True, False, False]])


def test_setup_drops_the_masked_pixels_from_every_frame_and_nothing_else(tmp_path):
    paths, bitmasks, datas = _frames(tmp_path)
    common = dict(chunk_maps=[np.zeros((H, W), np.int32)], grid_valid_weight=np.ones((H, W), np.float32),
                  apply_mask=True, ignore_list=[], outlier_thresh=None, max_workers=1, batch_size=3)
    n_sky = GRID[0] * GRID[1]
    plain = setup_lsqr(paths, GRID, **common)
    masked = setup_lsqr(paths, GRID, sky_mask=[([SRC], 0.0)], **common)
    drop, _, _ = flagged_fraction_mask(paths, GRID, [([SRC], 0.0)])
    counts_plain = np.asarray(plain.pixel_counts[:n_sky]).reshape(GRID)
    counts_masked = np.asarray(masked.pixel_counts[:n_sky]).reshape(GRID)
    assert counts_plain[drop].sum() > 0                          # the edge pixels: data from the unflagging frames
    assert counts_masked[drop].sum() == 0                        # gone from every frame
    np.testing.assert_array_equal(counts_masked[~drop], counts_plain[~drop])


def test_the_setting_lowers_to_the_setup_keyword_and_validates():
    import selfcal as sc
    from selfcal.config.base import ConfigError
    from selfcal.run.lower import setup_options

    class _Inst:
        def default_ignore_flags(self):
            return []

    compute = sc.Compute(workers=2)
    recipe = sc.Recipe(sc.continuum(), fit=sc.Fit(sky_mask=sc.SkyMask((SRC,), above=0.5)))
    assert setup_options(recipe, compute, _Inst())['sky_mask'] == [([SRC], 0.5)]
    two = sc.Fit(sky_mask=(sc.SkyMask((SRC,), above=0.5), sc.SkyMask((3, 7), above=0.0)))
    assert two.sky_mask == (sc.SkyMask((SRC,), above=0.5), sc.SkyMask((3, 7), above=0.0))
    assert setup_options(sc.Recipe(sc.continuum(), fit=two), compute, _Inst())['sky_mask'] == [([SRC], 0.5), ([3, 7], 0.0)]
    assert 'sky_mask' not in setup_options(sc.Recipe(sc.continuum()), compute, _Inst())
    assert sc.SkyMask([SRC, 4]).flags == (SRC, 4)
    for bad in (dict(flags=()), dict(flags=(-1,)), dict(flags=(SRC,), above=1.0), dict(flags=(SRC,), above=-0.1)):
        with pytest.raises(ConfigError):
            sc.SkyMask(**bad)
    with pytest.raises(ConfigError):
        sc.Fit(sky_mask=(SRC, 0.5))
