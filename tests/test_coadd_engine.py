"""The coadd engine (selfcal/core/coadd.py) on synthetic reprojected frames.

Contracts checked:
1. determinism: the maps are a pure function of (frames, batch size) — identical
   bytes for different worker counts, and identical whether the mean is taken
   from the cache+mean pass or a plain mean pass;
2. the sparse cache round-trips a prepared frame exactly, and the legacy dense
   cache format is still read;
3. the per-pixel arithmetic equals a plain dense reference (mean / std /
   sigma-clipped mean, with a det_aux map) up to summation order — the
   reference is float64, so agreement is at float32 rounding;
4. the mosaic-path _prep_subframe (bbox-local build) matches the full-frame
   build (the LSQR path) bit for bit, and the subset offset render equals the
   full render at the sampled pixels.

Runnable as ``python tests/test_coadd_engine.py`` or under pytest.
"""
import os
import shutil
import tempfile

import h5py
import numpy as np
from astropy.wcs import WCS

from selfcal.core import coadd
from selfcal.core.subframe import _prep_subframe
from selfcal.geometry.map_helper import chunk_to_det, det_to_sub, make_linear_interp_matrix

H = W = 24            # frame size
G = 20                # detector grid
REF = (40, 44)        # reference grid
F = 14                # frames
SEED = 11


def _header():
    return WCS(naxis=2).to_header().tostring()


def _write_frame(path, sub_data, sub_mapping, ref_coords, bitmask):
    with h5py.File(path, 'w', libver='latest') as hf:
        hf.create_dataset('sub_data', data=sub_data.astype(np.float32))
        hf.create_dataset('sub_foot', data=np.ones_like(sub_data, dtype=np.float16))
        hf.create_dataset('sub_bitmask', data=bitmask.astype(np.int32))
        hf.create_dataset('sub_mapping', data=sub_mapping.astype(np.float32))
        hf.attrs['sub_header'] = _header()
        hf.attrs['det_header'] = _header()
        hf.attrs['file_path'] = os.path.basename(path)
        hf.attrs['ref_coords'] = np.array(ref_coords, dtype=np.int32)


def _make_frames(tmp, rng):
    """F frames at random reference positions (some overhanging the map edge),
    each mapping to a random sub-window of the detector grid with sub-pixel
    shifts; NaN coordinates outside a disc, a few masked pixels."""
    files, truth = [], []
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    for k in range(F):
        y0 = int(rng.integers(-4, REF[0] - H + 4))
        x0 = int(rng.integers(-4, REF[1] - W + 4))
        dy, dx = rng.uniform(-3, 3, size=2)
        mapping = np.stack([xs * 0.7 + dx + 1.3, ys * 0.7 + dy + 1.1], axis=0)   # (x, y) detector coords
        disc = (ys - H / 2) ** 2 + (xs - W / 2) ** 2 < (H / 2 + 2) ** 2
        mapping[:, ~disc] = np.nan
        data = rng.normal(1.0, 0.3, size=(H, W)) + 0.01 * k
        data[rng.random((H, W)) < 0.02] = np.nan
        bitmask = np.zeros((H, W), dtype=np.int32)
        bitmask[rng.random((H, W)) < 0.05] = 1 << 3
        bitmask[rng.random((H, W)) < 0.05] = 1 << 21     # ignored bit
        p = os.path.join(tmp, f'exp_{k:06d}_det_0.h5')
        _write_frame(p, data, mapping, (y0, y0 + H, x0, x0 + W), bitmask)
        files.append(p)
        truth.append((y0, x0))
    return files, truth


def _geometry(rng):
    chunk_map = (np.arange(G * G).reshape(G, G) // 25).astype(np.int32)     # 16 chunks
    gvw = np.zeros((G, G), dtype=np.float32)
    gvw[6:15] = np.linspace(0.2, 1.0, 9)[:, None]                            # a horizontal band
    n_chunks = int(chunk_map.max()) + 1
    offsets = rng.normal(0, 0.05, size=(F, n_chunks)).astype(np.float32)
    det_aux = [np.linspace(1.0, 2.0, G * G).reshape(G, G).astype(np.float32)]
    return chunk_map, gvw, offsets, det_aux


def _common(files, chunk_map, gvw, offsets, det_aux, tmp, **kw):
    base = dict(ref_shape=REF, file_list=files, offset_lists=[offsets], apply_weight=True, apply_mask=True,
                chunk_maps=[chunk_map], grid_valid_weight=gvw, ignore_list=[21], det_offset_funcs=None,
                oversample_factor=1, det_aux=det_aux, cache_dir=os.path.join(tmp, 'cache'))
    base.update(kw)
    return base


def _dense_reference(files, chunk_map, gvw, offsets, det_aux, sigma):
    """Plain float64 dense accumulation of the prepared frames (mean, std, sc-mean, aux)."""
    prepped = []
    for k, f in enumerate(files):
        rc, d, w, _, a = _prep_subframe(file=f, chunk_maps=[chunk_map], apply_weight=True, apply_mask=True,
                                        chunk_offsets=[offsets[k]], ignore_list=[21], grid_valid_weight=gvw,
                                        oversample_factor=1, det_aux=det_aux)
        prepped.append((rc, d.astype(np.float64), w.astype(np.float64), a[0].astype(np.float64)))

    def accumulate(fn):
        S = {k: np.zeros(REF) for k in ('d', 'w', 'a')}
        for rc, d, w, a in prepped:
            y0, y1, x0, x1 = (int(v) for v in rc)
            sy, sx = slice(max(y0, 0), min(y1, REF[0])), slice(max(x0, 0), min(x1, REF[1]))
            fy, fx = slice(sy.start - y0, sy.stop - y0), slice(sx.start - x0, sx.stop - x0)
            dv, wv, av = fn(d[fy, fx], w[fy, fx], a[fy, fx], (sy, sx))
            S['d'][sy, sx] += dv; S['w'][sy, sx] += wv; S['a'][sy, sx] += av
        return S

    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    S = accumulate(lambda d, w, a, s: (d * w, w, a * w))
    mean, aux_mean = div(S['d'], S['w']), div(S['a'], S['w'])
    S = accumulate(lambda d, w, a, s: ((d - mean[s]) ** 2 * w, w, (a - mean[s]) ** 2 * w))
    std = np.sqrt(div(S['d'], S['w']))

    def clip(d, w, a, s):
        m = np.abs(d - mean[s]) <= sigma * std[s]
        return d * w * m, w * m, a * w * m
    S = accumulate(clip)
    return mean, aux_mean, std, div(S['d'], S['w']), S['w']


def _bytes_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def test_schedule_deterministic_and_matches_reference():
    rng = np.random.default_rng(SEED)
    tmp = tempfile.mkdtemp(prefix='selfcal_coadd_')
    try:
        files, _ = _make_frames(tmp, rng)
        chunk_map, gvw, offsets, det_aux = _geometry(rng)
        kw = _common(files, chunk_map, gvw, offsets, det_aux, tmp, cache_batch_size=3, coadd_batch_size=3,
                     cache_intermediate=True, make_std_map=True, apply_sigma_clipping=True, sigma=1.5,
                     wav_maps=(np.full((G, G), 1.5, np.float64) + np.arange(G)[None, :] * 0.01,
                               np.full((G, G), 0.03, np.float64)))
        maps2, cached = coadd.run_coadd_schedule(max_workers=2, **kw)
        assert len(cached) == F
        with h5py.File(cached[0], 'r') as hf:
            assert hf.attrs['format'] == coadd.CACHE_FORMAT and 'bc' in hf and 'mask' in hf
        maps5, _ = coadd.run_coadd_schedule(max_workers=5, **kw)
        for name in ('mean_map', 'std_map', 'sc_mean_map', 'wav_mean_map', 'wav_std_map'):
            for key in ('data', 'weight', 'aux'):
                a, b = maps2[name][key], maps5[name][key]
                if a is None:
                    assert b is None
                else:
                    assert _bytes_equal(a, b), f"{name}.{key} differs between 2 and 5 workers"
        # no cache: the mean pass is a plain prep pass -> same bytes
        maps_nc, cached_nc = coadd.run_coadd_schedule(max_workers=3, **dict(kw, cache_intermediate=False,
                                                                             wav_maps=None))
        assert cached_nc == []
        for name in ('mean_map', 'std_map', 'sc_mean_map'):
            assert _bytes_equal(maps_nc[name]['data'], maps2[name]['data']), name
            assert _bytes_equal(maps_nc[name]['weight'], maps2[name]['weight']), name
        # single-pass API from the sparse cache == schedule (same batch size: the
        # accumulation order is batch order, so the batch size is part of the contract)
        m, w, a = coadd.compute_coadd_map('mean', REF, cached, use_cached=True, max_workers=4, batch_size=3,
                                          det_aux=det_aux, cache_dir=os.path.join(tmp, 'cache'))
        assert _bytes_equal(m, maps2['mean_map']['data']) and _bytes_equal(w, maps2['mean_map']['weight'])
        assert _bytes_equal(a, maps2['mean_map']['aux'])
        # dense float64 reference
        mean, aux_mean, std, sc, sc_w = _dense_reference(files, chunk_map, gvw, offsets, det_aux, 1.5)
        cov = maps2['mean_map']['weight'] > 0
        assert cov.any() and cov.sum() < cov.size
        np.testing.assert_allclose(maps2['mean_map']['data'][cov], mean[cov], rtol=3e-6, atol=1e-6)
        np.testing.assert_allclose(maps2['mean_map']['aux'][0][cov], aux_mean[cov], rtol=3e-6, atol=1e-6)
        np.testing.assert_allclose(maps2['std_map']['data'][cov], std[cov], rtol=2e-4, atol=1e-6)
        np.testing.assert_allclose(maps2['sc_mean_map']['weight'][cov], sc_w[cov], rtol=3e-6, atol=1e-6)
        np.testing.assert_allclose(maps2['sc_mean_map']['data'][cov], sc[cov], rtol=3e-6, atol=1e-6)
        # wav: the mean band centre lies within the centre map's range and the band std
        # is at least the in-band term width/sqrt(12) (plus the centre scatter across frames)
        wm = maps2['wav_mean_map']['data']
        assert 1.5 <= wm[cov].min() and wm[cov].max() <= 1.5 + 0.01 * G
        # (the production formula sqrt(E[BW(BW^2/12+BC^2)]/E[BW] - mean^2) cancels two O(1)
        # float32 terms to get an O(1e-4) variance, so allow a generous margin)
        assert (maps2['wav_std_map']['data'][cov] >= 0.03 / np.sqrt(12) * 0.7).all()
        assert (maps2['wav_std_map']['data'][cov] <= 0.03 / np.sqrt(12) + 0.01 * G).all()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_roundtrip_and_legacy_format():
    rng = np.random.default_rng(SEED + 1)
    tmp = tempfile.mkdtemp(prefix='selfcal_coadd_')
    try:
        files, _ = _make_frames(tmp, rng)
        chunk_map, gvw, offsets, det_aux = _geometry(rng)
        rc, d, w, _, a = _prep_subframe(file=files[3], chunk_maps=[chunk_map], apply_weight=True, apply_mask=True,
                                        chunk_offsets=[offsets[3]], ignore_list=[21], grid_valid_weight=gvw,
                                        oversample_factor=1, det_aux=det_aux)
        fr = coadd.sparsify_frame(rc, d, w, a)
        p = os.path.join(tmp, 'sparse.h5')
        coadd.write_cached_frame(p, fr)
        back = coadd.read_cached_frame(p)
        for k in ('ref_coords', 'sub_bbox', 'flat', 'data', 'weight', 'aux'):
            assert _bytes_equal(np.asarray(fr[k]), np.asarray(back[k])), k
        # dense reconstruction == the dense crop the legacy cache stored
        rmin, rmax, cmin, cmax = fr['sub_bbox']
        rc2, d2, w2, bbox2 = coadd.load_cached_frame_dense(p)
        assert np.array_equal(w2, w[rmin:rmax, cmin:cmax])
        assert np.array_equal(d2 * (w2 != 0), d[rmin:rmax, cmin:cmax] * (w[rmin:rmax, cmin:cmax] != 0))
        # legacy dense file -> same sparse frame
        q = os.path.join(tmp, 'dense.h5')
        with h5py.File(q, 'w') as hf:
            hf.create_dataset('ref_coords', data=fr['ref_coords'])
            hf.create_dataset('sub_data', data=d[rmin:rmax, cmin:cmax])
            hf.create_dataset('sub_weight', data=w[rmin:rmax, cmin:cmax])
            hf.create_dataset('sub_aux', data=a[:, rmin:rmax, cmin:cmax])
            hf.create_dataset('sub_bbox', data=np.array([rmin, rmax, cmin, cmax], np.int32))
        leg = coadd.read_cached_frame(q)
        for k in ('ref_coords', 'sub_bbox', 'flat', 'data', 'weight', 'aux'):
            assert _bytes_equal(np.asarray(fr[k]), np.asarray(leg[k])), k
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prep_bbox_path_matches_full_frame_build():
    rng = np.random.default_rng(SEED + 2)
    tmp = tempfile.mkdtemp(prefix='selfcal_coadd_')
    try:
        files, _ = _make_frames(tmp, rng)
        chunk_map, gvw, offsets, det_aux = _geometry(rng)
        for k in (0, 5, 9):
            # mosaic path (bbox-local) vs LSQR path (full-frame matrix), no offsets
            base = dict(file=files[k], chunk_maps=[chunk_map], apply_weight=True, apply_mask=True,
                        ignore_list=[21], grid_valid_weight=gvw, oversample_factor=1, det_aux=det_aux)
            rc_m, d_m, w_m, _, a_m = _prep_subframe(for_lsqr=False, **base)
            rc_l, d_l, w_l, cc, a_l = _prep_subframe(for_lsqr=True, **base)
            assert _bytes_equal(d_m, d_l) and _bytes_equal(w_m, w_l) and _bytes_equal(a_m, a_l)
            assert cc and cc[0].shape[1] == H * W
            # offsets: bbox path vs a full-frame reference through the full interp matrix
            rc_o, d_o, w_o, _, _ = _prep_subframe(for_lsqr=False, chunk_offsets=[offsets[k]], **base)
            with h5py.File(files[k], 'r') as hf:
                raw = hf['sub_data'][()]
                mapping = hf['sub_mapping'][()]
            full = make_linear_interp_matrix(mapping.reshape(2, -1)[::-1], input_shape=(G, G))
            ref = raw.copy()
            ref -= det_to_sub(chunk_to_det(chunk_map, chunk_data=offsets[k]), interp_matrix=full)
            ref[np.isnan(ref)] = 0.0
            assert np.array_equal(d_o[w_o != 0], ref[w_o != 0])
            # subset render == full render at the sampled pixels
            needed = np.unique(full.indices)
            assert np.array_equal(chunk_to_det(chunk_map, offsets[k], needed=needed),
                                  chunk_to_det(chunk_map, offsets[k]).ravel()[needed])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test_schedule_deterministic_and_matches_reference()
    print("OK schedule: deterministic across worker counts, cache==no-cache, matches dense reference")
    test_cache_roundtrip_and_legacy_format()
    print("OK sparse cache round-trip + legacy dense format")
    test_prep_bbox_path_matches_full_frame_build()
    print("OK _prep_subframe bbox path == full-frame build")
