"""Any telescope, any model — each adapted with high-level functions only.

Every example is a different telescope or model, calibrated end to end on
synthetic data, and none of them changes selfcal: each is a run config plus
functions defined in THIS file (coefficients, bases, per-frame values, a reader,
a prior) and, only where the telescope has something new to say (a detector map,
a per-frame keyword, a raw format), a small ``Instrument`` subclass.

1. ``test_time_variable_sky`` — a sky term modulated with the season
   (``sin 2π(t − t0)/yr``, ``t`` a header keyword of each exposure), offsets that
   drift slowly in time with a temporal-smoothness prior.
2. ``test_imaging_polarimeter`` — Stokes I, Q, U maps: ``I + Q cos 2ψ + U sin 2ψ``
   with ``ψ`` a function of a per-frame half-wave-plate angle (header) and a
   per-pixel polariser angle (a detector map of the instrument).
3. ``test_thermal_pattern_and_gradients`` — a detector-fixed pattern scaled by
   each frame's temperature (an offset coefficient of a frame variable) plus a
   free 2-D gradient per frame (an offset basis in the built-in detector
   coordinates); the mosaic subtracts both at every observation.
4. ``test_data_cube_slices`` — an integral-field-like cube: a reader turns the
   slices of each file into frames with a per-slice wavelength and a variance
   plane; the model fits a line map (a Gaussian of the slice wavelength), shares
   the offsets over the slices of one exposure (grouping by a frame variable)
   and weights every observation by its inverse sigma (a stored layer).
5. ``test_frames_written_directly`` — samples that never were a FITS image: frames
   written with ``write_frame``; a per-frame gain against a known sky (an offset
   coefficient of a sky-grid variable) with the gains' scale fixed by a prior
   written here — through the library API, without the runner.
6. ``test_amplifier_ghost`` — each amplifier picks up a fraction of the mean
   signal of its mirror amplifier (crosstalk): the variable is computed by a
   function of the frame's own stored data, the fractions are a shared offset
   term with that variable as its coefficient.
7. ``test_heterogeneous_focal_plane`` — two detectors of different sizes on one
   focal plane: the reader places each detector's pixels on the focal-plane
   coordinates, the chunk map spans the focal plane, and each detector's
   read-out pattern is an offset term grouped by the detector.
"""
import dataclasses
import os
import shutil
import sys
import tempfile

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from selfcal import _state                                                  # noqa: E402
from selfcal.instruments import register_instrument                         # noqa: E402
from selfcal.instruments.base import ExposureLayout                         # noqa: E402
from selfcal.instruments.grid import GridInstrument, rect_grid_chunk_map    # noqa: E402
from selfcal.io.calfile import CalFile                                      # noqa: E402
from selfcal.io.frames import ExposureData, frame_header_values             # noqa: E402

H, W = 48, 64                     # detector (rows, cols)
REF = 160                         # side of the truth sky grid
PIX = 20.0                        # arcsec per pixel (detector = reference)
CHUNKS = (3, 4)


# ============================================================================
# the functions the models name (by import path: tests.test_any_telescope:<name>)
# ============================================================================
def annual(t, t0=60000.0, period=365.25):
    """A seasonal modulation of time ``t`` (days)."""
    return np.sin(2 * np.pi * (np.asarray(t, dtype=np.float64) - t0) / period)


def polariser_angle(hwp_deg, pix_angle):
    """Angle of the polarisation a pixel analyses: the half-wave plate rotates it by twice its angle."""
    return 2.0 * np.deg2rad(np.asarray(hwp_deg, dtype=np.float64)) + np.asarray(pix_angle, dtype=np.float64)


def cos2(psi):
    return np.cos(2.0 * psi)


def sin2(psi):
    return np.sin(2.0 * psi)


def centred(temperature, t0=80.0):
    """A thermal offset scales with the temperature above t0."""
    return np.asarray(temperature, dtype=np.float64) - t0


def plane(x, y):
    """A 2-D gradient basis in detector coordinates (two functions)."""
    return [(np.asarray(x) - W / 2) / W, (np.asarray(y) - H / 2) / H]


def gaussian_line(wave, center=1.60, sigma=0.02):
    return np.exp(-0.5 * ((np.asarray(wave, dtype=np.float64) - center) / sigma) ** 2)


def inverse_sigma(variance):
    return 1.0 / np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64), 1e-12))


def identity(x):
    return x


def partner_mean(frame):
    """Crosstalk source: at every observation, the mean of the frame's stored data
    over the partner amplifier (the mirrored chunk column)."""
    raw = frame.raw()
    sm = np.asarray(frame.sub_mapping, dtype=np.float64)
    x, y = np.rint(np.nan_to_num(sm[0], nan=-1)).astype(int), np.rint(np.nan_to_num(sm[1], nan=-1)).astype(int)
    inside = np.isfinite(raw) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    cm = rect_grid_chunk_map((H, W), *CHUNKS)
    n = int(cm.max()) + 1
    chunk = cm[y[inside], x[inside]]
    means = np.bincount(chunk, weights=raw[inside], minlength=n) / np.maximum(np.bincount(chunk, minlength=n), 1)
    ny, nx = CHUNKS
    ids = np.arange(n)
    partner = (ids // nx) * nx + (nx - 1 - ids % nx)
    out = np.zeros(raw.shape)
    out[inside] = means[partner][chunk]
    return out


# focal plane of test 7: detector A (40 x 40) at x 0..39, detector B (40 x 24) at x 48..71
FP_SHAPE = (40, 72)
FP_ORIGIN = {0: 0, 1: 48}
FP_WIDTH = {0: 40, 1: 24}


def read_focal_plane(path, sci_ext, dq_ext=None, header_only=False):
    """Two detectors per file, of different sizes; every pixel gets its
    focal-plane coordinates (x shifted by the detector's origin)."""
    with fits.open(path) as hdul:
        hdu = hdul[int(sci_ext)]
        hdr = hdu.header.copy()
        det = int(hdr['DETID'])
        if header_only:
            return ExposureData(header=hdr, shape=tuple(hdu.shape))
        data = np.array(hdu.data, dtype=np.float64)
    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]].astype(np.float64)
    return ExposureData(header=hdr, data=data, coords=(xx + FP_ORIGIN[det], yy))


def read_cube_slice(path, sci_ext, dq_ext=None, header_only=False):
    """A raw format that is not an image per extension: every file holds a data
    cube (slices of one exposure at different wavelengths) and a variance cube.
    ``sci_ext`` is the slice index; the slice's wavelength becomes the header
    keyword ``WAVE`` (a frame variable) and its variance a layer."""
    with fits.open(path) as hdul:
        hdr = hdul['SCI'].header
        cel = WCS(hdr).celestial.to_header()
        cel['WAVE'] = float(hdr[f'WAVE{int(sci_ext)}'])
        cel['NAXIS'], cel['NAXIS1'], cel['NAXIS2'] = 2, int(hdr['NAXIS1']), int(hdr['NAXIS2'])
        if header_only:
            return ExposureData(header=cel, shape=(int(hdr['NAXIS2']), int(hdr['NAXIS1'])))
        data = np.array(hdul['SCI'].data[int(sci_ext)], dtype=np.float64)
        var = np.array(hdul['VAR'].data[int(sci_ext)], dtype=np.float64)
    return ExposureData(header=cel, data=data, layers={'variance': var})


# ============================================================================
# instruments — only what the telescope has to say beyond the grid imager
# ============================================================================
def _pix_angle_map():
    """The polariser angle of every pixel (radians): varies across the detector."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    return (np.deg2rad(60.0) * x / W + np.deg2rad(20.0) * y / H).astype(np.float32)


@register_instrument('toy_polarimeter')
class ToyPolarimeter(GridInstrument):
    """A grid imager whose pixels carry polarisers and whose exposures record a
    half-wave-plate angle: one detector map and one frame variable more."""

    def detector_geometry(self, inst_cfg, oversample):
        geom = super().detector_geometry(inst_cfg, oversample)
        return dataclasses.replace(geom, aux={'pix_angle': _pix_angle_map()})

    def frame_variable_names(self, inst_cfg):
        return super().frame_variable_names(inst_cfg) + ('hwp',)

    def frame_variables(self, frames, inst_cfg=None):
        out = super().frame_variables(frames, inst_cfg)
        out['hwp'] = frame_header_values(frames, ['HWPANG'])['HWPANG']
        return out


@register_instrument('toy_ifu')
class ToyCube(GridInstrument):
    """Files hold cubes: four slices per exposure, read by ``read_cube_slice``."""

    def exposure_layout(self, inst_cfg):
        return ExposureLayout(sci_ext=[0, 1, 2, 3], dq_ext=None, detector_ids=[0, 1, 2, 3],
                              ref_use_ext=(0,), reader=read_cube_slice)


def _focal_plane_chunks():
    """Chunk map over the focal plane: each detector split into two halves (rows)
    per column block; -1 in the gap between the detectors."""
    cm = np.full(FP_SHAPE, -1, dtype=np.int32)
    k = 0
    for det in (0, 1):
        x0, w = FP_ORIGIN[det], FP_WIDTH[det]
        for r in range(2):
            for c in range(2):
                cm[r * 20:(r + 1) * 20, x0 + c * (w // 2):x0 + (c + 1) * (w // 2)] = k
                k += 1
    return cm


@register_instrument('toy_focal_plane')
class ToyFocalPlane(GridInstrument):
    """Two detectors of different sizes on one focal plane (no code for the
    geometry beyond the chunk map and the reader)."""

    def exposure_layout(self, inst_cfg):
        return ExposureLayout(sci_ext=[1, 2], dq_ext=None, detector_ids=[0, 1], ref_use_ext=(1, 2),
                              reader=read_focal_plane)

    def detector_geometry(self, inst_cfg, oversample):
        from selfcal.instruments.base import ChunkMap, DetectorGeometry
        from selfcal.models.offset_structure import ChunkAxes
        cm = _focal_plane_chunks()
        axes = ChunkAxes.row_major(('det', 'row', 'col'), (2, 2, 2), ('x', 'y', 'x'))
        chunk = ChunkMap(name='amps', det=cm, grid=cm, axes=axes, adjacency_axes=())
        return DetectorGeometry(shape=FP_SHAPE, chunk_maps={'amps': chunk}, primary='amps')

    def job_geometry(self, inst_cfg, geom, job):
        from selfcal.instruments.base import JobGeometry
        valid = (geom.chunk_map.det >= 0).astype(np.float32)
        n = geom.chunk_map.n_chunks
        return JobGeometry(det_valid_weight=valid, grid_valid_weight=valid, chunk_valid=np.ones(n, bool),
                           chunk_valid_strict=np.ones(n, bool), det_valid_mask=valid, grid_valid_mask=valid)

    def frame_tag(self, inst_cfg):
        return 'FocalPlane'


# ============================================================================
# synthetic data + comparison helpers
# ============================================================================
def _sky():
    y, x = np.mgrid[0:REF, 0:REF].astype(np.float64)
    return 5.0 + 0.02 * x + 0.01 * y + 0.5 * np.sin(x / 11.0) * np.cos(y / 13.0)


def _pattern(seed, scale=1.0):
    y, x = np.mgrid[0:REF, 0:REF].astype(np.float64)
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, 2)
    return scale * np.sin(x / (5.0 + seed) + ph[0]) * np.cos(y / (6.5 + seed) + ph[1])


def _wcs(ox, oy):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [W / 2 + 0.5 - ox, H / 2 + 0.5 - oy]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.cdelt = [-PIX / 3600.0, PIX / 3600.0]
    return w


def _write_image(path, img, ox, oy, keywords):
    hdr = _wcs(ox, oy).to_header()
    for k, v in keywords.items():
        hdr[k] = v
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.asarray(img, np.float32), header=hdr, name='SCI')]
                 ).writeto(path, overwrite=True)


def _truth_values(out, got, ok, truth):
    """``(got, truth)`` at the covered reference pixels ``ok``, matched through the WCS."""
    from selfcal.geometry import wcs_helper
    ref_wcs, _ = wcs_helper.load_from_fits(os.path.join(out, 'toy_run', 'ref.fits'))
    tw = _wcs(0, 0)
    py, px = np.nonzero(ok)
    tx, ty = tw.world_to_pixel(ref_wcs.pixel_to_world(px, py))
    tx, ty = np.round(tx).astype(int), np.round(ty).astype(int)
    inside = (tx >= 2) & (tx < REF - 2) & (ty >= 2) & (ty < REF - 2)
    return got[py[inside], px[inside]], truth[ty[inside], tx[inside]]


def _agreement(a, b, demean=True):
    """Pearson r and slope of ``a`` against ``b`` (means removed: a constant is a
    gauge of the solve — degenerate with the per-frame scalars)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if demean:
        a, b = a - a.mean(), b - b.mean()
    return float(np.corrcoef(a, b)[0, 1]), float(np.dot(a, b) / np.dot(b, b))


def _run(tmp, inst, model, n_exp, exp_dir, pattern='/toy_exp_*.fits', calibration=None, lsqr=None,
         mosaic=False):
    """Reproject + calibrate (+ mosaic) through the runner, as a user would."""
    from selfcal_scripts.runner import pipelines
    from tests.test_runner_e2e_toy import _write_config
    out, cache = os.path.join(tmp, 'out'), os.path.join(tmp, 'cache')
    os.makedirs(cache, exist_ok=True)
    rcfg = _write_config(os.path.join(tmp, 'reproject.toml'), 'reproject', out, cache, instrument=inst,
                         reproject=dict(input_dirs=[exp_dir], file_pattern=pattern, padding_pixels=8,
                                        max_workers=2, inner_parallel=1, reproj_func='interp',
                                        padding_percentage=0.05, replace_existing=True))
    pipelines.run(rcfg)
    cal = dict(apply_mask=False, apply_weight=False, outlier_thresh=8.0, ignore_list=[], batch_size=4,
               offset_regularization=True, weighted_damping=True, damp_weight=1e-4, max_workers=2)
    cal.update(calibration or {})
    lq = dict(atol=1e-12, btol=1e-12, damp=0, iter_lim=500, precondition=True, solver='lsqr')
    lq.update(lsqr or {})
    model = dict(model)
    model.setdefault('mosaic', 'full' if mosaic else 'none')
    ccfg = _write_config(os.path.join(tmp, 'cal.toml'), 'cal', out, cache, scalars={'mode': 'model'},
                         instrument=inst, model=model, calibration=cal, lsqr=lq,
                         mosaic=dict(apply_mask=False, apply_weight=False, make_std_map=True,
                                     apply_sigma_clipping=False, sigma=3.0, ignore_list=[],
                                     cache_batch_size=4, coadd_batch_size=4, cache_intermediate=False,
                                     max_workers=2))
    res = pipelines.run(ccfg)
    return out, res


def _grid_inst(**kw):
    d = {'name': 'grid', 'tag': 'Toy', 'detector_shape': [H, W], 'chunks': list(CHUNKS), 'sci_ext': 1,
         'dq_ext': -1}
    d.update(kw)
    return d


# ============================================================================
# 1. a time-variable sky
# ============================================================================
def test_time_variable_sky():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_anytel_time_')
    try:
        rng = np.random.default_rng(1)
        sky, s_annual = _sky(), _pattern(1, 0.8)
        cm = rect_grid_chunk_map((H, W), *CHUNKS)
        n_exp = 48
        mjd = 60000.0 + np.sort(rng.uniform(0, 365.25, n_exp))
        # chunk offsets that drift slowly in time (what the temporal prior expects)
        drift = 0.3 * np.sin(2 * np.pi * (mjd[:, None] - 60000.0) / 200.0 + rng.uniform(0, 6, cm.max() + 1))
        drift -= drift.mean(axis=1, keepdims=True)
        exp_dir = os.path.join(tmp, 'exposures')
        os.makedirs(exp_dir)
        for k in range(n_exp):
            oy, ox = rng.integers(0, REF - H), rng.integers(0, REF - W)
            img = (sky[oy:oy + H, ox:ox + W] + annual(mjd[k]) * s_annual[oy:oy + H, ox:ox + W]
                   + drift[k][cm] + rng.normal(0, 1.0) + rng.normal(0, 0.01, (H, W)))
            _write_image(os.path.join(exp_dir, f'toy_exp_{k:03d}.fits'), img, ox, oy, {'MJD-AVG': mjd[k]})
        model = {
            'variables': {'time': {'header': 'MJD-AVG'}},
            'sky': [{'name': 'continuum'},
                    {'name': 'annual', 'damp_weight': 1e-4,
                     'coefficient': {'variable': 'time', 'function': 'tests.test_any_telescope:annual'}}],
            'offset': [{'kind': 'free', 'reg_weight': 0.05, 'mean_zero': True, 'name': 'chunks'}],
            'prior': [{'term': 'chunks', 'function': 'frame_smoothness', 'variable': 'time', 'weight': 0.2}],
        }
        out, res = _run(tmp, _grid_inst(), model, n_exp, exp_dir)
        with CalFile(res.cal_paths[0]) as cal:
            got, cov = cal.sky('annual'), cal.sky_coverage('annual')
            offs = cal.offsets[0]
            names = [os.path.basename(p) for p in cal.reproj_list]
        g, t = _truth_values(out, got, (cov >= 6) & np.isfinite(got), s_annual)
        r, slope = _agreement(g, t)
        order = [int(n.split('_')[1]) for n in names]
        ro, _ = _agreement(offs.ravel(), drift[order].ravel())
        print(f"time-variable sky: annual map r = {r:.4f}, slope {slope:.3f} ({g.size} px); drifting offsets r = {ro:.3f}")
        assert r > 0.99 and abs(slope - 1) < 0.05, (r, slope)
        assert ro > 0.95, ro
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# 2. an imaging polarimeter
# ============================================================================
def test_imaging_polarimeter():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_anytel_pol_')
    try:
        rng = np.random.default_rng(2)
        sky_i, sky_q, sky_u = _sky(), _pattern(2, 0.6), _pattern(3, 0.6)
        pix = _pix_angle_map()
        cm = rect_grid_chunk_map((H, W), *CHUNKS)
        n_exp = 48
        exp_dir = os.path.join(tmp, 'exposures')
        os.makedirs(exp_dir)
        for k in range(n_exp):
            hwp = [0.0, 22.5, 45.0, 67.5][k % 4]
            oy, ox = rng.integers(0, REF - H), rng.integers(0, REF - W)
            psi = polariser_angle(hwp, pix)
            off = rng.normal(0, 0.2, cm.max() + 1)
            img = (sky_i[oy:oy + H, ox:ox + W] + sky_q[oy:oy + H, ox:ox + W] * np.cos(2 * psi)
                   + sky_u[oy:oy + H, ox:ox + W] * np.sin(2 * psi) + (off - off.mean())[cm]
                   + rng.normal(0, 1.0) + rng.normal(0, 0.01, (H, W)))
            _write_image(os.path.join(exp_dir, f'toy_exp_{k:03d}.fits'), img, ox, oy, {'HWPANG': hwp})
        model = {
            'variables': {'psi': {'function': 'tests.test_any_telescope:polariser_angle',
                                  'inputs': ['hwp', 'pix_angle']}},
            'sky': [{'name': 'I'},
                    {'name': 'Q', 'damp_weight': 1e-4,
                     'coefficient': {'variable': 'psi', 'function': 'tests.test_any_telescope:cos2'}},
                    {'name': 'U', 'damp_weight': 1e-4,
                     'coefficient': {'variable': 'psi', 'function': 'tests.test_any_telescope:sin2'}}],
            'offset': [{'kind': 'free', 'reg_weight': 0.05, 'mean_zero': True}],
        }
        out, res = _run(tmp, _grid_inst(name='toy_polarimeter'), model, n_exp, exp_dir)
        with CalFile(res.cal_paths[0]) as cal:
            assert cal.sky_names == ['I', 'Q', 'U']
            maps = {n: (cal.sky(n), cal.sky_coverage(n)) for n in ('Q', 'U')}
        for name, truth in (('Q', sky_q), ('U', sky_u)):
            got, cov = maps[name]
            g, t = _truth_values(out, got, (cov >= 6) & np.isfinite(got), truth)
            r, slope = _agreement(g, t)
            print(f"polarimeter: {name} r = {r:.4f}, slope {slope:.3f} ({g.size} px)")
            assert r > 0.99 and abs(slope - 1) < 0.05, (name, r, slope)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# 3. a thermal pattern and per-frame gradients (offset coefficient + basis)
# ============================================================================
def test_thermal_pattern_and_gradients():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_anytel_thermal_')
    try:
        rng = np.random.default_rng(3)
        sky = _sky()
        cm = rect_grid_chunk_map((H, W), *CHUNKS)
        pattern = rng.normal(0, 0.05, cm.max() + 1)
        pattern -= pattern.mean()
        n_exp = 48
        temp = 80.0 + rng.uniform(-4, 4, n_exp)
        grad = rng.normal(0, 0.3, (n_exp, 2))
        y, x = np.mgrid[0:H, 0:W].astype(np.float64)
        gx, gy = plane(x, y)
        exp_dir = os.path.join(tmp, 'exposures')
        os.makedirs(exp_dir)
        for k in range(n_exp):
            oy, ox = rng.integers(0, REF - H), rng.integers(0, REF - W)
            img = (sky[oy:oy + H, ox:ox + W] + centred(temp[k]) * pattern[cm]
                   + grad[k, 0] * gx + grad[k, 1] * gy + rng.normal(0, 1.0) + rng.normal(0, 0.01, (H, W)))
            _write_image(os.path.join(exp_dir, f'toy_exp_{k:03d}.fits'), img, ox, oy, {'TEMP': temp[k]})
        model = {
            'variables': {'temperature': {'header': 'TEMP'}},
            'sky': [{'name': 'continuum'}],
            'offset': [
                # a detector-fixed pattern times each frame's temperature
                {'kind': 'fixed', 'mean_zero': True, 'name': 'thermal',
                 'coefficient': {'variable': 'temperature', 'function': 'tests.test_any_telescope:centred'}},
                # a free 2-D gradient per frame over the whole detector
                {'map': 'detector', 'kind': 'free', 'name': 'gradient',
                 'basis': {'variable': ['det_x', 'det_y'], 'function': 'tests.test_any_telescope:plane', 'n': 2}},
            ],
        }
        out, res = _run(tmp, _grid_inst(), model, n_exp, exp_dir, mosaic=True)
        with CalFile(res.cal_paths[0]) as cal:
            got_pattern = cal.offsets[0][0]
            got_grad = cal.offsets[1]
            solved_sky = cal.sky(0)
            names = [os.path.basename(p) for p in cal.reproj_list]
            assert cal.offset_basis(1)[0] == 2 and cal.offset_basis(0)[0] == 1
        order = [int(n.split('_')[1]) for n in names]
        rp, sp = _agreement(got_pattern, pattern)
        # a gradient common to every frame is a gauge (a sky gradient + per-frame scalars
        # reproduce it): compare each direction's frame-to-frame variation
        gg = got_grad - got_grad.mean(axis=0)
        tt = grad[order] - grad[order].mean(axis=0)
        rg, sg = _agreement(gg.ravel(), tt.ravel(), demean=False)
        print(f"thermal pattern r = {rp:.4f}, slope {sp:.3f}; per-frame gradients r = {rg:.4f}, slope {sg:.3f}")
        assert rp > 0.99 and abs(sp - 1) < 0.05, (rp, sp)
        assert rg > 0.99 and abs(sg - 1) < 0.05, (rg, sg)
        # The mosaic subtracts both terms at every observation, so the coadd of the
        # corrected frames is the solved sky (the same gauge: a common gradient),
        # and the true sky up to that plane.
        with fits.open(res.mosaic_paths[0]) as hdul:
            mos = np.asarray(hdul['MEAN_MAP'].data, dtype=np.float64)
            wt = np.asarray(hdul['MEAN_MAP_WEIGHT'].data, dtype=np.float64)
        ok = np.isfinite(mos) & (wt > 0) & np.isfinite(solved_sky)
        d = mos[ok] - solved_sky[ok]
        print(f"mosaic vs solved sky: rms {np.std(d - d.mean()):.2e} over {ok.sum()} px")
        assert np.std(d - d.mean()) < 0.01, np.std(d - d.mean())
        m, t = _truth_values(out, mos, ok, sky)
        py, px = np.nonzero(ok)
        resid = m - t
        from selfcal.geometry import wcs_helper
        ref_wcs, _ = wcs_helper.load_from_fits(os.path.join(out, 'toy_run', 'ref.fits'))
        tx, ty = _wcs(0, 0).world_to_pixel(ref_wcs.pixel_to_world(px, py))
        inside = (np.round(tx) >= 2) & (np.round(tx) < REF - 2) & (np.round(ty) >= 2) & (np.round(ty) < REF - 2)
        A = np.stack([np.ones(inside.sum()), px[inside], py[inside]], axis=1).astype(float)
        coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
        plane_resid = resid - A @ coef
        print(f"mosaic vs true sky, a plane removed: rms {plane_resid.std():.4f} (gradient amplitude ~0.3)")
        assert plane_resid.std() < 0.01, plane_resid.std()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# 4. data cubes whose slices are frames
# ============================================================================
def test_data_cube_slices():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_anytel_cube_')
    try:
        rng = np.random.default_rng(4)
        sky, line = _sky(), 1.0 + _pattern(4, 0.7)
        cm = rect_grid_chunk_map((H, W), *CHUNKS)
        waves = np.array([1.56, 1.58, 1.60, 1.62])
        sigma = 0.004 * (1.0 + 4.0 * np.mgrid[0:H, 0:W][1] / W)       # noise grows across the detector
        n_exp = 16
        exp_dir = os.path.join(tmp, 'exposures')
        os.makedirs(exp_dir)
        for k in range(n_exp):
            oy, ox = rng.integers(0, REF - H), rng.integers(0, REF - W)
            off = rng.normal(0, 0.2, cm.max() + 1)                   # shared by the exposure's slices
            cube, var = [], []
            for s, lam in enumerate(waves):
                cube.append(sky[oy:oy + H, ox:ox + W] + gaussian_line(lam) * line[oy:oy + H, ox:ox + W]
                            + (off - off.mean())[cm] + rng.normal(0, 1.0) + rng.normal(0, 1.0, (H, W)) * sigma)
                var.append(sigma ** 2)
            hdr = _wcs(ox, oy).to_header()
            for s, lam in enumerate(waves):
                hdr[f'WAVE{s}'] = lam
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.asarray(cube, np.float32), header=hdr, name='SCI'),
                          fits.ImageHDU(np.asarray(var, np.float32), name='VAR')]
                         ).writeto(os.path.join(exp_dir, f'toy_exp_{k:03d}.fits'), overwrite=True)
        model = {
            'variables': {'wave': {'header': 'WAVE'}, 'variance': {'layer': 'variance'}},
            'weight': {'variable': 'variance', 'function': 'tests.test_any_telescope:inverse_sigma'},
            'sky': [{'name': 'continuum'},
                    {'name': 'line', 'damp_weight': 1e-4,
                     'coefficient': {'variable': 'wave', 'function': 'tests.test_any_telescope:gaussian_line'}}],
            'offset': [{'kind': 'grouped', 'groups': 'exposure', 'reg_weight': 0.05, 'mean_zero': True}],
        }
        out, res = _run(tmp, _grid_inst(name='toy_ifu', tag='Cube'), model, n_exp, exp_dir)
        from selfcal.io.reproj import load_reproj_file
        frames = sorted(p for p in os.listdir(os.path.join(out, 'toy_run', 'reprojected')) if p.endswith('.h5'))
        assert len(frames) == 4 * n_exp
        f0 = load_reproj_file(os.path.join(out, 'toy_run', 'reprojected', frames[0]), ['layers/variance'])
        assert f0['layers/variance'] is not None
        with CalFile(res.cal_paths[0]) as cal:
            got, cov = cal.sky('line'), cal.sky_coverage('line')
            offs = cal.offsets[0]
            names = [os.path.basename(p) for p in cal.reproj_list]
        # slices of one exposure share their offsets
        exps = np.array([int(n.split('_')[1]) for n in names])
        for e in np.unique(exps)[:3]:
            same = offs[exps == e]
            assert np.allclose(same, same[0]), e
        g, t = _truth_values(out, got, (cov >= 8) & np.isfinite(got), line)
        r, slope = _agreement(g, t)
        print(f"cube slices: line map r = {r:.4f}, slope {slope:.3f} ({g.size} px)")
        assert r > 0.98 and abs(slope - 1) < 0.08, (r, slope)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# 5. frames written directly (no image, no WCS, no runner)
# ============================================================================
def test_frames_written_directly():
    from selfcal.geometry import wcs_helper
    from selfcal.io.frames import standard_frame_path, write_frame
    from selfcal.models.offset_model import Basis, OffsetBlock, OffsetModel
    from selfcal.models.sky_model import Coefficient, SkyModel
    from selfcal.models.variables import VariableSet
    from selfcal.pipeline.pipeline_wrapper import Calibrator, PipelineConfig
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_anytel_direct_')
    try:
        rng = np.random.default_rng(5)
        sky = _sky()
        n = 40
        gains = rng.normal(0, 0.05, n)
        scalars = rng.normal(0, 1.0, n)
        pc = PipelineConfig(output_dir=os.path.join(tmp, 'out'), run_name='direct', resolution_arcsec=PIX)
        os.makedirs(os.path.dirname(pc.ref_path), exist_ok=True)
        wcs_helper.save_to_fits(_wcs(0, 0), (REF, REF), pc.ref_path)
        frame_dir = os.path.join(tmp, 'frames')
        os.makedirs(frame_dir)
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        for k in range(n):
            oy, ox = int(rng.integers(0, REF - H)), int(rng.integers(0, REF - W))
            vals = sky[oy:oy + H, ox:ox + W] * (1 + gains[k]) + scalars[k] + rng.normal(0, 0.005, (H, W))
            write_frame(standard_frame_path(frame_dir, k, 0), vals, [oy, oy + H, ox, ox + W], np.stack([x, y]))
        # the per-frame gain multiplies a known sky (a sky-grid variable); the gains'
        # overall scale is degenerate with the sky's, fixed here by a prior Σ g = 0
        gain = OffsetBlock(chunk_map=np.zeros((H, W), dtype=np.int32),
                           basis=Basis(Coefficient('S0', identity), 1))

        def gains_sum_to_zero(info):
            cols = np.arange(info.layout.offset_slice(0).start, info.layout.offset_slice(0).stop)
            return np.zeros(cols.size, dtype=np.int64), cols, np.full(cols.size, 10.0), np.zeros(1)

        cc = Calibrator(pc, reproj_dir=frame_dir)
        cc.setup_lsqr(offset_model=OffsetModel([gain], use_per_frame_scalar=True),
                      grid_valid_weight=np.ones((H, W), dtype=np.float32), sky_model=SkyModel.continuum_only(),
                      variables=VariableSet(sky={'S0': sky}), priors=[gains_sum_to_zero],
                      apply_mask=False, apply_weight=False, outlier_thresh=None, max_workers=2, batch_size=4,
                      weighted_damping=True, damp_weight=1e-6)
        from selfcal.core.solution import compute_x0_scalar_only
        x0 = compute_x0_scalar_only(cc.A, cc.b, cc.ref_shape, scalar_col_start=cc.col_bases[1],
                                    num_sky_blocks=1, active_mask=getattr(cc, 'active_mask', None))
        cc.apply_lsqr(x0=x0, atol=1e-12, btol=1e-12, iter_lim=800, n_threads=2, damp=0)
        path = cc.save_calibration(cal_dir=os.path.join(tmp, 'cal'), cal_file='cal_direct.h5')
        with CalFile(path) as cal:
            got = cal.offsets[0][:, 0]
            order = [int(os.path.basename(p).split('_')[1]) for p in cal.reproj_list]
        r, slope = _agreement(got, gains[order])
        print(f"frames written directly: per-frame gains r = {r:.4f}, slope {slope:.3f}")
        assert r > 0.99 and abs(slope - 1) < 0.05, (r, slope)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# 6. amplifier crosstalk (a variable computed from the frame's own data)
# ============================================================================
def test_amplifier_ghost():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_anytel_ghost_')
    try:
        rng = np.random.default_rng(6)
        sky = _sky()
        cm = rect_grid_chunk_map((H, W), *CHUNKS)
        n_chunk = int(cm.max()) + 1
        ny, nx = CHUNKS
        ids = np.arange(n_chunk)
        partner = (ids // nx) * nx + (nx - 1 - ids % nx)
        eps = rng.uniform(0.01, 0.05, n_chunk)                     # leak fraction into each amplifier
        n_exp = 40
        exp_dir = os.path.join(tmp, 'exposures')
        os.makedirs(exp_dir)
        for k in range(n_exp):
            oy, ox = rng.integers(0, REF - H), rng.integers(0, REF - W)
            clean = sky[oy:oy + H, ox:ox + W] + rng.normal(0, 1.0) + rng.normal(0, 0.01, (H, W))
            c = np.bincount(cm.ravel(), weights=clean.ravel(), minlength=n_chunk) / np.bincount(cm.ravel())
            # the chunk means of the final data solve m = c + eps * m[partner] (pairwise)
            m = (c + eps * c[partner]) / (1.0 - eps * eps[partner])
            img = clean + (eps * m[partner])[cm]
            _write_image(os.path.join(exp_dir, f'toy_exp_{k:03d}.fits'), img, ox, oy, {})
        model = {
            'variables': {'partner': {'frame_function': 'tests.test_any_telescope:partner_mean'}},
            'sky': [{'name': 'continuum'}],
            'offset': [{'kind': 'fixed', 'name': 'ghost', 'coefficient': {'variable': 'partner'}}],
        }
        out, res = _run(tmp, _grid_inst(), model, n_exp, exp_dir)
        with CalFile(res.cal_paths[0]) as cal:
            got = cal.offsets[0][0]
        r, slope = _agreement(got, eps, demean=False)
        print(f"amplifier ghost: leak fractions r = {r:.4f}, slope {slope:.3f}; "
              f"max |error| {np.max(np.abs(got - eps)):.2e}")
        assert r > 0.99 and abs(slope - 1) < 0.05, (r, slope)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# 7. two detectors of different sizes on one focal plane
# ============================================================================
def test_heterogeneous_focal_plane():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_anytel_fp_')
    try:
        rng = np.random.default_rng(7)
        sky = _sky()
        cm = _focal_plane_chunks()
        n_chunk = int(cm.max()) + 1
        readout = rng.normal(0, 0.2, n_chunk)                     # fixed pattern per amplifier
        for det in (0, 1):
            sel = np.arange(4) + 4 * det
            readout[sel] -= readout[sel].mean()
        n_exp = 40
        exp_dir = os.path.join(tmp, 'exposures')
        os.makedirs(exp_dir)
        fh, fw = FP_SHAPE
        for k in range(n_exp):
            oy, ox = rng.integers(0, REF - fh), rng.integers(0, REF - fw)
            hdus = [fits.PrimaryHDU()]
            for det in (0, 1):
                x0, w = FP_ORIGIN[det], FP_WIDTH[det]
                img = (sky[oy:oy + fh, ox + x0:ox + x0 + w] + readout[cm[:, x0:x0 + w]]
                       + rng.normal(0, 1.0) + rng.normal(0, 0.01, (fh, w)))
                wcs = WCS(naxis=2)
                wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
                wcs.wcs.crpix = [W / 2 + 0.5 - (ox + x0), H / 2 + 0.5 - oy]
                wcs.wcs.crval = [180.0, 30.0]
                wcs.wcs.cdelt = [-PIX / 3600.0, PIX / 3600.0]
                hdr = wcs.to_header()
                hdr['DETID'] = det
                hdus.append(fits.ImageHDU(img.astype(np.float32), header=hdr))
            fits.HDUList(hdus).writeto(os.path.join(exp_dir, f'toy_exp_{k:03d}.fits'), overwrite=True)
        # Each detector's frames share one pattern over the focal-plane map but observe
        # only their own amplifiers: a mean-zero anchor over the whole map would include
        # the other detector's (unobserved, free) chunks and fix nothing, so the constant
        # per detector (degenerate with the frames' scalars) is set by damping, which acts
        # on observed unknowns only.
        model = {
            'sky': [{'name': 'continuum'}],
            'offset': [{'map': 'amps', 'kind': 'grouped', 'groups': 'detector', 'damp': 1e-3}],
        }
        out, res = _run(tmp, {'name': 'toy_focal_plane', 'detector_shape': list(FP_SHAPE)}, model, n_exp,
                        exp_dir)
        with CalFile(res.cal_paths[0]) as cal:
            offs = cal.offsets[0]
            dets = np.array([int(os.path.basename(p).split('_')[3][:2]) for p in cal.reproj_list])
        # detector d's frames share one pattern; its own amplifiers carry it
        got = np.concatenate([offs[dets == d][0][4 * d:4 * d + 4] - offs[dets == d][0][4 * d:4 * d + 4].mean()
                              for d in (0, 1)])
        r, slope = _agreement(got, readout, demean=False)
        print(f"heterogeneous focal plane: read-out patterns r = {r:.4f}, slope {slope:.3f}")
        assert r > 0.99 and abs(slope - 1) < 0.05, (r, slope)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    import time
    for fn in (test_frames_written_directly, test_thermal_pattern_and_gradients, test_time_variable_sky,
               test_imaging_polarimeter, test_data_cube_slices, test_amplifier_ghost,
               test_heterogeneous_focal_plane):
        t0 = time.time()
        fn()
        print(f"  {fn.__name__}: OK ({time.time() - t0:.0f} s)")
