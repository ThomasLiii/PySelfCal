"""Any telescope, any model — each adapted with high-level functions only.

Every example is a different telescope or model, calibrated end to end on
synthetic data through the Python API, and none of them changes selfcal: each
is an instrument (``sc.Camera``, or a small ``sc.Instrument`` subclass where the
telescope has a file layout or a chunk geometry of its own), a model
(``sc.Model``) and functions defined in THIS file (coefficients, bases,
per-frame values, a reader, a prior), run with ``field.reproject`` and
``field.calibrate``.

1. ``test_time_variable_sky`` — a sky term modulated with the season
   (``sin 2π(t − t0)/yr``, ``t`` a header keyword of each exposure), offsets that
   drift slowly in time with a temporal-smoothness prior.
2. ``test_imaging_polarimeter`` — Stokes I, Q, U maps: ``I + Q cos 2ψ + U sin 2ψ``
   with ``ψ`` a function of a per-frame half-wave-plate angle (a header keyword
   the camera reads) and a per-pixel polariser angle (a detector map of the camera).
3. ``test_thermal_pattern_and_gradients`` — a detector-fixed pattern scaled by
   each frame's temperature (an offset times a function of a frame variable)
   plus a free 2-D gradient per frame (an offset basis in the built-in detector
   coordinates); the mosaic subtracts both at every observation.
4. ``test_data_cube_slices`` — an integral-field-like cube: the instrument's
   reader turns the slices of each file into frames with a per-slice wavelength
   and a variance plane; the model fits a line map (a Gaussian of the slice
   wavelength), shares the offsets over the slices of one exposure (grouping by
   a frame variable) and weights every observation by its inverse sigma (a
   stored layer).
5. ``test_frames_written_directly`` — samples that never were a FITS image:
   frames written with ``write_frame`` and calibrated where they are; a
   per-frame gain against a known sky (an offset times a sky-grid variable)
   with the gains' scale fixed by a prior written here.
6. ``test_amplifier_ghost`` — each amplifier picks up a fraction of the mean
   signal of its mirror amplifier (crosstalk): the variable is computed by a
   function of the frame's own stored data, the fractions are a shared offset
   term with that variable as its coefficient.
7. ``test_heterogeneous_focal_plane`` — two detectors of different sizes on one
   focal plane: the instrument's reader places each detector's pixels on the
   focal-plane coordinates, its chunk map spans the focal plane, and each
   detector's read-out pattern is an offset term grouped by the detector.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal.geometry import wcs_helper  # noqa: E402
from selfcal.instruments.grid import rect_grid_chunk_map  # noqa: E402
from selfcal.io.frames import ExposureData, standard_frame_path, write_frame  # noqa: E402
from selfcal.io.reproj import load_reproj_file, parse_reproj_basename  # noqa: E402
from selfcal.models.offset_structure import ChunkAxes  # noqa: E402

H, W = 48, 64                     # detector (rows, cols)
REF = 160                         # side of the truth sky grid
PIX = 20.0                        # arcsec per pixel (detector = reference)
CHUNKS = (3, 4)

# What the examples share: how a model is fitted, coadded and summed, and the machine.
FIT = sc.Fit(500, clip=8.0, use_mask=False, tolerance=1e-12)          # LSQR; an 8-sigma clip; no DQ mask
COADD = sc.Coadd(clip=None, use_mask=False)                            # mean and std maps, no clip
NUMERICS = sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4)      # two solver threads, small batches
TWO_WORKERS = sc.Compute(workers=2)                                    # frames read in place

sc.set_progress(False)


# ============================================================================
# synthetic data, and how a solution is compared with the truth
# ============================================================================
def true_sky():
    """The sky every example observes, on the truth grid (``REF`` x ``REF`` pixels)."""
    y, x = np.mgrid[0:REF, 0:REF].astype(np.float64)
    return 5.0 + 0.02 * x + 0.01 * y + 0.5 * np.sin(x / 11.0) * np.cos(y / 13.0)


def sky_pattern(seed, scale=1.0):
    """A wavy map on the truth grid: the signal of a sky term other than the constant one."""
    y, x = np.mgrid[0:REF, 0:REF].astype(np.float64)
    ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, 2)
    return scale * np.sin(x / (5.0 + seed) + ph[0]) * np.cos(y / (6.5 + seed) + ph[1])


def pointing_wcs(ox, oy):
    """The WCS of a detector whose pixel ``(0, 0)`` sees truth pixel ``(oy, ox)``
    (``pointing_wcs(0, 0)`` is the truth grid's own)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [W / 2 + 0.5 - ox, H / 2 + 0.5 - oy]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.cdelt = [-PIX / 3600.0, PIX / 3600.0]
    return w


def write_exposure(path, image, ox, oy, keywords):
    """A FITS exposure: the science image with its WCS and header ``keywords`` in extension 1."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = pointing_wcs(ox, oy).to_header()
    for k, v in keywords.items():
        hdr[k] = v
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.asarray(image, np.float32), header=hdr, name='SCI')]
                 ).writeto(path, overwrite=True)


def truth_pixels(field, ok):
    """The reference pixels ``ok`` of the field that fall inside the truth grid (matched through
    the WCS): their reference indices ``(py, px)`` and truth indices ``(ty, tx)``."""
    ref_wcs, _ = field.reference()
    py, px = np.nonzero(ok)
    tx, ty = pointing_wcs(0, 0).world_to_pixel(ref_wcs.pixel_to_world(px, py))
    tx, ty = np.round(tx).astype(int), np.round(ty).astype(int)
    inside = (tx >= 2) & (tx < REF - 2) & (ty >= 2) & (ty < REF - 2)
    return (py[inside], px[inside]), (ty[inside], tx[inside])


def against_truth(field, got, ok, truth):
    """``(got, truth)`` at the reference pixels ``ok`` that fall inside the truth grid."""
    ref_idx, truth_idx = truth_pixels(field, ok)
    return got[ref_idx], truth[truth_idx]


def agreement(a, b, demean=True):
    """Pearson r and slope of ``a`` against ``b`` (means removed: a constant is a
    gauge of the solve — degenerate with the per-frame scalars)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if demean:
        a, b = a - a.mean(), b - b.mean()
    return float(np.corrcoef(a, b)[0, 1]), float(np.dot(a, b) / np.dot(b, b))


def frame_ids(cal):
    """``(exposure, detector)``: the indices of every frame of a cal file, in its order."""
    exposure, detector = np.array([parse_reproj_basename(p) for p in cal.reproj_list]).T
    return exposure, detector


# ============================================================================
# 1. a time-variable sky
# ============================================================================
def annual(time, t0=60000.0, period=365.25):
    """A seasonal modulation of the time (days): ``sin 2π(t − t0)/P``."""
    return np.sin(2 * np.pi * (np.asarray(time, dtype=np.float64) - t0) / period)


def test_time_variable_sky(tmp_path):
    """A camera whose exposures record their time (header ``MJD-AVG``): a constant sky plus a sky
    term times ``annual(time)``; offsets per frame and chunk that drift slowly, pulled together
    along the time by a smoothness prior."""
    rng = np.random.default_rng(1)
    sky, s_annual = true_sky(), sky_pattern(1, 0.8)
    cm = rect_grid_chunk_map((H, W), *CHUNKS)
    n_exp = 48
    mjd = 60000.0 + np.sort(rng.uniform(0, 365.25, n_exp))
    # chunk offsets that drift slowly in time (what the temporal prior expects)
    drift = 0.3 * np.sin(2 * np.pi * (mjd[:, None] - 60000.0) / 200.0 + rng.uniform(0, 6, cm.max() + 1))
    drift -= drift.mean(axis=1, keepdims=True)
    for k in range(n_exp):
        oy, ox = rng.integers(0, REF - H), rng.integers(0, REF - W)
        img = (sky[oy:oy + H, ox:ox + W] + annual(mjd[k]) * s_annual[oy:oy + H, ox:ox + W]
               + drift[k][cm] + rng.normal(0, 1.0) + rng.normal(0, 0.01, (H, W)))
        write_exposure(tmp_path / 'exposures' / f'toy_exp_{k:03d}.fits', img, ox, oy, {'MJD-AVG': mjd[k]})

    field = sc.Field(tmp_path / 'toy_run', sc.Camera((H, W), chunks=CHUNKS, tag='Toy'), PIX, compute=TWO_WORKERS)
    field.reproject(tmp_path / 'exposures' / 'toy_exp_*.fits', method='interp', padding=8)
    model = sc.Model(sky=[sc.Sky(damping=1e-4), sc.Sky('annual', times=annual, damping=1e-4)],
                     offsets=[sc.Offsets('chunks', smooth=0.05, mean_zero=True)],
                     variables={'time': sc.Header('MJD-AVG')},
                     priors=[sc.priors.frame_smoothness('chunks', 'time', weight=0.2)])
    result = field.calibrate(sc.Recipe(model, fit=FIT, coadd=None, numerics=NUMERICS))

    with result.cal() as cal:
        got, cov = cal.sky('annual'), cal.sky_coverage('annual')
        offs = cal.offsets[0]
        exposure, _ = frame_ids(cal)
    g, t = against_truth(field, got, (cov >= 6) & np.isfinite(got), s_annual)
    r, slope = agreement(g, t)
    ro, _ = agreement(offs.ravel(), drift[exposure].ravel())
    print(f"time-variable sky: annual map r = {r:.4f}, slope {slope:.3f} ({g.size} px); drifting offsets r = {ro:.3f}")
    assert r > 0.99 and abs(slope - 1) < 0.05, (r, slope)
    assert ro > 0.95, ro


# ============================================================================
# 2. an imaging polarimeter
# ============================================================================
def pixel_polariser_angles():
    """The polariser angle of every pixel (radians): varies across the detector."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    return (np.deg2rad(60.0) * x / W + np.deg2rad(20.0) * y / H).astype(np.float32)


def polariser_angle(hwp, pix_angle):
    """Angle of the polarisation a pixel analyses: its polariser angle (``pix_angle``, radians),
    rotated by twice the half-wave plate's angle (``hwp``, degrees)."""
    return 2.0 * np.deg2rad(np.asarray(hwp, dtype=np.float64)) + np.asarray(pix_angle, dtype=np.float64)


def cos2(psi):
    """Q's coefficient: ``cos 2ψ``."""
    return np.cos(2.0 * psi)


def sin2(psi):
    """U's coefficient: ``sin 2ψ``."""
    return np.sin(2.0 * psi)


def test_imaging_polarimeter(tmp_path):
    """A camera whose pixels carry polarisers (a detector map, ``pix_angle``) and whose exposures
    record a half-wave-plate angle (header ``HWPANG``, the frame variable ``hwp``): Stokes I, Q
    and U maps, ``I + Q cos 2ψ + U sin 2ψ`` with ``ψ`` a function of both."""
    rng = np.random.default_rng(2)
    sky_i, sky_q, sky_u = true_sky(), sky_pattern(2, 0.6), sky_pattern(3, 0.6)
    pix = pixel_polariser_angles()
    cm = rect_grid_chunk_map((H, W), *CHUNKS)
    n_exp = 48
    for k in range(n_exp):
        hwp = [0.0, 22.5, 45.0, 67.5][k % 4]
        oy, ox = rng.integers(0, REF - H), rng.integers(0, REF - W)
        psi = polariser_angle(hwp, pix)
        off = rng.normal(0, 0.2, cm.max() + 1)
        img = (sky_i[oy:oy + H, ox:ox + W] + sky_q[oy:oy + H, ox:ox + W] * np.cos(2 * psi)
               + sky_u[oy:oy + H, ox:ox + W] * np.sin(2 * psi) + (off - off.mean())[cm]
               + rng.normal(0, 1.0) + rng.normal(0, 0.01, (H, W)))
        write_exposure(tmp_path / 'exposures' / f'toy_exp_{k:03d}.fits', img, ox, oy, {'HWPANG': hwp})

    polarimeter = sc.Camera((H, W), chunks=CHUNKS, detector_maps={'pix_angle': pix}, headers={'hwp': 'HWPANG'},
                            tag='Toy')
    field = sc.Field(tmp_path / 'toy_run', polarimeter, PIX, compute=TWO_WORKERS)
    field.reproject(tmp_path / 'exposures' / 'toy_exp_*.fits', method='interp', padding=8)
    model = sc.Model(sky=[sc.Sky('I', damping=1e-4), sc.Sky('Q', times=cos2, damping=1e-4),
                          sc.Sky('U', times=sin2, damping=1e-4)],
                     offsets=[sc.Offsets(smooth=0.05, mean_zero=True)],
                     variables={'psi': sc.Derived(polariser_angle)})
    result = field.calibrate(sc.Recipe(model, fit=FIT, coadd=None, numerics=NUMERICS))

    with result.cal() as cal:
        assert cal.sky_names == ['I', 'Q', 'U']
        maps = {n: (cal.sky(n), cal.sky_coverage(n)) for n in ('Q', 'U')}
    for name, truth in (('Q', sky_q), ('U', sky_u)):
        got, cov = maps[name]
        g, t = against_truth(field, got, (cov >= 6) & np.isfinite(got), truth)
        r, slope = agreement(g, t)
        print(f"polarimeter: {name} r = {r:.4f}, slope {slope:.3f} ({g.size} px)")
        assert r > 0.99 and abs(slope - 1) < 0.05, (name, r, slope)


# ============================================================================
# 3. a thermal pattern and per-frame gradients (an offset coefficient and a basis)
# ============================================================================
def centred(temperature, t0=80.0):
    """A thermal offset scales with the temperature above t0."""
    return np.asarray(temperature, dtype=np.float64) - t0


def plane(det_x, det_y):
    """A 2-D gradient basis in detector coordinates (two functions)."""
    return [(np.asarray(det_x) - W / 2) / W, (np.asarray(det_y) - H / 2) / H]


def test_thermal_pattern_and_gradients(tmp_path):
    """A camera whose exposures record a temperature (header ``TEMP``): a detector-fixed pattern
    shared by every frame, times ``centred(temperature)``, and a free gradient per frame over the
    whole detector (two basis functions of the detector coordinates); the mosaic subtracts both."""
    rng = np.random.default_rng(3)
    sky = true_sky()
    cm = rect_grid_chunk_map((H, W), *CHUNKS)
    pattern = rng.normal(0, 0.05, cm.max() + 1)
    pattern -= pattern.mean()
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

    field = sc.Field(tmp_path / 'toy_run', sc.Camera((H, W), chunks=CHUNKS, tag='Toy'), PIX, compute=TWO_WORKERS)
    field.reproject(tmp_path / 'exposures' / 'toy_exp_*.fits', method='interp', padding=8)
    model = sc.Model(sky=[sc.Sky(damping=1e-4)],
                     offsets=[sc.Offsets('thermal', per='all', times=centred, mean_zero=True),
                              sc.Offsets('gradient', on='detector', basis=plane, n=2)],
                     variables={'temperature': sc.Header('TEMP')})
    result = field.calibrate(sc.Recipe(model, fit=FIT, coadd=COADD, numerics=NUMERICS))

    with result.cal() as cal:
        got_pattern = cal.offsets[0][0]
        got_grad = cal.offsets[1]
        solved_sky = cal.sky(0)
        exposure, _ = frame_ids(cal)
        assert cal.offset_basis(1)[0] == 2 and cal.offset_basis(0)[0] == 1
    rp, sp = agreement(got_pattern, pattern)
    # a gradient common to every frame is a gauge (a sky gradient + per-frame scalars
    # reproduce it): compare each direction's frame-to-frame variation
    gg = got_grad - got_grad.mean(axis=0)
    tt = grad[exposure] - grad[exposure].mean(axis=0)
    rg, sg = agreement(gg.ravel(), tt.ravel(), demean=False)
    print(f"thermal pattern r = {rp:.4f}, slope {sp:.3f}; per-frame gradients r = {rg:.4f}, slope {sg:.3f}")
    assert rp > 0.99 and abs(sp - 1) < 0.05, (rp, sp)
    assert rg > 0.99 and abs(sg - 1) < 0.05, (rg, sg)
    # The mosaic subtracts both terms at every observation, so the coadd of the
    # corrected frames is the solved sky (the same gauge: a common gradient),
    # and the true sky up to that plane.
    with result.mosaic() as mos:
        mean = np.asarray(mos.mean, dtype=np.float64)
        weight = np.asarray(mos.weight('MEAN_MAP'), dtype=np.float64)
    ok = np.isfinite(mean) & (weight > 0) & np.isfinite(solved_sky)
    d = mean[ok] - solved_sky[ok]
    print(f"mosaic vs solved sky: rms {np.std(d - d.mean()):.2e} over {ok.sum()} px")
    assert np.std(d - d.mean()) < 0.01, np.std(d - d.mean())
    (py, px), truth_idx = truth_pixels(field, ok)
    resid = mean[py, px] - sky[truth_idx]
    A = np.stack([np.ones(px.size), px, py], axis=1).astype(float)
    coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
    plane_resid = resid - A @ coef
    print(f"mosaic vs true sky, a plane removed: rms {plane_resid.std():.4f} (gradient amplitude ~0.3)")
    assert plane_resid.std() < 0.01, plane_resid.std()


# ============================================================================
# 4. data cubes whose slices are frames
# ============================================================================
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


@dataclass(frozen=True, kw_only=True)
class ToyCube(sc.Instrument):
    """An integral-field unit whose files hold cubes: four slices of one exposure, each read
    as a frame by ``read_cube_slice``; rectangular chunks, as a camera's."""
    tag: str = 'Cube'

    def geometry(self, oversample):
        return sc.Geometry((H, W), oversample, maps=[sc.ChunkMap.rectangles('grid', (H, W), CHUNKS)])

    def layout(self):
        return sc.ExposureLayout(sci_ext=[0, 1, 2, 3], dq_ext=None, detector_ids=[0, 1, 2, 3], ref_use_ext=(0,),
                                 reader=read_cube_slice)


def gaussian_line(wave, center=1.60, sigma=0.02):
    """The line map's coefficient: a Gaussian of the slice's wavelength (µm)."""
    return np.exp(-0.5 * ((np.asarray(wave, dtype=np.float64) - center) / sigma) ** 2)


def inverse_sigma(variance):
    """The observation weight: ``1/σ``, from the stored variance."""
    return 1.0 / np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64), 1e-12))


def test_data_cube_slices(tmp_path):
    """Cubes read slice by slice (``ToyCube``): a constant sky plus a line map times a Gaussian of
    the slice's wavelength (the frame variable ``wave``); offsets shared by the slices of one
    exposure; every observation weighted by its inverse sigma, from the variance layer the reader
    stores with each frame (the mosaic weights by its square)."""
    rng = np.random.default_rng(4)
    sky, line = true_sky(), 1.0 + sky_pattern(4, 0.7)
    cm = rect_grid_chunk_map((H, W), *CHUNKS)
    waves = np.array([1.56, 1.58, 1.60, 1.62])
    sigma = 0.004 * (1.0 + 4.0 * np.mgrid[0:H, 0:W][1] / W)       # noise grows across the detector
    n_exp = 16
    os.makedirs(tmp_path / 'exposures')
    pointings = []
    for k in range(n_exp):
        oy, ox = rng.integers(0, REF - H), rng.integers(0, REF - W)
        pointings.append((int(oy), int(ox)))
        off = rng.normal(0, 0.2, cm.max() + 1)                   # shared by the exposure's slices
        cube, var = [], []
        for s, lam in enumerate(waves):
            cube.append(sky[oy:oy + H, ox:ox + W] + gaussian_line(lam) * line[oy:oy + H, ox:ox + W]
                        + (off - off.mean())[cm] + rng.normal(0, 1.0) + rng.normal(0, 1.0, (H, W)) * sigma)
            var.append(sigma ** 2)
        hdr = pointing_wcs(ox, oy).to_header()
        for s, lam in enumerate(waves):
            hdr[f'WAVE{s}'] = lam
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.asarray(cube, np.float32), header=hdr, name='SCI'),
                      fits.ImageHDU(np.asarray(var, np.float32), name='VAR')]
                     ).writeto(tmp_path / 'exposures' / f'toy_exp_{k:03d}.fits', overwrite=True)

    field = sc.Field(tmp_path / 'toy_run', ToyCube(), PIX, compute=TWO_WORKERS)
    field.reproject(tmp_path / 'exposures' / 'toy_exp_*.fits', method='interp', padding=8)
    model = sc.Model(sky=[sc.Sky(damping=1e-4), sc.Sky('line', times=gaussian_line, damping=1e-4)],
                     offsets=[sc.Offsets(per='exposure', smooth=0.05, mean_zero=True)],
                     variables={'wave': sc.Header('WAVE'), 'variance': sc.Layer()},
                     weight=inverse_sigma)
    result = field.calibrate(sc.Recipe(model, fit=FIT, coadd=COADD, numerics=NUMERICS))

    assert len(field.frames) == 4 * n_exp
    assert load_reproj_file(field.frames[0], ['layers/variance'])['layers/variance'] is not None
    with result.cal() as cal:
        got, cov = cal.sky('line'), cal.sky_coverage('line')
        offs = cal.offsets[0]
        exposure, _ = frame_ids(cal)
    # slices of one exposure share their offsets
    for e in np.unique(exposure)[:3]:
        same = offs[exposure == e]
        assert np.allclose(same, same[0]), e
    g, t = against_truth(field, got, (cov >= 8) & np.isfinite(got), line)
    r, slope = agreement(g, t)
    print(f"cube slices: line map r = {r:.4f}, slope {slope:.3f} ({g.size} px)")
    assert r > 0.98 and abs(slope - 1) < 0.08, (r, slope)
    # The mosaic weights every observation by the model's weight squared (1/σ²,
    # σ read from the stored variance layer): a pixel's coadd weight is the sum
    # of 1/σ² over the slices covering it.
    with result.mosaic() as mos:
        wmap = np.asarray(mos.weight('MEAN_MAP'), dtype=np.float64)
    expected = np.zeros((REF, REF))
    inv_var = 1.0 / sigma ** 2
    for oy, ox in pointings:
        expected[oy:oy + H, ox:ox + W] += len(waves) * inv_var
    m, e = against_truth(field, wmap, wmap > 0, expected)
    inner = e > 0
    ratio = m[inner] / e[inner]
    print(f"cube mosaic weights / Σ 1/σ²: median {np.median(ratio):.4f}, 5-95% "
          f"{np.percentile(ratio, 5):.4f}..{np.percentile(ratio, 95):.4f}")
    assert abs(np.median(ratio) - 1) < 0.01, np.median(ratio)


# ============================================================================
# 5. frames written directly (no image, no WCS, no reprojection)
# ============================================================================
def known_sky(ref_wcs, ref_shape):
    """The sky the gains multiply, on the reference grid (made here as the truth grid itself)."""
    return true_sky()


def gains_sum_to_zero(gain):
    """A prior's rows (the contract of :mod:`selfcal.models.priors`): one row, the per-frame
    gains sum to zero. Their overall scale is degenerate with the sky's; this fixes it."""
    cols = gain.col_base + np.arange(gain.size)
    return np.zeros(cols.size, dtype=np.int64), cols, np.ones(cols.size), np.zeros(1)


def test_frames_written_directly(tmp_path):
    """Samples that never were a FITS image: each frame file is written with ``write_frame`` (its
    values on a box of the reference grid and their detector coordinates) and the field is
    calibrated from those files where they are. The model: one gain per frame over the whole
    detector times a known sky (a sky-grid variable), the per-frame scalar, and a prior written
    here that fixes the gains' scale."""
    rng = np.random.default_rng(5)
    sky = true_sky()
    n = 40
    gains = rng.normal(0, 0.05, n)
    scalars = rng.normal(0, 1.0, n)
    field = sc.Field(tmp_path / 'toy_run', sc.Camera((H, W), tag='Toy'), PIX, compute=TWO_WORKERS)
    # the reference grid the frames' boxes refer to (no reprojection makes one here)
    os.makedirs(field.path)
    wcs_helper.save_to_fits(pointing_wcs(0, 0), (REF, REF), os.path.join(field.path, 'ref.fits'))
    frames = tmp_path / 'frames'
    os.makedirs(frames)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    for k in range(n):
        oy, ox = int(rng.integers(0, REF - H)), int(rng.integers(0, REF - W))
        vals = sky[oy:oy + H, ox:ox + W] * (1 + gains[k]) + scalars[k] + rng.normal(0, 0.005, (H, W))
        write_frame(standard_frame_path(frames, k, 0), vals, [oy, oy + H, ox, ox + W], np.stack([x, y]))

    model = sc.Model(sky=[sc.Sky(damping=1e-6)],
                     offsets=[sc.Offsets('gain', on='detector', times='known_sky')],
                     variables={'known_sky': sc.SkyMap(known_sky)},
                     priors=[sc.Prior(gains_sum_to_zero, 'gain', weight=10.0)])
    fit = sc.Fit(800, clip=None, use_mask=False, tolerance=1e-12, method='lsmr', float32=False)
    result = field.calibrate(sc.Recipe(model, fit=fit, coadd=None, numerics=sc.Numerics(2, batch=4)), frames=frames)

    with result.cal() as cal:
        got = cal.offsets[0][:, 0]
        exposure, _ = frame_ids(cal)
    r, slope = agreement(got, gains[exposure])
    print(f"frames written directly: per-frame gains r = {r:.4f}, slope {slope:.3f}")
    assert r > 0.99 and abs(slope - 1) < 0.05, (r, slope)


# ============================================================================
# 6. amplifier crosstalk (a variable computed from the frame's own data)
# ============================================================================
def mirror_chunk(chunk):
    """The partner amplifier of each chunk: the chunk mirrored across the detector's middle
    column (same chunk row)."""
    nx = CHUNKS[1]
    return (chunk // nx) * nx + (nx - 1 - chunk % nx)


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
    out = np.zeros(raw.shape)
    out[inside] = means[mirror_chunk(np.arange(n))][chunk]
    return out


def test_amplifier_ghost(tmp_path):
    """A camera whose amplifiers (the chunk columns) each pick up a fraction of the signal of
    their mirror amplifier: the variable ``partner`` is computed from each frame's own data
    (``partner_mean``), and the fractions are an offset shared by every frame, times it."""
    rng = np.random.default_rng(6)
    sky = true_sky()
    cm = rect_grid_chunk_map((H, W), *CHUNKS)
    n_chunk = int(cm.max()) + 1
    partner = mirror_chunk(np.arange(n_chunk))
    eps = rng.uniform(0.01, 0.05, n_chunk)                     # leak fraction into each amplifier
    n_exp = 40
    for k in range(n_exp):
        oy, ox = rng.integers(0, REF - H), rng.integers(0, REF - W)
        clean = sky[oy:oy + H, ox:ox + W] + rng.normal(0, 1.0) + rng.normal(0, 0.01, (H, W))
        c = np.bincount(cm.ravel(), weights=clean.ravel(), minlength=n_chunk) / np.bincount(cm.ravel())
        # the chunk means of the final data solve m = c + eps * m[partner] (pairwise)
        m = (c + eps * c[partner]) / (1.0 - eps * eps[partner])
        img = clean + (eps * m[partner])[cm]
        write_exposure(tmp_path / 'exposures' / f'toy_exp_{k:03d}.fits', img, ox, oy, {})

    field = sc.Field(tmp_path / 'toy_run', sc.Camera((H, W), chunks=CHUNKS, tag='Toy'), PIX, compute=TWO_WORKERS)
    field.reproject(tmp_path / 'exposures' / 'toy_exp_*.fits', method='interp', padding=8)
    model = sc.Model(sky=[sc.Sky(damping=1e-4)],
                     offsets=[sc.Offsets('ghost', per='all', times='partner')],
                     variables={'partner': sc.FrameFunction(partner_mean)})
    result = field.calibrate(sc.Recipe(model, fit=FIT, coadd=None, numerics=NUMERICS))

    got = result.offsets('ghost')[0]
    r, slope = agreement(got, eps, demean=False)
    print(f"amplifier ghost: leak fractions r = {r:.4f}, slope {slope:.3f}; "
          f"max |error| {np.max(np.abs(got - eps)):.2e}")
    assert r > 0.99 and abs(slope - 1) < 0.05, (r, slope)


# ============================================================================
# 7. two detectors of different sizes on one focal plane
# ============================================================================
FP_SHAPE = (40, 72)               # detector A (40 x 40) at x 0..39, detector B (40 x 24) at x 48..71
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


def focal_plane_chunks():
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


@dataclass(frozen=True, kw_only=True)
class ToyFocalPlane(sc.Instrument):
    """Two detectors of different sizes on one focal plane, both in every exposure file (read by
    ``read_focal_plane``): one chunk map over the focal plane, four amplifiers per detector."""
    tag: str = 'FocalPlane'

    def geometry(self, oversample):
        cm = focal_plane_chunks()
        axes = ChunkAxes.row_major(('det', 'row', 'col'), (2, 2, 2), ('x', 'y', 'x'))   # id = 4 det + 2 row + col
        return sc.Geometry(FP_SHAPE, oversample, maps=[sc.ChunkMap('amps', cm, cm, axes=axes)])

    def layout(self):
        return sc.ExposureLayout(sci_ext=[1, 2], dq_ext=None, detector_ids=[0, 1], ref_use_ext=(1, 2),
                                 reader=read_focal_plane)


def test_heterogeneous_focal_plane(tmp_path):
    """Two detectors of different sizes on one focal plane (``ToyFocalPlane``): a read-out pattern
    per amplifier, an offset on the focal-plane chunk map shared by the frames of each detector."""
    rng = np.random.default_rng(7)
    sky = true_sky()
    cm = focal_plane_chunks()
    n_chunk = int(cm.max()) + 1
    readout = rng.normal(0, 0.2, n_chunk)                     # fixed pattern per amplifier
    for det in (0, 1):
        sel = np.arange(4) + 4 * det
        readout[sel] -= readout[sel].mean()
    n_exp = 40
    os.makedirs(tmp_path / 'exposures')
    fh, fw = FP_SHAPE
    for k in range(n_exp):
        oy, ox = rng.integers(0, REF - fh), rng.integers(0, REF - fw)
        hdus = [fits.PrimaryHDU()]
        for det in (0, 1):
            x0, w = FP_ORIGIN[det], FP_WIDTH[det]
            img = (sky[oy:oy + fh, ox + x0:ox + x0 + w] + readout[cm[:, x0:x0 + w]]
                   + rng.normal(0, 1.0) + rng.normal(0, 0.01, (fh, w)))
            hdr = pointing_wcs(ox + x0, oy).to_header()
            hdr['DETID'] = det
            hdus.append(fits.ImageHDU(img.astype(np.float32), header=hdr))
        fits.HDUList(hdus).writeto(tmp_path / 'exposures' / f'toy_exp_{k:03d}.fits', overwrite=True)

    field = sc.Field(tmp_path / 'toy_run', ToyFocalPlane(), PIX, compute=TWO_WORKERS)
    field.reproject(tmp_path / 'exposures' / 'toy_exp_*.fits', method='interp', padding=8)
    # Each detector's frames share one pattern over the focal-plane map but observe
    # only their own amplifiers: a mean-zero anchor over the whole map would include
    # the other detector's (unobserved, free) chunks and fix nothing, so the constant
    # per detector (degenerate with the frames' scalars) is set by damping, which acts
    # on observed unknowns only.
    model = sc.Model(sky=[sc.Sky(damping=1e-4)], offsets=[sc.Offsets(on='amps', per='detector', damping=1e-3)])
    result = field.calibrate(sc.Recipe(model, fit=FIT, coadd=None, numerics=NUMERICS))

    with result.cal() as cal:
        offs = cal.offsets[0]
        _, detector = frame_ids(cal)
    # detector d's frames share one pattern; its own amplifiers (chunks 4d..4d+3) carry it
    amps = [offs[detector == d][0][4 * d:4 * d + 4] for d in (0, 1)]
    got = np.concatenate([a - a.mean() for a in amps])
    r, slope = agreement(got, readout, demean=False)
    print(f"heterogeneous focal plane: read-out patterns r = {r:.4f}, slope {slope:.3f}")
    assert r > 0.99 and abs(slope - 1) < 0.05, (r, slope)


if __name__ == '__main__':
    import tempfile
    import time
    from pathlib import Path
    for test in (test_frames_written_directly, test_thermal_pattern_and_gradients, test_time_variable_sky,
                 test_imaging_polarimeter, test_data_cube_slices, test_amplifier_ghost,
                 test_heterogeneous_focal_plane):
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix='selfcal_anytel_') as tmp:
            test(Path(tmp))
        print(f"  {test.__name__}: OK ({time.time() - t0:.0f} s)")
