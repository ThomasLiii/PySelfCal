"""Shared helpers for zodi offset analysis.

The self-calibration offset term (per-frame, per-chunk additive constant) is
dominated by the frame-averaged zodiacal light brightness. These helpers load
that offset, pair it with per-exposure pointing and time, compute ecliptic /
solar geometry, and provide the sinusoidal fit primitives used by the
downstream analysis scripts.

TODO(stable-advance): the analysis scripts here still import from the back-compat
``SelfCal`` shim. The package was renamed to ``selfcal`` (selfcal_scripts/ already
migrated). When the ``stable`` worktree advances to pick up the refactor, migrate
these imports ``from SelfCal...`` -> the real ``selfcal`` paths (e.g.
``SelfCal.SPHERExUtility`` -> ``selfcal.instruments.spherex.SPHERExUtility``,
``SelfCal.MakeMap`` -> ``selfcal.MakeMap``) and drop the ``SelfCal`` shim. Until
then the shim keeps these working unchanged. See workspace/selfcal-refactor/report.md.
"""
import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from astropy.io import fits
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u
from scipy.optimize import curve_fit

# analysis_script/ -> analysis/ -> selfcal/ (contains SelfCal/)
_SELFCAL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SELFCAL_ROOT not in sys.path:
    sys.path.insert(0, _SELFCAL_ROOT)

from SelfCal.MakeMap import load_reproj_file
from SelfCal.SPHERExUtility import make_stripped_chunk_valid_mask


CAL_OUTPUT_BASE = '/mnt/md124/thomasli/selfcal/outputs'
CAL_RUN_TEMPLATE = 'SPHEREx_nep_qr2_det{det}_6p2arcsec'
CAL_FILE_TEMPLATE = ('cal_Detector{det}_NumSub10_NumCh34_NumCol3_'
                     'Ch{ch}_damp0p1_reg0p1_outThresh5_sigma2.h5')

SIDEREAL_YEAR_DAYS = 365.25

# Output layout: pickles/npy in cache/, plots + csv/json in figures/.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_PKG_DIR, 'cache')
FIG_DIR = os.path.join(_PKG_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def data_path(name):
    return os.path.join(DATA_DIR, name)


def fig_path(name):
    return os.path.join(FIG_DIR, name)


def cal_path(detector, channel):
    return os.path.join(
        CAL_OUTPUT_BASE,
        CAL_RUN_TEMPLATE.format(det=detector),
        'calibration',
        CAL_FILE_TEMPLATE.format(det=detector, ch=channel),
    )


# ---------------------------------------------------------------------
# Zodi anchor (optional). The anchor lives next to the cal files at
# <run>/zodi_anchor/anchor_D{det}.h5 and sets the absolute level by a
# per-channel constant C (skymap += C, frame_scalar -= C). Analysis here
# is anchor-AWARE but opt-in: if the run has no anchor file, lookups
# return None / 0.0 and everything proceeds on the raw (un-anchored)
# offsets. See SelfCal/ZodiAnchor.py and PIPELINE.md (Zodi anchor stage).
# ---------------------------------------------------------------------

def anchor_path(detector):
    """Per-detector anchor file for the run cal_path targets."""
    return os.path.join(
        CAL_OUTPUT_BASE,
        CAL_RUN_TEMPLATE.format(det=detector),
        'zodi_anchor',
        f'anchor_D{detector}.h5',
    )


_ANCHOR_CACHE = {}


def load_anchor_for(detector):
    """Return the Anchor for this detector's run, or None if no anchor
    file exists. Cached per detector; imports SelfCal.ZodiAnchor lazily so
    runs without anchors don't pay the import."""
    if detector in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[detector]
    p = anchor_path(detector)
    anchor = None
    if os.path.exists(p):
        from SelfCal.ZodiAnchor import load_anchor
        anchor = load_anchor(p)
    _ANCHOR_CACHE[detector] = anchor
    return anchor


def anchor_C(detector, channel):
    """Final anchor C (MJy/sr) for (detector, channel), or 0.0 if the run
    has no anchor file / channel."""
    a = load_anchor_for(detector)
    if a is None or channel not in a.channels:
        return 0.0
    return a.C(channel)


def load_cal_offsets(path_or_file):
    """Read per-frame offsets from a cal_*.h5, handling both schemas.

    Returns ``{m: offset_m}`` where each ``offset_m`` is a per-frame
    ``(num_frames, num_chunks_m)`` array. For the legacy single-map schema
    this is ``{0: f['offset'][:]}``. For the multi-chunk-map schema this is
    ``{m: f['offsets/map_m'][:]}`` for each saved map; the shared per-frame
    scalar bias (top-level ``frame_scalar``, only written when any map uses
    ``det_groups``) is folded into map 0 so single-map analysis code that
    only reads ``[0]`` sees the same total bias the legacy schema baked in.

    Accepts either a path-like or an already-open ``h5py.File``.
    """
    if isinstance(path_or_file, h5py.File):
        return _read_cal_offsets(path_or_file)
    with h5py.File(path_or_file, 'r') as f:
        return _read_cal_offsets(f)


def _read_cal_offsets(f):
    if 'offsets' in f:
        K = int(f.attrs.get('num_maps', len(f['offsets'])))
        offsets = {m: f['offsets'][f'map_{m}'][:] for m in range(K)}
        if 'frame_scalar' in f:
            scalar = f['frame_scalar'][:][:, None]
            offsets[0] = offsets[0] + scalar
        return offsets
    return {0: f['offset'][:]}


def load_single_channel_offset(detector, channel,
                               num_subchannels=10, num_channels=34, num_columns=3,
                               apply_anchor=False):
    """Load one channel's calibration h5 and return per-exposure mean offset.

    Parameters
    ----------
    apply_anchor : bool
        If True, subtract the anchor C for this (detector, channel) from
        map 0 (the frame_scalar-folded offset) so the returned offsets are
        on the anchor's absolute zero-point — i.e. the per-frame DC the
        solver would carry after the anchor's `frame_scalar -= C` shift.
        No-op (with a one-line warning) if the run has no anchor file.

    Returns
    -------
    mean_offset : (num_frames,)
        Per-exposure mean of the offset term over valid chunks.
    raw_offset : (num_frames, num_chunks)
        Full per-chunk offset array from the cal file (map 0 in the
        multi-chunk-map schema, which is the standard chunk-shaped offset).
    chunk_valid_mask : (num_chunks,) bool-like
        Unpadded valid-chunk mask used to pick columns of `raw_offset`.
    reproj_list : list[str]
        Per-frame reproj HDF5 paths (axis-0 ordering of `raw_offset`).
    """
    path = cal_path(detector, channel)
    with h5py.File(path, 'r') as f:
        raw_offset = load_cal_offsets(f)[0]
        reproj_list = [s.decode('utf-8') for s in f['reproj_list'][:]]
    if apply_anchor:
        if load_anchor_for(detector) is None:
            print(f"  [zodi_utils] apply_anchor=True but no anchor file for "
                  f"D{detector} ({anchor_path(detector)}); using raw offsets.")
        # anchor moves a per-frame constant C out of frame_scalar (-> sky),
        # i.e. anchored map 0 = map0 - C (C broadcast over all chunks).
        raw_offset = raw_offset - anchor_C(detector, channel)
    mask = make_stripped_chunk_valid_mask(
        ch=[channel], num_subchannels=num_subchannels,
        num_channels=num_channels, num_columns=num_columns,
        subchannel_padding=0,
    )
    mask_bool = mask.astype(bool)
    mean_offset = np.mean(raw_offset[:, mask_bool], axis=1)
    return mean_offset, raw_offset, mask, reproj_list


def _extract_header(reproj_path):
    try:
        fpath = load_reproj_file(reproj_path, fields=['file_path'])['file_path']
        with fits.open(fpath) as hdul:
            h = hdul[1].header
            return (h.get('CRVAL1'), h.get('CRVAL2'), h.get('MJD-AVG'))
    except Exception:
        return (np.nan, np.nan, np.nan)


def build_header_table(reproj_list, max_workers=10, desc='FITS headers'):
    """Per-exposure [CRVAL1, CRVAL2, MJD_AVG] aligned to reproj_list."""
    records = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for rec in tqdm(ex.map(_extract_header, reproj_list, chunksize=20),
                        total=len(reproj_list), desc=desc):
            records.append(rec)
    return pd.DataFrame(records, columns=['CRVAL1', 'CRVAL2', 'MJD_AVG'])


def compute_ecliptic_geometry(ra_deg, dec_deg, mjd):
    """Vectorized ecliptic + solar geometry for the zodi analysis.

    Uses the geocentric-true ecliptic frame for target and Sun so the Sun's
    apparent 1 deg/day motion is preserved. For angular quantities
    (elongation, pa_to_sun) we use the Sun's APPARENT DIRECTION only -- if
    you pass the raw `get_sun(t)` return value (which carries a 1 AU
    distance) into astropy's `separation` or `position_angle` calls,
    astropy uses 3D positions and the ~1 AU Earth-Sun parallax flips the
    result by up to ~180 deg. We strip the distance to avoid this.

    Returns a dict of numpy arrays (all in degrees):
      ecl_lon, ecl_lat  : target ecliptic coordinates (geocentric)
      sun_ecl_lon       : Sun's apparent ecliptic longitude at MJD
      helio_lon         : ecl_lon - sun_ecl_lon wrapped to [-180, 180];
                          target's direction relative to the Sun
      elongation        : target-Sun angular separation on the sky
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    mjd = np.asarray(mjd, dtype=float)

    times = Time(mjd, format='mjd', scale='utc')
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs', obstime=times)
    target_ecl = target.transform_to('geocentrictrueecliptic')

    sun_gcrs = get_sun(times)
    sun_ecl = sun_gcrs.transform_to('geocentrictrueecliptic')
    # Direction-only sun in ICRS (drop the 1 AU distance to avoid parallax bias
    # in separation/position_angle below).
    sun_dir = SkyCoord(sun_gcrs.ra, sun_gcrs.dec, frame='icrs')

    elon = target.separation(sun_dir).deg
    helio_lon = np.mod(target_ecl.lon.deg - sun_ecl.lon.deg + 180.0, 360.0) - 180.0

    return {
        'ecl_lon': target_ecl.lon.deg,
        'ecl_lat': target_ecl.lat.deg,
        'sun_ecl_lon': sun_ecl.lon.deg,
        'helio_lon': helio_lon,
        'elongation': elon,
    }


def sine_model(t, A, phi, C, f=1.0 / SIDEREAL_YEAR_DAYS):
    return A * np.sin(2.0 * np.pi * f * t + phi) + C


def fit_sine(t, y, f_fixed=1.0 / SIDEREAL_YEAR_DAYS, p0=None, loss='soft_l1'):
    """Fit y = A sin(2 pi f t + phi) + C with f fixed (annual by default).

    Returns a dict of fit parameters and residual diagnostics.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if p0 is None:
        p0 = (0.5 * (np.nanmax(y) - np.nanmin(y)), 0.0, float(np.nanmedian(y)))

    def _m(t, A, phi, C):
        return sine_model(t, A, phi, C, f=f_fixed)

    popt, _ = curve_fit(_m, t, y, p0=p0, method='trf',
                        loss=loss, f_scale=0.1, max_nfev=5000)
    A, phi, C = popt
    # Canonical form: A >= 0, phi in [-pi, pi).
    if A < 0:
        A = -A
        phi = phi + np.pi
    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    res = y - _m(t, A, phi, C)
    return {
        'A': float(A),
        'phi': float(phi),
        'C': float(C),
        'f': float(f_fixed),
        'residual_std': float(np.std(res)),
        'residual_mad': float(1.4826 * np.median(np.abs(res - np.median(res)))),
        'n': int(len(t)),
    }


def bin_edges(values, n_bins, pct=(2.0, 98.0)):
    """Return uniform bin edges spanning the given percentile range of `values`."""
    lo, hi = np.nanpercentile(values, pct)
    return np.linspace(lo, hi, n_bins + 1)


def assign_2d_bin(x, y, x_edges, y_edges):
    """Return flat bin index for each (x, y); -1 for out-of-range points."""
    ix = np.digitize(x, x_edges) - 1
    iy = np.digitize(y, y_edges) - 1
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat = np.where(valid, ix * ny + iy, -1)
    return flat, ix, iy


def fit_sine_per_bin(t, y, bin_idx, min_points=30, f_fixed=1.0 / SIDEREAL_YEAR_DAYS):
    """Per-bin sine fits. Returns a DataFrame with one row per non-empty bin."""
    rows = []
    unique_bins = np.unique(bin_idx)
    unique_bins = unique_bins[unique_bins >= 0]
    for b in unique_bins:
        m = bin_idx == b
        if m.sum() < min_points:
            continue
        try:
            fit = fit_sine(t[m], y[m], f_fixed=f_fixed)
        except Exception:
            continue
        fit['bin'] = int(b)
        rows.append(fit)
    return pd.DataFrame(rows)
