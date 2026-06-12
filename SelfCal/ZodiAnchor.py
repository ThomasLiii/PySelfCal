"""Post-hoc zodiacal-light anchor for SelfCal.

The LSQR solve has one global additive degeneracy:
``sky -> sky + C`` and ``scalar[k] -> scalar[k] - C`` leaves the data
model unchanged. We pick ``C`` so that the per-frame DC of the LSQR
solution lines up with a Kelsall-style zodi prediction:

    full_DC[k] = slope * zodi_pred[k] + intercept       (linfit)
    full_DC[k] = scalar[k] + Σ_c (N_c[k] / N[k]) * offset[k, c]
                  (pixel-weighted; mean_offsets_list pins the
                   unit-weighted sum to 0, not this one)

The intercept is the anchor C: the global shift needed to put the
recovered sky at the correct absolute astrophysical level. Slope is a
validation check (should be ~1 if the zodi model captures per-frame
variation accurately).

A moving sigma-clip in MJD space iteratively rejects bright-source
outliers before refitting.

The anchor is **non-mutating**: SelfCal pipeline outputs (cal/mosaic)
stay pristine. The fit result is written to a per-detector anchor file
``<run>/zodi_anchor/anchor_D{N}.h5`` (``fit_anchor_for_channel`` +
``write_anchor`` / ``append_anchor_channel``) and applied to arrays at
read time by the ``Anchor`` consumer (``load_anchor``).

No zodipy dependency — that lives in
``selfcal_scripts/zodi_anchor/build_predictions.py``, which writes the
per-frame zodi-prediction ``.npz`` that this module consumes.
"""
import datetime
import hashlib
import os
import re

import h5py
import numpy as np
from astropy.io import fits


# Anchor-file schema version. Bump when the on-disk layout of
# anchor_D{N}.h5 changes in a backward-incompatible way.
ANCHOR_VERSION = 1

# Mosaic image HDUs the anchor C shift applies to (each has a sibling
# ``<EXTNAME>_WEIGHT`` used to mask the shift to covered pixels).
MOSAIC_MAP_EXTNAMES = ('MEAN_MAP', 'SC_MEAN_MAP')


# ---------------------------------------------------------------------
# Array-level functions
# ---------------------------------------------------------------------

def compute_full_dc(frame_scalar, offsets_map0, offset_coverage_map0):
    """Per-frame DC including the pixel-weighted chunk-mean leakage.

    Parameters
    ----------
    frame_scalar : (num_frames,) float
    offsets_map0 : (num_frames, num_chunks) float
    offset_coverage_map0 : (num_frames, num_chunks) numeric (pixel counts)

    Returns
    -------
    full_dc : (num_frames,) float64
    """
    frame_scalar = np.asarray(frame_scalar, dtype=np.float64)
    offsets = np.asarray(offsets_map0, dtype=np.float64)
    cov = np.asarray(offset_coverage_map0, dtype=np.float64)
    n_per_frame = cov.sum(axis=1)
    safe = n_per_frame > 0
    chunk_weighted = np.full_like(frame_scalar, np.nan)
    chunk_weighted[safe] = ((offsets[safe] * cov[safe]).sum(axis=1)
                            / n_per_frame[safe])
    return frame_scalar + chunk_weighted


def moving_sigma_clip_mask(mjds, residuals, window_days, sigma):
    """Boolean inlier mask via a sliding MJD-window MAD clip.

    For each frame, compute median + MAD of `residuals` within
    [mjd - W/2, mjd + W/2]; reject if
    ``|residual - local_med| > sigma * MAD / 0.6745``.
    Returns mask in the ORIGINAL (unsorted) frame order.
    """
    order = np.argsort(mjds, kind='stable')
    mjds_s = mjds[order]
    resid_s = residuals[order]
    n = len(mjds_s)
    keep_s = np.ones(n, dtype=bool)
    half = float(window_days) / 2.0
    los = np.searchsorted(mjds_s, mjds_s - half, side='left')
    his = np.searchsorted(mjds_s, mjds_s + half, side='right')
    for i in range(n):
        local = resid_s[los[i]:his[i]]
        if local.size < 5:
            continue
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        thresh = sigma * mad / 0.6745
        if thresh == 0:
            continue
        if abs(resid_s[i] - med) > thresh:
            keep_s[i] = False
    keep = np.empty_like(keep_s)
    keep[order] = keep_s
    return keep


def fit_with_clip(zp, fs, mjds, window_days=7.0, sigma=3.0, iters=2):
    """Iteratively linfit ``fs = slope*zp + intercept`` with a moving
    sigma-clip on residuals.

    Parameters
    ----------
    zp : zodi prediction, (num_frames,) float
    fs : per-frame DC (full_DC), (num_frames,) float
    mjds : (num_frames,) float or None. If None, no clipping is done.
    window_days, sigma, iters : clip parameters.

    Returns
    -------
    (slope, intercept, r, inlier_mask) tuple of (float, float, float, bool[num_frames])
    """
    inlier = np.isfinite(zp) & np.isfinite(fs)
    if mjds is not None:
        inlier &= np.isfinite(mjds)
    slope, intercept = np.polyfit(zp[inlier], fs[inlier], 1)
    for it in range(int(iters)):
        if mjds is None or window_days <= 0:
            break
        resid = fs - (slope * zp + intercept)
        keep = moving_sigma_clip_mask(
            mjds, np.where(inlier, resid, np.inf), window_days, sigma)
        new_inlier = inlier & keep
        n_new = int(new_inlier.sum())
        if n_new < 10:
            break
        if n_new == int(inlier.sum()):
            break
        inlier = new_inlier
        slope, intercept = np.polyfit(zp[inlier], fs[inlier], 1)
    r = float(np.corrcoef(zp[inlier], fs[inlier])[0, 1])
    return float(slope), float(intercept), r, inlier


# ---------------------------------------------------------------------
# .npz I/O
# ---------------------------------------------------------------------

def load_zodi_pred_npz(path, cal_reproj_list=None):
    """Load zodi_pred (+ optional mjds + reproj_list) from a build
    artifact. If ``cal_reproj_list`` is provided and the .npz carries a
    ``reproj_list`` key, the two are checked for strict equality."""
    npz = np.load(path, allow_pickle=False)
    if 'zodi_pred' not in npz.files:
        raise ValueError(
            f"{path} lacks 'zodi_pred' key. Available: {npz.files}")
    z = np.asarray(npz['zodi_pred'], dtype=np.float64).ravel()
    mjds = (np.asarray(npz['mjds'], dtype=np.float64).ravel()
            if 'mjds' in npz.files else None)
    if cal_reproj_list is not None and 'reproj_list' in npz.files:
        expected = [s.decode() if isinstance(s, (bytes, np.bytes_)) else s
                    for s in npz['reproj_list']]
        actual = [s.decode() if isinstance(s, (bytes, np.bytes_)) else s
                  for s in cal_reproj_list]
        if expected != actual:
            for i, (e, a) in enumerate(zip(expected, actual)):
                if e != a:
                    raise ValueError(
                        f"zodi_pred reproj_list does not match cal "
                        f"reproj_list at index {i}:\n"
                        f"  zodi_pred: {e}\n  cal:       {a}")
            if len(expected) != len(actual):
                raise ValueError(
                    f"zodi_pred reproj_list length {len(expected)} != "
                    f"cal reproj_list length {len(actual)}")
    return z, mjds


# ---------------------------------------------------------------------
# Fit-only core + non-mutating consumer
#
# The fit core computes the anchor (slope, C, r, ...) for one channel
# WITHOUT touching the cal/mosaic. Results are written to a per-detector
# anchor file (anchor_D{N}.h5) by build_anchor.py. The consumer
# (load_anchor / Anchor) applies the shift to arrays at read time so the
# pipeline outputs stay pristine. See workspace/zodi_anchor_refactor/refactor.md.
# ---------------------------------------------------------------------

def file_sha1(path, _bufsize=1 << 20):
    """SHA-1 of a file's bytes (for the anchor file's npz-identity check)."""
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(_bufsize), b''):
            h.update(chunk)
    return h.hexdigest()


def fit_anchor_for_channel(cal_path, zodi_pred_npz,
                           clip_window_days=7.0, clip_sigma=3.0,
                           clip_iters=2):
    """Fit the per-channel anchor from a PRISTINE cal + zodi-pred npz.

    Pure read + fit; never mutates cal or mosaic. Shared by build_anchor.py
    and the run_cal.py driver hook.

    Returns a dict of the per-channel summary scalars destined for the
    anchor-file Ch{c}/ group (plus npz identity fields).
    """
    with h5py.File(cal_path, 'r') as f:
        if 'frame_scalar' not in f:
            raise ValueError(
                "anchor fit requires use_per_frame_scalar=True cal runs; "
                f"{cal_path} lacks /frame_scalar.")
        if ('offsets' not in f or 'map_0' not in f['offsets']
                or 'offset_coverage' not in f
                or 'map_0' not in f['offset_coverage']):
            raise ValueError(
                "anchor fit requires multi-map schema; "
                f"{cal_path} lacks offsets/map_0 or offset_coverage/map_0.")
        frame_scalar = f['frame_scalar'][:].astype(np.float64)
        offsets_m0 = f['offsets/map_0'][:].astype(np.float64)
        cov_m0 = f['offset_coverage/map_0'][:].astype(np.float64)
        cal_reproj_list = list(f['reproj_list'][:])

    zodi_pred, mjds = load_zodi_pred_npz(zodi_pred_npz, cal_reproj_list)
    if len(zodi_pred) != len(frame_scalar):
        raise ValueError(
            f"zodi_pred length {len(zodi_pred)} != frame_scalar length "
            f"{len(frame_scalar)}")
    if mjds is not None and len(mjds) != len(frame_scalar):
        raise ValueError(
            f"mjds length {len(mjds)} != frame_scalar length "
            f"{len(frame_scalar)}")

    # wavelength / model name are metadata in the npz (optional).
    with np.load(zodi_pred_npz, allow_pickle=False) as z:
        wavelength_um = (float(z['wavelength_um'])
                         if 'wavelength_um' in z.files else float('nan'))
        model_name = (str(z['model_name'])
                      if 'model_name' in z.files else '')

    full_dc = compute_full_dc(frame_scalar, offsets_m0, cov_m0)
    slope, intercept, r, inlier = fit_with_clip(
        zodi_pred, full_dc, mjds,
        window_days=clip_window_days, sigma=clip_sigma, iters=clip_iters)
    n_finite = int((np.isfinite(zodi_pred) & np.isfinite(full_dc)).sum())
    n_used = int(inlier.sum())
    n_outl = n_finite - n_used

    return dict(
        wavelength_um=wavelength_um,
        slope=float(slope),
        intercept=float(intercept),       # == C
        pearson_r=float(r),
        n_inliers=n_used,
        n_outliers=n_outl,
        mean_full_dc=float(np.mean(full_dc[inlier])),
        mean_scalar=float(np.mean(frame_scalar[inlier])),
        mean_pred=float(np.mean(zodi_pred[inlier])),
        clip_window_days=float(clip_window_days),
        clip_sigma=float(clip_sigma),
        clip_iters=int(clip_iters),
        cal_path=os.path.abspath(cal_path),
        zodi_pred_npz=os.path.abspath(zodi_pred_npz),
        zodi_pred_n=int(len(zodi_pred)),
        zodi_pred_sha=file_sha1(zodi_pred_npz),
        model_name=model_name,
        # smoothing fields default to the raw fit until a Phase-1 pass runs.
        slope_final=float(slope),
        C_final=float(intercept),
        contaminated_flag=False,
        smooth_method='raw',
    )


# Keys written verbatim as Ch{c}/ attrs (order = doc order).
_ANCHOR_CHANNEL_KEYS = (
    'wavelength_um', 'slope', 'intercept', 'pearson_r',
    'n_inliers', 'n_outliers', 'mean_full_dc', 'mean_scalar', 'mean_pred',
    'clip_window_days', 'clip_sigma', 'clip_iters',
    'cal_path', 'zodi_pred_npz', 'zodi_pred_n', 'zodi_pred_sha', 'model_name',
    'slope_final', 'C_final', 'contaminated_flag', 'smooth_method',
)


def _write_root_attrs(f, detector, source_run, clip_defaults, anchor_method):
    f.attrs['anchor_version'] = int(ANCHOR_VERSION)
    f.attrs['source_run'] = str(source_run)
    f.attrs['detector'] = int(detector)
    f.attrs['created_iso'] = datetime.datetime.now().isoformat()
    f.attrs['clip_window_days'] = float(clip_defaults['clip_window_days'])
    f.attrs['clip_sigma'] = float(clip_defaults['clip_sigma'])
    f.attrs['clip_iters'] = int(clip_defaults['clip_iters'])
    f.attrs['anchor_method'] = str(anchor_method)


def _write_channel_group(channels_grp, ch, res):
    name = f'Ch{ch}'
    if name in channels_grp:
        del channels_grp[name]
    g = channels_grp.create_group(name)
    for k in _ANCHOR_CHANNEL_KEYS:
        g.attrs[k] = res[k]


def write_anchor(out_path, detector, source_run, channel_results,
                 clip_defaults, anchor_method='raw'):
    """Write a per-detector anchor file (summary-only schema), overwriting.

    Parameters
    ----------
    out_path : str
    detector : int
    source_run : str  (run dir basename, for provenance)
    channel_results : dict {channel_int: fit_dict}  (fit_dict from
        fit_anchor_for_channel, optionally with smoothing fields overwritten)
    clip_defaults : dict with clip_window_days/clip_sigma/clip_iters
    anchor_method : str  ("raw" | "rweighted_spline" | ...)
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        _write_root_attrs(f, detector, source_run, clip_defaults, anchor_method)
        ch_grp = f.create_group('channels')
        for ch in sorted(channel_results):
            _write_channel_group(ch_grp, ch, channel_results[ch])


def append_anchor_channel(out_path, detector, source_run, channel,
                          fit_result, clip_defaults, anchor_method='raw'):
    """Add/replace one channel in a per-detector anchor file, in place.

    Creates the file (and root attrs + channels group) if absent. Used by
    the run_cal.py driver hook, which fits channels sequentially and
    grows the detector anchor file as each finishes. Safe to re-run for a
    channel (overwrites its group).
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(out_path, 'a') as f:
        if 'anchor_version' not in f.attrs:
            _write_root_attrs(f, detector, source_run, clip_defaults,
                              anchor_method)
        ch_grp = f.require_group('channels')
        _write_channel_group(ch_grp, channel, fit_result)


class Anchor:
    """Non-mutating consumer of a per-detector anchor file.

    Loads anchor_D{N}.h5 and applies the (slope_final, C_final) shift to
    arrays at read time. Never writes to cal/mosaic.
    """

    def __init__(self, path):
        self.path = path
        self.channels = {}      # ch -> dict of attrs
        with h5py.File(path, 'r') as f:
            self.version = int(f.attrs.get('anchor_version', 0))
            self.detector = int(f.attrs.get('detector', -1))
            self.source_run = str(f.attrs.get('source_run', ''))
            self.anchor_method = str(f.attrs.get('anchor_method', 'raw'))
            for name, g in f['channels'].items():
                ch = int(name[2:])  # strip "Ch"
                d = {}
                for k, v in g.attrs.items():
                    d[k] = v.decode() if isinstance(v, bytes) else v
                self.channels[ch] = d

    def __repr__(self):
        return (f"Anchor(D{self.detector}, {len(self.channels)} channels, "
                f"method={self.anchor_method!r}, v{self.version})")

    def C(self, ch):
        """Final anchor constant for a channel (smoothing-aware)."""
        return float(self.channels[ch]['C_final'])

    def slope(self, ch):
        return float(self.channels[ch]['slope_final'])

    def apply_to_mosaic_array(self, data, weight, ch):
        """Return a copy of a mosaic image with +C on covered pixels.

        Pixels with weight <= 0 are left untouched so unobserved regions
        stay at their fill value.
        """
        out = np.array(data, copy=True)
        C = np.array(self.C(ch), dtype=out.dtype)
        if weight is not None:
            out[weight > 0] += C
        else:
            out += C
        return out

    def apply_to_skymap_array(self, skymap, coverage, ch):
        """Return a copy of a cal skymap with +C on covered pixels."""
        out = np.array(skymap, copy=True)
        C = self.C(ch)
        if coverage is not None:
            out[coverage > 0] += C
        else:
            out += C
        return out

    def apply_to_cal_scalar(self, frame_scalar, ch):
        """Return frame_scalar shifted by -C (the anchored per-frame DC)."""
        return np.asarray(frame_scalar, dtype=np.float64) - self.C(ch)

    def apply_to_mosaic_hdul(self, hdul, ch):
        """In-place: add C to the MEAN_MAP / SC_MEAN_MAP HDUs of an open
        mosaic HDUList (covered pixels only, via the *_WEIGHT siblings),
        and stamp a ZODIANCH header on each. Returns the shifted EXTNAMEs."""
        ext = {h.header.get('EXTNAME', ''): h for h in hdul[1:]}
        shifted = []
        for name in MOSAIC_MAP_EXTNAMES:
            hdu = ext.get(name)
            if hdu is None or hdu.data is None:
                continue
            w = ext.get(f'{name}_WEIGHT')
            hdu.data = self.apply_to_mosaic_array(
                hdu.data, w.data if (w is not None) else None, ch)
            hdu.header['ZODIANCH'] = (float(self.C(ch)),
                                      'Zodi anchor C added (MJy/sr)')
            shifted.append(name)
        return shifted


def load_anchor(path):
    """Load a per-detector anchor file into an Anchor consumer."""
    return Anchor(path)


def _channel_from_filename(path):
    m = re.search(r'_Ch(\d+)_', os.path.basename(path))
    if m is None:
        raise ValueError(f"cannot parse _Ch<n>_ from {os.path.basename(path)}; "
                         "pass ch= explicitly.")
    return int(m.group(1))


def load_anchored_mosaic(mosaic_path, anchor, ch=None, extname='MEAN_MAP'):
    """Open a PRISTINE mosaic and return its `extname` map with the anchor
    C applied in memory (covered pixels only). The file on disk is NOT
    modified.

    Parameters
    ----------
    mosaic_path : path to the pristine mosaic FITS
    anchor : an Anchor, or a path to an anchor_D{N}.h5
    ch : channel int; defaults to the channel parsed from the filename
    extname : which map to return ('MEAN_MAP' or 'SC_MEAN_MAP')

    Returns
    -------
    (data, header) — the anchored map array and its FITS header (with WCS).
    """
    if isinstance(anchor, str):
        anchor = load_anchor(anchor)
    if ch is None:
        ch = _channel_from_filename(mosaic_path)
    with fits.open(mosaic_path, memmap=False) as hdul:
        ext = {h.header.get('EXTNAME', ''): h for h in hdul[1:]}
        if extname not in ext:
            raise KeyError(f"{extname} not in {mosaic_path}; "
                           f"have {sorted(ext)}")
        hdu = ext[extname]
        w = ext.get(f'{extname}_WEIGHT')
        data = anchor.apply_to_mosaic_array(
            hdu.data, w.data if (w is not None) else None, ch)
        return data, hdu.header.copy()


# ---------------------------------------------------------------------
# Phase-1 slope smoothing of contaminated channels
# ---------------------------------------------------------------------

def rweighted_slope_smooth(wavelengths, slope, intercept, pearson_r,
                            mean_full_dc, mean_pred,
                            r_threshold=0.9, spline_k=3, s_factor=1.0,
                            r_eps=1e-3):
    """Targeted smoothing of contaminated channels: smooth the SLOPE only,
    then recompute C consistently (do NOT smooth C).

    Rationale: ``slope`` is the multiplicative zodi-SED calibration and
    should vary smoothly in wavelength. ``C`` is the non-zodi DC bucket
    (CIB + DGL + airglow); it is NOT smooth and must keep real features
    (e.g. the He I 1083 nm glow). At a contaminated channel the per-channel
    linfit slope is garbage (airglow scatter is uncorrelated with zodi),
    and because OLS couples them (``C = mean_full_dc - slope*mean_pred``)
    that garbage slope also corrupts C.

    So for flagged (``pearson_r < r_threshold``) channels we:
      1. set ``slope_final`` from a Pearson-r-weighted smoothing spline fit
         to the CLEAN channels only (a blown channel can't leak in), and
      2. set ``C_final = mean_full_dc - slope_final * mean_pred`` — the C
         implied by the corrected slope, which KEEPS the non-zodi signal
         (He glow etc.) and only removes the properly-calibrated zodi part.
    Clean channels keep their raw slope/intercept.

    The slope spline fits standardized y (zero weighted-mean, unit
    weighted-std over clean channels) with weight ``w = r^2/(1-r^2)`` (r
    clamped to [0, 0.999]) so the one knob ``s_factor`` is scale-free:
    ``s = s_factor * n_clean`` — ~0 interpolates the clean slopes, ~1
    flattens to their weighted mean; 1.0 follows the trend while smoothing.

    Parameters
    ----------
    wavelengths, slope, intercept, pearson_r : per-channel arrays (any order)
    mean_full_dc, mean_pred : per-channel inlier means (from the anchor file),
        used to recompute C_final for flagged channels
    r_threshold : channels with r below this are smoothed (default 0.9 —
        de-biases the slope of moderate-r channels like PAH/OI while their
        non-zodi C content is preserved by the recompute; lower to 0.5 to
        smooth only the hard blowouts)
    spline_k : slope-spline degree (default 3)
    s_factor : slope-spline smoothing strength (default 1.0)
    r_eps : stabilizer in the weight denominator

    Returns
    -------
    dict with (all in INPUT order):
      slope_final, C_final : smoothed arrays (raw where clean)
      contaminated : bool mask (r < r_threshold)
      slope_curve : the clean-fit slope spline evaluated at every channel
                    (for plotting/inspection)
      extrapolated : bool mask, True where a flagged channel lies outside
                     the clean-channel wavelength span (spline extrapolated)
    """
    from scipy.interpolate import UnivariateSpline
    wl = np.asarray(wavelengths, float)
    sl = np.asarray(slope, float)
    C = np.asarray(intercept, float)
    r = np.asarray(pearson_r, float)
    mfd = np.asarray(mean_full_dc, float)
    mpred = np.asarray(mean_pred, float)
    n = wl.size
    order = np.argsort(wl)
    inv = np.empty(n, int)
    inv[order] = np.arange(n)
    wl_s, sl_s, C_s, r_s = wl[order], sl[order], C[order], r[order]
    mfd_s, mpred_s = mfd[order], mpred[order]

    contam_s = r_s < r_threshold
    clean_s = ~contam_s
    n_clean = int(clean_s.sum())
    if n_clean < spline_k + 1:
        raise ValueError(
            f"only {n_clean} clean channels (r >= {r_threshold}); need "
            f">= {spline_k + 1} for a degree-{spline_k} spline.")

    rc = np.clip(r_s[clean_s], 0.0, 0.999)
    wbase_c = rc ** 2 / (1.0 - rc ** 2 + r_eps)
    wl_c = wl_s[clean_s]
    wnorm_c = wbase_c / np.median(wbase_c)   # typical clean weight ~1

    # Smooth the SLOPE only (clean-only fit, standardized).
    sl_c = sl_s[clean_s]
    sbar = np.average(sl_c, weights=wbase_c)
    svar = np.average((sl_c - sbar) ** 2, weights=wbase_c)
    sscale = float(np.sqrt(svar)) if svar > 0 else 1.0
    spl = UnivariateSpline(wl_c, (sl_c - sbar) / sscale, w=wnorm_c,
                           k=spline_k, s=s_factor * n_clean)
    sl_curve_s = spl(wl_s) * sscale + sbar

    sl_final_s = np.where(contam_s, sl_curve_s, sl_s)
    # C: keep raw on clean; on flagged recompute from the corrected slope
    # (preserves the non-zodi / airglow content of mean_full_dc).
    C_final_s = np.where(contam_s, mfd_s - sl_final_s * mpred_s, C_s)

    lo, hi = wl_c.min(), wl_c.max()
    extrap_s = contam_s & ((wl_s < lo) | (wl_s > hi))

    return dict(
        slope_final=sl_final_s[inv],
        C_final=C_final_s[inv],
        contaminated=contam_s[inv],
        slope_curve=sl_curve_s[inv],
        extrapolated=extrap_s[inv],
    )


def smooth_anchor_file(path, r_threshold=0.9, s_factor=1.0, spline_k=3,
                       dry_run=False):
    """Load an anchor file, compute the r-weighted slope smoothing, and (unless
    dry_run) write ``slope_final``/``C_final``/``contaminated_flag``/
    ``smooth_method`` back in-place plus root smoothing-provenance attrs. The
    raw ``slope``/``intercept`` are never touched, so this is re-runnable.

    Single write path shared by smooth_anchor.py and build_anchor.py
    (--smooth). Reads only the anchor file — no cal/npz I/O.

    Returns a dict ``{chs, wl, slope, intercept, pearson_r, result}`` (the
    raw per-channel arrays + the ``rweighted_slope_smooth`` output) so the
    caller can report/plot without recomputing.
    """
    a = load_anchor(path)
    chs = sorted(a.channels)
    wl = np.array([a.channels[c]['wavelength_um'] for c in chs])
    slope = np.array([a.channels[c]['slope'] for c in chs])
    C = np.array([a.channels[c]['intercept'] for c in chs])
    r = np.array([a.channels[c]['pearson_r'] for c in chs])
    mfd = np.array([a.channels[c]['mean_full_dc'] for c in chs])
    mpred = np.array([a.channels[c]['mean_pred'] for c in chs])

    res = rweighted_slope_smooth(
        wl, slope, C, r, mfd, mpred,
        r_threshold=r_threshold, spline_k=spline_k, s_factor=s_factor)
    contam = res['contaminated']

    if not dry_run:
        with h5py.File(path, 'r+') as f:
            for i, c in enumerate(chs):
                g = f['channels'][f'Ch{c}']
                g.attrs['slope_final'] = float(res['slope_final'][i])
                g.attrs['C_final'] = float(res['C_final'][i])
                g.attrs['contaminated_flag'] = bool(contam[i])
                g.attrs['smooth_method'] = ('rweighted_spline' if contam[i]
                                            else 'raw')
            f.attrs['anchor_method'] = 'rweighted_spline'
            f.attrs['smooth_r_threshold'] = float(r_threshold)
            f.attrs['smooth_s_factor'] = float(s_factor)
            f.attrs['smooth_spline_k'] = int(spline_k)
            f.attrs['smoothed_iso'] = datetime.datetime.now().isoformat()

    return dict(detector=a.detector, chs=chs, wl=wl, slope=slope,
                intercept=C, pearson_r=r, result=res)
