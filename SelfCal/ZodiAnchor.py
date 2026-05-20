"""Post-hoc zodiacal-light anchor for SelfCal cal files + mosaics.

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

The actual shift is applied uniformly to the cal file
(``skymap += C``, ``frame_scalar -= C`` — offsets are NOT touched)
and to the mosaic FITS (``MEAN_MAP += C``, ``SC_MEAN_MAP += C``).
This module exposes both array-level functions and a file-based
``apply_anchor_to_file`` driver.

No zodipy dependency — that lives in
``selfcal_scripts/zodi_anchor/build_zodi_predictions.py``, which writes
the per-frame zodi-prediction ``.npz`` that this module consumes.
"""
import datetime
import os
import shutil

import h5py
import numpy as np
from astropy.io import fits


SHIFTED_EXTNAMES = ('MEAN_MAP', 'SC_MEAN_MAP')


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
# File-level driver
# ---------------------------------------------------------------------

def _shift_cal_file(cal_in, cal_out, C, zodi_pred, slope, intercept, r,
                    mean_scalar, mean_full_dc, mean_zodi,
                    n_inliers, n_outliers,
                    clip_window_days, clip_sigma):
    """Copy cal -> cal_out (if distinct paths) and apply the shift
    (skymap += C, frame_scalar -= C) plus record metadata as root attrs
    + zodi_anchor_pred dataset."""
    if os.path.abspath(cal_in) != os.path.abspath(cal_out):
        shutil.copyfile(cal_in, cal_out)
    with h5py.File(cal_out, 'r+') as f:
        sky = f['skymap']
        sky[...] = sky[...] + C
        fs = f['frame_scalar']
        fs[...] = fs[...] - C
        f.attrs['zodi_anchor_C'] = float(C)
        f.attrs['zodi_anchor_slope'] = float(slope)
        f.attrs['zodi_anchor_intercept'] = float(intercept)
        f.attrs['zodi_anchor_pearson_r'] = float(r)
        f.attrs['zodi_anchor_mean_full_dc'] = float(mean_full_dc)
        f.attrs['zodi_anchor_mean_scalar'] = float(mean_scalar)
        f.attrs['zodi_anchor_mean_pred'] = float(mean_zodi)
        f.attrs['zodi_anchor_n_inliers'] = int(n_inliers)
        f.attrs['zodi_anchor_n_outliers'] = int(n_outliers)
        f.attrs['zodi_anchor_clip_window_days'] = float(clip_window_days)
        f.attrs['zodi_anchor_clip_sigma'] = float(clip_sigma)
        f.attrs['zodi_anchor_created_iso'] = datetime.datetime.now().isoformat()
        if 'zodi_anchor_pred' in f:
            del f['zodi_anchor_pred']
        f.create_dataset('zodi_anchor_pred',
                         data=zodi_pred.astype(np.float32),
                         compression='gzip')


def _shift_mosaic_file(mos_in, mos_out, C, slope, r, mean_zodi):
    """Copy mosaic -> mos_out (if distinct paths) and apply +C to
    MEAN_MAP / SC_MEAN_MAP HDUs, stamping ZODIANCH/ZODISLOP/
    ZODICORR/ZODIMEAN header keys."""
    if os.path.abspath(mos_in) != os.path.abspath(mos_out):
        shutil.copyfile(mos_in, mos_out)
    shifted = []
    def stamp(header):
        header['ZODIANCH'] = (float(C), 'Zodi-anchor shift (MJy/sr) = intercept')
        header['ZODISLOP'] = (float(slope), 'Linfit slope (validation; ~1 expected)')
        header['ZODICORR'] = (float(r), 'Pearson r of full_DC vs zodi_pred')
        header['ZODIMEAN'] = (float(mean_zodi), 'Mean predicted zodi (MJy/sr)')
    with fits.open(mos_out, mode='update') as hdul:
        stamp(hdul[0].header)
        for hdu in hdul[1:]:
            extname = hdu.header.get('EXTNAME', '')
            if extname in SHIFTED_EXTNAMES and hdu.data is not None:
                hdu.data += np.array(C, dtype=hdu.data.dtype)
                stamp(hdu.header)
                shifted.append(extname)
    return shifted


def apply_anchor_to_file(cal_in, mosaic_in, zodi_pred_npz,
                        out_dir=None, out_suffix='_zodianch',
                        in_place=False,
                        clip_window_days=7.0, clip_sigma=3.0, clip_iters=2,
                        overwrite=False):
    """Read cal + mosaic + zodi-prediction .npz, fit C, write anchored
    copies of cal and mosaic. Returns a dict of statistics.

    The mosaic_in argument can be None to skip the mosaic shift (e.g.
    when called from inside the pipeline before save_mosaic).

    Returns
    -------
    dict with: C, slope, intercept, r, mean_scalar, mean_full_dc,
    mean_zodi, n_inliers, n_outliers, cal_out, mosaic_out (or None),
    shifted_extnames (list).
    """
    # Read cal
    with h5py.File(cal_in, 'r') as f:
        if 'frame_scalar' not in f:
            raise ValueError(
                "apply_anchor requires use_per_frame_scalar=True cal "
                f"runs; {cal_in} lacks /frame_scalar.")
        if ('offsets' not in f or 'map_0' not in f['offsets']
                or 'offset_coverage' not in f
                or 'map_0' not in f['offset_coverage']):
            raise ValueError(
                "apply_anchor requires multi-map schema; "
                f"{cal_in} lacks offsets/map_0 or offset_coverage/map_0.")
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

    full_dc = compute_full_dc(frame_scalar, offsets_m0, cov_m0)
    slope, intercept, r, inlier = fit_with_clip(
        zodi_pred, full_dc, mjds,
        window_days=clip_window_days, sigma=clip_sigma, iters=clip_iters)
    n_finite = int((np.isfinite(zodi_pred) & np.isfinite(full_dc)).sum())
    n_used = int(inlier.sum())
    n_outl = n_finite - n_used
    C = float(intercept)
    mean_scalar = float(np.mean(frame_scalar[inlier]))
    mean_full_dc = float(np.mean(full_dc[inlier]))
    mean_zodi = float(np.mean(zodi_pred[inlier]))

    # Pick output paths
    if in_place:
        cal_out = cal_in
        mos_out = mosaic_in
    else:
        cal_out = _output_path(cal_in, out_suffix, out_dir)
        mos_out = _output_path(mosaic_in, out_suffix, out_dir) if mosaic_in else None
        for out in [p for p in (cal_out, mos_out) if p]:
            if os.path.exists(out):
                if overwrite:
                    os.remove(out)
                else:
                    raise FileExistsError(
                        f"{out} already exists; pass overwrite=True to replace.")

    _shift_cal_file(cal_in, cal_out, C, zodi_pred, slope, intercept, r,
                    mean_scalar, mean_full_dc, mean_zodi,
                    n_inliers=n_used, n_outliers=n_outl,
                    clip_window_days=clip_window_days,
                    clip_sigma=clip_sigma)
    shifted = []
    if mosaic_in:
        shifted = _shift_mosaic_file(
            mosaic_in, mos_out, C, slope, r, mean_zodi)

    return dict(
        C=C, slope=slope, intercept=intercept, r=r,
        mean_scalar=mean_scalar, mean_full_dc=mean_full_dc,
        mean_zodi=mean_zodi,
        n_inliers=n_used, n_outliers=n_outl,
        clip_window_days=clip_window_days, clip_sigma=clip_sigma,
        cal_out=cal_out, mosaic_out=mos_out,
        shifted_extnames=shifted,
    )


def _output_path(in_path, suffix, out_dir=None):
    if out_dir is None:
        root, ext = os.path.splitext(in_path)
        return f'{root}{suffix}{ext}'
    base = os.path.basename(in_path)
    root, ext = os.path.splitext(base)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f'{root}{suffix}{ext}')
