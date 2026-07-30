"""Sanity check for Figure 3:
stack the raw exposures used in the calibration of Detector 5 and compare
the bitmask-cleaned mean to the C(sub, col) detector-grid render.

Steps
-----
  1. Pull the reproj list from any one of the channel cal files (all 17
     channels share the same exposures).
  2. (Optional) random subsample to keep I/O time small.
  3. For each exposure: read HDU 1 (IMAGE) and HDU 2 (FLAGS), mask any
     pixel where FLAGS != 0, accumulate sum and count per pixel.
  4. mean = sum / count.  This is the stacked raw map.
  5. Compute per-subchannel mean across columns using the chunk_map and
     subtract -- gives the column-only residual (compare to Fig 3b).

Outputs:
  * cache/verify_stack_det{N}_n{N_used}.npz            (mean + count)
  * figures/verify_stack_det{N}_n{N_used}.png          (4 panels)
"""
import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SELFCAL_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
if _SELFCAL_ROOT not in sys.path:
    sys.path.insert(0, _SELFCAL_ROOT)

from selfcal.io.reproj import load_reproj_file
from selfcal.instruments.spherex.spherex_utility import (
    make_stripped_chunk_map,
    make_spherex_stripped_offset_map,
    load_lvf_params,
    load_calibration,
)
from scipy.interpolate import interp1d
from zodi_utils import data_path, fig_path, cal_path

N_CHUNKS = (10 * 34 + 2) * 3

NUM_SUB, NUM_CH, NUM_COL = 10, 34, 3
TOT_SUB = NUM_SUB * NUM_CH + 2


def _stack_batch(args):
    """Worker: accumulate sum + count over a batch of reproj paths."""
    reproj_paths = args
    s = np.zeros((2040, 2040), dtype=np.float64)
    c = np.zeros((2040, 2040), dtype=np.int32)
    for rp in reproj_paths:
        try:
            fp = load_reproj_file(rp, fields=['file_path'])['file_path']
            with fits.open(fp, memmap=False) as hdul:
                data = hdul[1].data.astype(np.float32)
                flags = hdul[2].data
            valid = (flags == 0) & np.isfinite(data)
            s[valid] += data[valid]
            c[valid] += 1
        except Exception:
            continue
    return s, c


def main(detector, n_frames, n_workers, batch_size, seed, reuse_cache=False):
    # Pull reproj_list (shared across all 17 channels of this detector).
    with h5py.File(cal_path(detector, 1), 'r') as f:
        reproj_list = [s.decode('utf-8') for s in f['reproj_list'][:]]
    n_total = len(reproj_list)
    print(f'{n_total} exposures available')
    if n_frames is not None and n_frames < n_total:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, n_frames, replace=False)
        reproj_list = [reproj_list[i] for i in sorted(idx.tolist())]
        n_used = n_frames
    else:
        n_used = n_total

    out_npz = data_path(f'verify_stack_det{detector}_n{n_used}.npz')
    if reuse_cache and os.path.exists(out_npz):
        print(f'reusing existing stack at {out_npz}')
        d = np.load(out_npz)
        mean_arr = d['mean']
        count_arr = d['count']
    else:
        print(f'stacking {n_used} exposures with {n_workers} workers '
              f'(batch={batch_size})')
        batches = [reproj_list[i:i + batch_size]
                   for i in range(0, n_used, batch_size)]
        sum_arr = np.zeros((2040, 2040), dtype=np.float64)
        count_arr = np.zeros((2040, 2040), dtype=np.int32)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for s, c in tqdm(ex.map(_stack_batch, batches),
                             total=len(batches), desc='stacking'):
                sum_arr += s
                count_arr += c
        mean_arr = np.where(count_arr > 0,
                            sum_arr / np.maximum(count_arr, 1),
                            np.nan)
        np.savez_compressed(out_npz, mean=mean_arr, count=count_arr,
                            n_used=n_used)
        print(f'wrote {out_npz}')

    # Build chunk_map; we need both detector geometry and the chunk index per
    # pixel.
    lvf = load_lvf_params(f'lvf_params_D{detector}.npy')
    det_chunk_map, _, r_edges, x_edges = make_stripped_chunk_map(
        detector, num_subchannels=NUM_SUB, num_channels=NUM_CH,
        num_columns=NUM_COL, oversample_factor=1, lvf_params=lvf,
    )

    # Reference vmin/vmax from the Fig 3b detector-fixed render, so panel (c)
    # below uses the same colour scale.
    fig3_cache = data_path(f'meeting_det{detector}_per_chunk_fits.npz')
    if os.path.exists(fig3_cache):
        D_meeting = np.load(fig3_cache)['D']                       # (342, 3)
        D_meeting_grid = make_spherex_stripped_offset_map(
            chunk_map=det_chunk_map,
            chunk_offset=np.nan_to_num(D_meeting.ravel(), nan=0.0),
            chunk_valid_mask=np.isfinite(D_meeting).ravel().astype(float),
            lvf_params=lvf,
            r_edges=r_edges, x_edges=x_edges,
            tot_subchannels=TOT_SUB, num_columns=NUM_COL,
            fill_invalid=True,
        )
        nz = D_meeting_grid != 0
        vabs_fig3 = float(np.nanpercentile(np.abs(D_meeting_grid[nz]), 99))
        print(f'using Fig 3b colour scale: vmin/vmax = ±{vabs_fig3:.5f}')
    else:
        vabs_fig3 = None
        print(f'Fig 3b cache {fig3_cache} not found; using auto colour scale')

    # ----- Per-subchannel ROBUST mean (median over pixels) -----
    # Plain mean has a few spikes from negative-outlier pixels not caught
    # by FLAGS==0. Median per subchannel is robust to those outliers.
    sub_per_pixel = (det_chunk_map // NUM_COL).astype(np.int32)
    sub_flat = sub_per_pixel.ravel()
    mean_flat = mean_arr.ravel()
    valid = np.isfinite(mean_flat)
    # Sort pixels by subchannel id once; per-subchannel median via slicing.
    sf = sub_flat[valid]; mf = mean_flat[valid]
    sort_idx = np.argsort(sf, kind='stable')
    sf_sorted = sf[sort_idx]; mf_sorted = mf[sort_idx]
    bounds = np.searchsorted(sf_sorted, np.arange(TOT_SUB + 1))
    sub_median = np.full(TOT_SUB, np.nan)
    for s in range(TOT_SUB):
        pix = mf_sorted[bounds[s]:bounds[s + 1]]
        if pix.size >= 50:
            sub_median[s] = float(np.median(pix))

    # Wavelength per subchannel for the spectrum panel.
    mean_wav = 0.5 * (lvf['wave_edges'][:-1] + lvf['wave_edges'][1:])
    sub_wav = np.full(TOT_SUB, np.nan)
    sub_wav[1:1 + len(mean_wav)] = mean_wav

    # ----- Per-subchannel median spectrum (no smoothing) -----
    valid_idx = np.where(np.isfinite(sub_median) & np.isfinite(sub_wav))[0]
    valid_idx = valid_idx[(valid_idx >= 1) & (valid_idx <= 340)]
    spec_w = sub_wav[valid_idx]
    spec_y = sub_median[valid_idx]
    sort_w = np.argsort(spec_w)
    spec_w = spec_w[sort_w]; spec_y = spec_y[sort_w]
    spec_interp = interp1d(spec_w, spec_y, kind='linear',
                           bounds_error=False, fill_value='extrapolate')

    # ----- Per-pixel spectral subtraction via BC_map -----
    # Restrict to pixels with BC_map inside the spline's valid wavelength
    # range so we don't extrapolate. Pixels outside become NaN in the result.
    BC_map, _ = load_calibration(
        band=detector,
        calibration_dir='/data3/SPHEREx/SpecCal_202509/ParameterFiles')
    w_min, w_max = float(spec_w.min()), float(spec_w.max())
    in_band = np.isfinite(BC_map) & (BC_map >= w_min) & (BC_map <= w_max)
    finite_pix = np.isfinite(mean_arr) & in_band
    spectral_grid = np.full_like(mean_arr, np.nan, dtype=np.float64)
    spectral_grid[finite_pix] = spec_interp(BC_map[finite_pix])
    column_only_pix = mean_arr - spectral_grid

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # (a) Raw stacked mean.
    ax = axes[0]
    finite = np.isfinite(mean_arr)
    vlo, vhi = np.nanpercentile(mean_arr[finite], [2, 98])
    im = ax.imshow(mean_arr, cmap='viridis', vmin=vlo, vmax=vhi,
                   origin='lower')
    ax.set(xlabel='Detector X [pix]', ylabel='Detector Y [pix]',
           title=f'(a) Raw Stack of {n_used} Exposures')
    plt.colorbar(im, ax=ax, label='Mean Intensity [MJy/sr]')

    # (b) Per-pixel residual after subtracting the per-subchannel median
    # spectrum evaluated at each pixel's wavelength via BC_map. Colour scale
    # matches Fig 3b for direct visual comparison.
    ax = axes[1]
    if vabs_fig3 is not None:
        vabs_c = vabs_fig3
    else:
        fc = np.isfinite(column_only_pix)
        vabs_c = np.nanpercentile(np.abs(column_only_pix[fc]), 99)
    im = ax.imshow(column_only_pix, cmap='RdBu_r',
                   vmin=-vabs_c, vmax=vabs_c, origin='lower')
    ax.set(xlabel='Detector X [pix]', ylabel='Detector Y [pix]',
           title=f'(b) Stack − Per-Subchannel Median')
    plt.colorbar(im, ax=ax, label='[MJy/sr]')

    fig.suptitle(
        f'Verification: Raw Exposure Stack — Detector {detector}, '
        f'{n_used} Frames',
        y=1.02,
    )
    fig.tight_layout()
    out_png = fig_path(f'verify_stack_det{detector}_n{n_used}.png')
    fig.savefig(out_png, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')

    # ----------------------------------------------------------------------
    # Sanity check: median-bin panel (b) per chunk, then re-interpolate with
    # the same mean-preserving spline used for Fig 3b, and plot side-by-side.
    # If they look similar, the calibration's time-averaged detector-fixed
    # component (Fig 3b) is consistent with what a raw bitmask-cleaned stack
    # of the same exposures would recover.
    # ----------------------------------------------------------------------
    if vabs_fig3 is not None:
        n_chunks_total = TOT_SUB * NUM_COL
        chunk_id_flat = det_chunk_map.ravel()
        val_flat = column_only_pix.ravel()
        finite = np.isfinite(val_flat)
        cf = chunk_id_flat[finite]
        vf = val_flat[finite]
        order = np.argsort(cf, kind='stable')
        cf_sorted = cf[order]
        vf_sorted = vf[order]
        cbounds = np.searchsorted(cf_sorted, np.arange(n_chunks_total + 1))
        chunk_field = np.full(n_chunks_total, np.nan)
        for k in range(n_chunks_total):
            seg = vf_sorted[cbounds[k]:cbounds[k + 1]]
            if seg.size >= 50:
                chunk_field[k] = float(np.median(seg))
        chunk_field_2d = chunk_field.reshape(TOT_SUB, NUM_COL)

        chunkbin_grid = make_spherex_stripped_offset_map(
            chunk_map=det_chunk_map,
            chunk_offset=np.nan_to_num(chunk_field_2d.ravel(), nan=0.0),
            chunk_valid_mask=np.isfinite(chunk_field_2d).ravel().astype(float),
            lvf_params=lvf,
            r_edges=r_edges, x_edges=x_edges,
            tot_subchannels=TOT_SUB, num_columns=NUM_COL,
            fill_invalid=True,
        )

        diff = chunkbin_grid - D_meeting_grid
        nz = (chunkbin_grid != 0) | (D_meeting_grid != 0)
        diff_rms = float(np.sqrt(np.nanmean(diff[nz] ** 2)))

        fig2, axes2 = plt.subplots(1, 3, figsize=(19, 6))
        ax = axes2[0]
        im = ax.imshow(chunkbin_grid, cmap='RdBu_r',
                       vmin=-vabs_fig3, vmax=vabs_fig3, origin='lower')
        ax.set(xlabel='Detector X [pix]', ylabel='Detector Y [pix]',
               title='(a) Panel (b) Median-Binned + Re-interpolated')
        plt.colorbar(im, ax=ax, label='[MJy/sr]')

        ax = axes2[1]
        im = ax.imshow(D_meeting_grid, cmap='RdBu_r',
                       vmin=-vabs_fig3, vmax=vabs_fig3, origin='lower')
        ax.set(xlabel='Detector X [pix]', ylabel='Detector Y [pix]',
               title='(b) Fig 3b: $C - \\langle C(\\lambda)\\rangle$')
        plt.colorbar(im, ax=ax, label='[MJy/sr]')

        ax = axes2[2]
        im = ax.imshow(diff, cmap='RdBu_r',
                       vmin=-vabs_fig3, vmax=vabs_fig3, origin='lower')
        ax.set(xlabel='Detector X [pix]', ylabel='Detector Y [pix]',
               title=f'(c) (a) − (b)   RMS = {diff_rms:.5f} MJy/sr')
        plt.colorbar(im, ax=ax, label='[MJy/sr]')

        fig2.suptitle(
            f'Verification: chunk-binned stack vs Fig 3b — '
            f'Detector {detector}, {n_used} frames', y=1.02,
        )
        fig2.tight_layout()
        out_cmp = fig_path(
            f'verify_stack_det{detector}_n{n_used}_chunkbin_vs_fig3b.png')
        fig2.savefig(out_cmp, dpi=170, bbox_inches='tight')
        plt.close(fig2)
        print(f'wrote {out_cmp}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    p.add_argument('--n-frames', type=int, default=1000,
                   help='random subsample size (None = use all exposures)')
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--reuse-cache', action='store_true',
                   help='if the stack cache for this (detector, n_frames) '
                        'already exists, reuse it instead of re-stacking')
    args = p.parse_args()
    main(args.detector, args.n_frames, args.workers, args.batch_size,
         args.seed, reuse_cache=args.reuse_cache)
