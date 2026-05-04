"""Task 3: Build a detector-fixed pattern template from channels 1-17
and map it back to the detector pixel grid.

What we're computing
--------------------
For each calibration run (one per channel), the offset array has shape
(N_exposures, 342 subchannels * 3 columns). Each entry is a per-frame
additive constant applied to a (subchannel arc x vertical strip) region
of the detector.

The "detector-fixed pattern" is the part of those offsets that is the
same across all exposures (independent of pointing and time). We
isolate it by:

  1. For each exposure, subtract the mean-over-valid-chunks (the
     exposure-level DC). This removes the annual zodi modulation and
     any per-exposure DC.
  2. Time-average the result per (subchannel, column) bin. Because each
     channel's LSQR solution has an arbitrary DC set by its own
     regularization, naively concatenating channels produces a saw-tooth
     pattern at channel boundaries.
  3. **Stitch channels together** using the padded overlap between
     adjacent channels: shift each channel's template by a single scalar
     (cumulative) so the values agree at the padded overlap. This is the
     same trick as `combined_offset` in analysis/offset_analysis.ipynb.
  4. Average stitched templates across channels (overlapping subchannels
     contribute to multiple channels; nanmean combines them).
  5. Render on the detector grid by looking up every detector pixel's
     chunk id in det_chunk_map and plotting the template value.

Outputs
-------
    det{det}__det_fixed_template.png
    det{det}__det_fixed_template_grid.npy  (raw 2040x2040 map)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SELFCAL_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
if _SELFCAL_ROOT not in sys.path:
    sys.path.insert(0, _SELFCAL_ROOT)

from SelfCal.SPHERExUtility import (
    make_stripped_chunk_map,
    load_lvf_params,
    make_stripped_chunk_valid_mask,
    make_spherex_stripped_offset_map,
)
from zodi_utils import data_path, fig_path

NUM_SUB, NUM_CH, NUM_COL = 10, 34, 3
TOT_SUB = NUM_SUB * NUM_CH + 2


def _padded_mask(channel):
    """Boolean (342, 3) mask of the channel's padded valid region."""
    m = make_stripped_chunk_valid_mask(
        ch=[channel], num_subchannels=NUM_SUB, num_channels=NUM_CH,
        num_columns=NUM_COL, subchannel_padding=1,
    ).reshape(TOT_SUB, NUM_COL).astype(bool)
    return m


def stitch_templates(templates, channels):
    """Apply cumulative additive shifts so adjacent channels agree on their
    padded overlap. Returns {ch: shifted template}.

    Continuity constraint at the overlap between channel k-1 and k:
        <stitched_{k-1} - stitched_k>_overlap = 0
    Setting shift_1 = 0 and solving iteratively:
        shift_k = shift_{k-1} + <tpl_{k-1} - tpl_k>_overlap
    """
    sorted_channels = sorted(channels)
    masks = {ch: _padded_mask(ch) for ch in sorted_channels}

    shifts = {sorted_channels[0]: 0.0}
    for i in range(1, len(sorted_channels)):
        prev_ch = sorted_channels[i - 1]
        curr_ch = sorted_channels[i]
        overlap = masks[prev_ch] & masks[curr_ch]
        prev_t = templates[prev_ch]
        curr_t = templates[curr_ch]
        good = overlap & ~np.isnan(prev_t) & ~np.isnan(curr_t)
        if good.any():
            d = float(np.mean(prev_t[good] - curr_t[good]))
        else:
            d = 0.0
        shifts[curr_ch] = shifts[prev_ch] + d
    stitched = {ch: templates[ch] + shifts[ch] for ch in sorted_channels}
    return stitched, shifts


def main(detector):
    templates = pd.read_pickle(data_path(f'detector_templates_det{detector}.pkl'))
    channels = sorted(templates.keys())
    print(f'Stitching templates from channels {channels[0]}..{channels[-1]} '
          f'({len(channels)} channels).')

    stitched, shifts = stitch_templates(templates, channels)
    for ch in channels:
        print(f'  ch{ch:2d}: cumulative shift = {shifts[ch]:+.5f} MJy/sr')

    # Stack stitched templates, nanmean to merge overlapping bins.
    stack = np.stack([stitched[ch] for ch in channels], axis=0)   # (Nch, 342, 3)
    with np.errstate(invalid='ignore'):
        combined = np.nanmean(stack, axis=0)                      # (342, 3)
    # Centre the final map on zero so the plotted colormap is symmetric.
    combined = combined - np.nanmean(combined)
    coverage = np.sum(~np.isnan(stack), axis=0)                   # (342, 3)

    # Subtract per-subchannel mean across columns to remove the spectral
    # trend (time-averaged zodi + detector spectral response). What remains
    # is the pure column-to-column detector-fixed pattern.
    with np.errstate(invalid='ignore'):
        subchannel_mean = np.nanmean(combined, axis=1, keepdims=True)
    combined_col_only = combined - subchannel_mean                # (342, 3)

    # Render BOTH on the detector pixel grid:
    #   * Blocky lookup via det_chunk_map  (one value per chunk)
    #   * SMOOTH interpolation via make_spherex_stripped_offset_map which
    #     fits a 2-D mean-preserving spline in (R, x) -- same path the
    #     pipeline uses to evaluate offset maps during mosaicking.
    lvf = load_lvf_params(f'lvf_params_D{detector}.npy')
    det_chunk_map, _, r_edges, x_edges = make_stripped_chunk_map(
        detector, num_subchannels=NUM_SUB, num_channels=NUM_CH,
        num_columns=NUM_COL, oversample_factor=1, lvf_params=lvf,
    )

    def _blocky(grid_2d):
        flat = grid_2d.reshape(-1)
        flat = np.where(np.isnan(flat), 0.0, flat)
        return flat[det_chunk_map]

    # Build a common chunk_valid_mask for spline fitting: any (sub, col)
    # covered by at least one channel.
    chunk_valid_mask = (coverage.reshape(-1) > 0).astype(float)

    def _smooth(grid_2d):
        chunk_offset = np.nan_to_num(grid_2d.reshape(-1), nan=0.0)
        return make_spherex_stripped_offset_map(
            chunk_map=det_chunk_map,
            chunk_offset=chunk_offset,
            chunk_valid_mask=chunk_valid_mask,
            lvf_params=lvf,
            r_edges=r_edges,
            x_edges=x_edges,
            tot_subchannels=TOT_SUB,
            num_columns=NUM_COL,
            fill_invalid=True,
        )

    det_grid_full = _blocky(combined)                             # (2040, 2040)
    det_grid_col = _blocky(combined_col_only)                     # (2040, 2040)
    det_grid_col_smooth = _smooth(combined_col_only)              # (2040, 2040)
    coverage_grid = coverage.reshape(-1)[det_chunk_map]

    np.save(data_path(f'det{detector}__det_fixed_template_grid.npy'), det_grid_full)
    np.save(data_path(f'det{detector}__det_fixed_template_colonly_grid.npy'), det_grid_col)
    np.save(data_path(f'det{detector}__det_fixed_template_colonly_smooth_grid.npy'),
            det_grid_col_smooth)
    print(f'wrote cache/det{detector}__det_fixed_template_grid.npy  (full blocky)')
    print(f'wrote cache/det{detector}__det_fixed_template_colonly_grid.npy  (col-only blocky)')
    print(f'wrote cache/det{detector}__det_fixed_template_colonly_smooth_grid.npy  (col-only smooth)')

    # ---------------- Plots ----------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # Row 1: the full stitched template (still contains spectral trend).
    ax = axes[0, 0]
    vabs = np.nanpercentile(np.abs(combined), 99)
    im = ax.imshow(combined, aspect='auto', cmap='RdBu_r',
                   vmin=-vabs, vmax=vabs, origin='lower')
    ax.set(xlabel='column index', ylabel='subchannel index',
           title='(a) full template (subchannel x column)')
    plt.colorbar(im, ax=ax, label='[MJy/sr]')

    ax = axes[0, 1]
    for col in range(NUM_COL):
        ax.plot(np.arange(TOT_SUB), combined[:, col],
                lw=1.5, alpha=0.85, label=f'col {col}')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='subchannel index', ylabel='template [MJy/sr]',
           title='(b) full template vs subchannel  (spectral trend + col gradient)')
    ax.legend(); ax.grid(alpha=0.3)

    # Row 2: column-only residual (per-subchannel mean across cols removed).
    ax = axes[1, 0]
    vabs_co = np.nanpercentile(np.abs(combined_col_only), 99)
    im = ax.imshow(combined_col_only, aspect='auto', cmap='RdBu_r',
                   vmin=-vabs_co, vmax=vabs_co, origin='lower')
    ax.set(xlabel='column index', ylabel='subchannel index',
           title='(c) column-only residual  (template - mean_over_cols)')
    plt.colorbar(im, ax=ax, label='[MJy/sr]')

    ax = axes[1, 1]
    for col in range(NUM_COL):
        ax.plot(np.arange(TOT_SUB), combined_col_only[:, col],
                lw=1.5, alpha=0.85, label=f'col {col}')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='subchannel index', ylabel='column-only residual [MJy/sr]',
           title='(d) column-only residual vs subchannel')
    ax.legend(); ax.grid(alpha=0.3)

    # Row 3: detector-pixel rendering of the column-only pattern.
    # (e) smooth spline interpolation; (f) original blocky lookup for comparison.
    ax = axes[2, 0]
    vabs_det = np.nanpercentile(np.abs(det_grid_col_smooth[det_grid_col_smooth != 0]), 99)
    im = ax.imshow(det_grid_col_smooth, cmap='RdBu_r',
                   vmin=-vabs_det, vmax=vabs_det, origin='lower')
    ax.set(xlabel='detector x [pix]', ylabel='detector y [pix]',
           title='(e) column-only residual (smooth spline interpolation)')
    plt.colorbar(im, ax=ax, label='[MJy/sr]')

    ax = axes[2, 1]
    im = ax.imshow(det_grid_col, cmap='RdBu_r',
                   vmin=-vabs_det, vmax=vabs_det, origin='lower')
    ax.set(xlabel='detector x [pix]', ylabel='detector y [pix]',
           title='(f) column-only residual (blocky chunk lookup)')
    plt.colorbar(im, ax=ax, label='[MJy/sr]')

    fig.suptitle(
        f'Detector-fixed offset template  det{detector}  '
        f'(channels {channels[0]}..{channels[-1]}, '
        f'stitched across channels, mean-over-columns subtracted for col-only)',
        y=1.005,
    )
    fig.tight_layout()
    out = fig_path(f'det{detector}__det_fixed_template.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    args = p.parse_args()
    main(args.detector)
