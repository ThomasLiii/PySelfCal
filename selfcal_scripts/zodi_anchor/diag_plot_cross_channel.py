"""Cross-channel continuity diagnostic.

After anchoring each channel independently, the per-chunk values should
be continuous in wavelength space (the SPHEREx LVF maps subchannels ->
wavelengths). Three panels:

  (a) anchor C per channel vs channel-mean wavelength.
  (b) per-chunk mean offset (averaged over frames) vs subchannel
      wavelength, one curve per channel.
  (c) per-chunk anchored "absolute" value (sky_in_chunk_anch +
      offset_avg + scalar_anch_avg) vs subchannel wavelength.
      The 7 channel curves should overlap at adjacent subchannels
      (where adjacent channels' masks overlap with padding=1).

Run in selfcal env (needs matplotlib).
"""
import argparse
import glob
import os
import re

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from SelfCal.ZodiAnchor import load_anchor


DEFAULT_CALIBRATION_DIR = '/home/thomasli/spherex/SPHEREx_Spectral_Calibration'
DET_BC_TEMPLATE = '20250901_SSDC_BC_Band{detector}.fits'
VALID_CHUNK_THRESH = 0.05


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--run-dir',
                     help='Run dir with calibration/ (pristine cal_*.h5) + '
                          'zodi_anchor/anchor_D{N}.h5 sidecar. The anchor is '
                          'applied in-memory from the sidecar; cal files are '
                          'NOT modified.')
    src.add_argument('--cal-glob',
                     help='Glob for pristine cal_*.h5 files. Requires '
                          '--sidecar.')
    p.add_argument('--sidecar', default=None,
                   help='Path to anchor_D{N}.h5 sidecar. Required with '
                        '--cal-glob; auto-located under <run>/zodi_anchor/ '
                        'with --run-dir.')
    p.add_argument('--detector', type=int, default=None,
                   help='Detector index. Auto-parsed from filename if '
                        'omitted.')
    p.add_argument('--calibration-dir', default=DEFAULT_CALIBRATION_DIR)
    p.add_argument('--out', default=None,
                   help='Output PNG path (default: cross_channel.png in '
                        '<run>/zodi_anchor/ or next to the cal files).')
    return p.parse_args()


def parse_detector_from_filename(path):
    m = re.search(r'Detector(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def parse_channel_from_filename(path):
    m = re.search(r'_Ch(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def per_chunk_wavelengths(det_chunk_map, det_BC, valid_chunks):
    """Return per-chunk-id mean wavelength."""
    n_chunks = int(det_chunk_map.max()) + 1
    out = np.full(n_chunks, np.nan, dtype=np.float64)
    for c in valid_chunks:
        mask = det_chunk_map == c
        out[c] = det_BC[mask].mean()
    return out


def main():
    args = parse_args()

    if args.run_dir:
        cal_paths = sorted(glob.glob(
            os.path.join(args.run_dir, 'calibration', 'cal_*.h5')))
        detector = args.detector or (
            parse_detector_from_filename(cal_paths[0]) if cal_paths else None)
        sidecar_path = args.sidecar or (
            os.path.join(args.run_dir, 'zodi_anchor',
                         f'anchor_D{detector}.h5') if detector else None)
        out_path_default = os.path.join(
            args.run_dir, 'zodi_anchor', 'cross_channel.png')
    else:
        cal_paths = sorted(glob.glob(args.cal_glob))
        if not args.sidecar:
            raise SystemExit("--cal-glob requires --sidecar.")
        sidecar_path = args.sidecar
        out_path_default = os.path.join(
            os.path.dirname(cal_paths[0]) if cal_paths else '.',
            'cross_channel.png')
        detector = args.detector or (
            parse_detector_from_filename(cal_paths[0]) if cal_paths else None)
    out_path = args.out or out_path_default

    if not cal_paths:
        raise SystemExit("no cal files found.")
    if detector is None:
        raise SystemExit("could not determine detector; pass --detector.")
    if not sidecar_path or not os.path.exists(sidecar_path):
        raise SystemExit(f"sidecar not found: {sidecar_path}")
    anchor = load_anchor(sidecar_path)
    print(f"loaded sidecar {sidecar_path} ({anchor})")
    bc_path = os.path.join(
        args.calibration_dir, DET_BC_TEMPLATE.format(detector=detector))
    det_BC = fits.getdata(bc_path)
    print(f"detector: {detector}, det_BC from {bc_path}")
    print(f"loading {len(cal_paths)} pristine cal files "
          f"(anchor applied in-memory from sidecar)")

    channels = []
    for cal in cal_paths:
        ch = parse_channel_from_filename(cal)
        if ch not in anchor.channels:
            print(f"  Ch{ch}: not in sidecar; skipping")
            continue
        with h5py.File(cal, 'r') as f:
            det_chunk_map = f['chunk_maps/map_0'][:]
            cov_frac = f['offset_coverage_frac/map_0'][:]
            offsets = f['offsets/map_0'][:]
            frame_scalar = f['frame_scalar'][:]
            skymap = f['skymap'][:]
            skymap_cov = f['skymap_coverage'][:] if 'skymap_coverage' in f else None
        # Anchor params from the sidecar (cals are pristine; shift applied
        # in-memory). C_final/slope_final are repair-aware.
        C = anchor.C(ch)
        slope = anchor.slope(ch)
        r = float(anchor.channels[ch]['pearson_r'])
        # Apply the anchor in-memory: skymap += C (covered), frame_scalar -= C.
        skymap = anchor.apply_to_skymap_array(skymap, skymap_cov, ch)
        frame_scalar = anchor.apply_to_cal_scalar(frame_scalar, ch)
        valid_chunks = np.where((cov_frac > VALID_CHUNK_THRESH).any(axis=0))[0]
        # Channel-mean wavelength (representative)
        det_valid_mask = np.isin(det_chunk_map, valid_chunks)
        ch_wavelength = float(np.mean(det_BC[det_valid_mask]))
        # Per-chunk wavelength + per-chunk mean offset across frames
        chunk_wls = per_chunk_wavelengths(det_chunk_map, det_BC, valid_chunks)
        chunk_mean_offset = np.full(int(det_chunk_map.max()) + 1, np.nan)
        for c in valid_chunks:
            chunk_mean_offset[c] = offsets[:, c].mean()
        # Per-chunk absolute "abs value":
        # = mean_pixels_in_c (sky_anch) + mean_k offset[k, c] + mean_k scalar_anch
        # skymap/frame_scalar were just anchored in-memory above.
        mean_scalar_anch = float(np.mean(frame_scalar))
        chunk_abs_value = np.full_like(chunk_mean_offset, np.nan)
        # sky_in_chunk: average skymap pixels that this chunk covers (after mosaic-time det_to_sub).
        # We don't have a direct mapping from chunk -> sky pixels here; use skymap-only when coverage permits.
        # As a pragmatic proxy, use skymap_coverage-weighted mean of skymap across all pixels:
        if skymap_cov is not None:
            cov_mask = skymap_cov > 0
            mean_sky = float(np.average(skymap[cov_mask], weights=skymap_cov[cov_mask].astype(float)))
        else:
            mean_sky = float(np.nanmean(skymap))
        for c in valid_chunks:
            chunk_abs_value[c] = mean_sky + chunk_mean_offset[c] + mean_scalar_anch
        channels.append(dict(
            ch=ch, cal=cal, wavelength_um=ch_wavelength,
            C=C, slope=slope, r=r,
            valid_chunks=valid_chunks,
            chunk_wls=chunk_wls,
            chunk_mean_offset=chunk_mean_offset,
            chunk_abs_value=chunk_abs_value,
            mean_sky=mean_sky,
            mean_scalar_anch=mean_scalar_anch,
        ))
    channels.sort(key=lambda c: c['ch'])

    # Print summary
    print()
    print(f"{'Ch':>3} {'wavelength_um':>12} {'C':>10} {'slope':>7} {'r':>7} "
          f"{'mean_sky':>10} {'mean_scalar_a':>12}")
    for c in channels:
        print(f"{c['ch']:>3} {c['wavelength_um']:>12.4f} {c['C']:>10.4g} "
              f"{c['slope']:>7.3f} {c['r']:>7.3f} "
              f"{c['mean_sky']:>10.4g} {c['mean_scalar_anch']:>12.4g}")

    # Plot
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(channels)))
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    # (a) C per channel vs wavelength
    ax = axes[0]
    chs = [c['ch'] for c in channels]
    wls = [c['wavelength_um'] for c in channels]
    Cs = [c['C'] for c in channels]
    slopes = [c['slope'] for c in channels]
    rs = [c['r'] for c in channels]
    ax.plot(wls, Cs, 'o-', color='C0')
    # Per-point slope + r labels, alternating above/below to reduce overlap.
    for i, (x, y, s, r) in enumerate(zip(wls, Cs, slopes, rs)):
        above = (i % 2 == 0)
        ax.annotate(
            f's={s:.2f}\nr={r:.2f}',
            xy=(x, y),
            xytext=(0, 9 if above else -9),
            textcoords='offset points',
            fontsize=6, ha='center',
            va='bottom' if above else 'top',
            color='gray',
        )
    # Bit of vertical headroom so labels at the top/bottom don't clip.
    ymin, ymax = ax.get_ylim()
    pad = 0.10 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel('Channel mean wavelength (um)')
    ax.set_ylabel('Anchor C (MJy/sr)')
    ax.set_title('Per-channel anchor constant (label = slope / Pearson r)')
    ax.grid(alpha=0.3)

    # (b) per-chunk mean offset vs wavelength
    ax = axes[1]
    for color, c in zip(colors, channels):
        vc = c['valid_chunks']
        ax.scatter(c['chunk_wls'][vc], c['chunk_mean_offset'][vc],
                   s=4, alpha=0.7, color=color)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('Chunk mean wavelength (um)')
    ax.set_ylabel('mean over frames of offset (MJy/sr)')
    ax.set_title('Per-chunk mean offset across channels (should be continuous)')
    ax.grid(alpha=0.3)

    # (c) per-chunk "absolute" value vs wavelength
    ax = axes[2]
    for color, c in zip(colors, channels):
        vc = c['valid_chunks']
        ax.scatter(c['chunk_wls'][vc], c['chunk_abs_value'][vc],
                   s=4, alpha=0.7, color=color)
    ax.set_xlabel('Chunk mean wavelength (um)')
    ax.set_ylabel('mean_sky + mean_offset[c] + mean_scalar_anch (MJy/sr)')
    ax.set_title('Per-chunk anchored absolute value across channels')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
