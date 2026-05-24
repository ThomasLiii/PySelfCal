"""Build zodi predictions for ALL channels of a detector without
needing per-channel cal files.

Useful when you want to extract predictions concurrently with the
calibration solve: the per-channel mask is purely a function of the
LVF geometry and channel number (via
``SelfCal.SPHERExUtility.make_stripped_chunk_valid_mask``), so we only
need:

  - one canonical ``reproj_list`` (grab it from ANY existing cal —
    all channels of one detector share the same exposure set), AND
  - ``lvf_params`` for that detector (already cached as
    ``selfcal_scripts/lvf_params/lvf_params_D{D}.npy``).

Runs in the sidecar `selfcal-zodipy` env. Saves zodi_pred_<tag>.npz
per channel where ``<tag>`` matches the cal/mosaic naming the rest of
the pipeline produces.
"""
import argparse
import datetime
import os
import sys
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from astropy.io import fits

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
# Add the repo root to sys.path so SelfCal/ is importable without
# pip-installing it in the sidecar selfcal-zodipy env (the SelfCal
# package's deps include reproject/zarr/opencv that bump numpy past
# zodipy's <2.0 pin).
for p in (_HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_predictions import (  # noqa: E402
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_METADATA_CACHE_TEMPLATE,
    DET_BC_TEMPLATE,
    build_for_channel_theoretical,
    extract_metadata_for_reproj_list,
    save_predictions_npz,
)
from SelfCal.SPHERExUtility import (  # noqa: E402
    make_stripped_chunk_map, load_lvf_params,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--detector', type=int, required=True)
    p.add_argument('--num-subchannels', type=int, default=10)
    p.add_argument('--num-channels', type=int, default=34,
                   help='Number of spectral channels (e.g. 34 for production NumCh=34).')
    p.add_argument('--num-columns', type=int, default=10,
                   help='Per-subchannel column count (NumCol).')
    p.add_argument('--channels', default=None,
                   help='Channel range/list, e.g. "1-34" or "1,2,5-7". '
                        'Default: 1..num_channels.')
    p.add_argument('--reference-cal', required=True,
                   help='Path to one existing cal_*.h5 file to source '
                        'reproj_list (which exposures to use). Any '
                        'channel\'s cal works since all channels of a '
                        'detector share the same exposures.')
    p.add_argument('--out-dir', required=True,
                   help='Output dir for zodi_pred_<tag>.npz files.')
    p.add_argument('--file-suffix', default='_damp0p1_reg0p1_outThresh5_sigma2_polyK1',
                   help='Suffix in the cal/mosaic filename pattern; the '
                        'output .npz tag matches '
                        'Detector{D}_NumSub{S}_NumCh{C}_NumCol{Co}_Ch{ch}<suffix>.')
    p.add_argument('--subchannel-padding', type=int, default=0,
                   help='Channel-mask padding (matches '
                        'make_stripped_chunk_valid_mask). Production '
                        "default is 0 (strict). Use 1 to match LSQR's "
                        'padded mask.')
    p.add_argument('--calibration-dir', default=DEFAULT_CALIBRATION_DIR)
    p.add_argument('--lvf-params-dir', default=os.path.join(
        os.path.dirname(_HERE), 'lvf_params'),
        help='Dir with lvf_params_D{detector}.npy.')
    p.add_argument('--metadata-cache', default=None,
                   help='Persistent metadata cache (per detector). '
                        f'Default: {DEFAULT_METADATA_CACHE_TEMPLATE}')
    p.add_argument('--model', default='dirbe')
    p.add_argument('--grid-size', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=30)
    p.add_argument('--nprocesses', type=int, default=20)
    p.add_argument('--skip-existing', action='store_true',
                   help='Per-channel: skip if the .npz already exists.')
    return p.parse_args()


def parse_channels(s, num_channels):
    if s is None:
        return list(range(1, num_channels + 1))
    out = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def make_tag(detector, num_subchannels, num_channels, num_columns,
             ch, file_suffix):
    return (f'Detector{detector}_NumSub{num_subchannels}'
            f'_NumCh{num_channels}_NumCol{num_columns}'
            f'_Ch{ch}{file_suffix}')


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    channels = parse_channels(args.channels, args.num_channels)
    print(f"detector:      {args.detector}")
    print(f"channels:      {channels}")
    print(f"reference cal: {args.reference_cal}")
    print(f"out-dir:       {args.out_dir}")
    print(f"file-suffix:   {args.file_suffix}")

    # Reference cal: get reproj_list only.
    with h5py.File(args.reference_cal, 'r') as f:
        reproj_list_bytes = f['reproj_list'][:]
    reproj_paths = [s.decode() if isinstance(s, (bytes, np.bytes_)) else s
                    for s in reproj_list_bytes]
    print(f"{len(reproj_paths)} reproj files in reference cal.")

    # lvf_params + det_BC + det_chunk_map (all derivable; no cal data needed).
    lvf_path = os.path.join(args.lvf_params_dir,
                            f'lvf_params_D{args.detector}.npy')
    if not os.path.exists(lvf_path):
        raise SystemExit(f"lvf_params file not found: {lvf_path}")
    lvf_params = load_lvf_params(f'lvf_params_D{args.detector}.npy',
                                 input_dir=args.lvf_params_dir)
    print(f"lvf_params:    {lvf_path}")
    det_chunk_map, _, _, _ = make_stripped_chunk_map(
        args.detector, num_subchannels=args.num_subchannels,
        num_channels=args.num_channels, num_columns=args.num_columns,
        oversample_factor=1, lvf_params=lvf_params,
        calibration_dir=args.calibration_dir)
    bc_path = os.path.join(args.calibration_dir,
                           DET_BC_TEMPLATE.format(detector=args.detector))
    det_BC = fits.getdata(bc_path)
    print(f"det_chunk_map: shape={det_chunk_map.shape}, n_chunks="
          f"{det_chunk_map.max()+1}; det_BC from {bc_path}")

    # Extract per-frame MJD + WCS (cached across runs).
    meta_cache_path = (args.metadata_cache
                       or DEFAULT_METADATA_CACHE_TEMPLATE.format(
                           detector=args.detector))
    print(f"meta cache:    {meta_cache_path}")
    t0 = time.time()
    wcs_list, mjds, errors = extract_metadata_for_reproj_list(
        reproj_paths, num_workers=args.num_workers,
        desc=f"Reading {len(reproj_paths)} (reproj + source FITS) headers "
             f"with {args.num_workers} workers...",
        metadata_cache_path=meta_cache_path)
    print(f"metadata stage finished in {time.time() - t0:.1f}s "
          f"({len(errors)} read errors).")

    # Loop channels.
    summary = []
    for ch in channels:
        tag = make_tag(args.detector, args.num_subchannels,
                       args.num_channels, args.num_columns,
                       ch, args.file_suffix)
        npz_path = os.path.join(args.out_dir, f'zodi_pred_{tag}.npz')
        print(f"\n=== Ch{ch} ===")
        print(f"  out: {npz_path}")
        if args.skip_existing and os.path.exists(npz_path):
            print(f"  already exists; skipping (--skip-existing)")
            continue
        try:
            t_b = time.time()
            result = build_for_channel_theoretical(
                ch=ch,
                num_subchannels=args.num_subchannels,
                num_channels=args.num_channels,
                num_columns=args.num_columns,
                wcs_list=wcs_list, mjds=mjds,
                det_BC=det_BC, det_chunk_map=det_chunk_map,
                reproj_list_bytes=reproj_list_bytes,
                detector=args.detector,
                model_name=args.model,
                grid_size=args.grid_size,
                nprocesses=args.nprocesses,
                subchannel_padding=args.subchannel_padding,
            )
            save_predictions_npz(npz_path, result)
            print(f"  build: {time.time() - t_b:.1f}s")
            summary.append({'ch': ch, 'wavelength_um': result['wavelength_um']})
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    print()
    print(f"=== Summary: built {len(summary)} / {len(channels)} channels ===")
    if summary:
        for row in summary:
            print(f"  Ch{row['ch']:>2}: wavelength = {row['wavelength_um']:.4f} um")


if __name__ == '__main__':
    main()
