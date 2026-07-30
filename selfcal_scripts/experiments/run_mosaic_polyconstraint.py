"""Build mosaic FITS files for the polynomial-constraint calibration variants
defined in ``benchmarks/run_cal_baseline_test.py``: poly_k1 = one LVF chunk
map (K=1, where K = number of chunk maps) with a linear column-polynomial
constraint at NumCol=10; poly_k2 = the same plus a second 32-stripe
detector-fixed readout map (K=2).

Matches the production ``make_mosaic`` recipe (as run by the generic runner's
mosaic step; formerly the ``run_cal.py`` driver, kept in git history): the
LVF chunk map goes through ``make_spherex_stripped_offset_map``
(mean-preserving spline);
for ``poly_k2`` the second 32-stripe detector-fixed map uses the simple
chunk_to_det broadcast (``det_offset_funcs[1] = None``). Wavelength coadd
is appended via ``wav_coadd`` so the FITS layout matches the existing
production mosaics on disk.

The per-frame DC scalar saved in K=2 cal files is folded into ``offsets[0]``
automatically by ``Mosaicker.load_calibration``.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
import sys
import shutil
import time
import gc
import glob as glob_module
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm

# Sibling `from run_cal_baseline_test import ...` lives at
# <repo>/selfcal_scripts/benchmarks/run_cal_baseline_test.py.
sys.path.append(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'benchmarks'))

from selfcal.pipeline import pipeline_wrapper
from selfcal._state import set_hdd_io_limit
from selfcal.instruments.spherex.spherex_utility import (
    load_calibration,
    load_lvf_params,
    make_stripped_chunk_map,
    make_stripped_chunk_valid_mask,
    fast_vertical_dist,
    make_spherex_stripped_offset_map,
)
from run_cal_baseline_test import (
    prepare_detector_inputs,
    prepare_channel_inputs,
    build_detector_stripe_map,
    VARIANT_NUMCOL,
)


def variant_chunk_maps_and_funcs(variant, detector_inputs, channel_inputs,
                                 frame_setting, mosaic_oversample_factor):
    """Per-variant ``(chunk_maps, det_offset_funcs)`` for ``Mosaicker.make_mosaic``.

    Map 0 (the LVF map) is the same for every variant: the grid-resolution
    stripped chunk map rendered through ``make_spherex_stripped_offset_map``.
    For ``poly_k2`` we add a second grid-resolution 32-stripe detector-fixed
    map using the default ``chunk_to_det`` lookup (``det_offset_func=None``).
    """
    grid_chunk_map = detector_inputs['grid_chunk_map']
    chunk_valid_mask = channel_inputs['chunk_valid_mask']
    lvf_params = detector_inputs['lvf_params']
    r_edges = detector_inputs['r_edges']
    x_edges = detector_inputs['x_edges']

    partial_make_offset_map = partial(
        make_spherex_stripped_offset_map,
        chunk_valid_mask=chunk_valid_mask,
        lvf_params=lvf_params,
        r_edges=r_edges,
        x_edges=x_edges,
        tot_subchannels=frame_setting['NumSub'] * frame_setting['NumCh'] + 2,
        num_columns=frame_setting['NumCol'],
        fill_invalid=True,
    )

    chunk_maps = [grid_chunk_map]
    det_offset_funcs = [partial_make_offset_map]

    if variant in ('poly_k2', 'poly_k2_fixed'):
        # 32-stripe detector-fixed map at the mosaic oversample factor.
        # 60 / 64x30 / 60 in detector px → 120 / 128x30 / 120 at oversample=2.
        ds = mosaic_oversample_factor
        grid_stripe_map = build_detector_stripe_map(
            (grid_chunk_map.shape[0], grid_chunk_map.shape[1]),
            mid_width=64 * ds,
            edge_width=60 * ds,
            dtype=grid_chunk_map.dtype,
        )
        chunk_maps.append(grid_stripe_map)
        det_offset_funcs.append(None)  # chunk_to_det fallback (stripe lookup)

    return chunk_maps, det_offset_funcs


if __name__ == "__main__":
    # selfcal library logs -> plain stdout, matching the historical print()
    # console output byte-for-byte (log parsers match on these lines).
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # ----------------------------- Settings -----------------------------
    base_frame_setting = {
        'Detector': 5,
        'NumSub': 10,
        'NumCh': 34,
        'NumCol': 3,
    }
    selfcal_config = pipeline_wrapper.PipelineConfig(
        output_dir='/mnt/md124/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_nep_qr2_det{base_frame_setting["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2,
    )

    mosaic_kwargs = {
        'apply_mask': True,
        'apply_weight': False,
        'make_std_map': True,
        'apply_sigma_clipping': True,
        'sigma': 2.0,
        'ignore_list': [21],
        'cache_batch_size': 20,
        'coadd_batch_size': 30,
        'cache_intermediate': True,
        'max_workers': 32,
    }

    mosaic_oversample_factor = 2

    CACHE_DIR = '/home/thomasli/selfcal-project/selfcal/cache/'

    # Variant => (NumCol, file_suffix) for cal/mosaic naming. The 'baseline'
    # variant targets the existing NumCol=3 regression-test cal file
    # (`cal_*_baseline_after_poly_off.h5`); the new variants use the cal files
    # produced by run_cal_baseline_test.py.
    # (num_col, cal_suffix, mos_suffix). When the cal and mosaic should share
    # a suffix, pass the same value twice. For the diagnostic 'production'
    # variant we keep the original cal suffix but write a `_remosaic_v2` mosaic
    # so the March 2026 production mosaic (5GB) isn't clobbered.
    # 'multi_chunk_pr' targets the regression-reference cal written when the
    # multi-chunk-map schema first landed; its on-disk suffix
    # '_baseline_after_commit3' is historical and must match the existing file.
    # The '*_fixed' rows target the cal files currently produced by
    # run_cal_baseline_test.py (its FILE_SUFFIX appends '_fixed'); the
    # un-suffixed rows are the earlier pre-fix generation of the same
    # variants, kept so old mosaics remain reproducible.
    VARIANT_SPEC = {
        'baseline':         (3,  '_baseline_after_poly_off',           '_baseline_after_poly_off'),
        'production':       (3,  '_damp0p1_reg0p1_outThresh5_sigma2',  '_damp0p1_reg0p1_outThresh5_sigma2_remosaic_v2'),
        'multi_chunk_pr':   (3,  '_baseline_after_commit3',            '_baseline_after_commit3_mosaic'),
        'oldx0_off':        (3,  '_baseline_oldx0_off',                '_baseline_oldx0_off'),
        'scalar_off':       (3,  '_baseline_scalar_off',               '_baseline_scalar_off'),
        'poly_off':         (3,  '_baseline_poly_off',                 '_baseline_poly_off'),
        'poly_off_fixed':   (3,  '_baseline_poly_off_fixed',           '_baseline_poly_off_fixed'),
        'poly_k1':          (10, '_baseline_poly_k1',                  '_baseline_poly_k1'),
        'poly_k1_fixed':    (10, '_baseline_poly_k1_fixed',            '_baseline_poly_k1_fixed'),
        'poly_k2':          (10, '_baseline_poly_k2',                  '_baseline_poly_k2'),
        'poly_k2_fixed':    (10, '_baseline_poly_k2_fixed',            '_baseline_poly_k2_fixed'),
    }
    VARIANTS = ['poly_off_fixed', 'poly_k1_fixed', 'poly_k2_fixed']

    HDD_IO_LIMIT = 20
    chs = [[3]]
    # ----------------------------- End of Settings -----------------------------

    set_hdd_io_limit(HDD_IO_LIMIT)

    nvme_reproj_dir = os.path.join(CACHE_DIR, f'reproj_nvme_{selfcal_config.run_name}')
    os.makedirs(nvme_reproj_dir, exist_ok=True)

    hdd_reproj_files = sorted(glob_module.glob(os.path.join(selfcal_config.reproj_dir, '*.h5')))

    def copy_to_nvme(src_path):
        dst = os.path.join(nvme_reproj_dir, os.path.basename(src_path))
        if not os.path.exists(dst):
            shutil.copy2(src_path, dst)
        return dst

    print(f"Copying {len(hdd_reproj_files)} reproj files to NVMe ({nvme_reproj_dir})...")
    t_copy = time.time()
    with ThreadPoolExecutor(max_workers=HDD_IO_LIMIT or 20) as ex:
        for _ in tqdm(ex.map(copy_to_nvme, hdd_reproj_files),
                      total=len(hdd_reproj_files), desc="HDD->NVMe", unit="file"):
            pass
    print(f"Reproj file copy complete in {time.time() - t_copy:.2f} seconds.")
    set_hdd_io_limit(None)

    def remap_to_nvme(file_list):
        return [os.path.join(nvme_reproj_dir, os.path.basename(f)) for f in file_list]

    for variant in VARIANTS:
        num_col, CAL_SUFFIX, MOS_SUFFIX = VARIANT_SPEC[variant]
        frame_setting = dict(base_frame_setting, NumCol=num_col)
        frame_setting_str = '_'.join([f'{k}{v}' for k, v in frame_setting.items()])

        print(f"\n{'=' * 70}\nVariant {variant} (NumCol={frame_setting['NumCol']})\n{'=' * 70}")

        detector_inputs = prepare_detector_inputs(frame_setting, mosaic_oversample_factor)

        for ch in chs:
            job_name = f'Ch{"-".join(map(str, ch))}' if isinstance(ch, list) else ch
            t0 = time.time()

            cal_file = f'cal_{frame_setting_str}_{job_name}{CAL_SUFFIX}.h5'
            mos_file = f'mosaic_{frame_setting_str}_{job_name}{MOS_SUFFIX}.fits'
            cache_dir = os.path.join(CACHE_DIR, f'cache_{frame_setting_str}_{job_name}{MOS_SUFFIX}')
            cal_path = os.path.join(selfcal_config.cal_dir, cal_file)
            mos_path = os.path.join(selfcal_config.mos_dir, mos_file)

            if not os.path.exists(cal_path):
                raise FileNotFoundError(f"missing calibration: {cal_path}")
            if os.path.exists(mos_path):
                print(f"Mosaic file {mos_path} already exists. Skipping.")
                continue

            channel_inputs = prepare_channel_inputs(
                ch, frame_setting,
                detector_inputs['det_chunk_map'],
                detector_inputs['grid_chunk_map'],
            )

            chunk_maps, det_offset_funcs = variant_chunk_maps_and_funcs(
                variant, detector_inputs, channel_inputs,
                frame_setting, mosaic_oversample_factor,
            )
            print(f"K={len(chunk_maps)}; "
                  f"map shapes={[m.shape for m in chunk_maps]}; "
                  f"num_chunks={[int(m.max())+1 for m in chunk_maps]}")

            mm = pipeline_wrapper.Mosaicker(selfcal_config, reproj_dir=nvme_reproj_dir)
            mm.load_calibration(cal_path=cal_path)
            mm.reproj_list = remap_to_nvme(mm.reproj_list)

            maps = mm.make_mosaic(
                chunk_maps=chunk_maps,
                grid_valid_weight=channel_inputs['grid_valid_weight'],
                oversample_factor=mosaic_oversample_factor,
                det_offset_funcs=det_offset_funcs,
                cache_dir=cache_dir,
                **mosaic_kwargs,
            )

            mm.save_mosaic(mos_file=mos_file, overwrite=True)

            del mm, maps
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            gc.collect()
            print(f"Finished {variant}/{job_name} in {time.time() - t0:.2f} seconds.")
            print("-" * 50 + "\n")

    print(f"NVMe reproj cache preserved at: {nvme_reproj_dir}")
