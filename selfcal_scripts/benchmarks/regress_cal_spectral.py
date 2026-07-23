"""Fixed-subset SPECTRAL cal generator for byte-equality regression testing of
the spectral cal path.

Companion to ``regress_cal.py`` (continuum). Runs the real spectral_fit=True
path — Calibrator.setup_lsqr(spectral_fit=True, det_aux=[BC, BW]) + apply_lsqr +
save_calibration — on a fixed, sorted subset of the NVMe-staged D4 NEP reproj
cache (see NVME_DIR below), with the production PAHfit config
(selfcal_scripts/configs/pahfit.toml: NumCol=5, Aromatic_PAHfit window subch
210-250, per-frame scalar, linear column poly, damp_weight_line=0.0). This exercises the 2-block sky layout (continuum +
PAH 3.29 um line amplitude), the per-pixel Gaussian G(lambda) row coefficients,
the line-block Fisher accumulation, and the skymap_line* cal datasets — none of
which the continuum gate touches.

By default setup_lsqr is driven via the explicit sky_model= object API (the
path the standing byte-equality regression check exercises); it is byte-equal
to the deprecated spectral_fit shim it replaces. Pass --flat-kwargs to run
spectral_fit=True instead, as a legacy-path check that the old flag still
reproduces the baseline cal.

Save BEFORE a change as the baseline, then after each change diff with
diff_cal_h5.py — the offset blocks AND skymap_line/skymap_line_fisher MUST be
byte-equal:

    python regress_cal_spectral.py --suffix _gate_golden --n-frames 150   # sky_model= (default)
    python regress_cal_spectral.py --suffix _gate_flat --flat-kwargs      # legacy spectral_fit
    python diff_cal_h5.py <cal_dir>/cal_..._gate_golden.h5 \
                          <cal_dir>/cal_..._gate_flat.h5

WARNING: --batch-size and --max-workers must be IDENTICAL across baseline and
candidate runs for byte-equality (cal accumulation flush ordering depends on
batch_size; see regress_cal.py). --n-frames and --iter-lim must also match.
Frame count / iters are kept small here: bit-identity does not need science
convergence, only that the same code on the same inputs reproduces the arrays.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import glob as glob_module
import time

import numpy as np

from selfcal.pipeline import pipeline_wrapper
from selfcal._state import set_hdd_io_limit
from selfcal.core.solution import compute_x0_scalar_only
from selfcal.instruments.spherex.spherex_utility import (compute_column_polynomial_chains,
                                    make_stripped_chunk_valid_mask, fast_vertical_dist)
from run_cal_baseline_test import prepare_detector_inputs

FRAME_SETTING = {'Detector': 4, 'NumSub': 10, 'NumCh': 34, 'NumCol': 5}
MOSAIC_OVERSAMPLE = 2
# NVMe-staged copy of 1000 reprojected D4 NEP frames (run
# SPHEREx_NEP_2026W17_D4_6p2arcsec); if absent, re-stage by copying reproj *.h5
# from that run's reproj_dir (the same frame subset is required for
# byte-equality).
NVME_DIR = '/home/thomasli/selfcal-project/selfcal/cache/reproj_nvme_pahfit_sanity_1k'
# PAH 3.29 um window: 40 subchannels centered on the line (matches the
# production PAHfit config, selfcal_scripts/configs/pahfit.toml).
PAH_SUBCH = np.arange(210, 250)
POLY_DEGREE = 1
POLY_WEIGHT = 0.5


def prepare_channel_inputs_pahfit(frame_setting, det_chunk_map, grid_chunk_map):
    """Aromatic_PAHfit valid masks / weights — same computation as
    prepare_channel_inputs's string-channel branch in run_cal_baseline_test.py,
    applied to the 40-subchannel PAH window."""
    nsub = frame_setting['NumSub']
    nch = frame_setting['NumCh']
    ncol = frame_setting['NumCol']
    chunk_valid_mask_padded = make_stripped_chunk_valid_mask(
        subch=PAH_SUBCH, num_subchannels=nsub, num_channels=nch, num_columns=ncol,
        subchannel_padding=1)
    chunk_valid_mask = make_stripped_chunk_valid_mask(
        subch=PAH_SUBCH, num_subchannels=nsub, num_channels=nch, num_columns=ncol,
        subchannel_padding=0)

    det_valid_mask = chunk_valid_mask[det_chunk_map]
    det_valid_weight = fast_vertical_dist(det_valid_mask)
    if np.max(det_valid_weight) > 0:
        det_valid_weight /= np.max(det_valid_weight)
    det_valid_mask_padded = chunk_valid_mask_padded[det_chunk_map]

    grid_valid_mask = chunk_valid_mask[grid_chunk_map]
    grid_valid_weight = fast_vertical_dist(grid_valid_mask)
    if np.max(grid_valid_weight) > 0:
        grid_valid_weight /= np.max(grid_valid_weight)

    return {
        'chunk_valid_mask': chunk_valid_mask,
        'det_valid_mask_padded': det_valid_mask_padded,
        'grid_valid_weight': grid_valid_weight,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', required=True)
    ap.add_argument('--n-frames', type=int, default=150)
    ap.add_argument('--iter-lim', type=int, default=20)
    ap.add_argument('--max-workers', type=int, default=48)
    ap.add_argument('--batch-size', type=int, default=50)
    ap.add_argument('--flat-kwargs', action='store_true',
                    help='drive setup_lsqr via the deprecated spectral_fit flag instead '
                         'of the default sky_model= object (legacy-path regression '
                         'check; must still be byte-equal to the golden)')
    args = ap.parse_args()

    cfg = pipeline_wrapper.PipelineConfig(
        output_dir='/mnt/md124/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_NEP_2026W17_D{FRAME_SETTING["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2,
    )

    set_hdd_io_limit(None)
    det_inputs = prepare_detector_inputs(FRAME_SETTING, MOSAIC_OVERSAMPLE)
    ch_inputs = prepare_channel_inputs_pahfit(
        FRAME_SETTING, det_inputs['det_chunk_map'], det_inputs['grid_chunk_map'])

    cc = pipeline_wrapper.Calibrator(cfg, reproj_dir=NVME_DIR)
    files = sorted(glob_module.glob(os.path.join(NVME_DIR, '*.h5')))[:args.n_frames]
    cc.reproj_list = files
    num_frames = len(cc.reproj_list)
    print(f"SPECTRAL gate: n_frames={num_frames}  "
          f"num_chunks={int(det_inputs['det_chunk_map'].max())+1}  iter_lim={args.iter_lim}")

    poly_chains, poly_stencil = compute_column_polynomial_chains(
        det_inputs['det_chunk_map'], FRAME_SETTING['NumCol'], degree=POLY_DEGREE)
    poly_constraints_list = [[
        {'chains': poly_chains, 'stencil': poly_stencil, 'weight': POLY_WEIGHT},
    ]]

    frame_str = '_'.join(f'{k}{v}' for k, v in FRAME_SETTING.items())
    cal_file = f'cal_{frame_str}_AromaticPAHfit{args.suffix}.h5'

    common = dict(
        chunk_maps=[det_inputs['det_chunk_map']],
        grid_valid_weight=ch_inputs['det_valid_mask_padded'],
        oversample_factor=1,
        adj_infos=[det_inputs['adj_info']],
        poly_constraints_list=poly_constraints_list,
        mean_offsets_list=[np.zeros(num_frames)],
        use_per_frame_scalar=True,
        det_aux=[det_inputs['det_BC'], det_inputs['det_BW']],
        apply_mask=True, apply_weight=True, outlier_thresh=5.0, ignore_list=[21],
        batch_size=args.batch_size, offset_regularization=True, reg_weights=[0.1],
        weighted_damping=True, damp_weight=0.1, damp_weight_line=0.0,
        max_workers=args.max_workers,
    )
    t0 = time.time()
    if args.flat_kwargs:
        # Legacy-path regression check: the deprecated spectral_fit flag. Must
        # be byte-equal to the baseline generated via the default sky_model= path.
        cc.setup_lsqr(spectral_fit=True, **common)
    else:
        # Default path the standing byte-equality regression check exercises:
        # the explicit sky_model= object API. Byte-equal to the spectral_fit
        # shim it replaces.
        from selfcal.models.sky_model import SkyModel
        cc.setup_lsqr(sky_model=SkyModel.continuum_plus_pah_gaussian(), **common)
    print(f"setup_lsqr: {time.time() - t0:.2f} s  num_sky_blocks={cc.num_sky_blocks}")

    x0 = compute_x0_scalar_only(
        cc.A, cc.b, cc.ref_shape,
        scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
        num_sky_blocks=cc.num_sky_blocks,
        active_mask=cc.active_mask,
    )
    cc.apply_lsqr(x0=x0, atol=1e-06, btol=1e-06, damp=0, iter_lim=args.iter_lim,
                  precondition=True, solver='lsqr', use_float32=True, n_threads=args.max_workers)
    cc.line_fisher_threshold = 10.0

    nvme_list = cc.reproj_list
    cc.reproj_list = [os.path.join(cfg.reproj_dir, os.path.basename(f)) for f in nvme_list]
    cal_path = cc.save_calibration(cal_file=cal_file)
    print(f"wrote {cal_path}  (total {time.time()-t0:.2f} s)")


if __name__ == '__main__':
    main()
