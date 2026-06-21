"""Tiled NEP D4 PAHfit calibration — ONE parameterized driver replacing the four
copy-pasted quadrant drivers (run_cal_NEP_{NW,NE,SW,SE}.py) + stitch_cals.py +
launch_chain.sh.

It drives :class:`selfcal.pipeline.tiled.TiledCalibration`:

  * ``make_tile_grid((12676, 12672), 2, 2, overlap_px=50, names=[NW,NE,SW,SE])``
    reproduces the four quadrant bboxes EXACTLY
    (NW=(0,6363,0,6361) NE=(0,6363,6311,12672) SW=(6313,12676,0,6361)
     SE=(6313,12676,6311,12672)) — asserted in cache/refactor_gate.
  * ``assign_frames(frame_filter='center', halo=0)`` reproduces each quadrant's
    frame list — it calls the same ``selfcal.io.frame_select.filter_by_center``
    the quadrant drivers used (via the chunk_filter shim), on the same
    exposure-number-sorted file list and the same bbox.
  * ``run_tile`` is the migrated per-tile PAHfit recipe (OffsetModel dual
    column+subchannel poly + SkyModel.continuum_plus_pah_gaussian), byte-identical
    to the Wave-2 quadrant drivers' setup_lsqr/apply_lsqr/save_calibration.
  * ``stitch`` is the Fisher-weighted merge — byte-equal on the same inputs to the
    old stitch_cals.py (gated by cache/refactor_gate/verify_stitch.py).

Sequencing (launch_chain.sh semantics): tiles run sequentially, abort on the
first failure; per-tile peak RSS is large so the RSS guardrail below force-exits
before a kernel OOM-kill. Run with ``python -u run_cal_tiled_NEP.py``.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import shutil
import time
import gc
import glob as glob_module
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm


# --- RSS guardrail: avoid silent OOM-kill at end-of-build / Phase 4 transient ---
# (identical to the quadrant drivers): poll VmRSS, track peak, force a clean
# os._exit(2) before the kernel SIGKILLs us mid-allocation with no traceback.
RSS_POLL_SEC = 15.0
RSS_ABORT_FRACTION = 0.85
_RSS_STATE = {'peak_kb': 0, 'aborting': False}


def _read_meminfo_kb(field='MemTotal'):
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith(field + ':'):
                return int(line.split()[1])
    return 0


def _read_self_rss_kb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def _rss_guardrail_loop(mem_total_kb, abort_threshold_kb):
    while True:
        try:
            rss_kb = _read_self_rss_kb()
            if rss_kb > _RSS_STATE['peak_kb']:
                _RSS_STATE['peak_kb'] = rss_kb
            rss_gb = rss_kb / 1024 / 1024
            peak_gb = _RSS_STATE['peak_kb'] / 1024 / 1024
            total_gb = mem_total_kb / 1024 / 1024
            pct = 100.0 * rss_kb / mem_total_kb if mem_total_kb else 0
            print(f'[RSS] {time.strftime("%H:%M:%S")}  RSS={rss_gb:6.1f} GB  '
                  f'peak={peak_gb:6.1f} GB  ({pct:5.1f}% of {total_gb:.0f} GB)',
                  file=sys.stderr, flush=True)
            if rss_kb >= abort_threshold_kb and not _RSS_STATE['aborting']:
                _RSS_STATE['aborting'] = True
                print(f'\n*** RSS GUARDRAIL TRIPPED ***\n'
                      f'    RSS={rss_gb:.1f} GB >= abort threshold '
                      f'{abort_threshold_kb/1024/1024:.0f} GB '
                      f'({100*RSS_ABORT_FRACTION:.0f}% of {total_gb:.0f} GB).\n'
                      f'    Forcing clean exit before kernel OOM-kill.\n',
                      file=sys.stderr, flush=True)
                os._exit(2)
        except Exception as e:
            print(f'[RSS] poll error: {e}', file=sys.stderr, flush=True)
        time.sleep(RSS_POLL_SEC)


def _start_rss_guardrail():
    mem_total_kb = _read_meminfo_kb('MemTotal')
    abort_threshold_kb = int(mem_total_kb * RSS_ABORT_FRACTION)
    total_gb = mem_total_kb / 1024 / 1024
    abort_gb = abort_threshold_kb / 1024 / 1024
    print(f'[RSS] starting guardrail: poll={RSS_POLL_SEC:.0f}s  '
          f'abort_threshold={abort_gb:.0f} GB ({100*RSS_ABORT_FRACTION:.0f}% of {total_gb:.0f} GB)',
          flush=True)
    t = threading.Thread(target=_rss_guardrail_loop,
                         args=(mem_total_kb, abort_threshold_kb),
                         daemon=True, name='rss-guardrail')
    t.start()


def _rss_checkpoint(label):
    rss_kb = _read_self_rss_kb()
    peak_kb = max(rss_kb, _RSS_STATE['peak_kb'])
    print(f'[RSS] checkpoint {label!r}: RSS={rss_kb/1024/1024:.1f} GB  '
          f'peak so far={peak_kb/1024/1024:.1f} GB', flush=True)


_REPO_ROOT = '/home/thomasli/selfcal-project/selfcal'
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from selfcal.pipeline import pipeline_wrapper
from selfcal._state import set_hdd_io_limit
from selfcal.models.offset_model import OffsetModel, OffsetBlock
from selfcal.models.sky_model import SkyModel
from selfcal.core.solution import compute_x0_scalar_only
from selfcal.instruments.spherex.spherex_utility import (load_calibration, load_lvf_params,
    compute_column_adjacency, make_stripped_chunk_map, make_stripped_chunk_valid_mask,
    fast_vertical_dist, compute_column_polynomial_chains,
    compute_subchannel_polynomial_chains)
from selfcal.pipeline.tiled import make_tile_grid, TiledCalibration

FULL_REPROJ_DIR = '/data3/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec/reprojected'
REF_SHAPE = (12676, 12672)         # full NEP D4 mosaic
TILE_OVERLAP_PX = 50               # 50 px quadrant seam (footprints provide the blend)
TILE_NAMES = ['NW', 'NE', 'SW', 'SE']  # row-major over the 2x2 grid -> quadrant bboxes
SINGLESHOT_ITER_LIM = 400          # v15 production convergence


def prepare_detector_inputs(frame_setting, mosaic_setting_oversample):
    detector = frame_setting['Detector']
    num_subchannels = frame_setting['NumSub']
    num_channels = frame_setting['NumCh']
    num_columns = frame_setting['NumCol']

    lvf_filename = f'lvf_params_D{detector}.npy'
    lvf_params = load_lvf_params(lvf_filename)

    det_BC, det_BW = load_calibration(band=detector, calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration')
    grid_chunk_map, _, _, _ = make_stripped_chunk_map(detector, num_subchannels=num_subchannels, num_channels=num_channels, num_columns=num_columns,
                                                    oversample_factor=mosaic_setting_oversample, lvf_params=lvf_params)
    det_chunk_map, _, r_edges, x_edges = make_stripped_chunk_map(detector, num_subchannels=num_subchannels, num_channels=num_channels, num_columns=num_columns,
                                            oversample_factor=1, lvf_params=lvf_params)
    adj_info = compute_column_adjacency(det_chunk_map, num_columns)
    return {
        'lvf_params': lvf_params, 'det_BC': det_BC, 'det_BW': det_BW,
        'grid_chunk_map': grid_chunk_map, 'det_chunk_map': det_chunk_map,
        'r_edges': r_edges, 'x_edges': x_edges, 'adj_info': adj_info,
    }


def prepare_channel_inputs(ch, frame_setting, det_chunk_map, grid_chunk_map):
    num_subchannels = frame_setting['NumSub']
    num_channels = frame_setting['NumCh']
    num_columns = frame_setting['NumCol']
    if ch == 'Aromatic_PAHfit':
        # 60-subchannel PAH 3.29 um window (200-260) — matches the quadrant drivers.
        subch = np.arange(200, 260)
    else:
        raise ValueError(f"Unknown channel tag {ch!r}")
    chunk_valid_mask_padded = make_stripped_chunk_valid_mask(subch=subch, num_subchannels=num_subchannels, num_channels=num_channels,
                                    num_columns=num_columns, subchannel_padding=1)
    chunk_valid_mask = make_stripped_chunk_valid_mask(subch=subch, num_subchannels=num_subchannels, num_channels=num_channels,
                                    num_columns=num_columns, subchannel_padding=0)
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
        'chunk_valid_mask_padded': chunk_valid_mask_padded,
        'chunk_valid_mask': chunk_valid_mask,
        'det_valid_mask_padded': det_valid_mask_padded,
        'grid_valid_weight': grid_valid_weight,
    }


if __name__ == "__main__":
    # ----------------------------- Settings -----------------------------
    frame_setting = {'Detector': 4, 'NumSub': 10, 'NumCh': 34, 'NumCol': 5}

    selfcal_config = pipeline_wrapper.PipelineConfig(
        output_dir='/data3/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_NEP_2026W17_D{frame_setting["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2,
    )

    # Global solver settings (NOT per-block); reg_weight + poly constraints +
    # mean-anchor live on the OffsetBlock built in run_tile.
    calibration_kwargs = {
        'apply_mask': True,
        'apply_weight': True,
        'outlier_thresh': 5.0,
        'ignore_list': [21],
        'batch_size': 50,
        'offset_regularization': True,
        'weighted_damping': True,
        'damp_weight': 0.1,
        'damp_weight_line': 5e-3,
        'max_workers': 48,
        'postprocess_func': None,
    }
    lsqr_kwargs = {
        'atol': 1e-06, 'btol': 1e-06, 'damp': 0,
        'iter_lim': SINGLESHOT_ITER_LIM, 'precondition': True, 'solver': 'lsqr',
    }

    CACHE_DIR = '/home/thomasli/selfcal-project/selfcal/cache/'
    nvme_reproj_dir = os.path.join(CACHE_DIR, 'reproj_nvme_pahfit_region_10k')
    os.makedirs(nvme_reproj_dir, exist_ok=True)
    HDD_IO_LIMIT = 20

    # Offset constraints (same constants as the quadrant drivers).
    POLY_DEGREE = 1
    POLY_WEIGHT = 0.5
    SUBCH_POLY_DEGREE = 3
    SUBCH_POLY_WEIGHT = 100.0
    SUBCH_POLY_LO = 200
    SUBCH_POLY_HI = 259
    SUBCH_TOT = 342

    CH = 'Aromatic_PAHfit'
    # Per-tile cal filename suffix; {tile} fills NW/NE/SW/SE (matches the legacy
    # per-quadrant cal names so existing stitch/analysis tooling keeps resolving).
    FILE_SUFFIX_TMPL = ('_damp0p1_reg0p1_applyWt_PAHfit_dampL5e-3_subch60_nosrcmask'
                        '_NumCol5_full_{tile}_iter400_subchPoly3_w100_outThresh5_sigma2_polyK1')
    frame_setting_str = '_'.join(f'{k}{v}' for k, v in frame_setting.items())
    STITCHED_SUFFIX = ('_damp0p1_reg0p1_applyWt_PAHfit_dampL5e-3_subch60_nosrcmask'
                       '_NumCol5_full_NEP_iter400_stitched')
    # -------------------------- End of settings --------------------------

    set_hdd_io_limit(HDD_IO_LIMIT)
    _start_rss_guardrail()
    _rss_checkpoint('startup')

    # Tile geometry — exactly the 4 quadrant bboxes (NW/NE/SW/SE row-major).
    tiles = make_tile_grid(REF_SHAPE, 2, 2, overlap_px=TILE_OVERLAP_PX, names=TILE_NAMES)
    print("[tiled] tiles:", flush=True)
    for t in tiles:
        print(f"    {t.name}: bbox={t.bbox}", flush=True)

    # Enumerate the full D4 NEP reproj dataset, exposure-number sorted (the order
    # the quadrant drivers used; filter_by_center preserves it).
    hdd_reproj_files = sorted(
        glob_module.glob(os.path.join(FULL_REPROJ_DIR, 'exp_*_det_00.h5')),
        key=lambda p: int(os.path.basename(p).split('_')[1]))
    print(f"[tiled] {len(hdd_reproj_files)} reproj files in {FULL_REPROJ_DIR}", flush=True)

    # Detector inputs are tile-independent — build once.
    detector_inputs = prepare_detector_inputs(frame_setting, 2)
    channel_inputs = prepare_channel_inputs(CH, frame_setting,
                                            detector_inputs['det_chunk_map'],
                                            detector_inputs['grid_chunk_map'])

    tiled = TiledCalibration(hdd_reproj_files, tiles, frame_filter='center', halo=0)
    assignment = tiled.assign_frames()
    for t in tiles:
        files, _ = assignment[t.name]
        print(f"[tiled] {t.name}: {len(files)} frames "
              f"({100*len(files)/len(hdd_reproj_files):.1f}% of {len(hdd_reproj_files)})",
              flush=True)

    def copy_to_nvme(src_path):
        dst_path = os.path.join(nvme_reproj_dir, os.path.basename(src_path))
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
        return dst_path

    def run_tile(tile, files):
        """Per-tile PAHfit recipe — byte-identical to the migrated quadrant
        driver's setup_lsqr/apply_lsqr/save_calibration for this tile's frames."""
        cal_file = f'cal_{frame_setting_str}_{CH}{FILE_SUFFIX_TMPL.format(tile=tile.name)}.h5'
        cal_path = os.path.join(selfcal_config.cal_dir, cal_file)
        if os.path.exists(cal_path):
            print(f"[tiled] [{tile.name}] cal exists, skipping: {cal_path}", flush=True)
            return cal_path

        t0 = time.time()
        print(f"\n[tiled] === {tile.name}: staging {len(files)} frames -> NVMe ===", flush=True)
        set_hdd_io_limit(HDD_IO_LIMIT)
        with ThreadPoolExecutor(max_workers=HDD_IO_LIMIT or 20) as ex:
            for _ in tqdm(ex.map(copy_to_nvme, files), total=len(files),
                          desc=f"{tile.name} HDD->NVMe", unit="file"):
                pass
        set_hdd_io_limit(None)
        # Lexically sorted NVMe paths — same frame ordering the quadrant drivers
        # fed to cc.reproj_list (keeps the cal byte-identical).
        nvme_frame_list = sorted(os.path.join(nvme_reproj_dir, os.path.basename(f))
                                 for f in files)

        cc = pipeline_wrapper.Calibrator(selfcal_config, reproj_dir=nvme_reproj_dir)
        cc.reproj_list = list(nvme_frame_list)
        num_frames_run = len(cc.reproj_list)

        poly_chains, poly_stencil = compute_column_polynomial_chains(
            detector_inputs['det_chunk_map'], frame_setting['NumCol'], degree=POLY_DEGREE)
        subch_chains, subch_stencil = compute_subchannel_polynomial_chains(
            num_subchannels=SUBCH_TOT, num_columns=frame_setting['NumCol'],
            degree=SUBCH_POLY_DEGREE, subch_lo=SUBCH_POLY_LO, subch_hi=SUBCH_POLY_HI)
        poly_groups = [
            {'chains': poly_chains, 'stencil': poly_stencil, 'weight': POLY_WEIGHT},
            {'chains': subch_chains, 'stencil': subch_stencil, 'weight': SUBCH_POLY_WEIGHT},
        ]
        offset_model = OffsetModel([
            OffsetBlock(chunk_map=detector_inputs['det_chunk_map'],
                        adj_info=detector_inputs['adj_info'],
                        reg_weight=0.1,
                        poly_constraints=poly_groups,
                        mean_offset=np.zeros(num_frames_run)),
        ], use_per_frame_scalar=True)

        _rss_checkpoint(f'{tile.name} pre-setup_lsqr')
        cc.setup_lsqr(
            offset_model=offset_model,
            grid_valid_weight=channel_inputs['det_valid_mask_padded'],
            oversample_factor=1,
            sky_model=SkyModel.continuum_plus_pah_gaussian(),
            det_aux=[detector_inputs['det_BC'], detector_inputs['det_BW']],
            **calibration_kwargs)
        _rss_checkpoint(f'{tile.name} post-setup_lsqr')

        x0 = compute_x0_scalar_only(
            cc.A, cc.b, cc.ref_shape,
            scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
            num_sky_blocks=cc.num_sky_blocks,
            active_mask=getattr(cc, "active_mask", None))
        _rss_checkpoint(f'{tile.name} pre-apply_lsqr')
        cc.apply_lsqr(x0=x0, use_float32=True, n_threads=48, **lsqr_kwargs)
        _rss_checkpoint(f'{tile.name} post-apply_lsqr')
        cc.line_fisher_threshold = 10.0

        # Save with HDD paths so the cal survives NVMe cleanup.
        nvme_list = cc.reproj_list
        cc.reproj_list = [os.path.join(selfcal_config.reproj_dir, os.path.basename(f))
                          for f in nvme_list]
        cc.save_calibration(cal_file=cal_file)
        cc.reproj_list = nvme_list
        del cc; gc.collect()
        print(f"[tiled] === {tile.name} cal saved to {cal_path} "
              f"({time.time()-t0:.1f}s) ===", flush=True)
        return cal_path

    # Sequential per-tile calibration (launch_chain.sh semantics: abort on first
    # failure; per-tile peak RSS is large so they cannot overlap).
    cal_paths = tiled.run(run_tile, sequential=True)

    # Fisher-weighted stitch into one cal-shaped h5 (byte-equal to stitch_cals.py).
    stitched = os.path.join(
        selfcal_config.cal_dir,
        f'cal_{frame_setting_str}_{CH}{STITCHED_SUFFIX}.h5')
    print(f"\n[tiled] stitching {len(cal_paths)} tile cals -> {stitched}", flush=True)
    tiled.stitch(cal_paths, stitched, ref_shape=REF_SHAPE, line=True)
    print(f"[tiled] DONE. stitched cal: {stitched}", flush=True)
