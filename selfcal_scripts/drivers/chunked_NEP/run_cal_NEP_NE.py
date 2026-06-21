# DEPRECATED: superseded by run_cal_tiled_NEP.py — one parameterized
# TiledCalibration driver calibrates all 4 quadrants + Fisher-stitches them.
# Kept for reference pending archival after the tiled driver's first validated
# production run.
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
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits


# --- RSS guardrail: avoid silent OOM-kill at end-of-build / Phase 4 transient ---
# Spawn a daemon thread that polls /proc/self/status every RSS_POLL_SEC and:
#   1. prints VmRSS to stderr each poll (visible in run log)
#   2. tracks peak RSS
#   3. if RSS exceeds RSS_ABORT_FRACTION * MemTotal -> print + os._exit(2)
#   to avoid the kernel killing us mid-allocation with no traceback.
RSS_POLL_SEC = 15.0
RSS_ABORT_FRACTION = 0.85  # 85% of MemTotal (~642 GB). Bumped from 0.80 since
# v6 7k never reached LSQR iter despite peaking at 610 GB - more headroom needed
# for the actual solve loop to enter.
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
            print(
                f'[RSS] {time.strftime("%H:%M:%S")}  RSS={rss_gb:6.1f} GB  '
                f'peak={peak_gb:6.1f} GB  ({pct:5.1f}% of {total_gb:.0f} GB)',
                file=sys.stderr,
                flush=True,
            )
            if rss_kb >= abort_threshold_kb and not _RSS_STATE['aborting']:
                _RSS_STATE['aborting'] = True
                print(
                    f'\n*** RSS GUARDRAIL TRIPPED ***\n'
                    f'    RSS={rss_gb:.1f} GB >= abort threshold '
                    f'{abort_threshold_kb/1024/1024:.0f} GB '
                    f'({100*RSS_ABORT_FRACTION:.0f}% of {total_gb:.0f} GB).\n'
                    f'    Forcing clean exit before kernel OOM-kill.\n',
                    file=sys.stderr,
                    flush=True,
                )
                # os._exit bypasses Python finalizers - clean enough for our purpose
                os._exit(2)
        except Exception as e:
            print(f'[RSS] poll error: {e}', file=sys.stderr, flush=True)
        time.sleep(RSS_POLL_SEC)


def _start_rss_guardrail():
    mem_total_kb = _read_meminfo_kb('MemTotal')
    abort_threshold_kb = int(mem_total_kb * RSS_ABORT_FRACTION)
    total_gb = mem_total_kb / 1024 / 1024
    abort_gb = abort_threshold_kb / 1024 / 1024
    print(
        f'[RSS] starting guardrail: poll={RSS_POLL_SEC:.0f}s  '
        f'abort_threshold={abort_gb:.0f} GB ({100*RSS_ABORT_FRACTION:.0f}% of {total_gb:.0f} GB)',
        flush=True,
    )
    t = threading.Thread(
        target=_rss_guardrail_loop,
        args=(mem_total_kb, abort_threshold_kb),
        daemon=True,
        name='rss-guardrail',
    )
    t.start()


def _rss_checkpoint(label):
    """Print current RSS at a named phase boundary (synchronous, not in the loop)."""
    rss_kb = _read_self_rss_kb()
    peak_kb = max(rss_kb, _RSS_STATE['peak_kb'])
    print(
        f'[RSS] checkpoint {label!r}: RSS={rss_kb/1024/1024:.1f} GB  '
        f'peak so far={peak_kb/1024/1024:.1f} GB',
        flush=True,
    )

# Resolve to the repo root so SelfCal/ + selfcal_scripts/ are importable even
# though this driver lives under workspace/spectral-pah-fit/region_10k/.
_REPO_ROOT = '/home/thomasli/selfcal-project/selfcal'
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from selfcal.pipeline import PipelineWrapper
from selfcal.MakeMap import (set_hdd_io_limit, compute_x0_from_Ab,
                             OffsetModel, OffsetBlock, SkyModel)
from selfcal.core.solution import compute_x0_scalar_only
from selfcal.instruments.spherex.SPHERExUtility import load_calibration, load_lvf_params, compute_column_adjacency, \
make_stripped_chunk_map, make_stripped_chunk_valid_mask, make_spherex_stripped_offset_map, fast_vertical_dist, \
compute_column_polynomial_chains, compute_subchannel_polynomial_chains
from selfcal.instruments.spherex.wavemap import wav_coadd


# Region-7k v6: 7000 frames closest to the 10k bbox center (radial cut).
# See plot_pointing_distribution.py for selection rule (top-7000 by distance to
# 10k bbox center; r_max = 2477 px). Wider subch=60 window with smaller frame
# count keeps the SciPy LSQR f64 working buffers (~m+4n) below the OOM
# threshold (10k-frame subch=60 hit 624 GB at LSQR init; 7k-frame should
# land ~440 GB).
FULL_REPROJ_DIR = '/data3/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec/reprojected'

# --- SINGLE-SHOT v8 (no staircase) ---
# One continuous LSQR call with iter_lim=200. Tests whether the v7 staircase
# oscillation was caused by restarted-Krylov warm-restart (would NOT appear
# here since there's no restart) vs frame-count instability (WOULD still appear).
SINGLESHOT_ITER_LIM = 400  # v15 production convergence
# --- chunk filter: NW quadrant of full NEP (12676 x 12672) with 50 px overlap ---
CHUNK_BBOX = (0, 6363, 6311, 12672)  # NE quadrant of D4 NEP


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
        'lvf_params': lvf_params,
        'det_BC': det_BC,
        'det_BW': det_BW,
        'grid_chunk_map': grid_chunk_map,
        'det_chunk_map': det_chunk_map,
        'r_edges': r_edges,
        'x_edges': x_edges,
        'adj_info': adj_info
    }


def prepare_channel_inputs(ch, frame_setting, det_chunk_map, grid_chunk_map):
    num_subchannels = frame_setting['NumSub']
    num_channels = frame_setting['NumCh']
    num_columns = frame_setting['NumCol']

    if isinstance(ch, list) or isinstance(ch, np.ndarray):
        chunk_valid_mask_padded = make_stripped_chunk_valid_mask(ch=ch, num_subchannels=num_subchannels, num_channels=num_channels,
                                        num_columns=num_columns, subchannel_padding=1)
        chunk_valid_mask = make_stripped_chunk_valid_mask(ch=ch, num_subchannels=num_subchannels, num_channels=num_channels,
                                        num_columns=num_columns, subchannel_padding=0)
    elif isinstance(ch, str):
        if ch == 'Aromatic':
            subch = np.arange(225, 236)
        elif ch == 'Aliphatic':
            subch = np.arange(249, 260)
        elif ch == 'Aromatic_PAHfit':
            # v5 widens the PAH 3.29 um window back to the original
            # 60-subchannel range (200-260, +/-30 around peak at ~subch 230)
            # for the in-LSQR per-pixel continuum + line amplitude joint
            # fit. +/-3sigma around line, G(lambda_edge) ~ 0.008 - well-posed
            # continuum-line decoupling. Memory audit on 10k frames at
            # subch=60 lands ~420 GB peak RSS (well under the 600 GB
            # safety target); the 17.6k-frame OOM that motivated the
            # subch=40 reduction is not in play at 10k.
            subch = np.arange(200, 260)
        else:
            raise ValueError(f"Unknown channel tag {ch!r}")
        chunk_valid_mask_padded = make_stripped_chunk_valid_mask(subch=subch, num_subchannels=num_subchannels, num_channels=num_channels,
                                        num_columns=num_columns, subchannel_padding=1)
        chunk_valid_mask = make_stripped_chunk_valid_mask(subch=subch, num_subchannels=num_subchannels, num_channels=num_channels,
                                        num_columns=num_columns, subchannel_padding=0)

    # Pre-calculate weights safely
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
        'det_valid_mask': det_valid_mask,
        'grid_valid_mask': grid_valid_mask,
        'det_valid_mask_padded': det_valid_mask_padded,
        'det_valid_weight': det_valid_weight,
        'grid_valid_weight': grid_valid_weight
    }

def mask_bright_pixels(local_vars):
    sub_data = local_vars['sub_data']
    sub_weight = local_vars['sub_weight']

    valid_mask = sub_weight > 0
    if np.sum(valid_mask) > 0:
        threshold = np.nanpercentile(sub_data[valid_mask], 25)
        sub_data[sub_data > threshold] = np.nan

    return sub_data


def load_frame_list(path, reproj_dir):
    """Read explicit frame list. Entries may be bare basenames OR absolute paths;
    both are normalized to absolute HDD paths anchored at reproj_dir. Blank
    lines and lines starting with '#' are skipped."""
    with open(path, 'r') as f:
        raw = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]
    abs_paths = []
    for entry in raw:
        if os.path.isabs(entry):
            abs_paths.append(entry)
        else:
            abs_paths.append(os.path.join(reproj_dir, entry))
    return sorted(abs_paths)


if __name__ == "__main__":
    # ----------------------------- Start of Settings -----------------------------
    frame_setting = {
        'Detector': 4,
        'NumSub': 10,
        'NumCh': 34,
        'NumCol': 5,
    }

    selfcal_config = PipelineWrapper.PipelineConfig(
        output_dir='/data3/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_NEP_2026W17_D{frame_setting["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2
    )

    calibration_kwargs = {
        'apply_mask': True,
        'apply_weight': True,  # Poisson inverse-variance weighting (production fix). Off-peak PAH pixels get less weight, breaking sky->offset leakage.
        'outlier_thresh': 5.0,
        # drop source-mask bit (match mosaic); sources participate in LSQR for source-included spectral mosaic.
        # Verified on cirrus 1k: doubles covered pixels, raises corr(sky_total,baseline) 0.58->0.91, sky_line uncontaminated.
        # See memory project_pahfit_ignore_list_21.md.
        'ignore_list': [21],
        'batch_size': 50,
        'offset_regularization': True,
        # reg_weights moved onto the OffsetBlock built below (per-block config).
        'weighted_damping': True,
        'damp_weight': 0.1,
        # --- Spectral-fit (PAHfit) mode params: 2-block sky (continuum + line amplitude) ---
        # det_aux + spectral_fit are wired below into the cc.setup_lsqr call,
        # not in this dict (det_aux depends on detector_inputs which isn't
        # available at module load time).
        'damp_weight_line': 5e-3,  # v10: 5x v9 damping (5e-3 vs 1e-3) to test the synthesizer's hypothesis that v9 iter200's growing planar mode is an LSQR low-k near-null-space mode. If 5x damping collapses the plane gradient back toward iter40 levels at iter200, mechanism is confirmed.
        # damp_offset reverted to 0 - hybrid test (apply_weight + damp_offset=0.1) was WORSE than applyWt alone (cirrus dark_spread widened further, dark-ring re-emerged). The two levers aren't orthogonal - both reweight toward bright/well-covered chunks. See aromatic-map-tuning session.
        'max_workers': 48,
        'postprocess_func': None, #mask_bright_pixels,
    }

    lsqr_kwargs = {
        'atol': 1e-06,
        'btol': 1e-06,
        'damp': 0,
        # Per-step iter_lim for the staircase loop (overridden via
        # chunk_lsqr_kwargs['iter_lim'] = STAIRCASE_ITER_LIM each iter).
        'iter_lim': SINGLESHOT_ITER_LIM,
        'precondition': True,
        'solver': 'lsqr',
    }

    mosaic_oversample_factor = 2

    CACHE_DIR = '/home/thomasli/selfcal-project/selfcal/cache/'
    FILE_SUFFIX = f'_damp0p1_reg0p1_applyWt_PAHfit_dampL5e-3_subch60_nosrcmask_NumCol5_full_NE_iter400_subchPoly3_w100_outThresh5_sigma2_polyK1'

    # --- v14: subchannel-direction polynomial constraint inside the PAH window ---
    # Forces per-frame offset[k, :, c] to be at most a cubic in subch index s for
    # s in [200, 259], so any Gaussian-shaped PAH line emission is pushed into the
    # sky_line block (which already has the correct G(lambda) basis in the forward
    # model). Replaces v11's soft orthG penalty (which only got -6% on A_G at
    # w_orth=1.0) with a hard polynomial-basis constraint. Pattern proven at 1k
    # in v13c (subch-poly3 w=100, iter=40): A_G dropped -69%, sky_line median
    # shifted +0.0048 (clean handoff), canonical poly3 baseline preserved at R^2=0.97.
    SUBCH_POLY_DEGREE = 3
    SUBCH_POLY_WEIGHT = 100.0
    SUBCH_POLY_LO = 200   # matches the LSQR data window subch=np.arange(200, 260)
    SUBCH_POLY_HI = 259   # inclusive
    SUBCH_TOT = 342       # 10 subchannels x 34 channels + 2 padding

    # Linear column constraint weight (compute_column_polynomial_chains, degree=1)
    POLY_DEGREE = 1
    POLY_WEIGHT = 0.5

    # Channels to process - per-pixel PAH 3.29 um continuum + line amplitude
    # joint solve, 60-subch window (200-260) for well-posed cont/line decoupling.
    chs = ['Aromatic_PAHfit']
    # Max concurrent HDD reads - prevents RAID thrashing when multiple instances run.
    # Tune based on RAID config: ~4-8 for most RAID arrays. Set to None to disable.
    HDD_IO_LIMIT = 20
    # ----------------------------- End of Settings -----------------------------

    set_hdd_io_limit(HDD_IO_LIMIT)

    # ---- RSS guardrail: catch OOM before the kernel SIGKILLs us silently ----
    _start_rss_guardrail()
    _rss_checkpoint('startup')

    # ---- Region-10k mode: explicit frame list instead of glob(reproj_dir/*.h5) ----
    # REUSE the existing region-10k NVMe cache dir so the 10k staged files
    # don't need to be re-copied for this v5 staircase rerun. The standard
    # region-10k driver also writes here with KEEP_NVME_CACHE=True.
    nvme_reproj_dir = os.path.join(CACHE_DIR, 'reproj_nvme_pahfit_region_10k')
    os.makedirs(nvme_reproj_dir, exist_ok=True)

    # Enumerate full D4 NEP reproj dataset (17,647 frames)
    import glob as _glob
    hdd_reproj_files = sorted(_glob.glob(os.path.join(FULL_REPROJ_DIR, 'exp_*_det_00.h5')),
                              key=lambda p: int(os.path.basename(p).split('_')[1]))
    print(f"Found {len(hdd_reproj_files)} reproj files in {FULL_REPROJ_DIR}")

    # ----- Center-based spatial chunk filter (halo=0) -----
    # Frames whose CENTER falls in CHUNK_BBOX are kept; frame footprints (~3156x3156 px)
    # naturally extend sky coverage past chunk_bbox by ~1500 px, so the overlap with
    # adjacent chunks comes from FOOTPRINTS (not from shared frames). Stitcher Fisher-
    # weights each chunk's contribution at every shared pixel.
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from chunk_filter import load_ref_coords_table, filter_by_center
    _t = time.time()
    _rc_table = load_ref_coords_table(hdd_reproj_files)
    print(f"load_ref_coords_table on {len(hdd_reproj_files)} frames: {time.time()-_t:.1f}s")
    hdd_reproj_files, _kept_idx = filter_by_center(
        hdd_reproj_files, _rc_table, CHUNK_BBOX, halo=0)
    print(f"After NE center filter (bbox={CHUNK_BBOX}): "
          f"{len(hdd_reproj_files)} frames kept "
          f"({100*len(hdd_reproj_files)/_rc_table.shape[0]:.1f}% of 17,647 total)")

    def copy_to_nvme(src_path):
        dst_path = os.path.join(nvme_reproj_dir, os.path.basename(src_path))
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
        return dst_path

    print(f"Copying {len(hdd_reproj_files)} reproj files to NVMe ({nvme_reproj_dir})...")
    t_copy = time.time()
    with ThreadPoolExecutor(max_workers=HDD_IO_LIMIT or 20) as executor:
        for _ in tqdm(executor.map(copy_to_nvme, hdd_reproj_files),
                      total=len(hdd_reproj_files), desc="HDD->NVMe", unit="file"):
            pass
    print(f"Reproj file copy complete in {time.time() - t_copy:.2f} seconds.")

    # NVMe can handle massively parallel reads - disable the HDD I/O throttle
    set_hdd_io_limit(None)

    # Pre-build the NVMe path list once (one entry per file in frame_list.txt).
    # We assign this directly into cc.reproj_list / mm.reproj_list to override
    # the default glob(nvme_reproj_dir/*.h5) which would also pick up stale
    # files left in the cache dir from earlier runs.
    nvme_frame_list = sorted([
        os.path.join(nvme_reproj_dir, os.path.basename(f)) for f in hdd_reproj_files
    ])

    def remap_to_nvme(file_list):
        """Replace directory prefix with nvme_reproj_dir, keeping filenames."""
        return [os.path.join(nvme_reproj_dir, os.path.basename(f)) for f in file_list]

    frame_setting_str = '_'.join([f'{key}{value}' for key, value in frame_setting.items()])

    # 1. Prepare overarching detector inputs
    detector_inputs = prepare_detector_inputs(frame_setting, mosaic_oversample_factor)

    # 2. Iterate through channels
    for ch in chs:
        if isinstance(ch, list):
            job_name = f'Ch{"-".join(map(str, ch))}'
        else:
            job_name = ch
        t0 = time.time()
        print(f"Processing channel {job_name} for detector {frame_setting['Detector']}...")

        job_tag = f'{frame_setting_str}_{job_name}{FILE_SUFFIX}'
        cache_dir = f'{CACHE_DIR}cache_{job_tag}'

        # Prepare specific inputs for this channel
        channel_inputs = prepare_channel_inputs(ch, frame_setting, detector_inputs['det_chunk_map'], detector_inputs['grid_chunk_map'])

        # ----------------------------- Calibration (STAIRCASE v5) -----------------------------
        cc = PipelineWrapper.Calibrator(selfcal_config, reproj_dir=nvme_reproj_dir)
        # Override the auto-globbed reproj_list with our explicit region-10k list.
        # Calibrator.__init__ already called get_reproj_files(nvme_reproj_dir),
        # which would pick up every *.h5 sitting in the NVMe cache dir; we
        # want only the ~10k frames named in frame_list.txt.
        cc.reproj_list = list(nvme_frame_list)

        # Per-frame scalar absorbs DC; mean-anchor on map 0 chunks forces
        # within-frame structure only on the chunks. Required to avoid
        # scan-stripe residuals on narrow channel masks - see PIPELINE.md.
        num_frames_run = len(cc.reproj_list)
        poly_chains, poly_stencil = compute_column_polynomial_chains(
            detector_inputs['det_chunk_map'], frame_setting['NumCol'], degree=POLY_DEGREE,
        )
        subch_chains, subch_stencil = compute_subchannel_polynomial_chains(
            num_subchannels=SUBCH_TOT,
            num_columns=frame_setting['NumCol'],
            degree=SUBCH_POLY_DEGREE,
            subch_lo=SUBCH_POLY_LO,
            subch_hi=SUBCH_POLY_HI,
        )
        print(
            f"v14 subch poly3 constraint: window subch=[{SUBCH_POLY_LO},{SUBCH_POLY_HI}], "
            f"{subch_chains.shape[0]} chains/frame x {num_frames_run} frames = "
            f"{subch_chains.shape[0] * num_frames_run:,} rows, "
            f"stencil={subch_stencil.tolist()}, weight={SUBCH_POLY_WEIGHT}",
            flush=True,
        )
        # Two poly-constraint groups on map 0: linear column + cubic subchannel.
        poly_groups = [
            {'chains': poly_chains, 'stencil': poly_stencil, 'weight': POLY_WEIGHT},
            {'chains': subch_chains, 'stencil': subch_stencil, 'weight': SUBCH_POLY_WEIGHT},
        ]
        # Single offset block: per-frame scalar absorbs DC; mean-anchor +
        # column/subchannel poly-constraints shape the within-frame structure.
        # Lowers to the exact parallel-list kwargs the flat call used.
        offset_model = OffsetModel([
            OffsetBlock(chunk_map=detector_inputs['det_chunk_map'],
                        adj_info=detector_inputs['adj_info'],
                        reg_weight=0.1,
                        poly_constraints=poly_groups,
                        mean_offset=np.zeros(num_frames_run)),
        ], use_per_frame_scalar=True)

        _rss_checkpoint('pre-setup_lsqr')
        cc.setup_lsqr(
            offset_model=offset_model,
            grid_valid_weight=channel_inputs['det_valid_mask_padded'],
            oversample_factor=1,
            # --- Spectral-fit (PAHfit) mode ---
            # 2-block sky: x = [sky_cont | sky_line | offsets | scalar].
            # det_aux = [BC_map, BW_map] gives per-(frame, sub-pixel)
            # wavelength and per-pixel sigma; the line-amp column coefficient is
            # the Gaussian profile G(lambda_i) per row. continuum_plus_pah_gaussian()
            # is byte-identical to the legacy spectral_fit=True path.
            sky_model=SkyModel.continuum_plus_pah_gaussian(),
            det_aux=[detector_inputs['det_BC'], detector_inputs['det_BW']],
            **calibration_kwargs
        )
        _rss_checkpoint('post-setup_lsqr')

        # Single-shot apply_lsqr with iter_lim=SINGLESHOT_ITER_LIM. No staircase,
        # no warm-restart, no keep_state. One continuous Lanczos basis over
        # SINGLESHOT_ITER_LIM iterations - directly tests whether restarted-Krylov
        # was the source of the v7 staircase oscillation.
        x0 = compute_x0_scalar_only(
            cc.A, cc.b, cc.ref_shape,
            scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
            num_sky_blocks=cc.num_sky_blocks,
            active_mask=getattr(cc, "active_mask", None),
        )
        _rss_checkpoint('pre-apply_lsqr (single-shot)')
        cc.apply_lsqr(
            x0=x0, use_float32=True, n_threads=48,
            **lsqr_kwargs,  # iter_lim=SINGLESHOT_ITER_LIM
        )
        _rss_checkpoint('post-apply_lsqr (single-shot)')

        # Phase 6 recommended Fisher mask threshold - saved as cal attr only;
        # apply at read time via SelfCal.MakeMap.apply_line_fisher_mask.
        cc.line_fisher_threshold = 10.0

        # Save cal. Swap reproj_list to HDD paths so cal references survive
        # NVMe cleanup.
        cal_file = f"cal_{job_tag}.h5"
        cal_path = os.path.join(selfcal_config.cal_dir, cal_file)
        nvme_list = cc.reproj_list
        cc.reproj_list = [
            os.path.join(selfcal_config.reproj_dir, os.path.basename(f))
            for f in nvme_list
        ]
        cc.save_calibration(cal_file=cal_file)
        cc.reproj_list = nvme_list
        print(f"=== SINGLE-SHOT iter={SINGLESHOT_ITER_LIM} cal saved to {cal_path} ===", flush=True)

        # NOTE: this driver intentionally does NOT run the mosaic step.
        del cc; gc.collect()

        print(f"Finished channel {job_name} for detector {frame_setting['Detector']} in {time.time() - t0:.2f} seconds.")
        print("-" * 50 + "\n")

    # Cleanup NVMe reproj cache (skip when iterating - set KEEP_NVME_CACHE
    # to avoid re-staging from HDD on every rerun). REUSE of the region-10k
    # NVMe dir requires KEEP_NVME_CACHE=True so the standard region-10k
    # driver's cache survives this staircase run (and vice versa).
    KEEP_NVME_CACHE = True
    if not KEEP_NVME_CACHE and os.path.exists(nvme_reproj_dir):
        shutil.rmtree(nvme_reproj_dir)
        print("NVMe reproj cache cleaned up.")
    elif KEEP_NVME_CACHE and os.path.exists(nvme_reproj_dir):
        print(f"NVMe reproj cache preserved at {nvme_reproj_dir} (KEEP_NVME_CACHE=True).")
