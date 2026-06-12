"""Time-bounded end-to-end before/after harness for the perf/algo-optimizations work.

Runs the full D3 Ch17 NumCol=3 pipeline (cal setup_lsqr + apply + mosaic
cache/mean/std/sigma_clip + wav) on a fixed --n-frames subset of the NVMe-staged
reproj files, with the production 48/50 config, phase-timed (wall + peak RSS) via
the shared PhaseTracker. Same harness on the pre-change SelfCal (git checkout
<parent> -- SelfCal/) vs the changed SelfCal gives a clean per-phase delta on
identical machine state.

Usage (single-shot):
    python bench_e2e.py --n-frames 4000 --label before
    python bench_e2e.py --n-frames 4000 --label after

Usage (orchestrated before-vs-after with crash-safe SelfCal restore):
    python bench_e2e.py --before-ref <sha-or-branch> --after-ref <sha-or-branch> \\
                        --n-frames 4000 --label myrun
    # writes e2e_myrun_before_summary.txt and e2e_myrun_after_summary.txt under
    # figures/benchmark/; restores SelfCal/ to the recorded HEAD on completion
    # AND on SIGINT/SIGTERM/exception.

In orchestrated mode the script:
  1. records the current HEAD sha for SelfCal/,
  2. installs a try/finally + SIGINT/SIGTERM handlers that always restore
     SelfCal/ to that recorded sha,
  3. checks out --before-ref into SelfCal/ and re-launches the same script
     as a child subprocess (single-shot mode, label=<label>_before),
  4. checks out --after-ref into SelfCal/ and re-launches the same script
     as a child subprocess (single-shot mode, label=<label>_after),
  5. restores SelfCal/ to the recorded sha via the finally block.
The two phases run as separate child Python processes specifically because
in-process re-import of SelfCal cannot pick up a newly-checked-out version
(modules are already cached in sys.modules from the orchestrator's own
imports).

Reading the per-phase RSS columns
---------------------------------
PhaseTracker emits two RSS columns. ``peak_rss_gb`` is the max sampled
whole-process-tree RSS during the phase; for nested phases (mosaic_coadd_mean
/ _std / _sigma_clip inside mosaic_make_mosaic_total) it includes carry-over
from prior sub-phases that is still live and is therefore monotonically rising
even when individual phases free what they allocate. ``delta_rss_gb`` is
``peak_rss_gb - start_rss_gb`` where ``start_rss_gb`` is the RSS at phase
entry; this is the per-phase NEW-allocation signal. When auditing whether a
specific sub-phase allocated more memory than its baseline, read
``delta_rss_gb``, not ``peak_rss_gb``.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import signal
import subprocess
import sys


FRAME_SETTING = {'Detector': 3, 'NumSub': 10, 'NumCh': 34, 'NumCol': 3}
MOSAIC_OVERSAMPLE = 2
NVME_DIR = '/home/thomasli/selfcal-project/selfcal/cache/reproj_nvme_SPHEREx_nep_qr2_det3_6p2arcsec'
CACHE_ROOT = '/home/thomasli/selfcal-project/selfcal/cache'
BENCH_DIR = '/home/thomasli/selfcal-project/selfcal/figures/benchmark/'
REPO_ROOT = '/home/thomasli/selfcal-project/selfcal'


def _run_once(label, n_frames, max_workers):
    """Single-shot pipeline run with the production config and PhaseTracker.

    Kept as a function (rather than inline in main) so the before/after
    orchestration can subprocess-launch it twice on different SelfCal/ checkouts
    without colliding in module state.
    """
    # Imports inside the function so the orchestrator can parse args BEFORE
    # importing SelfCal (which it doesn't need; only the child does).
    import gc
    import glob as glob_module
    import shutil
    from functools import partial

    import matplotlib
    matplotlib.use('Agg')
    import numpy as np

    from SelfCal import PipelineWrapper, MakeMap
    from SelfCal.MakeMap import set_hdd_io_limit
    from SelfCal.solution import compute_x0_scalar_only
    from SelfCal.SPHERExUtility import make_spherex_stripped_offset_map
    from SelfCal.SPHERExAppendWav import wav_coadd
    from run_cal_baseline_test import prepare_detector_inputs, prepare_channel_inputs
    from benchmark_d3_ch17_poly import PhaseTracker

    os.makedirs(BENCH_DIR, exist_ok=True)

    cfg = PipelineWrapper.PipelineConfig(
        output_dir='/mnt/md124/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_nep_qr2_det{FRAME_SETTING["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2,
    )
    set_hdd_io_limit(None)

    tracker = PhaseTracker(sample_interval_s=0.5)
    tracker.start()

    with tracker.phase('detector_inputs'):
        det_inputs = prepare_detector_inputs(FRAME_SETTING, MOSAIC_OVERSAMPLE)
    with tracker.phase('channel_inputs'):
        ch_inputs = prepare_channel_inputs(
            [17], FRAME_SETTING, det_inputs['det_chunk_map'], det_inputs['grid_chunk_map'])

    files = sorted(glob_module.glob(os.path.join(NVME_DIR, '*.h5')))[:n_frames]
    frame_str = '_'.join(f'{k}{v}' for k, v in FRAME_SETTING.items())
    suffix = f'_e2e_{label}'
    cal_file = f'cal_{frame_str}_Ch17{suffix}.h5'
    cache_dir = os.path.join(CACHE_ROOT, f'cache_e2e_{label}')

    cc = PipelineWrapper.Calibrator(cfg, reproj_dir=NVME_DIR)
    cc.reproj_list = files
    num_frames = len(cc.reproj_list)
    print(f"[e2e:{label}] n_frames={num_frames}", flush=True)

    with tracker.phase('cal_setup_lsqr'):
        cc.setup_lsqr(
            chunk_maps=[det_inputs['det_chunk_map']],
            grid_valid_weight=ch_inputs['det_valid_mask_padded'],
            oversample_factor=1,
            adj_infos=[det_inputs['adj_info']],
            mean_offsets_list=[np.zeros(num_frames)],
            use_per_frame_scalar=True,
            apply_mask=True, apply_weight=False, outlier_thresh=5.0, ignore_list=[],
            batch_size=50, offset_regularization=True, reg_weights=[0.1],
            weighted_damping=True, damp_weight=0.1, max_workers=max_workers,
        )
    with tracker.phase('cal_warmstart'):
        x0 = compute_x0_scalar_only(cc.A, cc.b, cc.ref_shape,
                                    scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
                                    active_mask=cc.active_mask)
    with tracker.phase('cal_apply_lsqr'):
        cc.apply_lsqr(x0=x0, atol=1e-06, btol=1e-06, damp=0, iter_lim=50,
                      precondition=True, solver='lsqr', use_float32=True, n_threads=max_workers)
    with tracker.phase('cal_save'):
        cal_path = cc.save_calibration(cal_file=cal_file)
    del cc
    gc.collect()

    # Instrument each compute_coadd_map call by mode
    _orig = MakeMap.compute_coadd_map
    def _instr(mode, *a, **k):
        with tracker.phase(f'mosaic_coadd_{mode}'):
            return _orig(mode, *a, **k)
    MakeMap.compute_coadd_map = _instr

    try:
        mm = PipelineWrapper.Mosaicker(cfg, reproj_dir=NVME_DIR)
        with tracker.phase('mosaic_load_cal'):
            mm.load_calibration(cal_path=cal_path)
            mm.reproj_list = [os.path.join(NVME_DIR, os.path.basename(f)) for f in mm.reproj_list]
        offset_fn = partial(
            make_spherex_stripped_offset_map,
            chunk_valid_mask=ch_inputs['chunk_valid_mask'], lvf_params=det_inputs['lvf_params'],
            r_edges=det_inputs['r_edges'], x_edges=det_inputs['x_edges'],
            tot_subchannels=FRAME_SETTING['NumSub'] * FRAME_SETTING['NumCh'] + 2,
            num_columns=FRAME_SETTING['NumCol'], fill_invalid=True,
        )
        with tracker.phase('mosaic_make_mosaic_total'):
            maps = mm.make_mosaic(
                chunk_maps=[det_inputs['grid_chunk_map']],
                grid_valid_weight=ch_inputs['grid_valid_weight'],
                oversample_factor=MOSAIC_OVERSAMPLE,
                det_offset_funcs=[offset_fn], cache_dir=cache_dir,
                apply_mask=True, apply_weight=False,
                make_std_map=True, apply_sigma_clipping=True, sigma=2.0,
                ignore_list=[21], cache_batch_size=50, coadd_batch_size=50,
                cache_intermediate=True, max_workers=max_workers,
            )
    finally:
        MakeMap.compute_coadd_map = _orig

    with tracker.phase('mosaic_wav_coadd'):
        wav_coadd(det_inputs['det_BC'], det_inputs['det_BW'],
                  mean_map=maps['mean_map']['data'], std_map=maps['std_map']['data'],
                  reproj_list=mm.reproj_list, cache_list=mm.cached_list,
                  ref_shape=maps['mean_map']['data'].shape, sigma=2.0,
                  batch_size=50, max_workers=max_workers)

    tracker.stop()
    summary = tracker.summary_table()
    print("\n" + "=" * 130 + f"\n[e2e:{label}] n_frames={num_frames}\n" + summary + "\n" + "=" * 130, flush=True)
    txt = os.path.join(BENCH_DIR, f'e2e_{label}_summary.txt')
    with open(txt, 'w') as f:
        f.write(f"n_frames={num_frames}\n" + summary + "\n")
    print(f"[e2e:{label}] wrote {txt}")

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


# ---------------------------------------------------------------------------
# Orchestrated before/after with crash-safe restore
# ---------------------------------------------------------------------------

def _git(*args, capture=False):
    """Thin wrapper around git so we can swap a dry-run flag in later if needed."""
    cmd = ['git'] + list(args)
    if capture:
        out = subprocess.run(cmd, cwd=REPO_ROOT, check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return out.stdout.strip()
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return None


def _checkout_selfcal_to(ref):
    """Run `git checkout <ref> -- SelfCal/` from the repo root."""
    print(f"[orchestrator] git checkout {ref} -- SelfCal/", flush=True)
    _git('checkout', ref, '--', 'SelfCal/')


def _orchestrate(before_ref, after_ref, label, n_frames, max_workers):
    """Run the pipeline twice (before then after) with crash-safe SelfCal restore."""
    head_sha = _git('rev-parse', 'HEAD', capture=True)
    print(f"[orchestrator] recorded HEAD = {head_sha}  (restore target)", flush=True)
    print(f"[orchestrator] before_ref = {before_ref}", flush=True)
    print(f"[orchestrator] after_ref  = {after_ref}", flush=True)

    restored = {'done': False}

    def restore():
        if restored['done']:
            return
        restored['done'] = True
        try:
            _checkout_selfcal_to(head_sha)
            print(f"[orchestrator] SelfCal/ restored to {head_sha}", flush=True)
        except Exception as e:
            # Last-ditch — we're already in an exit path, so don't re-raise.
            print(f"[orchestrator] WARNING: restore failed: {e}", file=sys.stderr, flush=True)

    def _sig_handler(signum, frame):  # pylint: disable=unused-argument
        sigtxt = {signal.SIGINT: 'SIGINT', signal.SIGTERM: 'SIGTERM'}.get(signum, str(signum))
        print(f"\n[orchestrator] received {sigtxt}; restoring SelfCal/ before exit", flush=True)
        restore()
        # POSIX: 128 + signal number is the conventional shell exit code.
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    child_env = os.environ.copy()
    # The child needs to import benchmark_d3_ch17_poly + run_cal_baseline_test
    # the same way this orchestrator can — both live next to this script.
    child_cwd = os.path.dirname(os.path.abspath(__file__))

    def run_phase(ref, sub_label):
        _checkout_selfcal_to(ref)
        cmd = [sys.executable, os.path.abspath(__file__),
               '--n-frames', str(n_frames),
               '--max-workers', str(max_workers),
               '--label', sub_label]
        print(f"[orchestrator] launching child: {' '.join(cmd)}", flush=True)
        # Use check=True so a child-side failure surfaces and the finally block
        # runs the restore.
        subprocess.run(cmd, env=child_env, cwd=child_cwd, check=True)

    try:
        run_phase(before_ref, f'{label}_before')
        run_phase(after_ref, f'{label}_after')
    finally:
        restore()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-frames', type=int, default=4000)
    ap.add_argument('--label', required=True)
    ap.add_argument('--max-workers', type=int, default=48)
    ap.add_argument('--before-ref', default=None,
                    help='Git ref to checkout into SelfCal/ for the BEFORE run. '
                         'Requires --after-ref. Enables orchestrated mode.')
    ap.add_argument('--after-ref', default=None,
                    help='Git ref to checkout into SelfCal/ for the AFTER run. '
                         'Requires --before-ref. Enables orchestrated mode.')
    args = ap.parse_args()

    if (args.before_ref is None) != (args.after_ref is None):
        ap.error('--before-ref and --after-ref must be set together '
                 '(or neither, for single-shot mode).')

    if args.before_ref is not None:
        _orchestrate(args.before_ref, args.after_ref,
                     args.label, args.n_frames, args.max_workers)
    else:
        _run_once(args.label, args.n_frames, args.max_workers)


if __name__ == '__main__':
    main()
