"""Generic pipeline entry point.

    python -m selfcal_scripts.run --config selfcal_scripts/configs/<run>.toml

Pins the BLAS/OpenMP thread envs to 1 BEFORE importing numpy/selfcal (so the
in-process LSQR threadpool is the only source of parallelism), loads the TOML
config, and dispatches on its ``task``. ``--dry-run`` loads + validates the
config and resolves the instrument's jobs without executing the pipeline — a
cheap way to confirm a config resolves to the intended jobs/mode, the same
resolution the byte-equality regression checks in cache/refactor_gate/ verify.

Every real run also writes its console output (including worker processes and
any traceback) to ``<output_dir>/<run_name>/logs/<task>_<timestamp>_<pid>.log``,
headed by the command, git commit and the full config text. ``--log PATH``
chooses the file; ``--no-log`` turns it off.
"""
import argparse
import os

# Must precede any numpy/scipy/selfcal import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import logging
import sys
import time


def main():
    # selfcal library logs -> plain stdout, matching the historical print()
    # console output byte-for-byte (log parsers match on these lines).
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, help='path to the run TOML config')
    ap.add_argument('--dry-run', action='store_true',
                    help='load config + resolve jobs without running the pipeline')
    ap.add_argument('--log', metavar='PATH', default=None,
                    help='log file for this run (default: <output_dir>/<run_name>/logs/'
                         '<task>_<timestamp>_<pid>.log)')
    ap.add_argument('--no-log', action='store_true', help='do not write a log file')
    args = ap.parse_args()

    from selfcal_scripts.runner.config import load_config, get_instrument
    from selfcal_scripts.runner import pipelines

    cfg = load_config(args.config)

    run_log = None
    if not args.dry_run and not args.no_log:
        from selfcal_scripts.runner.runlog import default_log_path, start_run_log
        log_path = args.log or default_log_path(cfg)
        if log_path is None:
            print("[run] no output_dir/run_name or cache_dir in the config: running without a log file")
        else:
            repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            run_log = start_run_log(log_path, config_path=args.config, repo=repo)
            if run_log is not None:
                print(f"[run] log: {run_log.path}")
    print(f"[run] task={cfg.task} instrument={cfg.instrument} mode={cfg.mode} "
          f"run_name={cfg.resolved_run_name()}")

    if args.dry_run:
        inst = get_instrument(cfg.instrument)
        if cfg.task in ('cal', 'tiled'):
            jobs = inst.jobs(cfg.instrument_cfg)
            print(f"[dry-run] {len(jobs)} job(s): {[j.name for j in jobs]}")
            from selfcal_scripts.runner.modes import get_mode
            mode = get_mode(cfg.mode)
            print(f"[dry-run] mode={mode.name} pipeline={mode.pipeline} "
                  f"mosaic_mode={mode.mosaic_mode} requires={mode.requires}")
        print("[dry-run] config OK")
        return

    t0 = time.time()
    try:
        pipelines.run(cfg)
    except BaseException as e:
        print(f"[run] FAILED after {time.time() - t0:.1f} s: {type(e).__name__}: {e}", flush=True)
        raise
    print(f"[run] finished in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
