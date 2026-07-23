"""Generic pipeline entry point.

    python -m selfcal_scripts.run --config selfcal_scripts/configs/<run>.toml

Pins the BLAS/OpenMP thread envs to 1 BEFORE importing numpy/selfcal (so the
in-process LSQR threadpool is the only source of parallelism), loads the TOML
config, and dispatches on its ``task``. ``--dry-run`` loads + validates the
config and resolves the instrument's jobs without executing the pipeline — a
cheap way to confirm a config resolves to the intended jobs/mode, the same
resolution the byte-equality regression checks in cache/refactor_gate/ verify.
"""
import argparse
import os

# Must precede any numpy/scipy/selfcal import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, help='path to the run TOML config')
    ap.add_argument('--dry-run', action='store_true',
                    help='load config + resolve jobs without running the pipeline')
    args = ap.parse_args()

    from selfcal_scripts.runner.config import load_config, get_instrument
    from selfcal_scripts.runner import pipelines

    cfg = load_config(args.config)
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

    pipelines.run(cfg)


if __name__ == "__main__":
    main()
