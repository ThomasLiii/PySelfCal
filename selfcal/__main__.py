"""The selfcal command line: ``selfcal <command> ...`` (or ``python -m selfcal ...``).

  selfcal run SCRIPT [ARGS...]         run a script with the BLAS / OpenMP threads pinned before numpy loads
  selfcal plan SCRIPT                  print the plan of a run script (FIELD, RECIPE and RUN; nothing is run)
  selfcal adopt SCRIPT                 record the existing products of a run script as its own (made by TOML)
  selfcal convert RUN.toml [-o RUN.py] write the Python form of a TOML run config, checked to run identically
  selfcal rerun RECORD.json            run an action again from its record (--overwrite: make its products again)
  selfcal compare A B                  compare two products: byte-identical, equal values, or different and why
"""
import os
import sys

# Before numpy is imported anywhere: one BLAS / OpenMP thread per process (the solver's thread
# pool is the parallelism), as every action of the Python API sets.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
             'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'


def _run(args):
    import runpy
    script = os.path.abspath(args.script)
    sys.argv = [script] + list(args.args)
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name='__main__')


def _load_script(path):
    """A run script imported as a module under its own name (its functions are recorded so),
    without running its ``__main__`` block."""
    import importlib.util
    script = os.path.abspath(path)
    sys.path.insert(0, os.path.dirname(script))
    name = os.path.splitext(os.path.basename(script))[0]
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run_settings(module, accepted):
    return {k: v for k, v in module.RUN.items() if k in accepted}


def _plan(args):
    module = _load_script(args.script)
    if hasattr(module, 'RUN') and hasattr(module, 'FIELD'):
        plan = module.FIELD.plan(getattr(module, 'RECIPE', None), **_run_settings(
            module, ('jobs', 'tiles', 'passes', 'frames', 'overwrite', 'compute')))
        print(plan)
        if any(p.state == 'unrecorded' for p in plan.refused):
            print(f"\nTo record the unrecorded products as made by this script (each is checked): "
                  f"selfcal adopt {args.script}")
    elif hasattr(module, 'REPROJECT') and hasattr(module, 'FIELD'):
        cfg = module.FIELD.reprojection_config(**module.REPROJECT)
        print(f"Plan: reproject {module.FIELD.name} ({module.FIELD.path})\n  exposures   {cfg.reproject['input_dirs']}"
              f"\n  method      {cfg.reproject['reproj_func']}, padding {cfg.reproject['padding_pixels']}"
              f"\n  reference   {cfg.reproject['source_ref_path'] or 'ref.fits, or fitted to the exposures'}")
    else:
        print(f"{args.script}: no FIELD with RUN (or REPROJECT) to plan; a run script defines them at its top level "
              f"and runs FIELD.calibrate(RECIPE, **RUN) under its __main__ guard")
        return 2
    return 0


def _adopt(args):
    from selfcal.config import ConfigError
    module = _load_script(args.script)
    if not (hasattr(module, 'RUN') and hasattr(module, 'FIELD')):
        print(f"{args.script}: no FIELD with RUN to adopt products for")
        return 2
    try:
        adopted = module.FIELD.adopt(getattr(module, 'RECIPE', None),
                                     **_run_settings(module, ('jobs', 'tiles', 'passes', 'frames', 'compute')))
    except ConfigError as e:
        print(e)
        return 1
    print(f"adopted {len(adopted)} products" + ''.join(f"\n  {os.path.basename(p)}" for p in adopted))
    return 0


def _convert(args):
    from selfcal.run.convert import convert_file
    out = args.output or os.path.splitext(args.config)[0] + '.py'
    notes = convert_file(args.config, out, check=not args.no_check, force=args.force)
    print(f"wrote {out}")
    for n in notes:
        print(f"note: {n}")


def _rerun(args):
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    from selfcal.run.records import rerun
    print(rerun(args.record, overwrite=args.overwrite))


def _compare(args):
    from selfcal.run.compare import compare
    result = compare(args.a, args.b)
    print(result)
    return 0 if result.verdict != 'different' else 1


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='selfcal', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='run a script with the threads pinned before numpy loads')
    p.add_argument('script')
    p.add_argument('args', nargs=argparse.REMAINDER)
    p.set_defaults(func=_run)
    p = sub.add_parser('plan', help="print a run script's plan (nothing is run)")
    p.add_argument('script')
    p.set_defaults(func=_plan)
    p = sub.add_parser('adopt', help="record a run script's existing products as made by it (each is checked)")
    p.add_argument('script')
    p.set_defaults(func=_adopt)
    p = sub.add_parser('convert', help='write the Python form of a TOML run config')
    p.add_argument('config')
    p.add_argument('-o', '--output', default=None, help='the script to write (default: next to the config)')
    p.add_argument('--no-check', action='store_true', help='skip the check that the script runs identically')
    p.add_argument('--force', action='store_true', help='replace an existing output script')
    p.set_defaults(func=_convert)
    p = sub.add_parser('rerun', help='run an action again from its record')
    p.add_argument('record')
    p.add_argument('--overwrite', action='store_true', help='make the products again (default: reuse current ones)')
    p.set_defaults(func=_rerun)
    p = sub.add_parser('compare', help='compare two products (cal .h5 or mosaic .fits)')
    p.add_argument('a')
    p.add_argument('b')
    p.set_defaults(func=_compare)
    args = ap.parse_args(argv)
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
