"""The selfcal command line: ``selfcal <command> ...`` (or ``python -m selfcal ...``).

  selfcal run SCRIPT [ARGS...]         run a script with the BLAS / OpenMP threads pinned before numpy loads
  selfcal plan SCRIPT                  print the plan of a run script: its FIELD (or FIELDS), RECIPE and RUN (the
                                       keyword arguments of calibrate), defined at its top level; nothing is run
  selfcal adopt SCRIPT                 record the existing products of a run script as its own (made before records)
  selfcal convert RUN.toml [-o RUN.py] write the Python form of an old TOML run config (unchecked: read it)
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
    """The keyword arguments of the script's action (its RUN; none without one)."""
    return {k: v for k, v in getattr(module, 'RUN', {}).items() if k in accepted}


def _fields(module):
    """The fields of a run script: its FIELDS (a campaign), or its FIELD."""
    return list(module.FIELDS) if hasattr(module, 'FIELDS') else [module.FIELD]


def _has_run(module):
    """A run script: FIELD (or FIELDS) with RECIPE and/or RUN at its top level."""
    return (hasattr(module, 'FIELD') or hasattr(module, 'FIELDS')) and (hasattr(module, 'RECIPE')
                                                                        or hasattr(module, 'RUN'))


def _plan(args):
    module = _load_script(args.script)
    if _has_run(module):
        unrecorded = False
        for i, field in enumerate(_fields(module)):
            plan = field.plan(getattr(module, 'RECIPE', None), **_run_settings(
                module, ('jobs', 'tiles', 'passes', 'frames', 'overwrite', 'compute', 'start', 'snapshots',
                         'monitor')))
            print(('\n' if i else '') + str(plan))
            unrecorded |= any(p.state == 'unrecorded' for p in plan.refused)
        if unrecorded:
            print(f"\nTo record the unrecorded products as made by this script (each is checked): "
                  f"selfcal adopt {args.script}")
    elif hasattr(module, 'PRECOMPUTE'):
        settings = dict(module.PRECOMPUTE)
        print(f"Plan: precompute the SPHEREx LVF parameters of detectors {list(settings.pop('detectors'))}"
              + ''.join(f"\n  {k:<11} {v}" for k, v in settings.items()))
    elif hasattr(module, 'REPROJECT') and hasattr(module, 'FIELD'):
        r = module.FIELD.reprojection_spec(**module.REPROJECT).reproject
        print(f"Plan: reproject {module.FIELD.name} ({module.FIELD.path})\n  exposures   {list(r.exposures)}"
              f"\n  method      {r.method}, padding {r.padding}"
              f"\n  reference   {r.reference or 'ref.fits, or fitted to the exposures'}")
    else:
        print(f"{args.script}: no FIELD (or FIELDS) with RUN, or REPROJECT, to plan; a run script defines them at its "
              f"top level and runs FIELD.calibrate(RECIPE, **RUN) under its __main__ guard")
        return 2
    return 0


def _adopt(args):
    from selfcal.config import ConfigError
    module = _load_script(args.script)
    if not _has_run(module):
        print(f"{args.script}: no FIELD (or FIELDS) with RUN to adopt products for")
        return 2
    status = 0
    for field in _fields(module):
        try:
            adopted = field.adopt(getattr(module, 'RECIPE', None),
                                  **_run_settings(module, ('jobs', 'tiles', 'passes', 'frames', 'compute')))
        except ConfigError as e:
            print(f"{field.name}: {e}")
            status = 1
            continue
        print(f"{field.name}: adopted {len(adopted)} products" +
              ''.join(f"\n  {os.path.basename(p)}" for p in adopted))
    return status


def _convert(args):
    from selfcal.run.convert import convert_file
    out = args.output or os.path.splitext(args.config)[0] + '.py'
    notes = convert_file(args.config, out, force=args.force)
    print(f"wrote {out} (not run: read it, and plan it with `selfcal plan {out}`)")
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
    p = sub.add_parser('convert', help='write the Python form of an old TOML run config')
    p.add_argument('config')
    p.add_argument('-o', '--output', default=None, help='the script to write (default: next to the config)')
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
    if args.command in ('run', 'plan', 'adopt') and args.script.endswith('.toml'):
        print(f"TOML configs are no longer run; convert it: selfcal convert {args.script}", file=sys.stderr)
        return 2
    return args.func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
