"""The transfer-function run: the production calibration and mosaic of one SPHEREx channel, on frames
that carry a simulated sky, from six inputs:

    python selfcal_scripts/transfer_function/transfer_function.py \\
        --detector 3 --channel 9 --frames <simsky_frames_dir> --ref <ref.fits> \\
        --output-dir <output_dir> --run-name TF_D3

Each input is a flag, else its environment variable, else its default (flag > env var > default).
--dry-run prints the plan and runs nothing. Another recipe: copy this script and change RECIPE (its model
carries the partition it is solved on, ``maps(num_ch=17, num_col=1)`` of ``selfcal_scripts.recipes.spherex``).
"""
import argparse
import os

import selfcal as sc
from selfcal_scripts.recipes.spherex import SLOPE_GAUGE
from selfcal_scripts.spherex import jobs

#: The production recipe of 2026-10, unchanged (SLOPE_GAUGE): per frame, a constant per subchannel and
#: a free gradient along detector x, undamped and unregularised; every observation weighted by its shot
#: noise floored at its frame's median; deflated CGLS to the chi-squared stop, no clip; the post-solve
#: plane gauge, added to the sky term and the maps; the coadd weighted the same way (mean, std and
#: 2-sigma clipped mean, flag 21 ignored). The transfer function is read off MEAN_MAP.
RECIPE = SLOPE_GAUGE

#: The channels RECIPE's chunk maps cut the band into (the channels are 1..17).
NUM_CH = 17

#: The six inputs: flags, environment variable, default, what it is.
INPUTS = (
    (('-d', '--detector'), 'DETECTOR', 3, '1..6'),
    (('-c', '--channel'), 'CHANNEL', 9, f'1..{NUM_CH}, a single LVF channel'),
    (('-f', '--frames', '--reproj-frame-dir'), 'REPROJ_FRAME_DIR', '/scratch/tf/D3_simsky_frames',
     'the simulated-sky frames, read in place'),
    (('-r', '--ref', '--ref-fits'), 'REF_FITS',
     '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D3_6p2arcsec/ref.fits', "the detector's ref.fits"),
    (('-o', '--output-dir'), 'OUTPUT_DIR', '/mnt/md124/thomasli/selfcal/outputs', "where the run's folder goes"),
    (('-n', '--run-name'), 'RUN_NAME', 'TF_D3', "the run's folder: <output-dir>/<run-name>/"),
)


def arguments(argv=None):
    """The inputs (an empty environment variable counts as unset)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flags, variable, default, what in INPUTS:
        parser.add_argument(*flags, type=type(default), default=os.environ.get(variable) or default,
                            help=f'{what}; env {variable}, default {default}')
    # The machine never changes a byte; the defaults are the processing host's.
    parser.add_argument('--scratch', default='/home/thomasli/selfcal-project/selfcal/cache/',
                        help="fast local disk for the solver's spill files and the coadd's frame cache; "
                             "default %(default)s")
    parser.add_argument('--workers', type=int, default=48, help='worker processes; default %(default)s')
    parser.add_argument('--dry-run', action='store_true', help='print the plan and run nothing')
    return parser.parse_args(argv)


def setup(args):
    """The field and the calibrate() arguments of the run."""
    compute = sc.Compute(args.scratch, workers=args.workers)
    field = sc.Field(os.path.join(args.output_dir, args.run_name), sc.SPHEREx(args.detector), 6.2, compute=compute)
    return field, dict(jobs=jobs.channel(args.channel), frames=args.frames)


def link_reference(ref, run_dir):
    """Point ``<run_dir>/ref.fits``, where the engine reads the reference grid, at ``ref``: a
    symlink (the file is not copied or modified)."""
    link = os.path.join(run_dir, 'ref.fits')
    if not os.path.isfile(ref):
        raise SystemExit(f'--ref {ref}: no such file')
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(link) and os.path.samefile(link, ref):
        return link
    if os.path.islink(link):
        os.remove(link)
    elif os.path.exists(link):        # a run folder's own grid: never replace it
        raise SystemExit(f'{link} is a file, not a link to a ref.fits: give the run another --run-name')
    os.symlink(os.path.abspath(ref), link)      # absolute: a relative target would resolve from run_dir
    return link


if __name__ == '__main__':
    args = arguments()
    field, run = setup(args)
    print(f'{link_reference(args.ref, field.path)} -> {args.ref}')
    print(field.plan(RECIPE, **run))
    if not args.dry_run:
        print(field.calibrate(RECIPE, **run))
