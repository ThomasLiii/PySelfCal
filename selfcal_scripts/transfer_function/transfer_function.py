"""The transfer-function run: the fiducial calibration and mosaic of one SPHEREx channel, on frames
that carry a simulated sky, from six inputs:

    python selfcal_scripts/transfer_function/transfer_function.py \\
        --detector 3 --channel 17 --frames <simsky_frames_dir> --ref <ref.fits> \\
        --output-dir <output_dir> --run-name TF_D3

Each input is a flag, else its environment variable, else its default (flag > env var > default).
--dry-run prints the plan and runs nothing. Another recipe: copy this script and change RECIPE.
"""
import argparse
import os

import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.spherex import POLY_K1

#: The fiducial NumCol 10 recipe (damp0p1 / reg0p1 / outThresh5 / polyK1) with a mean-only coadd:
#: the transfer function is read off MEAN_MAP, which is the same map with or without the std,
#: sigma-clipped and wavelength maps. No name, so the products carry no suffix.
FIDUCIAL = POLY_K1.replace(coadd=POLY_K1.coadd.replace(clip=None, std=False, instrument_maps=False), name='')

#: The recipe this script runs. A copy of the script changes it under a name of its own, which its
#: products carry, e.g. RECIPE = FIDUCIAL.replace(fit=sc.Fit(100, clip=5.0), name='iter100').
RECIPE = FIDUCIAL

#: The six inputs: flags, environment variable, default, what it is.
INPUTS = (
    (('-d', '--detector'), 'DETECTOR', 3, '1..6'),
    (('-c', '--channel'), 'CHANNEL', 17, '1..34, a single LVF channel'),
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
                        help="fast local disk for the solver's spill files; default %(default)s")
    parser.add_argument('--workers', type=int, default=48, help='worker processes; default %(default)s')
    parser.add_argument('--dry-run', action='store_true', help='print the plan and run nothing')
    return parser.parse_args(argv)


def setup(args):
    """The field and the calibrate() arguments of the run."""
    # No frame cache: the mean map is one pass over the frames, and nothing else reads the cache.
    compute = sc.Compute(args.scratch, workers=args.workers, cache_frames=False)
    field = sc.Field(os.path.join(args.output_dir, args.run_name), sc.SPHEREx(args.detector, num_col=10), 6.2,
                     compute=compute)
    return field, dict(jobs=spherex.channel(args.channel), frames=args.frames)


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
