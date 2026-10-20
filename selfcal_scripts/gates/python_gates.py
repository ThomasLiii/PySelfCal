"""The byte-equality gates: each a run of the Python API on a fixture of this machine. Their
products carry a ``py`` suffix and must equal the float64-norm goldens (``run_gates.sh`` runs them
and compares; ``make_goldens.sh`` makes the goldens from a committed tree).

    python -m selfcal_scripts.gates.python_gates continuum | spectral | e2e | npass3 | euclid

Paths are this machine's (the fixtures of ``README.md``): the outputs on the RAID and the frame fixtures in
``$SELFCAL_GATE_FIXTURES``; the scratch is the ``cache/`` of the repository this file is in.
"""
import os
import sys

import numpy as np  # noqa: F401  (imported before selfcal on purpose: the actions pin the threads)

import selfcal as sc
from selfcal.instruments import spherex

OUTPUTS = '/mnt/md124/thomasli/selfcal/outputs'
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the tree under test
FIXTURES = os.environ.get('SELFCAL_GATE_FIXTURES', '/home/thomasli/selfcal-project/gate_fixtures')
QR2_FIXTURE = f'{FIXTURES}/qr2_d3_links400'         # D3 qr2: 400 links to the field's frames (continuum)
E2E_FIXTURE = f'{FIXTURES}/qr2_d3_frames300'        # D3 qr2: 300 frames (e2e)
PAH_FIXTURE = f'{FIXTURES}/pahfit_sanity'           # D4: 150 frames (spectral, npass3)

SUBCHANNEL = sc.ChunkGroups.along('subchannel')           # clip within each subchannel's chunks
LINES = [spherex.line(n, damping=5e-3) for n in ('aromatic', 'aliphatic', 'plateau')]

QR2_D3 = sc.Field(f'{OUTPUTS}/SPHEREx_nep_qr2_det3_6p2arcsec', sc.SPHEREx(3), 6.2)
NEP_D4 = sc.Field(f'{OUTPUTS}/SPHEREx_NEP_2026W17_D4_6p2arcsec', sc.SPHEREx(4), 6.2)
NEP_D4_COL5 = sc.Field(f'{OUTPUTS}/SPHEREx_NEP_2026W17_D4_6p2arcsec', sc.SPHEREx(4, num_col=5), 6.2)
EDFN_Y = sc.Field(f'{OUTPUTS}/EDFN_Y_1p5arcsec_unifygolden', sc.Euclid(band='Y', chunks=40, strips=40), 1.5)


def continuum():
    """D3 Ch17, 300 frames, no mosaic."""
    recipe = sc.Recipe(sc.continuum(smooth=0.1), fit=sc.Fit(50, clip=5.0, ignore_flags=[], method='lsqr'), coadd=None,
                       name='unify_gate_py')
    return QR2_D3.calibrate(recipe, jobs=spherex.channel(17), frames=sc.frames_in(QR2_FIXTURE)[:300],
                            compute=sc.Compute(workers=48))


def spectral():
    """The PAH catalogue line on D4, 150 frames, no mosaic."""
    model = sc.spectral([sc.Sky('pah_3p29', times=sc.catalog('pah_3p29'), damping=0.0)],
                        poly_prior=sc.Poly(1, weight=0.5))
    recipe = sc.Recipe(model, fit=sc.Fit(20, clip=5.0, ignore_flags=[21], shot_noise_weights=True, method='lsqr'), coadd=None,
                       name='unify_gate_py')
    return NEP_D4_COL5.calibrate(recipe, jobs=spherex.window('AromaticPAHfit', subchannels=range(210, 250)),
                                 frames=sc.frames_in(PAH_FIXTURE)[:150], compute=sc.Compute(workers=48))


def e2e():
    """D3 Ch17, 300 frames, cal + full mosaic (wavelength maps included)."""
    recipe = sc.Recipe(sc.continuum(smooth=0.1, poly_prior=sc.Poly(1, weight=0.5)),
                       fit=sc.Fit(50, clip=5.0, ignore_flags=[], method='lsqr'),
                       coadd=sc.Coadd(clip=2.0, ignore_flags=[21], oversample=2), name='unify_e2e_gate_py')
    compute = sc.Compute(f'{REPO}/cache', workers=48, stage='reuse', keep_staged=True)
    return QR2_D3.calibrate(recipe, jobs=spherex.channel(17), frames=sc.frames_in(E2E_FIXTURE)[:300],
                            compute=compute)


def npass3():
    """Three lines, the hard polynomial, INIT + SKY + OFFSET."""
    recipe = sc.Recipe(sc.spectral(LINES, polynomial=sc.Poly(2, window=range(200, 321))),
                       fit=sc.Fit(20, clip=5.0, ignore_flags=[21], shot_noise_weights=True, method='lsqr'), coadd=None,
                       name='unify_npass3_gate_py')
    passes = sc.Passes(3, order='sky_first', sky_clip=sc.Clip(5.0, per=SUBCHANNEL),
                       offset=sc.Refit(clip=sc.Clip(2.5, per=SUBCHANNEL)), ends_on_offset=True)
    return NEP_D4.calibrate(recipe, jobs=spherex.window('Multiline3', subchannels=range(200, 321)), passes=passes,
                            frames=sc.frames_in(PAH_FIXTURE)[:150], compute=sc.Compute(f'{REPO}/cache', workers=48))


def euclid():
    """The EDFN stripe recipe on 3 exposures x 16 detectors."""
    model = sc.Model(offsets=[
        sc.Offsets(on='grid', per='detector', smooth=0.1, smooth_along=('row', 'col'), mean_zero=True,
                   exact_group_rows=True),
        sc.Offsets(on='col_strips', smooth_along=(), damping=0.3),
        sc.Offsets(on='row_strips', smooth_along=(), damping=0.3)])
    recipe = sc.Recipe(model, fit=sc.Fit(30, clip=5.0, float32=False, method='lsqr'),
                       coadd=sc.Coadd(clip=2.0, instrument_maps=False, min_chunk_coverage=0.0),
                       numerics=sc.Numerics(8, batch=8, mosaic_batch=8, coadd_batch=8), name='unify_gate_py')
    return EDFN_Y.calibrate(recipe, compute=sc.Compute(f'{REPO}/cache', workers=8, stage=None))


GATES = {'continuum': continuum, 'spectral': spectral, 'e2e': e2e, 'npass3': npass3, 'euclid': euclid}

if __name__ == '__main__':
    names = sys.argv[1:] or list(GATES)
    for name in names:
        print(f'=== python gate {name}', flush=True)
        print(GATES[name](), flush=True)
