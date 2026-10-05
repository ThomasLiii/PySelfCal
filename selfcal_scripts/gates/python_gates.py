"""The byte-equality gates, written in Python: each TOML gate config of ``configs/`` as a run
script of the Python API. Their products carry a ``py`` suffix and must equal the same goldens
as the TOML gates (``run_python_gates.sh`` runs and compares them).

    python -m selfcal_scripts.gates.python_gates continuum | spectral | e2e | npass3 | euclid | m13

Paths are this machine's (the fixtures of ``README.md``).
"""
import sys

import numpy as np  # noqa: F401  (imported before selfcal on purpose: the actions pin the threads)

import selfcal as sc
from selfcal.instruments import spherex

OUTPUTS = '/mnt/md124/thomasli/selfcal/outputs'
REPO = '/home/thomasli/selfcal-project/selfcal-memopt'
QR2_FIXTURE = '/home/thomasli/selfcal-project/selfcal/cache/reproj_nvme_SPHEREx_nep_qr2_det3_6p2arcsec'
E2E_FIXTURE = f'{REPO}/cache/reproj_nvme_SPHEREx_nep_qr2_det3_6p2arcsec'
PAH_FIXTURE = '/home/thomasli/selfcal-project/selfcal/cache/reproj_nvme_pahfit_sanity_1k'

SUBCHANNEL = sc.ChunkGroups.along('subchannel')           # clip within each subchannel's chunks
LINES = [spherex.line(n, damping=5e-3) for n in ('aromatic', 'aliphatic', 'plateau')]

QR2_D3 = sc.Field(f'{OUTPUTS}/SPHEREx_nep_qr2_det3_6p2arcsec', sc.SPHEREx(3), 6.2)
NEP_D4 = sc.Field(f'{OUTPUTS}/SPHEREx_NEP_2026W17_D4_6p2arcsec', sc.SPHEREx(4), 6.2)
NEP_D4_COL5 = sc.Field(f'{OUTPUTS}/SPHEREx_NEP_2026W17_D4_6p2arcsec', sc.SPHEREx(4, num_col=5), 6.2)
EDFN_Y = sc.Field(f'{OUTPUTS}/EDFN_Y_1p5arcsec_unifygolden', sc.Euclid(band='Y', chunks=40, strips=40), 1.5)


def continuum():
    """configs/gate_continuum_unify.toml: D3 Ch17, 300 frames, no mosaic."""
    recipe = sc.Recipe(sc.continuum(smooth=0.1), fit=sc.Fit(50, clip=5.0, ignore_flags=[]), coadd=None,
                       name='unify_gate_py')
    return QR2_D3.calibrate(recipe, jobs=spherex.channel(17), frames=sc.frames_in(QR2_FIXTURE)[:300],
                            compute=sc.Compute(workers=48))


def spectral():
    """configs/gate_spectral_unify.toml: the PAH catalogue line on D4, 150 frames, no mosaic."""
    model = sc.spectral([sc.Sky('pah_3p29', times=sc.catalog('pah_3p29'), damping=0.0)],
                        poly_prior=sc.Poly(1, weight=0.5))
    recipe = sc.Recipe(model, fit=sc.Fit(20, clip=5.0, ignore_flags=[21], shot_noise_weights=True), coadd=None,
                       name='unify_gate_py')
    return NEP_D4_COL5.calibrate(recipe, jobs=spherex.window('AromaticPAHfit', subchannels=range(210, 250)),
                                 frames=sc.frames_in(PAH_FIXTURE)[:150], compute=sc.Compute(workers=48))


def e2e():
    """configs/gate_e2e.toml: D3 Ch17, 300 frames, cal + full mosaic (wavelength maps included)."""
    recipe = sc.Recipe(sc.continuum(smooth=0.1, poly_prior=sc.Poly(1, weight=0.5)),
                       fit=sc.Fit(50, clip=5.0, ignore_flags=[]),
                       coadd=sc.Coadd(clip=2.0, ignore_flags=[21], oversample=2), name='unify_e2e_gate_py')
    compute = sc.Compute(f'{REPO}/workspace/coaddopt/cache', workers=48, stage='reuse', keep_staged=True)
    return QR2_D3.calibrate(recipe, jobs=spherex.channel(17), frames=sc.frames_in(E2E_FIXTURE)[:300],
                            compute=compute)


def npass3():
    """configs/gate_npass3_unify.toml: three lines, the hard polynomial, INIT + SKY + OFFSET."""
    recipe = sc.Recipe(sc.spectral(LINES, polynomial=sc.Poly(2, window=range(200, 321))),
                       fit=sc.Fit(20, clip=5.0, ignore_flags=[21], shot_noise_weights=True), coadd=None,
                       name='unify_npass3_gate_py')
    passes = sc.Passes(3, order='sky_first', sky_clip=sc.Clip(5.0, per=SUBCHANNEL),
                       offset=sc.Refit(2, clip=sc.Clip(2.5, per=SUBCHANNEL)), ends_on_offset=True)
    return NEP_D4.calibrate(recipe, jobs=spherex.window('Multiline3', subchannels=range(200, 321)), passes=passes,
                            frames=sc.frames_in(PAH_FIXTURE)[:150], compute=sc.Compute(f'{REPO}/cache', workers=48))


def euclid():
    """configs/gate_euclid_unify.toml: the EDFN stripe recipe on 3 exposures x 16 detectors."""
    model = sc.Model(offsets=[
        sc.Offsets(on='grid', per='detector', smooth=0.1, smooth_along=('row', 'col'), mean_zero=True,
                   exact_group_rows=True),
        sc.Offsets(on='col_strips', smooth_along=(), damping=0.3),
        sc.Offsets(on='row_strips', smooth_along=(), damping=0.3)])
    recipe = sc.Recipe(model, fit=sc.Fit(30, clip=5.0, float32=False),
                       coadd=sc.Coadd(clip=2.0, instrument_maps=False, min_chunk_coverage=0.0),
                       numerics=sc.Numerics(8, batch=8, mosaic_batch=8, coadd_batch=8), name='unify_gate_py')
    return EDFN_Y.calibrate(recipe, compute=sc.Compute(f'{REPO}/cache', workers=8, stage=None))


M13_BOXES = {
    'M01': (0, 6489, 0, 4723), 'M02': (0, 4911, 3145, 6938), 'M03': (3333, 6799, 0, 4972),
    'M04': (4911, 6199, 4972, 6188), 'M05': (0, 5048, 4610, 7558), 'M06': (5048, 6199, 6188, 7558),
    'M07': (0, 6217, 7558, 12672), 'M08': (3061, 7249, 7558, 12672), 'M09': (4699, 7567, 0, 4991),
    'M10': (6199, 7567, 4991, 6120), 'M11': (5989, 12676, 0, 4801), 'M12': (7567, 12676, 3073, 6870),
    'M13': (6199, 6907, 6120, 7060), 'M14': (6907, 12676, 5520, 7660), 'M15': (4849, 7622, 7060, 12672),
    'M16': (6044, 12676, 7060, 12672)}


def m13():
    """configs/gate_npass1_M13_unify_gate.toml: the NEP multi-line INIT on the adaptive tile M13."""
    recipe = sc.Recipe(sc.spectral(LINES, polynomial=sc.Poly(2, window=range(200, 321))),
                       fit=sc.Fit(300, clip=5.0, ignore_flags=[21], shot_noise_weights=True), coadd=None,
                       numerics=sc.Numerics(32),
                       name='multiline3_NEPovlp_UNIFYNPASS1GATEPY_STITCHED_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2')
    tiles = sc.Tiles(boxes=M13_BOXES, only=['M13'], stitched_name='{name}',
                     tile_name='multiline3_NEPovlp_{tile}_UNIFYNPASS1GATEPY_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2')
    compute = sc.Compute(f'{REPO}/cache', workers=24, stage_dir='reproj_nvme_unify_npass1_M13')
    return NEP_D4.calibrate(recipe, jobs=spherex.window('Multiline3', subchannels=range(200, 321)), tiles=tiles,
                            passes=sc.Passes(1), compute=compute)


GATES = {'continuum': continuum, 'spectral': spectral, 'e2e': e2e, 'npass3': npass3, 'euclid': euclid, 'm13': m13}

if __name__ == '__main__':
    names = sys.argv[1:] or list(GATES)
    for name in names:
        print(f'=== python gate {name}', flush=True)
        print(GATES[name](), flush=True)
