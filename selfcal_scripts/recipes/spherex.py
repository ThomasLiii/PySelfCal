"""SPHEREx: the fields, recipes and tilings of the NEP and SEP campaigns.

Shared by the run scripts (``selfcal_scripts/runs/``), the gates, the transfer-function kit and
the notebooks::

    from selfcal.instruments import spherex
    from selfcal_scripts.recipes.spherex import NUMCOL3, nep

    if __name__ == "__main__":
        nep(3).calibrate(NUMCOL3, jobs=spherex.channels(1, 34))

A recipe's ``name`` is its products' suffix, so the names below are those of the existing
products; ``field.result(recipe, jobs=...)`` finds them.
"""
import selfcal as sc
from selfcal.instruments import spherex

from .site import ORCA, OUTPUTS

__all__ = ['nep', 'sep', 'nep_qr2', 'NUMCOL3', 'POLY_K1', 'LINES3', 'SUBCHANNEL', 'NEP_OVERLAP_TILES',
           'NEP_BIG6_TILES', 'SEP6_TILES', 'SEP_HALVES', 'multiline3', 'MULTILINE3_PASSES', 'pahfit_tiled']


# ---------------------------------------------------------------------------------------------- fields
def nep(detector, num_col=3, compute=ORCA, outputs=OUTPUTS):
    """The NEP deep field (2026W17 data), 6.2" pixels."""
    return sc.Field(f'{outputs}/SPHEREx_NEP_2026W17_D{detector}_6p2arcsec', sc.SPHEREx(detector, num_col=num_col),
                    6.2, compute=compute)


def sep(detector, num_col=3, compute=ORCA, outputs=OUTPUTS):
    """The SEP deep field (2025 data), 6.2" pixels."""
    return sc.Field(f'{outputs}/SPHEREx_SEP_2025_D{detector}_6p2arcsec', sc.SPHEREx(detector, num_col=num_col),
                    6.2, compute=compute)


def nep_qr2(detector, num_col=3, compute=ORCA, outputs=OUTPUTS):
    """The NEP QR2 data set, 6.2" pixels (the continuum gates' field)."""
    return sc.Field(f'{outputs}/SPHEREx_nep_qr2_det{detector}_6p2arcsec', sc.SPHEREx(detector, num_col=num_col),
                    6.2, compute=compute)


# ---------------------------------------------------------------------------------------------- channel maps
#: The channel-map production recipe since 2026-09-29: NumCol 3, no column polynomial (the transfer
#: function showed NumCol 10 with a linear column polynomial overfits), a 2x oversampled coadd.
NUMCOL3 = sc.Recipe(sc.continuum(smooth=0.1),
                    fit=sc.Fit(50, clip=5.0),
                    coadd=sc.Coadd(clip=2.0, oversample=2, ignore_flags=[21]),
                    name='damp0p1_reg0p1_outThresh5_sigma2')

#: The NumCol 10 era fiducial: a linear polynomial along the columns (weight 0.5).
POLY_K1 = sc.Recipe(sc.continuum(smooth=0.1, poly_prior=sc.Poly(1, weight=0.5)),
                    fit=sc.Fit(50, clip=5.0),
                    coadd=sc.Coadd(clip=2.0, oversample=2, ignore_flags=[21]),
                    name='damp0p1_reg0p1_outThresh5_sigma2_polyK1')


# ---------------------------------------------------------------------------------------------- PAH lines
#: The three PAH features of the multi-line fits, with the shipped realistic templates.
LINES3 = [spherex.line(name, damping=5e-3) for name in ('aromatic', 'aliphatic', 'plateau')]

#: Clip within each subchannel's chunks (the N-pass passes' clip).
SUBCHANNEL = sc.ChunkGroups.along('subchannel')

#: The subchannel window of the multi-line fits (3.1 to 3.6 um on Detector 4).
MULTILINE3_WINDOW = range(200, 321)


def multiline3(name, *, iterations=300, threads=32, lines=LINES3, window=MULTILINE3_WINDOW, degree=2,
               segments=None):
    """The multi-line recipe: a constant sky plus ``lines``, the offset a hard degree-``degree``
    polynomial along the subchannels of ``window`` (per column), per-pixel shot-noise weights,
    the source-mask bit ignored; no mosaic (tiled and N-pass runs)."""
    return sc.Recipe(sc.spectral(lines, polynomial=sc.Poly(degree, window=window, segments=segments)),
                     fit=sc.Fit(iterations, clip=5.0, ignore_flags=[21], shot_noise_weights=True),
                     coadd=None, numerics=sc.Numerics(threads), name=name)


def pahfit_tiled(name, *, iterations):
    """The tiled PAH 3.29 um recipe (2026-08): a constant sky plus the catalogue PAH line (damping 5e-3),
    free offsets smoothed along the columns with a linear column polynomial and a cubic along the
    subchannels 200-259 (weight 100), shot-noise weights, the source-mask bit ignored; no mosaic."""
    model = sc.spectral([sc.Sky('pah_3p29', times=sc.catalog('pah_3p29'), damping=5e-3)],
                        poly_prior=[sc.Poly(1, weight=0.5), sc.Poly(3, along='subchannel', window=range(200, 260),
                                                                  weight=100.0)])
    return sc.Recipe(model, fit=sc.Fit(iterations, clip=5.0, ignore_flags=[21], shot_noise_weights=True), coadd=None,
                     name=name)


#: The N-pass schedule of the multi-line fits: five passes ending on a sky; every clip per subchannel.
MULTILINE3_PASSES = sc.Passes(5, order='offset_first',
                              init_clip=sc.Clip(2.5, per=SUBCHANNEL, ignore_flags=[21]),
                              sky_clip=sc.Clip(5.0, per=SUBCHANNEL),
                              offset=sc.Refit(4, clip=sc.Clip(2.5, per=SUBCHANNEL)))


# ---------------------------------------------------------------------------------------------- tilings
#: NEP D4: 16 adaptive-overlap tiles (design: analysis/analysis_script/multiline/design_overlap_tiles.py).
NEP_OVERLAP_TILES = {
    'M01': (0, 6489, 0, 4723), 'M02': (0, 4911, 3145, 6938), 'M03': (3333, 6799, 0, 4972),
    'M04': (4911, 6199, 4972, 6188), 'M05': (0, 5048, 4610, 7558), 'M06': (5048, 6199, 6188, 7558),
    'M07': (0, 6217, 7558, 12672), 'M08': (3061, 7249, 7558, 12672), 'M09': (4699, 7567, 0, 4991),
    'M10': (6199, 7567, 4991, 6120), 'M11': (5989, 12676, 0, 4801), 'M12': (7567, 12676, 3073, 6870),
    'M13': (6199, 6907, 6120, 7060), 'M14': (6907, 12676, 5520, 7660), 'M15': (4849, 7622, 7060, 12672),
    'M16': (6044, 12676, 7060, 12672)}

#: NEP D4: six large tiles.
NEP_BIG6_TILES = {
    'B01': (0, 5401, 0, 6192), 'B02': (0, 5401, 6192, 12672), 'B03': (5401, 6879, 0, 6194),
    'B04': (5401, 6879, 6194, 12672), 'B05': (6879, 12676, 0, 5940), 'B06': (6879, 12676, 5939, 12672)}

#: SEP D4: the west and east halves (order matters: the N-pass passes keep a frame two tiles share
#: in the first one listed).
SEP_HALVES = {'WEST': (0, 12343, 0, 6177), 'EAST': (0, 12343, 6177, 12303)}

#: SEP D4: six tiles.
SEP6_TILES = {
    'S01': (0, 5031, 0, 6294), 'S02': (0, 5031, 6294, 12303), 'S03': (5031, 6508, 0, 6376),
    'S04': (5031, 6508, 6376, 12303), 'S05': (6508, 12343, 0, 5375), 'S06': (6508, 12343, 5375, 12303)}
