"""SPHEREx: the fields, recipes and tilings of the NEP and SEP campaigns.

Shared by the run scripts (``selfcal_scripts/runs/``), the gates, the transfer-function kit and
the notebooks::

    from selfcal_scripts.spherex import jobs
    from selfcal_scripts.recipes.spherex import CHANNEL_MAPS, nep

    if __name__ == "__main__":
        nep(3).calibrate(CHANNEL_MAPS, jobs=jobs.channels(1, 17))

A field is the detector alone (``sc.SPHEREx(detector)``); the partition of the detector a recipe solves
on is the recipe's (:func:`maps`: the LVF subchannels, ``NumSub10_NumCh17_NumCol1`` for the channel maps,
and the detector columns), and the jobs name their subchannels on it (``selfcal_scripts.spherex.jobs``).
A recipe's ``name`` is its products' suffix, so the names below are those of the existing
products; ``field.result(recipe, jobs=...)`` finds them. The recipes of products made before
2026-10-13 name their solver, LSQR (CGLS has been the default since), so they still describe how
those products were made.
"""
import selfcal as sc
from selfcal.models.variables import identity
from selfcal_scripts.spherex import chunks as C, terms as T

from .site import ORCA, OUTPUTS

__all__ = ['nep', 'sep', 'nep_qr2', 'maps', 'on_maps', 'CHANNEL_MAPS', 'SLOPE_GAUGE', 'SLOPE_GAUGE_VARWEIGHT',
           'SLOPE_GAUGE_COLUMNS', 'SLOPE_GAUGE_VARWEIGHT_COLUMNS', 'SUBCH_VARWEIGHT_COLUMNS',
           'SUBCH_VARWEIGHT_COLUMNS_FINAL', 'continuous_final', 'NUMCOL3', 'POLY_K1', 'LINES3', 'SUBCHANNEL',
           'NEP_OVERLAP_TILES', 'NEP_BIG6_TILES', 'SEP6_TILES', 'SEP_HALVES', 'multiline3', 'MULTILINE3_PASSES',
           'pahfit_tiled']


# ---------------------------------------------------------------------------------------------- fields
def nep(detector, compute=ORCA, outputs=OUTPUTS):
    """The NEP deep field (2026W17 data), 6.2" pixels."""
    return sc.Field(f'{outputs}/SPHEREx_NEP_2026W17_D{detector}_6p2arcsec', sc.SPHEREx(detector), 6.2,
                    compute=compute)


def sep(detector, compute=ORCA, outputs=OUTPUTS):
    """The SEP deep field (2025 data), 6.2" pixels."""
    return sc.Field(f'{outputs}/SPHEREx_SEP_2025_D{detector}_6p2arcsec', sc.SPHEREx(detector), 6.2, compute=compute)


def nep_qr2(detector, compute=ORCA, outputs=OUTPUTS):
    """The NEP QR2 data set, 6.2" pixels (the continuum gates' field)."""
    return sc.Field(f'{outputs}/SPHEREx_nep_qr2_det{detector}_6p2arcsec', sc.SPHEREx(detector), 6.2, compute=compute)


# ---------------------------------------------------------------------------------------------- chunk maps
def maps(num_sub=10, num_ch=17, num_col=1):
    """The chunk maps of the recipes: the LVF arcs cut into ``num_ch`` channels of ``num_sub``
    subchannels and ``num_col`` columns (``subchannel``, primary: the products carry its tag
    ``NumSub<num_sub>_NumCh<num_ch>_NumCol<num_col>``) and the detector columns (``detector_column``)."""
    return {'subchannel': C.subchannel_map(num_sub, num_ch, num_col), 'detector_column': sc.ChunkMap(C.detector_columns)}


def on_maps(recipe, num_sub=10, num_ch=17, num_col=1):
    """``recipe`` solved on :func:`maps` of another partition (the multi-line fits: 34 channels, 1 or 3
    columns); its name, fit and coadd unchanged."""
    return recipe.replace(model=recipe.model.replace(chunk_maps=maps(num_sub, num_ch, num_col)))


# ---------------------------------------------------------------------------------------------- channel maps
#: The channel-map recipe from 2026-09-29 to 2026-10-14 (its maps: Detectors 1 to 4, Detector 5 channels 1 to 16):
#: NumCol 3 over 34 channels, no column polynomial (the transfer function showed NumCol 10 with a linear column
#: polynomial overfits), a 2x oversampled coadd; the sky damped with weight 0.1 (the library default of that time).
NUMCOL3 = sc.Recipe(sc.continuum(smooth=0.1, damping=0.1).replace(chunk_maps=maps(num_ch=34, num_col=3)),
                    fit=sc.Fit(50, clip=5.0, method='lsqr'),
                    coadd=sc.Coadd(clip=2.0, oversample=2, ignore_flags=[21]),
                    name='damp0p1_reg0p1_outThresh5_sigma2')

#: The channel recipe of 2026-10 (the transfer-function and zodi-leak work on Ch9, NumCol 1 with 17
#: channels): subchannel constants with no constant over a frame's subchannels, a free gradient per frame along
#: detector x and a per-frame scalar (the frame's level), undamped and unregularised, no clip, every observation
#: weighted by its shot noise floored at the frame's median
#: (``sc.shot_noise()``), deflated CGLS to the chi-squared stop, the post-solve plane gauge applied on
#: the map (``sc.Gauge``: the plane is a freedom of the model, a quadratic only nearly one); the coadd weighted the same way (mean, std, 2-sigma clipped mean, flag 21
#: ignored). The gate `production` runs it on 100 frames of the qr2 set.
SLOPE_GAUGE = sc.Recipe(sc.Model(chunk_maps=maps(num_ch=17, num_col=1), sky=[sc.Sky(damping=0.0)],
                                 offsets=[sc.Offsets(on='subchannel', exclude=sc.Poly(0)), T.frame_slope(),
                                          sc.Offsets('scalar', on='detector')],
                                 weight=sc.shot_noise()),
                        fit=sc.Fit(600, clip=None, ignore_flags=[], tolerance=0.0, deflate=sc.Deflate(4),
                                   stop=sc.Stop(chi2=sc.ChiSquaredRule(0.1, window=50, every=10)),
                                   gauge=sc.Gauge('xs_term', degree=1)),
                        coadd=sc.Coadd(clip=2.0, std=True, ignore_flags=[21], instrument_maps=False),
                        numerics=sc.Numerics(30, batch=40),
                        name='slope_gauge')

#: The fiducial channel maps since 2026-10-14: the transfer function's recipe (SLOPE_GAUGE, solved on 17
#: channels and one column), the mosaics also carrying the wavelength maps (WAV_MEAN_MAP, WAV_STD_MAP). The same
#: name and the same cal; the mosaic's other maps are SLOPE_GAUGE's.
CHANNEL_MAPS = SLOPE_GAUGE.replace(coadd=SLOPE_GAUGE.coadd.replace(instrument_maps=True))

#: SLOPE_GAUGE weighted by the L2b ``VARIANCE`` instead of the shot-noise proxy (2026-10-15, the covariance
#: work): the frame layer ``variance`` (the SPHEREx reader stores it; frames made before 2026-10-15 get it
#: from ``selfcal.io.reprojection.add_layers``) weights the solve's rows by 1/sigma and the coadd by
#: 1/variance. The same model, fit, gauge and coadd.
SLOPE_GAUGE_VARWEIGHT = SLOPE_GAUGE.replace(
    model=SLOPE_GAUGE.model.replace(variables={'variance': sc.Layer()}, weight=sc.Variance(identity, of='variance')),
    name='slope_gauge_varweight')

#: The shared-column term (the quadrant-bowl work, 2026-10-17): one constant per detector column
#: (2,040) shared by EVERY frame, on the ``detector_column`` chunk map. It holds the
#: survey-mean detector-fixed pattern along x that the per-frame subchannel constants + x-slope do
#: not represent and that one-sided coverage amplifies into a footprint's outskirts (the D2 bulge,
#: the quadrant rims). The deflation eliminates the shared columns by a Schur complement.
SHARED_COLUMNS = sc.Offsets('columns', on='detector_column', per='all')

#: SLOPE_GAUGE plus the shared columns: the recipe of the Ch9 W17 maps `*_slope_gauge_colonly_defl`
#: (D1-D6, 2026-10-18; made from the workspace driver with the same term, then under the name
#: ``readout`` patched to one chunk per column). Footprint-invariant to 0.16e-3 MJy/sr on D2 and
#: the Gaussian-sky transfer function of the production recipe or better on the D3 / D6 scenes.
SLOPE_GAUGE_COLUMNS = SLOPE_GAUGE.replace(
    model=SLOPE_GAUGE.model.replace(offsets=tuple(SLOPE_GAUGE.model.offsets) + (SHARED_COLUMNS,)),
    name='slope_gauge_columns')

#: SLOPE_GAUGE_VARWEIGHT plus the shared columns (the covariance work's recipe from 2026-10-18).
SLOPE_GAUGE_VARWEIGHT_COLUMNS = SLOPE_GAUGE_VARWEIGHT.replace(
    model=SLOPE_GAUGE_VARWEIGHT.model.replace(offsets=tuple(SLOPE_GAUGE_VARWEIGHT.model.offsets) + (SHARED_COLUMNS,)),
    name='slope_gauge_varweight_columns')

#: The base recipe of the weak-mode work (2026-10-21; workspace/weak-modes/plan.md): per frame one free constant per
#: subchannel (they hold the frame's level: no per-frame scalar term and no constraint on their mean; the default
#: seed starts each at the inverse-variance weighted mean of its pixels) and the x-slope per subchannel; one
#: constant per detector column shared by every
#: frame, without its constant and slope across the columns (those two duplicate the per-frame terms exactly);
#: weighted by the frames' L2b VARIANCE layer; no clip, damping, regularisation, degenerate-mode exclusion or
#: post-solve gauge; deflated CGLS to the chi-squared stop; the coadd as SLOPE_GAUGE's (inverse-variance weights,
#: VAR_MAP). The weak-mode runs add a single-observation drop (a workspace frame hook).
SUBCH_VARWEIGHT_COLUMNS = sc.Recipe(
    sc.Model(chunk_maps=maps(num_ch=17, num_col=1), sky=[sc.Sky(damping=0.0)],
             offsets=[sc.Offsets(on='subchannel'), T.frame_slope(),
                      sc.Offsets('columns', on='detector_column', per='all', exclude=sc.Poly(1))],
             variables={'variance': sc.Layer()}, weight=sc.Variance(identity, of='variance')),
    fit=sc.Fit(600, clip=None, ignore_flags=[], tolerance=0.0, deflate=sc.Deflate(4),
               stop=sc.Stop(chi2=sc.ChiSquaredRule(0.1, window=50, every=10))),
    coadd=sc.Coadd(clip=2.0, std=True, ignore_flags=[21], instrument_maps=False),
    numerics=sc.Numerics(30, batch=40),
    name='subch_varweight_columns')

#: The final channel recipe of the weak-mode work (the user, 2026-10-21 and 10-22): SUBCH_VARWEIGHT_COLUMNS with the
#: x-slope term kept from every smooth sky pattern of the deflation but the constant (exclude=SkySignature('all'): a
#: sky field cannot hide in the per-frame slopes), and the sky pixels that a single observation covers left out of the
#: solve (Fit(min_observations=2); the coadd still covers them). Its products carry 'subch_varweight_columns_final'.
SUBCH_VARWEIGHT_COLUMNS_FINAL = SUBCH_VARWEIGHT_COLUMNS.replace(
    model=SUBCH_VARWEIGHT_COLUMNS.model.replace(offsets=tuple(
        o.replace(exclude=sc.SkySignature('all')) if o.name == 'xs_term' else o
        for o in SUBCH_VARWEIGHT_COLUMNS.model.offsets)),
    fit=SUBCH_VARWEIGHT_COLUMNS.fit.replace(min_observations=2),
    name='subch_varweight_columns_final')


def continuous_final(channel, num_sub=10, num_ch=17, num_col=1):
    """SUBCH_VARWEIGHT_COLUMNS_FINAL on :func:`maps` of ``num_sub`` x ``num_ch`` x ``num_col``, for one channel job
    (``jobs.channel(channel)``) with its per-frame subchannel constants replaced by ``terms.spectral_offsets``: a
    continuous, piecewise-linear function of the position across the LVF arcs, one coefficient per frame and
    subchannel edge of the job (its ``num_sub`` subchannels and the one on each side: ``num_sub + 3`` edges), so
    a background that varies within a subchannel (an airglow line's edge) is followed without steps. The
    x-slopes and the shared column offsets are the final recipe's. Products carry
    'subch_varweight_columns_final_continuous' (the user, 2026-10-26: the channel recipe from the weak-mode work
    on)."""
    base = on_maps(SUBCH_VARWEIGHT_COLUMNS_FINAL, num_sub, num_ch, num_col)
    offsets = (T.spectral_offsets(range(num_sub * (channel - 1), num_sub * channel + 2), position='arc_position'),) + tuple(
        o for o in base.model.offsets if not (o.on == 'subchannel' and o.times is None and o.basis is None))
    model = base.model.replace(offsets=offsets,
                               variables={**dict(base.model.variables), 'arc_position': sc.DetectorMap(T.arc_position)})
    return base.replace(model=model, name=base.name + '_continuous')


def pixels_final(channel, num_sub=10, num_ch=17, num_col=1):
    """:func:`continuous_final` with its column constants shared by every frame replaced by one constant per
    detector pixel shared by every frame, over the job's subchannels and the one beyond each end (the pixels the
    bilinear sampling of the job's observations reaches): a detector-fixed pattern of any shape, such as a dark
    current's pixel-to-pixel structure. It carries no pattern that the per-frame spectral term or the x-slopes can
    make the same way in every frame (``exclude=sc.Overlap('spectral', 'xs_term', weights='information')``: measured
    with each pixel's information, so the pixels the data barely see cannot take values that cancel such a pattern
    held by the others; counted alike, 99.2 % of the real D2 Ch9 constants' variance lay along those patterns,
    held by S and L in mirror image); every column pattern is one of its patterns, so the column term goes. Its deflation solves the per-frame terms' response alone
    (``Deflate(offsets=('spectral', 'xs_term'))``): the per-pixel constants follow none of the non-constant sky
    patterns, which cross the detector differently in every frame, and holding their response at zero keeps their
    couplings to every frame's offsets out of the deflation (real D2 / D6 Ch9: 252 / 442 s of deflation). The pixels a single observation sees leave the solve (``min_observations=2``; real D2 /
    D6 Ch9: 11 / 59 pixels): such a pixel's constant fits that one observation exactly and informs nothing else.
    Products carry 'subch_varweight_pixels_final_continuous'."""
    base = continuous_final(channel, num_sub, num_ch, num_col)
    first, last = num_sub * (channel - 1) - 1, num_sub * channel + 2
    offsets = tuple(o for o in base.model.offsets if o.on != 'detector_column') + (
        sc.Offsets('pixels', on='pixel', per='all', exclude=sc.Overlap('spectral', 'xs_term', weights='information'),
                   min_observations=2),)
    maps = {'subchannel': base.model.chunk_maps['subchannel'],
            'pixel': C.pixel_map(first, last, num_sub, num_ch, num_col)}
    return base.replace(model=base.model.replace(offsets=offsets, chunk_maps=maps),
                        fit=base.fit.replace(deflate=base.fit.deflate.replace(offsets=('spectral', 'xs_term'))),
                        name='subch_varweight_pixels_final_continuous')


#: The NumCol 10 era fiducial: a linear polynomial along the columns (weight 0.5), 34 channels of 10 columns; the sky
#: damped with weight 0.1 (the library default of that time).
POLY_K1 = sc.Recipe(sc.continuum(smooth=0.1, poly_prior=sc.Poly(1, weight=0.5), damping=0.1).replace(chunk_maps=maps(num_ch=34, num_col=10)),
                    fit=sc.Fit(50, clip=5.0, method='lsqr'),
                    coadd=sc.Coadd(clip=2.0, oversample=2, ignore_flags=[21]),
                    name='damp0p1_reg0p1_outThresh5_sigma2_polyK1')


# ---------------------------------------------------------------------------------------------- PAH lines
#: The three PAH features of the multi-line fits, with the shipped realistic templates.
LINES3 = [T.line(name, damping=5e-3) for name in ('aromatic', 'aliphatic', 'plateau')]

#: Clip within each subchannel's chunks (the N-pass passes' clip).
SUBCHANNEL = sc.ChunkGroups.along('subchannel')

#: The subchannel window of the multi-line fits (3.1 to 3.6 um on Detector 4, over 34 channels).
MULTILINE3_WINDOW = range(200, 321)


def multiline3(name, *, iterations=300, threads=32, lines=LINES3, window=MULTILINE3_WINDOW, degree=2,
               segments=None, num_ch=34, num_col=3):
    """The multi-line recipe on :func:`maps` of 34 channels and ``num_col`` columns: a constant sky (damping 0.1,
    the library default when the recipe was made) plus ``lines``, the offset a hard degree-``degree`` polynomial
    along the subchannels of ``window`` (per column), per-pixel shot-noise weights, the source-mask bit ignored; no
    mosaic (tiled and N-pass runs)."""
    model = sc.spectral(lines, polynomial=sc.Poly(degree, window=window, segments=segments), damping=0.1)
    return sc.Recipe(model.replace(chunk_maps=maps(num_ch=num_ch, num_col=num_col)),
                     fit=sc.Fit(iterations, clip=5.0, ignore_flags=[21], shot_noise_weights=True,
                                method='lsqr'),
                     coadd=None, numerics=sc.Numerics(threads), name=name)


def pahfit_tiled(name, *, iterations, num_ch=34, num_col=3):
    """The tiled PAH 3.29 um recipe (2026-08) on :func:`maps` of 34 channels and ``num_col`` columns: a constant
    sky (damping 0.1) plus the PAH line (``terms.pah_3p29``, damping 5e-3), free offsets smoothed along the columns
    with a linear column polynomial and a cubic along the subchannels 200-259 (weight 100), shot-noise weights, the
    source-mask bit ignored; no mosaic."""
    model = sc.spectral([sc.Sky('pah_3p29', times=T.pah_3p29(), damping=5e-3)],
                        poly_prior=[sc.Poly(1, weight=0.5), sc.Poly(3, along='subchannel', window=range(200, 260),
                                                                  weight=100.0)], damping=0.1)
    return sc.Recipe(model.replace(chunk_maps=maps(num_ch=num_ch, num_col=num_col)),
                     fit=sc.Fit(iterations, clip=5.0, ignore_flags=[21], shot_noise_weights=True, method='lsqr'),
                     coadd=None, name=name)


#: The N-pass schedule of the multi-line fits: five passes ending on a sky; every clip per subchannel; the
#: OFFSET passes fit only observations whose modelled sky is below 0.05 MJy/sr. The OFFSET passes refit the
#: recipe's own offset terms (until 2026-10 they refitted a separate degree-4 polynomial per column).
MULTILINE3_PASSES = sc.Passes(5, order='offset_first',
                              init_clip=sc.Clip(2.5, per=SUBCHANNEL, ignore_flags=[21]),
                              sky_clip=sc.Clip(5.0, per=SUBCHANNEL),
                              offset=sc.Refit(clip=sc.Clip(2.5, per=SUBCHANNEL), bright_cut=0.05))


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
