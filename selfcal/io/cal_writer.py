"""The v3 sky-block layout of a ``cal_*.h5`` file — one writer for every producer.

``Calibrator.save_calibration`` (joint solves) and the N-pass
``combine_moments`` (sky-only closed-form solves) both write::

    attrs: num_sky_blocks, schema_version = 3, sky_components, [line_fisher_threshold]
    sky/<name>, sky_coverage/<name>, sky_fisher/<name>      one per block
    sky_separability/<name>                                 per spectral block (I_P)
    skymap, skymap_coverage, skymap_fisher                  hard-link aliases -> block 0
    skymap_line, skymap_line_coverage, skymap_line_fisher   aliases -> the LAST spectral block

Keeping this in one place means a cal produced by either path reads
identically through ``selfcal.pipeline.pipeline_wrapper`` / the analysis
loaders. Bit-identity contract: the datasets written here are compared
element-wise by the refactor gates (``selfcal_scripts/drivers/diff_cal_h5.py``).
"""
from __future__ import annotations

import numpy as np

__all__ = ["write_sky_groups"]


def write_sky_groups(f, *, sky_names, sky_maps, sky_coverages, sky_fishers,
                     pixel_cross, pixel_fisher, ref_shape, num_sky_blocks,
                     line_fisher_threshold=None):
    """Write the sky blocks (+ separability + aliases) into an open h5py file.

    Parameters
    ----------
    f : h5py.File (writable)
    sky_names : list[str]                 block names, block 0 = continuum
    sky_maps : list[np.ndarray]           per-block maps on the reference grid
    sky_coverages : list[np.ndarray]      per-block observation counts
    sky_fishers : list[np.ndarray | None] per-block Fisher information (None = skip)
    pixel_cross, pixel_fisher : moments for the per-block separability ``I_P``
        (``pixel_cross`` the bare (0,1) array for J == 2 or the pair dict;
        either may be None to skip ``sky_separability``)
    ref_shape : (ref_h, ref_w)
    num_sky_blocks : int
    line_fisher_threshold : float or None
        Informational read-time mask threshold (not applied destructively).
    """
    from ..core.system import parse_line_separability

    f.attrs['num_sky_blocks'] = int(num_sky_blocks)
    f.attrs['schema_version'] = 3
    f.attrs['sky_components'] = np.array(list(sky_names), dtype='S')
    sky_grp = f.create_group('sky')
    skycov_grp = f.create_group('sky_coverage')
    skyfish_grp = f.create_group('sky_fisher')
    for j, name in enumerate(sky_names):
        sky_grp.create_dataset(name, data=sky_maps[j], compression='gzip')
        skycov_grp.create_dataset(name, data=sky_coverages[j], compression='gzip')
        if sky_fishers[j] is not None:
            skyfish_grp.create_dataset(name, data=np.asarray(sky_fishers[j]).astype('float32'),
                                       compression='gzip')
    # Per-pixel SEPARABILITY I_P (each spectral block's Schur complement against
    # all other sky blocks): wavelength diversity, the quantity that bounds
    # per-pixel amplitude variance. One dataset per spectral block.
    if pixel_cross is not None and num_sky_blocks >= 2 and pixel_fisher is not None:
        sep_grp = f.create_group('sky_separability')
        for j in range(1, num_sky_blocks):
            sep = parse_line_separability(pixel_cross, pixel_fisher, ref_shape,
                                          num_sky_blocks=num_sky_blocks, block=j)
            sep_grp.create_dataset(sky_names[j], data=sep.astype('float32'), compression='gzip')
    # Back-compat hard-link aliases (v2 readers resolve transparently).
    cont = sky_names[0]
    f['skymap'] = sky_grp[cont]
    f['skymap_coverage'] = skycov_grp[cont]
    if cont in skyfish_grp:
        f['skymap_fisher'] = skyfish_grp[cont]
    extra_names = list(sky_names[1:])
    if extra_names:
        ln = extra_names[-1]
        f['skymap_line'] = sky_grp[ln]
        f['skymap_line_coverage'] = skycov_grp[ln]
        if ln in skyfish_grp:
            f['skymap_line_fisher'] = skyfish_grp[ln]
    if line_fisher_threshold is not None:
        f.attrs['line_fisher_threshold'] = float(line_fisher_threshold)
