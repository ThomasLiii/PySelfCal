"""SPHEREx as settings: ``sc.SPHEREx(detector)``, its jobs and its geometry.

A job is one map: a spectral channel (:func:`channel`), several channels solved together
(:func:`group`), or a window of subchannels (:func:`window`)::

    from selfcal.instruments import spherex
    jobs = spherex.channels(1, 34)                     # 34 jobs, Ch1 .. Ch34
    jobs = spherex.window("Multiline3", subchannels=range(200, 321))

:class:`SPHEREx` implements the instrument contract
(:class:`~selfcal.instruments.contract.Instrument`): every LVF / subchannel specific of a run
(the stripped (arc) chunk map with its ``(subchannel, column)`` axes, the H2RG readout-channel
map, the ``BC`` / ``BW`` wavelength maps, the subchannel masks of a job, the smooth arc offset
renderer, the wavelength coadd of the mosaic, the FINAST astrometry filter) lives here.

:func:`line` is a sky term with one of the shipped line templates; :func:`precompute_lvf` fits
the LVF arcs the geometry reads; :func:`zodi_anchor` fits a calibration's zodiacal-light anchor.
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import KW_ONLY, dataclass
from functools import partial

import numpy as np

from ...config import (
    ENV_LVF_PARAMS_DIR,
    ENV_SPHEREX_CALIB_DIR,
    ENV_SPHEREX_CHANNEL_FILE,
    resolve_path,
)
from ...config.base import ConfigError
from ...models.model import Sky, template
from ...models.offset_structure import ChunkAxes
from ..base import ChunkMap, DetectorGeometry, ExposureLayout, JobGeometry
from ..contract import Instrument, Job
from ..grid import upsample_chunk_map
from .adapter import SUBCH_WINDOWS, make_readout_chunk_map
from .spherex_utility import (
    _LVF_PARAMS_DIR,
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_CHANNEL_FILE,
    fast_vertical_dist,
    load_calibration,
    load_lvf_params,
    make_spherex_stripped_offset_map,
    make_stripped_chunk_map,
    make_stripped_chunk_valid_mask,
)
from .wavemap import wav_coadd

__all__ = ['SPHEREx', 'channel', 'channels', 'group', 'window', 'line', 'LINE_TEMPLATES', 'precompute_lvf',
           'zodi_anchor']

logger = logging.getLogger(__name__)

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'line_templates')

#: The shipped line templates (``data/line_templates``): name -> file.
LINE_TEMPLATES = {
    'aromatic': 'aromatic_3p289.npz',
    'aliphatic': 'aliphatic_3p400.npz',
    'plateau': 'plateau_3p470.npz',
}


@dataclass(frozen=True)
class SPHEREx(Instrument):
    """A SPHEREx detector (1 to 6): its LVF chunk map of ``num_ch`` channels x ``num_sub``
    subchannels x ``num_col`` columns, and its wavelength maps. ``calib_dir`` / ``lvf_dir``:
    the directories of the calibration maps and of the LVF parameters (default: the
    ``SELFCAL_SPHEREX_CALIB_DIR`` / ``SELFCAL_LVF_PARAMS_DIR`` variables, else the defaults)."""
    detector: int
    _: KW_ONLY
    num_col: int = 3
    num_sub: int = 10
    num_ch: int = 34
    calib_dir: str | None = None
    lvf_dir: str | None = None

    # A constant of the instrument, not a setting: without an annotation it is no dataclass field,
    # so it stays out of the products' fingerprints.
    unit = 'MJy/sr'

    def _validate(self):
        if self.detector not in range(1, 7):
            raise ConfigError(f"SPHEREx(detector={self.detector}): 1 to 6")
        for k in ('num_col', 'num_sub', 'num_ch'):
            if getattr(self, k) < 1:
                raise ConfigError(f"SPHEREx({k}={getattr(self, k)}): at least 1")

    @property
    def product_tag(self) -> str:
        return f"Detector{self.detector}_NumSub{self.num_sub}_NumCh{self.num_ch}_NumCol{self.num_col}"

    def default_jobs(self):
        raise ConfigError("SPHEREx runs name their maps: jobs=spherex.channel(17), spherex.channels(1, 34), "
                          "spherex.group(17, 18) or spherex.window(\"Aromatic\")")

    def default_ignore_flags(self):
        return ()

    def check_jobs(self, jobs):
        for j in jobs:
            if not isinstance(j, Job) or j.kind not in ('channels', 'window'):
                raise ConfigError(f"SPHEREx: a job is spherex.channel(...), .group(...) or .window(...), got {j!r}")
            if j.kind == 'channels':
                bad = [c for c in j.value if not 1 <= c <= self.num_ch]
                if bad:
                    raise ConfigError(f"SPHEREx: channels {bad} outside 1..{self.num_ch}")
            else:
                lo, hi = j.value
                n_sub = self.num_sub * self.num_ch + 2
                if not 0 <= lo < hi <= n_sub:
                    raise ConfigError(f"SPHEREx: window {j.name!r} = subchannels {lo}..{hi - 1}, outside "
                                      f"0..{n_sub - 1}")

    # ---- the contract -------------------------------------------------------------------
    def layout(self) -> ExposureLayout:
        """L2b exposures: science in extension 1, DQ bitmask in extension 2, one detector per
        file; only exposures with a converged astrometric solution (``FINAST == 0``) are kept."""
        return ExposureLayout(
            sci_ext=[1], dq_ext=[2], detector_ids=[0], ref_use_ext=(1,),
            header_predicate=lambda h: h.get('FINAST', 2) == 0,
            header_keys=('FINAST',), header_ext=1,
            cache_tag=f"finast_D{self.detector}")

    def geometry(self, oversample=1) -> DetectorGeometry:
        """The LVF parameters, the ``BC`` / ``BW`` maps, the stripped chunk map at detector and
        grid resolution with its (subchannel, column) axes, and the readout-channel map (no
        adjacency: that is the model's). ``calib_dir`` / ``lvf_dir`` choose the directories of the
        calibration maps and the LVF parameters (default: see
        :func:`~selfcal.instruments.spherex.spherex_utility.load_lvf_params`)."""
        det, ns, nch, ncol = self.detector, self.num_sub, self.num_ch, self.num_col
        lvf_params = load_lvf_params(f'lvf_params_D{det}.npy', input_dir=self.lvf_dir)
        det_BC, det_BW = load_calibration(
            band=det, calibration_dir=self.calib_dir)     # None: $SELFCAL_SPHEREX_CALIB_DIR, else the default
        # The stripped maps read BC from $SELFCAL_SPHEREX_CALIB_DIR / the default, never from
        # calib_dir: what every product was made with.
        grid_chunk_map, _, _, _ = make_stripped_chunk_map(
            det, num_subchannels=ns, num_channels=nch, num_columns=ncol,
            oversample_factor=oversample, lvf_params=lvf_params)
        det_chunk_map, _, r_edges, x_edges = make_stripped_chunk_map(
            det, num_subchannels=ns, num_channels=nch, num_columns=ncol,
            oversample_factor=1, lvf_params=lvf_params)
        n_sub = (int(det_chunk_map.max()) + 1) // int(ncol)      # == ns * nch + 2 (padding subchannels)
        # The chunk encoding chunk = subchannel * num_col + column, expressed
        # ONCE as chunk axes: subchannels change along y (arcs), columns along x.
        sub_axes = ChunkAxes.row_major(('subchannel', 'column'), (n_sub, int(ncol)), ('y', 'x'))
        stripped = ChunkMap(name='subchannel', det=det_chunk_map, grid=grid_chunk_map, axes=sub_axes,
                            adjacency_axes=('column',), spectral_axis='subchannel', group_axis='column')
        det_ro, n_ro = make_readout_chunk_map(det_chunk_map.shape)
        readout = ChunkMap(name='readout', det=det_ro, grid=upsample_chunk_map(det_ro, oversample),
                           axes=ChunkAxes.row_major(('readout',), (n_ro,), ('x',)))
        return DetectorGeometry(
            shape=det_chunk_map.shape, chunk_maps={'subchannel': stripped, 'readout': readout},
            primary='subchannel', aux={'BC': det_BC, 'BW': det_BW}, wavelength_key='BC', width_key='BW',
            extra={'lvf_params': lvf_params, 'r_edges': r_edges, 'x_edges': x_edges,
                   'num_sub': ns, 'num_ch': nch, 'num_col': ncol})

    def geometry_files(self):
        """The files :meth:`geometry` reads, resolved as its loaders resolve them: the LVF arc fit,
        the band-centre and band-width maps (of ``calib_dir``, and of the default directory the arc
        maps read) and the channel table."""
        lvf = resolve_path(self.lvf_dir, env_var=ENV_LVF_PARAMS_DIR, default=_LVF_PARAMS_DIR, must_exist=False)
        files = [os.path.join(lvf, f'lvf_params_D{self.detector}.npy')]
        for calib in dict.fromkeys((self.calib_dir, None)):
            directory = resolve_path(calib, env_var=ENV_SPHEREX_CALIB_DIR, default=DEFAULT_CALIBRATION_DIR,
                                     must_exist=False)
            for kind in ('BC', 'BW'):
                files += sorted(glob.glob(os.path.join(directory, f'*{kind}_Band{self.detector}.fits')))
        files.append(resolve_path(None, env_var=ENV_SPHEREX_CHANNEL_FILE, default=DEFAULT_CHANNEL_FILE,
                                  must_exist=False))
        return tuple(files)

    def job_geometry(self, geom, job) -> JobGeometry:
        """Return the valid masks and the solve and mosaic weights of one job's subchannels.

        A ``'window'`` job (``value = (lo, hi)``) selects subchannels ``lo`` to
        ``hi - 1``, indexed over all ``num_sub * num_ch + 2`` subchannels of the
        stripped map (``0`` and the last are the padding subchannels). A
        ``'channels'`` job selects the ``num_sub`` subchannels of each listed
        channel (numbered from 1), and its padded set adds one subchannel on each
        side, which overlaps the neighbouring channels for stitching; a window
        job's padded and strict sets are the same. A selected subchannel is
        selected in every column. Any other ``kind`` raises ``ValueError``.

        ``det_valid_weight``, the solve's weight, is the padded set's 0/1 mask on
        the detector grid (float64 for a channel job, boolean for a window job). The mosaic's
        ``grid_valid_weight`` (float32, on the primary map's ``grid``) tapers the strict set
        linearly with each pixel's distance in rows to the set's edge
        (:func:`~selfcal.instruments.spherex.spherex_utility.fast_vertical_dist`),
        divided by its maximum. ``chunk_valid`` / ``chunk_valid_strict`` hold the
        padded / strict sets per chunk of the primary map, and ``det_valid_mask``
        / ``grid_valid_mask`` the strict set on the two grids."""
        ns, nch, ncol = self.num_sub, self.num_ch, self.num_col
        det_chunk_map = geom.chunk_map.det
        grid_chunk_map = geom.chunk_map.grid
        kw = dict(num_subchannels=ns, num_channels=nch, num_columns=ncol)
        if job.kind == 'window':
            lo, hi = job.value
            sel = dict(subch=np.arange(lo, hi))
        elif job.kind == 'channels':
            sel = dict(ch=job.value)
        else:
            raise ValueError(f"unknown job kind {job.kind!r}")
        cvm_pad = make_stripped_chunk_valid_mask(**sel, **kw, subchannel_padding=1)
        cvm = make_stripped_chunk_valid_mask(**sel, **kw, subchannel_padding=0)
        det_valid_mask = cvm[det_chunk_map]
        det_valid_mask_padded = cvm_pad[det_chunk_map]
        grid_valid_mask = cvm[grid_chunk_map]
        grid_valid_weight = fast_vertical_dist(grid_valid_mask)
        if np.max(grid_valid_weight) > 0:
            grid_valid_weight /= np.max(grid_valid_weight)
        # The solve weights every pixel of the padded window equally (the
        # padding subchannels overlap the neighbouring jobs for stitching); the
        # mosaic tapers with the distance to the window's arc edges.
        return JobGeometry(det_valid_weight=det_valid_mask_padded, grid_valid_weight=grid_valid_weight,
                           chunk_valid=cvm_pad, chunk_valid_strict=cvm,
                           det_valid_mask=det_valid_mask, grid_valid_mask=grid_valid_mask)

    def offset_renderer(self, geom, jobgeom, map_name=None, render=None):
        """The smooth subchannel-arc offset renderer of the primary (stripped) map, for one job's
        mosaic, whatever ``render`` asks for; ``None`` (block-constant) for the readout map."""
        if map_name not in (None, geom.primary):
            return None
        ns, nch, ncol = self.num_sub, self.num_ch, self.num_col
        x = geom.extra
        return partial(
            make_spherex_stripped_offset_map,
            chunk_valid_mask=jobgeom.chunk_valid_strict,
            lvf_params=x['lvf_params'], r_edges=x['r_edges'], x_edges=x['x_edges'],
            tot_subchannels=ns * nch + 2, num_columns=ncol, fill_invalid=True)

    @staticmethod
    def aux_coadds(geom):
        """(band centre, band width) LVF maps, coadded by the mosaic's sigma-clip pass."""
        return geom.aux['BC'], geom.aux['BW']

    @staticmethod
    def finalize_mosaic(geom, mm, maps, sigma):
        """LVF wavelength maps for the full mosaic. When ``make_mosaic`` was given
        the band maps (``wav_maps=self.aux_coadds(...)``) it has already coadded
        them inside the sigma-clip pass and this only labels the units;
        otherwise the standalone ``wav_coadd`` runs over the intermediate cache
        (the pre-2026-09 path, which needs ``cache_intermediate``)."""
        if 'wav_mean_map' in maps and maps['wav_mean_map'].get('data') is not None:
            for k in ('wav_mean_map', 'wav_std_map'):
                mm.maps[k]['unit'] = 'um'
            return
        import time
        logger.info("Coadding wavelength maps...")
        t00 = time.time()
        wav_mean, wav_std = wav_coadd(
            geom.aux['BC'], geom.aux['BW'],
            mean_map=maps['mean_map']['data'], std_map=maps['std_map']['data'],
            reproj_list=mm.reproj_list, cache_list=mm.cached_list,
            ref_shape=maps['mean_map']['data'].shape, sigma=sigma,
            batch_size=40, max_workers=30)
        logger.info(f"Wavelength coaddition finished in {time.time() - t00:.2f} seconds.")
        mm.append_maps({'wav_mean_map': {'data': wav_mean, 'unit': 'um'},
                        'wav_std_map': {'data': wav_std, 'unit': 'um'}})

    @staticmethod
    def coefficient_catalog():
        """Return the catalogue of named SPHEREx coefficients; its one entry is ``pah_3p29``.

        ``pah_3p29``
        (:func:`~selfcal.instruments.spherex.line_catalog.pah_3p29_coefficient`)
        is a Gaussian of the band-centre map ``BC`` at the PAH 3.29 um feature,
        whose per-observation width combines the band-width map ``BW`` with the
        intrinsic PAH width. A sky term selects it with ``sc.catalog("pah_3p29")``, optionally
        overriding the factory's ``center`` and ``sigma`` (um). Each call returns a new copy of
        :data:`~selfcal.instruments.spherex.line_catalog.CATALOG`."""
        from .line_catalog import CATALOG
        return dict(CATALOG)


def channel(c) -> Job:
    """The job of spectral channel ``c`` (1 to 34): products ``..._Ch<c>...``."""
    return group(c)


def channels(first, last=None) -> tuple:
    """One job per channel: ``channels(1, 34)`` (first to last, inclusive), or ``channels([3, 17])``."""
    if last is None:
        return tuple(channel(int(c)) for c in first)
    return tuple(channel(c) for c in range(int(first), int(last) + 1))


def group(*chans) -> Job:
    """One job over several channels solved together: ``group(17, 18)`` -> ``Ch17-18``."""
    chans = tuple(int(c) for c in chans)
    if not chans:
        raise ConfigError("spherex.group(): at least one channel")
    return Job('Ch' + '-'.join(map(str, chans)), kind='channels', value=chans)


def window(name, subchannels=None) -> Job:
    """The job of a window of subchannels (``range(lo, hi)``, indexed over the 342 subchannels
    of the LVF map); a preset window by name alone (``"Aromatic"``, ``"Aliphatic"``)."""
    if subchannels is None:
        if name not in SUBCH_WINDOWS:
            raise ConfigError(f"spherex.window({name!r}): not a preset window ({sorted(SUBCH_WINDOWS)}); "
                              f"give subchannels=range(lo, hi)")
        lo, hi = SUBCH_WINDOWS[name]
    else:
        if not isinstance(subchannels, range) or subchannels.step != 1:
            raise ConfigError(f"spherex.window({name!r}, subchannels=...): a range of subchannels, "
                              f"e.g. range(200, 321)")
        lo, hi = subchannels.start, subchannels.stop
    return Job(str(name), kind='window', value=(int(lo), int(hi)))


def line(name, damping=None, *, file=None) -> Sky:
    """A sky term times one of the shipped line templates of the wavelength
    (:data:`LINE_TEMPLATES`: ``aromatic``, ``aliphatic``, ``plateau``), or the template ``file``."""
    if file is None:
        if name not in LINE_TEMPLATES:
            raise ConfigError(f"spherex.line({name!r}): no shipped template ({sorted(LINE_TEMPLATES)}); "
                              f"give file=")
        file = os.path.join(_DATA, LINE_TEMPLATES[name])
    return Sky(name, times=template(file), damping=damping)


def precompute_lvf(detectors, *, output_dir=None, calib_dir=None, num_sub=10, num_ch=34):
    """Fit and save the LVF arc parameters ``lvf_params_D<n>.npy`` of each detector in
    ``detectors`` (into ``output_dir``; default: ``$SELFCAL_LVF_PARAMS_DIR``, else the package's
    data), from its band-centre map (in ``calib_dir``; default: ``$SELFCAL_SPHEREX_CALIB_DIR``, else
    the default directory). The package ships them for detectors 1 to 6; this regenerates them."""
    from .spherex_utility import make_fiducial_chunk_map, save_lvf_params
    for det in detectors:
        det_BC, _ = load_calibration(band=int(det), calibration_dir=calib_dir)
        _, lvf_params, _ = make_fiducial_chunk_map(int(det), det_BC, num_subchannels=int(num_sub),
                                                   num_channels=int(num_ch), oversample_factor=1)
        lvf_params['filename'] = f'lvf_params_D{int(det)}.npy'
        save_lvf_params(lvf_params, output_dir=None if output_dir is None else str(output_dir))


def zodi_anchor(result, predictions, *, clip_window_days=7.0, clip_sigma=3.0, clip_iters=2):
    """Fit the zodiacal-light anchor of each single-channel job of ``result`` (a calibration's
    :class:`~selfcal.run.result.Result`) against the zodipy predictions
    ``<predictions>/zodi_pred_<stem>.npz``, and record it in the run's
    ``zodi_anchor/anchor_D<n>.h5`` (non-mutating: the cal and mosaic files stay as they are).
    A job without a prediction file is skipped. Returns ``{job name: fit}``."""
    import re

    from ...zodi_anchor import append_anchor_channel, fit_anchor_for_channel
    inst = result.field.instrument
    if not isinstance(inst, SPHEREx):
        raise ConfigError("spherex.zodi_anchor(): the result of a SPHEREx run")
    clip = dict(clip_window_days=clip_window_days, clip_sigma=clip_sigma, clip_iters=clip_iters)
    anchor_path = os.path.join(result.field.path, 'zodi_anchor', f'anchor_D{inst.detector}.h5')
    fits = {}
    for job, cal_path in zip(result.jobs, result.cal_paths):
        stem = os.path.basename(cal_path)[len('cal_'):-len('.h5')]
        npz = os.path.join(str(predictions), f'zodi_pred_{stem}.npz')
        m = re.search(r'_Ch(\d+)_', os.path.basename(cal_path))
        if not os.path.exists(npz) or m is None:
            print(f"Zodi anchor skipped for {stem}: "
                  + (f"{npz} not found." if m is not None else "not a single-channel job."))
            continue
        fit = fit_anchor_for_channel(cal_path, npz, **clip)
        append_anchor_channel(anchor_path, inst.detector, result.field.name, int(m.group(1)), fit, clip,
                              anchor_method='raw')
        fits[job.name] = fit
    return fits
