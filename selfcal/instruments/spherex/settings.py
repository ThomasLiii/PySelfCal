"""SPHEREx as settings: ``sc.SPHEREx(detector)``, what is intrinsic to the instrument.

:class:`SPHEREx` implements the instrument contract
(:class:`~selfcal.instruments.contract.Instrument`): the L2b exposure layout and reader (the
``VARIANCE`` as the frame layer ``variance``, the FINAST astrometry filter), the detector's
band-centre and band-width maps (``BC`` / ``BW``, the wavelength and its width), the LVF arc fit
every chunk map on the arcs reads (``geometry.extra['lvf_params']``), and the wavelength coadd of
the mosaic. The chunk maps a model solves on (the arcs cut into subchannels and columns, the
detector columns, the readout channels), the jobs (a channel, a window of subchannels) and the
terms are the model's: ``selfcal_scripts/spherex/{chunks,jobs,terms}.py`` build them.

:func:`precompute_lvf` fits the LVF arcs the geometry reads; :func:`zodi_anchor` fits a
calibration's zodiacal-light anchor.
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import KW_ONLY, dataclass

from ...config import (
    ENV_LVF_PARAMS_DIR,
    ENV_SPHEREX_CALIB_DIR,
    ENV_SPHEREX_CHANNEL_FILE,
    resolve_path,
)
from ...config.base import ConfigError
from ..base import DetectorGeometry, ExposureLayout
from ..contract import Geometry, Instrument
from .exposures import read_l2b
from .spherex_utility import (
    _LVF_PARAMS_DIR,
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_CHANNEL_FILE,
    load_calibration,
    load_lvf_params,
)
from .wavemap import wav_coadd

__all__ = ['SPHEREx', 'LINE_TEMPLATES', 'precompute_lvf', 'zodi_anchor']

logger = logging.getLogger(__name__)

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'line_templates')

#: Pixels per side of a SPHEREx detector (the frames are 2040 x 2040).
DETECTOR_SIZE = 2040

#: The shipped line templates (``data/line_templates``, the directory ``_DATA``): name -> file; a sky term on one
#: is ``selfcal_scripts.spherex.terms.line(name)``.
LINE_TEMPLATES = {
    'aromatic': 'aromatic_3p289.npz',
    'aliphatic': 'aliphatic_3p400.npz',
    'plateau': 'plateau_3p470.npz',
}


@dataclass(frozen=True)
class SPHEREx(Instrument):
    """A SPHEREx detector (1 to 6): its L2b exposures, its band-centre and band-width maps and its
    LVF arc fit. ``calib_dir`` / ``lvf_dir``: the directories of the calibration maps and of the
    LVF parameters (default: the ``SELFCAL_SPHEREX_CALIB_DIR`` / ``SELFCAL_LVF_PARAMS_DIR``
    variables, else the defaults). Products are named ``cal_Detector<n>_<primary map tag>_<job>...``."""
    detector: int
    _: KW_ONLY
    calib_dir: str | None = None
    lvf_dir: str | None = None

    # Constants of the instrument, not settings: without an annotation they are no dataclass fields,
    # so they stay out of the products' fingerprints. The geometry reads the settings and the
    # geometry_files() only: the engine may keep it between actions.
    unit = 'MJy/sr'
    geometry_is_pure = True

    def _validate(self):
        if self.detector not in range(1, 7):
            raise ConfigError(f"SPHEREx(detector={self.detector}): 1 to 6")

    @property
    def product_tag(self) -> str:
        return f"Detector{self.detector}"

    def default_ignore_flags(self):
        return ()

    # ---- the contract -------------------------------------------------------------------
    def layout(self) -> ExposureLayout:
        """L2b exposures (read by :func:`~selfcal.instruments.spherex.exposures.read_l2b`): science
        in extension 1, DQ bitmask in extension 2, the file's ``VARIANCE`` stored with the frame as
        the layer ``variance``, one detector per file; only exposures with a converged astrometric
        solution (``FINAST == 0``) are kept."""
        return ExposureLayout(
            sci_ext=[1], dq_ext=[2], detector_ids=[0], ref_use_ext=(1,),
            header_predicate=lambda h: h.get('FINAST', 2) == 0,
            header_keys=('FINAST',), header_ext=1,
            cache_tag=f"finast_D{self.detector}", reader=read_l2b)

    def geometry(self, oversample=1) -> DetectorGeometry:
        """The detector: the ``BC`` / ``BW`` maps as the aux maps (the wavelength and its width)
        and, in ``extra``, the LVF arc fit ``lvf_params`` and the ``detector`` number, which a
        chunk map on the arcs reads. ``calib_dir`` / ``lvf_dir`` choose the directories of the
        calibration maps and the LVF parameters (default: see
        :func:`~selfcal.instruments.spherex.spherex_utility.load_lvf_params`)."""
        det = self.detector
        lvf_params = load_lvf_params(f'lvf_params_D{det}.npy', input_dir=self.lvf_dir)
        det_BC, det_BW = load_calibration(
            band=det, calibration_dir=self.calib_dir)     # None: $SELFCAL_SPHEREX_CALIB_DIR, else the default
        return Geometry(det_BC.shape, oversample, aux={'BC': det_BC, 'BW': det_BW}, wavelength='BC', width='BW',
                        extra={'lvf_params': lvf_params, 'detector': det})

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
    A job without a prediction file is skipped, as is a job of several channels or a window (its
    cal's name holds no ``_Ch<n>`` followed by ``_`` or the end of the name). Returns
    ``{job name: fit}``."""
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
        m = re.search(r'_Ch(\d+)(?:_|$)', stem)         # a suffix follows the job's name, or nothing does
        if not os.path.exists(npz) or m is None:
            print(f"Zodi anchor skipped for {stem}: "
                  + (f"{npz} not found." if m is not None else "not a single-channel job."))
            continue
        fit = fit_anchor_for_channel(cal_path, npz, **clip)
        append_anchor_channel(anchor_path, inst.detector, result.field.name, int(m.group(1)), fit, clip,
                              anchor_method='raw')
        fits[job.name] = fit
    return fits
