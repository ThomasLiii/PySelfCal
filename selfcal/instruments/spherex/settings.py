"""SPHEREx as settings: ``sc.SPHEREx(detector)`` and its jobs.

A job is one map: a spectral channel (:func:`channel`), several channels solved together
(:func:`group`), or a window of subchannels (:func:`window`)::

    from selfcal.instruments import spherex
    jobs = spherex.channels(1, 34)                     # 34 jobs, Ch1 .. Ch34
    jobs = spherex.window("Multiline3", subchannels=range(200, 321))

:func:`line` is a sky term with one of the shipped line templates.
"""
from __future__ import annotations

import os
from dataclasses import KW_ONLY, dataclass

from ...config.base import ConfigError
from ...models.model import Sky, template
from ..contract import Instrument, Job
from .adapter import SUBCH_WINDOWS

__all__ = ['SPHEREx', 'channel', 'channels', 'group', 'window', 'line', 'LINE_TEMPLATES', 'precompute_lvf',
           'zodi_anchor']

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

    def engine(self, jobs):
        """``("spherex", table)``: the registered instrument and its ``[instrument]`` table."""
        table = {'name': 'spherex', 'detector': self.detector, 'num_sub': self.num_sub, 'num_ch': self.num_ch,
                 'num_col': self.num_col}
        if self.calib_dir is not None:
            table['calib_dir'] = self.calib_dir
        if self.lvf_dir is not None:
            table['lvf_dir'] = self.lvf_dir
        kinds = {j.kind for j in jobs}
        if len(kinds) > 1:
            raise ConfigError("SPHEREx: channel jobs and window jobs run separately")
        if kinds == {'channels'}:
            table['channels'] = [list(j.value) for j in jobs]
        elif kinds == {'window'}:
            table['windows'] = [j.name for j in jobs]
            table['window_defs'] = {j.name: list(j.value) for j in jobs}
        return 'spherex', table


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
    data). The package ships them for detectors 1 to 6; this regenerates them."""
    from .adapter import SPHERExInstrument
    table = {'detectors': [int(d) for d in detectors], 'num_sub': int(num_sub), 'num_ch': int(num_ch)}
    if output_dir is not None:
        table['lvf_output_dir'] = str(output_dir)
    if calib_dir is not None:
        table['calib_dir'] = str(calib_dir)
    SPHERExInstrument().precompute(table)


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
