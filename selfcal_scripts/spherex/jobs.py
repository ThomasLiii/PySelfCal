"""The jobs of a SPHEREx run: one map each, named on the ``subchannel`` axis of the primary chunk map.

``channel(c)`` is the job of spectral channel ``c``: its window is the channel's ``num_sub`` subchannels
padded by one on each side (the solve overlaps the neighbouring channels, for stitching), its strict set
the channel's own (the pixels the coadd keeps). ``window(name, subchannels)`` solves and keeps a range of
subchannels as given. The subchannel indices count over the whole map, ``num_sub * num_ch + 2`` of them,
0 and the last lying beyond the band's edges, so channel ``c`` owns ``(c - 1) * num_sub + 1`` to
``c * num_sub``::

    from selfcal_scripts.spherex import jobs
    field.calibrate(recipe, jobs=jobs.channel(9))
    field.calibrate(recipe, jobs=jobs.channels(1, 17))
    field.calibrate(recipe, jobs=jobs.window('Multiline3', range(200, 321)))
"""
import selfcal as sc

__all__ = ['channel', 'channels', 'group', 'window', 'WINDOWS']

#: Preset windows of subchannels, by name: the aromatic and the aliphatic PAH features.
WINDOWS = {
    'Aromatic': range(225, 236),
    'Aliphatic': range(249, 260),
}


def group(*chans, num_sub=10) -> sc.Job:
    """One job over consecutive channels solved together: ``group(17, 18)`` -> ``Ch17-18``, the window
    their subchannels padded by one on each side, the strict set their own."""
    chans = tuple(int(c) for c in chans)
    if not chans or chans != tuple(range(chans[0], chans[-1] + 1)):
        raise sc.ConfigError(f"jobs.group{chans!r}: consecutive channels, in order")
    lo, hi = (chans[0] - 1) * num_sub + 1, chans[-1] * num_sub + 1
    return sc.Job('Ch' + '-'.join(map(str, chans)), window={'subchannel': range(lo - 1, hi + 1)},
                  strict={'subchannel': range(lo, hi)})


def channel(c, num_sub=10) -> sc.Job:
    """The job of spectral channel ``c`` (counted from 1): products ``..._Ch<c>...``."""
    return group(c, num_sub=num_sub)


def channels(first, last=None, num_sub=10) -> tuple:
    """One job per channel: ``channels(1, 17)`` (first to last, inclusive), or ``channels([3, 17])``."""
    if last is None:
        return tuple(channel(int(c), num_sub=num_sub) for c in first)
    return tuple(channel(c, num_sub=num_sub) for c in range(int(first), int(last) + 1))


def window(name, subchannels=None) -> sc.Job:
    """The job of a window of subchannels, solved and kept as given (``range(lo, hi)``); a preset
    window (:data:`WINDOWS`) by its name alone."""
    if subchannels is None:
        if name not in WINDOWS:
            raise sc.ConfigError(f"jobs.window({name!r}): not a preset window ({sorted(WINDOWS)}); give "
                                 f"subchannels=range(lo, hi)")
        subchannels = WINDOWS[name]
    if not isinstance(subchannels, range) or subchannels.step != 1:
        raise sc.ConfigError(f"jobs.window({name!r}, subchannels=...): a range of subchannels, e.g. range(200, 321)")
    return sc.Job(str(name), window={'subchannel': subchannels})
