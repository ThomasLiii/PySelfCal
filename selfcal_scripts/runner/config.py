"""RunConfig + TOML loader + instrument resolver.

A run is fully described by one TOML file. The loader splits it into:
  * generic top-level scalars (task / instrument / mode / output paths / staging),
  * stage tables consumed verbatim as kwargs (``[calibration]`` ``[lsqr]``
    ``[mosaic]`` ``[zodi]`` ``[reproject]`` ``[tiled]`` ``[passes]``),
  * ``[instrument]`` — instrument-specific knobs (SPHEREx: detector, num_*,
    channel/window selection, calib_dir),
  * ``[params]`` — mode knobs (poly degree/weight, reg weight, line params, ...).

The engine and the modes only ever read these dicts; nothing here knows what any
particular key means. ``get_instrument`` maps the ``instrument`` name to an
adapter, so adding a telescope is a new adapter + a name, not an engine edit.
"""
import tomllib
from dataclasses import dataclass, field


@dataclass
class RunConfig:
    task: str                          # cal | mosaic | npass | reproject | precompute ('tiled' = cal + [tiling])
    instrument: str = None             # from [instrument].name (required)
    mode: str = None                   # cal/tiled mode name (None for reproject/precompute)
    output_dir: str = None
    run_name: str = None               # may contain "{detector}"
    resolution_arcsec: float = None    # required for reproject / cal / mosaic / npass
    cache_dir: str = None              # staging / scratch area (required for cal / mosaic / npass)
    suffix: str = ""
    oversample: int = 1
    staging: str = "copy"              # copy | reuse
    keep_nvme: bool = False
    hdd_io_limit: int = 20
    apply_n_threads: int = 48
    postprocess: str = None            # named postprocess_func, or None
    # Operational / gating knobs (optional):
    n_frames: int = None               # limit cal to the first N sorted reproj files
    skip_mosaic: bool = False          # cal only (no mosaic / wavelength)
    wavelength_coadd: bool = True      # append the LVF wav_mean/wav_std maps
    reproj_override: str = None        # use this reproj dir directly (skip NVMe staging)

    instrument_cfg: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    lsqr: dict = field(default_factory=dict)
    mosaic: dict = field(default_factory=dict)
    zodi: dict = field(default_factory=dict)
    reproject: dict = field(default_factory=dict)
    tiling: dict = field(default_factory=dict)   # [tiling] — tile the field (task 'cal'); old spelling [tiled]
    passes: dict = field(default_factory=dict)   # [passes] — the N-pass alternating solve (task = 'npass')

    @property
    def tiled(self):
        """Old spelling of ``tiling``."""
        return self.tiling

    def resolved_run_name(self):
        det = self.instrument_cfg.get('detector')
        return self.run_name.format(detector=det) if self.run_name else None


# Top-level scalar keys (everything else must be a recognized table). The
# instrument *name* lives inside the [instrument] table as `name`, not at the top
# level (a top-level `instrument = "..."` would collide with the [instrument]
# table in TOML).
_SCALAR_KEYS = {
    'task', 'mode', 'output_dir', 'run_name', 'resolution_arcsec',
    'cache_dir', 'suffix', 'oversample', 'staging', 'keep_nvme', 'hdd_io_limit',
    'apply_n_threads', 'postprocess', 'n_frames', 'skip_mosaic', 'reproj_override',
    'wavelength_coadd',
}
_TABLE_KEYS = {
    'instrument': 'instrument_cfg', 'params': 'params', 'calibration': 'calibration',
    'lsqr': 'lsqr', 'mosaic': 'mosaic', 'zodi': 'zodi', 'reproject': 'reproject',
    'tiling': 'tiling', 'tiled': 'tiling', 'passes': 'passes',
}


def load_config(path):
    with open(path, 'rb') as f:
        raw = tomllib.load(f)

    kwargs = {}
    for k, v in raw.items():
        if k in _TABLE_KEYS:
            if _TABLE_KEYS[k] in kwargs:
                raise ValueError(f"{path}: both [tiling] and [tiled] given; keep one")
            kwargs[_TABLE_KEYS[k]] = v
        elif k in _SCALAR_KEYS:
            kwargs[k] = v
        else:
            raise ValueError(
                f"unknown top-level key {k!r} in {path}; expected one of "
                f"{sorted(_SCALAR_KEYS)} or a table {sorted(_TABLE_KEYS)}")
    if 'task' not in kwargs:
        raise ValueError(f"{path} missing required 'task'")
    cfg = RunConfig(**kwargs)
    # Instrument selector lives inside [instrument].name (defaults to spherex).
    cfg.instrument = cfg.instrument_cfg.get('name')
    if not cfg.instrument:
        raise ValueError(f"{path}: [instrument] needs a 'name' (an instrument registered with "
                         f"selfcal.instruments, e.g. \"spherex\" or \"grid\")")
    if cfg.task != 'precompute' and cfg.resolution_arcsec is None:
        raise ValueError(f"{path}: resolution_arcsec is required")
    # task 'tiled' is the 'cal' task with a [tiling] table.
    if cfg.task == 'tiled':
        if not cfg.tiling:
            raise ValueError(f"{path}: task = 'tiled' needs a [tiling] table")
        cfg.task = 'cal'
    return cfg


def get_instrument(name):
    """The registered instrument (built-in or entry-point plugin); see selfcal.instruments."""
    from selfcal.instruments import get_instrument as _get
    return _get(name)


# Named postprocess functions selectable from config (default None).
def get_postprocess(name):
    if name is None:
        return None
    if name == 'mask_bright_pixels':
        from selfcal_scripts.runner.postprocess import mask_bright_pixels
        return mask_bright_pixels
    raise ValueError(f"unknown postprocess func {name!r}")
