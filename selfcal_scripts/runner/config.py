"""RunConfig + TOML loader + instrument resolver.

A run is fully described by one TOML file. The loader splits it into:
  * generic top-level scalars (task / instrument / mode / output paths / staging),
  * stage tables consumed verbatim as kwargs (``[calibration]`` ``[lsqr]``
    ``[mosaic]`` ``[zodi]`` ``[reproject]`` ``[tiled]``),
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
    task: str                          # cal | tiled | reproject | precompute
    instrument: str = "spherex"
    mode: str = None                   # cal/tiled mode name (None for reproject/precompute)
    output_dir: str = None
    run_name: str = None               # may contain "{detector}"
    resolution_arcsec: float = 6.2
    cache_dir: str = "/home/thomasli/selfcal-project/selfcal/cache/"
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
    reproj_override: str = None        # use this reproj dir directly (skip NVMe staging)

    instrument_cfg: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    lsqr: dict = field(default_factory=dict)
    mosaic: dict = field(default_factory=dict)
    zodi: dict = field(default_factory=dict)
    reproject: dict = field(default_factory=dict)
    tiled: dict = field(default_factory=dict)

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
}
_TABLE_KEYS = {
    'instrument': 'instrument_cfg', 'params': 'params', 'calibration': 'calibration',
    'lsqr': 'lsqr', 'mosaic': 'mosaic', 'zodi': 'zodi', 'reproject': 'reproject',
    'tiled': 'tiled',
}


def load_config(path):
    with open(path, 'rb') as f:
        raw = tomllib.load(f)

    kwargs = {}
    for k, v in raw.items():
        if k in _TABLE_KEYS:
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
    cfg.instrument = cfg.instrument_cfg.get('name', cfg.instrument)
    return cfg


def get_instrument(name):
    if name == 'spherex':
        from selfcal.instruments.spherex.adapter import SPHERExInstrument
        return SPHERExInstrument()
    raise ValueError(f"unknown instrument {name!r} (known: 'spherex')")


# Named postprocess functions selectable from config (default None).
def get_postprocess(name):
    if name is None:
        return None
    if name == 'mask_bright_pixels':
        from selfcal_scripts.runner.postprocess import mask_bright_pixels
        return mask_bright_pixels
    raise ValueError(f"unknown postprocess func {name!r}")
