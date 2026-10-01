"""``model`` mode — the self-calibration model spelled out in the config.

The named recipes are presets of one thing: a list of sky terms and offset
terms with their priors (:class:`selfcal.models.spec.ModelSpec`). This mode
reads that list verbatim from the ``[model]`` table, so a new combination —
another line, a second detector-fixed pattern, a different smoothness or a
polynomial along another axis — needs no Python::

    mode = "model"

    [model]
    scalar = true                           # per-frame scalar (the offset's DC)
    mosaic = "full"                         # full | no_wav | none

    [[model.sky]]
    name = "continuum"                      # no coefficient: c = 1

    [[model.sky]]
    name = "aromatic"                       # a map times a coefficient c(v) of data variable(s)
    coefficient = { variable = "wavelength", function = "template", file = ".../aromatic_3p289.npz" }
    damp_weight = 5e-3

    [[model.sky]]
    name = "mine"                           # any Python function of any data variables
    coefficient = { variable = ["wavelength", "bandwidth"], function = "mypkg.shapes:smeared",
                    params = { center = 3.3 } }

    [[model.offset]]
    map = "subchannel"                      # a chunk map of the instrument (omit: the primary)
    kind = "free"                           # free | polybasis | fixed
    reg_weight = 0.1
    adjacency = ["column"]                  # omit: the map's default axes
    mean_zero = true
    poly = [ { axis = "column", degree = 1, weight = 0.5 },
             { axis = "subchannel", degree = 3, lo = 200, hi = 320, weight = 1.0 } ]

    [[model.offset]]
    map = "readout"
    kind = "fixed"
    mean_zero = true

Every name the model uses — data variables, chunk maps, chunk axes,
catalogue entries — is checked against the instrument when the spec is built.
"""
from selfcal.models.spec import ModelSpec

from .base import CalMode, register_mode


@register_mode("model")
class ModelMode(CalMode):
    mosaic_mode = "full"
    requires = ()

    def model_spec(self, cfg, inst, geom):
        if not cfg.model:
            raise ValueError("mode = \"model\" needs a [model] table (sky terms, offset terms, scalar)")
        spec = ModelSpec.from_config(cfg.model)
        # variables, maps, axes, catalogue entries, prior terms exist
        spec.check(geom, inst.coefficient_catalog(), frame_variables=self.frame_variable_names(cfg, inst))
        self.mosaic_mode = spec.mosaic
        return spec
