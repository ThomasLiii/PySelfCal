"""Back-compat re-export shim: the monolithic lsqr.py was split into
``assembly`` (worker row assembly), ``system`` (setup_lsqr orchestration +
coverage/Fisher parsers), and ``solve`` (apply_lsqr). Import from those for
clarity; this module re-exports the public surface so consumers and
``selfcal.core.lsqr`` keep working unchanged.
"""
from .assembly import _prep_lsqr, _prep_lsqr_batch_worker  # noqa: F401
from .system import (  # noqa: F401
    setup_lsqr, parse_pixel_counts, parse_pixel_fisher,
    parse_pixel_counts_sky, parse_pixel_fisher_sky, apply_line_fisher_mask,
    parse_line_separability, apply_line_separability_mask,
)
from .solve import apply_lsqr, _partition_csr, _make_parallel_operator  # noqa: F401
