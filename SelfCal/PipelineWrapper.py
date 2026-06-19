"""Back-compat shim: ``SelfCal.PipelineWrapper`` moved to ``selfcal.PipelineWrapper`` (refactor/selfcal-package).

Prefer ``import selfcal.PipelineWrapper``. This redirect makes ``SelfCal.PipelineWrapper`` the *same* module
object as ``selfcal.PipelineWrapper`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.PipelineWrapper as _m
_sys.modules[__name__] = _m
