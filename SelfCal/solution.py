"""Back-compat shim: ``SelfCal.solution`` moved to ``selfcal.solution`` (refactor/selfcal-package).

Prefer ``import selfcal.solution``. This redirect makes ``SelfCal.solution`` the *same* module
object as ``selfcal.solution`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.core.solution as _m
_sys.modules[__name__] = _m
