"""Back-compat shim: ``SelfCal.lsqr`` moved to ``selfcal.lsqr`` (refactor/selfcal-package).

Prefer ``import selfcal.lsqr``. This redirect makes ``SelfCal.lsqr`` the *same* module
object as ``selfcal.lsqr`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.core.lsqr as _m
_sys.modules[__name__] = _m
