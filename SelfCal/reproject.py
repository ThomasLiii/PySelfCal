"""Back-compat shim: ``SelfCal.reproject`` moved to ``selfcal.reproject`` (refactor/selfcal-package).

Prefer ``import selfcal.reproject``. This redirect makes ``SelfCal.reproject`` the *same* module
object as ``selfcal.reproject`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.io.reprojection as _m
_sys.modules[__name__] = _m
