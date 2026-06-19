"""Back-compat shim: ``SelfCal.EuclidUtility`` moved to ``selfcal.EuclidUtility`` (refactor/selfcal-package).

Prefer ``import selfcal.EuclidUtility``. This redirect makes ``SelfCal.EuclidUtility`` the *same* module
object as ``selfcal.EuclidUtility`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.EuclidUtility as _m
_sys.modules[__name__] = _m
