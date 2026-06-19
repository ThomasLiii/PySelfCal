"""Back-compat shim: ``SelfCal.exposure_filter`` moved to ``selfcal.exposure_filter`` (refactor/selfcal-package).

Prefer ``import selfcal.exposure_filter``. This redirect makes ``SelfCal.exposure_filter`` the *same* module
object as ``selfcal.exposure_filter`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.exposure_filter as _m
_sys.modules[__name__] = _m
