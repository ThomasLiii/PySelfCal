"""Back-compat shim: ``SelfCal.WCSHelper`` moved to ``selfcal.WCSHelper`` (refactor/selfcal-package).

Prefer ``import selfcal.WCSHelper``. This redirect makes ``SelfCal.WCSHelper`` the *same* module
object as ``selfcal.WCSHelper`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.WCSHelper as _m
_sys.modules[__name__] = _m
