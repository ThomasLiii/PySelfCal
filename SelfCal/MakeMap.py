"""Back-compat shim: ``SelfCal.MakeMap`` moved to ``selfcal.MakeMap`` (refactor/selfcal-package).

Prefer ``import selfcal.MakeMap``. This redirect makes ``SelfCal.MakeMap`` the *same* module
object as ``selfcal.MakeMap`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.MakeMap as _m
_sys.modules[__name__] = _m
