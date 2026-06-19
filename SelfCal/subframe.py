"""Back-compat shim: ``SelfCal.subframe`` moved to ``selfcal.subframe`` (refactor/selfcal-package).

Prefer ``import selfcal.subframe``. This redirect makes ``SelfCal.subframe`` the *same* module
object as ``selfcal.subframe`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.subframe as _m
_sys.modules[__name__] = _m
