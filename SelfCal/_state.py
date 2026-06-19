"""Back-compat shim: ``SelfCal._state`` moved to ``selfcal._state`` (refactor/selfcal-package).

Prefer ``import selfcal._state``. This redirect makes ``SelfCal._state`` the *same* module
object as ``selfcal._state`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal._state as _m
_sys.modules[__name__] = _m
