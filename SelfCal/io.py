"""Back-compat shim: ``SelfCal.io`` moved to ``selfcal.io`` (refactor/selfcal-package).

Prefer ``import selfcal.io``. This redirect makes ``SelfCal.io`` the *same* module
object as ``selfcal.io`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.io.reproj as _m
_sys.modules[__name__] = _m
