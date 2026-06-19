"""Back-compat shim: ``SelfCal.MapHelper`` moved to ``selfcal.MapHelper`` (refactor/selfcal-package).

Prefer ``import selfcal.MapHelper``. This redirect makes ``SelfCal.MapHelper`` the *same* module
object as ``selfcal.MapHelper`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.MapHelper as _m
_sys.modules[__name__] = _m
