"""Back-compat shim: ``SelfCal.ZodiAnchor`` moved to ``selfcal.ZodiAnchor`` (refactor/selfcal-package).

Prefer ``import selfcal.ZodiAnchor``. This redirect makes ``SelfCal.ZodiAnchor`` the *same* module
object as ``selfcal.ZodiAnchor`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.ZodiAnchor as _m
_sys.modules[__name__] = _m
