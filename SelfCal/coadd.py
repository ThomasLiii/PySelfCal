"""Back-compat shim: ``SelfCal.coadd`` moved to ``selfcal.coadd`` (refactor/selfcal-package).

Prefer ``import selfcal.coadd``. This redirect makes ``SelfCal.coadd`` the *same* module
object as ``selfcal.coadd`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.coadd as _m
_sys.modules[__name__] = _m
