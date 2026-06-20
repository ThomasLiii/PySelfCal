"""Back-compat shim: ``SelfCal.SPHERExUtility`` moved to ``selfcal.SPHERExUtility`` (refactor/selfcal-package).

Prefer ``import selfcal.SPHERExUtility``. This redirect makes ``SelfCal.SPHERExUtility`` the *same* module
object as ``selfcal.SPHERExUtility`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.instruments.spherex.SPHERExUtility as _m
_sys.modules[__name__] = _m
