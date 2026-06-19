"""Back-compat shim: ``SelfCal.SPHERExAppendWav`` moved to ``selfcal.SPHERExAppendWav`` (refactor/selfcal-package).

Prefer ``import selfcal.SPHERExAppendWav``. This redirect makes ``SelfCal.SPHERExAppendWav`` the *same* module
object as ``selfcal.SPHERExAppendWav`` (no duplicate state) so existing imports keep working
during the migration.
"""
import sys as _sys
import selfcal.SPHERExAppendWav as _m
_sys.modules[__name__] = _m
