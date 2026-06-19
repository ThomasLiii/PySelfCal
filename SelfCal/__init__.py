"""Back-compat shim package: ``SelfCal`` was renamed to ``selfcal`` (refactor/selfcal-package).

Import from ``selfcal`` going forward. The ``SelfCal.*`` submodules are thin
redirects to the corresponding ``selfcal.*`` modules (the same module objects),
so existing scripts keep working unchanged during the migration.
"""
