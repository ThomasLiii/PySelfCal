"""selfcal_scripts — drivers, the generic runner, benchmarks, and analysis glue.

Not part of the installed ``selfcal`` package (pyproject ships ``selfcal*`` only);
this is the operational layer. The generic entry point is ``selfcal_scripts.run``
(``python -m selfcal_scripts.run --config <toml>``); see ``runner/`` and the
repo configs/ directory.
"""
