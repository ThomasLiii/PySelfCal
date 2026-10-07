"""Quickstart, step 5: the same run with the damping of the production SPHEREx recipes, 0.1, under
a name of its own, so that it makes new products next to the first ones.

    python examples/quickstart/damping.py
    python examples/quickstart/inspect_results.py --suffix _damp0p1
"""
from quickstart import FIELD  # quickstart.py, next to this script
from quickstart import RECIPE as QUICKSTART

import selfcal as sc

RECIPE = QUICKSTART.replace(model=sc.continuum(smooth=0.1, damping=0.1), name="damp0p1")

if __name__ == "__main__":
    print(FIELD.calibrate(RECIPE))
