"""The quickstart example (examples/quickstart/, docs/getting-started/quickstart.md) end to end.

The example's files are copied into a temporary directory and run there exactly as the
quickstart page runs them from the repository root, each in its own process:
simulate.py, the reproject and cal configs through the runner (``python -m
selfcal_scripts.run``, what ``run.sh`` calls), then inspect_results.py. Nothing is written
into the repository. Checks that every product exists and that the recovered offsets and
scalars follow the injected ones. 10 to 15 s.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
from astropy.io import fits

from selfcal.io.calfile import CalFile
from selfcal.io.reproj import parse_reproj_basename

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "examples", "quickstart")
STEM = "Sim_Chunks4x4_All_quickstart"


def _run(args, cwd):
    """Run ``python <args>`` in ``cwd`` with the repository importable; return its stdout."""
    env = dict(os.environ, MPLBACKEND="Agg",
               PYTHONPATH=os.pathsep.join(p for p in (REPO, os.environ.get("PYTHONPATH")) if p))
    proc = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, f"{args} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    return proc.stdout


def _remove_gauge(a):
    """Each frame's mean offset and each chunk's mean over the frames (see inspect_results.py)."""
    a = a - a.mean(axis=1, keepdims=True)
    return a - a.mean(axis=0, keepdims=True)


def test_quickstart_example(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "examples" / "quickstart",
                    ignore=shutil.ignore_patterns("__pycache__"))
    _run(["examples/quickstart/simulate.py"], tmp_path)
    _run(["-m", "selfcal_scripts.run", "--config", "examples/quickstart/reproject.toml"], tmp_path)
    _run(["-m", "selfcal_scripts.run", "--config", "examples/quickstart/cal.toml"], tmp_path)
    report = _run(["examples/quickstart/inspect_results.py"], tmp_path)

    out = tmp_path / "quickstart_output"
    run_dir = out / "quickstart"
    assert (run_dir / "ref.fits").is_file()
    assert len(list((run_dir / "reprojected").glob("exp_*_det_00.h5"))) == 24
    cal_path = run_dir / "calibration" / f"cal_{STEM}.h5"
    mosaic_path = run_dir / "mosaic" / f"mosaic_{STEM}.fits"
    assert cal_path.is_file() and mosaic_path.is_file()
    assert (out / "results_quickstart.png").is_file()
    assert "Offsets, gauge removed: correlation" in report
    assert list((out / "cache").iterdir()) == []            # staged frames and caches cleaned up

    with fits.open(mosaic_path) as hdul:
        names = [hdu.name for hdu in hdul[1:]]
        mosaic = hdul["SC_MEAN_MAP"].data
        weight = hdul["SC_MEAN_MAP_WEIGHT"].data
    assert names == ["MEAN_MAP", "MEAN_MAP_WEIGHT", "STD_MAP", "STD_MAP_WEIGHT",
                     "SC_MEAN_MAP", "SC_MEAN_MAP_WEIGHT"]
    assert (weight > 0).sum() > 0.5 * weight.size and np.isfinite(mosaic[weight > 0]).all()

    truth = np.load(out / "truth.npz")
    with CalFile(cal_path) as cal:
        offsets, scalars = cal.offsets[0], cal.frame_scalar
        exposure = [parse_reproj_basename(p)[0] for p in cal.reproj_list]
    assert offsets.shape == (24, 16) and scalars.shape == (24,)
    r_off = np.corrcoef(_remove_gauge(offsets).ravel(),
                        _remove_gauge(truth["offsets"][exposure]).ravel())[0, 1]
    r_scalar = np.corrcoef(scalars, truth["scalars"][exposure])[0, 1]
    assert r_off > 0.95, r_off
    assert r_scalar > 0.95, r_scalar
