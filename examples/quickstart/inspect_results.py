"""Inspect the products of the selfcal quickstart and compare them with the injected truth.

Run it after the calibration (quickstart.py), from the directory that
holds quickstart_output/ (the repository root in the quickstart):

    python examples/quickstart/inspect_results.py

It prints what the calibration file and the mosaic contain, compares the
recovered offsets, scalars and sky with the ones simulate.py injected, and saves
a figure, quickstart_output/results_quickstart.png. For the products of another
recipe name, pass it with a leading underscore: --suffix _damp0p1 (the figure is
then results_damp0p1.png).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.colors import SymLogNorm
from simulate import CHUNKS, DET_SHAPE, NOISE_RMS, OUT_DIR, true_sky  # next to this script

from selfcal.io.calfile import CalFile
from selfcal.io.reproj import parse_reproj_basename

RUN_DIR = os.path.join(OUT_DIR, "quickstart")   # the field's directory
STEM = "Sim_Chunks4x4_All"                      # <tag>_Chunks<rows>x<cols>_<job>, then _<recipe name>


def remove_gauge(offsets):
    """Remove what the data cannot fix: the mean offset of each frame, which trades
    against the frame's scalar, and the mean of each chunk over all frames, a pattern
    fixed on the detector (a ramp in it, with a change of every frame's scalar,
    trades against a gradient of the sky)."""
    offsets = offsets - offsets.mean(axis=1, keepdims=True)
    return offsets - offsets.mean(axis=0, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="Inspect the quickstart products.")
    parser.add_argument("--suffix", default="_quickstart",
                        help="_<recipe name> (the products' suffix; default: _quickstart)")
    suffix = parser.parse_args().suffix
    cal_path = os.path.join(RUN_DIR, "calibration", f"cal_{STEM}{suffix}.h5")
    mosaic_path = os.path.join(RUN_DIR, "mosaic", f"mosaic_{STEM}{suffix}.fits")
    truth = np.load(os.path.join(OUT_DIR, "truth.npz"))

    # The calibration file: the solved sky, offsets and per-frame scalars.
    with CalFile(cal_path) as cal:
        print(cal.describe())
        offsets = cal.offsets[0]               # (frames, chunks): offset map 0, the 4 x 4 chunks
        scalars = cal.frame_scalar             # (frames,)
        coverage = cal.sky_coverage(0)         # observations of each pixel of the sky map
        # Frame exp_<k>_det_00.h5 comes from exposure k of the sorted list, sim_<k>.fits,
        # whose truth is row k of truth.npz.
        exposure = [parse_reproj_basename(path)[0] for path in cal.reproj_list]
    print(f"  offsets {offsets.shape}, scalars {scalars.shape}; the sky map is observed on "
          f"{np.count_nonzero(coverage)} pixels, up to {coverage.max()} times")

    # The mosaic: one image extension per map, all on the reference grid of ref.fits.
    print(f"{os.path.basename(mosaic_path)}:")
    with fits.open(mosaic_path) as hdul:
        for hdu in hdul[1:]:
            print(f"  {hdu.name:<19} {hdu.data.shape[0]} x {hdu.data.shape[1]}")
        mosaic = hdul["SC_MEAN_MAP"].data.astype(float)
        weight = hdul["SC_MEAN_MAP_WEIGHT"].data
        wcs = WCS(hdul["SC_MEAN_MAP"].header)
    seen = weight > 0
    mosaic[~seen] = np.nan

    # Compare with the truth.
    recovered = remove_gauge(offsets)
    injected = remove_gauge(truth["offsets"][exposure])
    r = np.corrcoef(recovered.ravel(), injected.ravel())[0, 1]
    print(f"Offsets, gauge removed: correlation {r:.4f}, rms difference "
          f"{np.std(recovered - injected):.4f} "
          f"(rms of the injected offsets {np.std(injected):.4f})")
    shift = scalars - truth["scalars"][exposure]
    print(f"Scalars: recovered - injected = {shift.mean():+.4f} on average, "
          f"rms about it {shift.std():.4f}")
    rows, cols = np.mgrid[0:mosaic.shape[0], 0:mosaic.shape[1]]
    sky = true_sky(*wcs.pixel_to_world_values(cols, rows))     # the injected sky, mosaic grid
    zero_point = np.nanmedian(mosaic - sky)
    residual = mosaic - sky - zero_point
    print(f"Mosaic - injected sky: median {zero_point:+.4f}; "
          f"rms about it {np.nanstd(residual):.4f} on {np.count_nonzero(seen)} pixels")

    # The figure: one raw exposure, the mosaic, the mosaic's residual, the offsets.
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    raw = fits.getdata(os.path.join(OUT_DIR, "exposures", "sim_000.fits"), ext=1)
    limit = np.nanmax(np.abs(residual))
    symlog = SymLogNorm(linthresh=NOISE_RMS, vmin=-limit, vmax=limit)   # linear within the noise
    panels = [(axes[0, 0], raw, "Exposure sim_000.fits (detector pixels)", {}),
              (axes[0, 1], mosaic, "Mosaic SC_MEAN_MAP", {}),
              (axes[1, 0], residual, f"Mosaic - injected sky - ({zero_point:+.3f})",
               dict(cmap="RdBu_r", norm=symlog))]
    for ax, image, title, style in panels:
        im = ax.imshow(image, origin="lower", **style)
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
    for k in range(1, CHUNKS[0]):                              # chunk boundaries on the exposure
        axes[0, 0].axhline(k * DET_SHAPE[0] / CHUNKS[0] - 0.5, color="white", lw=0.6)
    for k in range(1, CHUNKS[1]):
        axes[0, 0].axvline(k * DET_SHAPE[1] / CHUNKS[1] - 0.5, color="white", lw=0.6)
    ax = axes[1, 1]
    ax.scatter(injected.ravel(), recovered.ravel(), s=5)
    edge = 1.05 * max(np.abs(injected).max(), np.abs(recovered).max())
    ax.plot([-edge, edge], [-edge, edge], color="black", lw=0.6)
    ax.set_xlim(-edge, edge)
    ax.set_ylim(-edge, edge)
    ax.set_aspect("equal")
    ax.set_xlabel("injected offset, gauge removed")
    ax.set_ylabel("recovered offset, gauge removed")
    ax.set_title(f"{recovered.size} offsets (frame, chunk): r = {r:.4f}")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"results{suffix}.png")
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
