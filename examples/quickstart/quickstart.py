"""Quickstart, steps 2 and 3: reproject the simulated exposures onto one grid, then solve for the
sky and the offsets and coadd the calibrated frames.

    python examples/quickstart/quickstart.py

Run it from the directory that holds quickstart_output/ (the repository root in the quickstart):
relative paths are used as written. reproject.toml and cal.toml are the same run as TOML configs
for the runner; both forms make the same products.
"""
import selfcal as sc

# How to read an exposure: a 64 x 64-pixel camera whose offsets are a 4 x 4 grid of chunks;
# extension 1 holds the image and its WCS, extension 2 an integer data-quality (DQ) mask. The tag
# starts the product names.
CAMERA = sc.Camera((64, 64), chunks=(4, 4), dq_ext=2, tag="Sim")

# Where the products go (quickstart_output/quickstart/), the camera, the pixel scale of the
# reference grid in arcsec, and the machine: scratch space for staged frames and caches, two
# worker processes.
FIELD = sc.Field("quickstart_output/quickstart", CAMERA, pixel_scale=10.0,
                 compute=sc.Compute("quickstart_output/cache", workers=2))

# What to solve: a sky map, an offset per frame and chunk (smoothed between neighbouring chunks,
# zero mean in each frame) and a scalar per frame, the sky damped toward zero (this fixes its zero
# point); up to 200 LSQR iterations; a coadd with a 3-sigma clipped mean. Two threads and small
# batches suit 24 frames. The name ends the product names.
RECIPE = sc.Recipe(sc.continuum(smooth=0.1, damping=0.001),
                   fit=sc.Fit(200, tolerance=1e-8),
                   coadd=sc.Coadd(clip=3.0),
                   numerics=sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4),
                   name="quickstart")

if __name__ == "__main__":
    FIELD.reproject("quickstart_output/exposures/sim_*.fits", method="interp", padding=8)
    print(FIELD.plan(RECIPE))
    print(FIELD.calibrate(RECIPE))
