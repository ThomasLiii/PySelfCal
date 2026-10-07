"""This machine (orca): where products and fast scratch disk are, and the production worker counts.

Machine settings never change a product's bytes (:class:`selfcal.Compute`); another machine needs
only another site module.
"""
import selfcal as sc

#: Products: <OUTPUTS>/<run name>/{ref.fits, reprojected/, calibration/, mosaic/, logs/, records/}.
OUTPUTS = '/mnt/md124/thomasli/selfcal/outputs'

#: Fast local disk for staged frames, the solver's spill files and the coadd's frame cache.
NVME = '/home/thomasli/selfcal-project/selfcal/cache'

#: The production machine settings (tuned 2026-05: 48 processes for the assembly and the coadd).
ORCA = sc.Compute(NVME, workers=48, coadd_workers=48)
