import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import numpy as np

from SelfCal import PipelineWrapper
from SelfCal.exposure_filter import filter_exposures_by_header

frame_setting = {
    'Detector': 5,
    'NumSub': 10,
    'NumCh': 34,
    'NumCol': 3,
}
selfcal_config = PipelineWrapper.PipelineConfig(
    output_dir='/mnt/md124/thomasli/selfcal/outputs/',
    run_name=f'SPHEREx_NEP_2026W17_D{frame_setting["Detector"]}_6p2arcsec',
    resolution_arcsec=6.2
)

# Optional: point a new run at an existing ref.fits to share the projection
# (same CRVAL/CDELT/CTYPE/PC) across runs that cover different areas around
# the same center. The new run's ref_shape + CRPIX are computed from its
# own exposure list. Leave None to fit an optimal frame from scratch.
SOURCE_REF_PATH = None
# e.g. '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D5_6p2arcsec/ref.fits'

qr1_dir = '/mnt/md124/SPHEREx/SPHEREx_nep_data/qr1_newgain'
qr2_dir = '/mnt/md124/SPHEREx/SPHEREx_nep_data/qr2'
file_pattern = f'/*/*/*/*D{frame_setting["Detector"]}*.fits'

exposure_list = sorted(glob.glob(qr1_dir + file_pattern) +
                       glob.glob(qr2_dir + file_pattern))
print(f"Globbed {len(exposure_list)} candidate exposures")

# Drop poor-astrometry exposures (FINAST != 0). The (path, mtime)-keyed
# cache reuses prior header reads on identical reruns, killing the slow
# per-file fits.open loop. Cache lives alongside outputs; safe to delete.
finast_cache = os.path.join(
    selfcal_config.output_dir, '_exposure_cache',
    f'finast_D{frame_setting["Detector"]}.json')
exposure_list, dropped = filter_exposures_by_header(
    exposure_list,
    predicate=lambda h: h.get('FINAST', 2) == 0,
    keys=['FINAST'],
    ext=1,
    cache_path=finast_cache,
    max_workers=16,
)
print(f"Kept {len(exposure_list)} exposures, dropped {len(dropped)} for poor astrometry")

# Initialize Reprojector and run reprojection
rr = PipelineWrapper.Reprojector(selfcal_config, exposure_list=exposure_list)
rr.define_reference(padding_pixels=100, use_ext=[1],
                    source_ref_path=SOURCE_REF_PATH)

# The reproject library can also parallelize internally; keep one level of
# parallelism to avoid oversubscription and high kernel/system CPU time.
reproject_max_workers = 50
reproject_inner_parallel = 1
print(
    f"Running reprojection with max_workers={reproject_max_workers}, "
    f"reproject_kwargs.parallel={reproject_inner_parallel}"
)

# run_reproject is resume-safe: it writes a manifest of the intended task
# set, filters already-done outputs before dispatch (so a Ctrl-C + rerun
# is cheap), writes h5 files atomically (.tmp + rename), and appends any
# worker failures to {reproj_dir}/failed.jsonl.
rr.run_reproject(max_workers=reproject_max_workers,
                 reproj_func='exact',
                 padding_percentage=0.05,
                 sci_ext_list=[1],
                 dq_ext_list=[2],
                 exp_idx_list=np.arange(0, len(exposure_list)),
                 det_idx_list=[0] * len(exposure_list),
                 replace_existing=False,
                 reproject_kwargs={'parallel': reproject_inner_parallel}
                )

# One-line summary of this run's reprojection state.
rr.status()
print("Reprojection complete")