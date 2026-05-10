import os
import h5py
import sys 
import glob
from tqdm import tqdm
import numpy as np

from . import WCSHelper
from . import MakeMap

from astropy.io import fits
from dataclasses import dataclass, field

from contextlib import contextmanager
import time

@contextmanager
def timer(description):
    start = time.perf_counter() # distinct from time.time(), better for execution duration
    yield
    elapsed = time.perf_counter() - start
    print(f"{description} finished in {elapsed:.2f} seconds.")

@dataclass
class PipelineConfig:
    output_dir: str
    run_name: str
    resolution_arcsec: float
    ref_path: str = None
    reproj_dir: str = None
    cal_dir: str = None
    mos_dir: str = None

    def __post_init__(self):
        # Auto-fill dependent paths if they weren't provided
        base_path = os.path.join(self.output_dir, self.run_name)
        if self.ref_path is None:
            self.ref_path = os.path.join(base_path, 'ref.fits')
        if self.reproj_dir is None:
            self.reproj_dir = os.path.join(base_path, 'reprojected')
        if self.cal_dir is None:
            self.cal_dir = os.path.join(base_path, 'calibration')
        if self.mos_dir is None:
            self.mos_dir = os.path.join(base_path, 'mosaic')

class Reprojector:
    def __init__(self, config: PipelineConfig, exposure_list=None):
        '''Initialize path to reference WCS and reprojected files'''
        self.config = config

        self.exposure_list = exposure_list
        if not os.path.exists(self.config.reproj_dir):
            os.makedirs(self.config.reproj_dir)

        self.ref_shape = None
        self.ref_wcs = None

    def define_reference(self, padding_pixels=100, use_ext=[1]):
        '''Define the smallest WCS oriented north-up, east-left frame that can contain all exposures'''
        if not os.path.exists(self.config.ref_path):
            print(f"Reference WCS not found at {self.config.ref_path}. Creating a new reference frame.")
            self.ref_wcs, self.ref_shape = WCSHelper.find_optimal_frame(
                exposure_list=self.exposure_list,
                resolution_arcsec=self.config.resolution_arcsec,
                padding_pixels=padding_pixels,
                use_ext=use_ext
            )
            WCSHelper.save_to_fits(self.ref_wcs, self.ref_shape, self.config.ref_path)
            print(f"Reference WCS saved to {self.config.ref_path}")
        else:
            self.ref_wcs, self.ref_shape = WCSHelper.load_from_fits(self.config.ref_path)
        print(f'Mosaic shape: {self.ref_shape}')
        print(f'Mosaic WCS: {self.ref_wcs}')

    def run_reproject(self, max_workers=50, reproj_func='exact', padding_percentage=0.05, 
                      sci_ext_list=None, dq_ext_list=None, exp_idx_list=None, det_idx_list=None,
                      output_dir=None, replace_existing=False, reproject_kwargs={}):
        if self.ref_wcs is None or self.ref_shape is None:
            raise ValueError("Reference WCS and shape must be defined before running reprojection. Call define_reference() first.")
        if output_dir is None:
            output_dir = self.config.reproj_dir

        with timer("Reprojection"):
            self.reproj_list = MakeMap.batch_reproject(
                # Can edit
                num_processes = max_workers, 
                reproj_func = reproj_func,  # interp: fastest, adaptive: conserves flux, exact: most accurate
                # Porbably don't want to edit
                exposure_list = self.exposure_list,
                ref_wcs = self.ref_wcs, 
                ref_shape = self.ref_shape,
                output_dir = output_dir, 
                padding_percentage = padding_percentage,
                sci_ext_list = sci_ext_list, 
                dq_ext_list = dq_ext_list,
                exp_idx_list = exp_idx_list,
                det_idx_list = det_idx_list,
                replace_existing = replace_existing,
                reproject_kwargs = reproject_kwargs
                )
            
    def check_reproj_files(self):
        for f in tqdm(self.reproj_list):
            result = MakeMap.load_reproj_file(f, fields=['sub_data',])
            if result['_is_missing_']:
                os.remove(f)
                print(f"Removed {f} due to missing data")

    def get_reproj_files(self, reproj_dir=None):
        if reproj_dir is None:
            reproj_dir = self.config.reproj_dir
        self.reproj_list = sorted(glob.glob(os.path.join(reproj_dir, '*.h5')))
        self.det_idx_list = []
        self.exp_idx_list = []
        for file in tqdm(self.reproj_list):
            file_name = os.path.basename(file)
            exp_idx, det_idx = int(file_name.split('_')[1]), int(file_name.split('_')[3].removesuffix('.h5'))
            self.det_idx_list.append(det_idx)
            self.exp_idx_list.append(exp_idx)
        
class Calibrator(Reprojector):
    def __init__(self, config: PipelineConfig, reproj_dir=None):
        super().__init__(config)
        self.get_reproj_files(reproj_dir)
        self.ref_wcs, self.ref_shape = WCSHelper.load_from_fits(self.config.ref_path)
        self.A = None
        self.b = None
        self.x = None
        self.pixel_counts = None
        # Multi-chunk-map state — always lists, even at K=1.
        self.chunk_maps = []
        self.frame_to_groups = []
        self.num_offset_groups_list = []
        self.num_chunks_list = []
        self.det_templates = []
        self.col_bases = None  # length K+1; col_bases[K] == scalar_col_start
        self.num_scalar_cols = 0

    def setup_lsqr(self, chunk_maps, grid_valid_weight, oversample_factor=1,
                   apply_mask=True, apply_weight=True, max_workers=20,
                   outlier_thresh=3.0, ignore_list=[], batch_size=10,
                   offset_regularization=False,
                   reg_weights=None, adj_infos=None, poly_constraints_list=None,
                   mean_offsets_list=None,
                   det_groups_list=None, det_templates=None,
                   postprocess_func=None, preprocess_func=None,
                   weighted_damping=False, damp_weight=0.1):
        """Build the LSQR system for K chunk maps.

        ``chunk_maps`` must be a list of K ndarrays sharing one shape. Per-map
        configuration arguments (``reg_weights``, ``adj_infos``,
        ``poly_constraints_list``, ``mean_offsets_list``, ``det_groups_list``,
        ``det_templates``) are each either ``None`` (default for every map) or
        length-K lists. A ``ValueError`` is raised on length mismatch.

        ``poly_constraints_list[m]`` is ``None`` (no polynomial-order
        constraints on map ``m``) or a list of constraint groups. Each group is
        a dict ``{'chains': (num_chains, L) int ndarray, 'stencil': (L,) float
        ndarray, 'weight': float}``. The constraint
        ``λ · Σ_ℓ stencil[ℓ] · o[chains[r, ℓ]] = 0`` is added per chain ``r`` per
        frame and generalizes adjacency reg (``stencil=[1,-1]``) to arbitrary
        finite-difference operators. See ``SPHERExUtility.compute_column_polynomial_chains``
        for the SPHEREx column-linearity helper.
        """
        assert isinstance(chunk_maps, list) and len(chunk_maps) >= 1, \
            "chunk_maps must be a non-empty list of ndarrays"
        K = len(chunk_maps)

        def _check_len(name, val):
            if val is not None and len(val) != K:
                raise ValueError(f"{name} must have length {K} (got {len(val)})")
        _check_len('reg_weights', reg_weights)
        _check_len('adj_infos', adj_infos)
        _check_len('poly_constraints_list', poly_constraints_list)
        _check_len('mean_offsets_list', mean_offsets_list)
        _check_len('det_groups_list', det_groups_list)
        _check_len('det_templates', det_templates)

        with timer("Setup LSQR"):
            self.A, self.b, self.pixel_counts = MakeMap.setup_lsqr(
                self.reproj_list, self.ref_shape,
                chunk_maps=chunk_maps,
                grid_valid_weight=grid_valid_weight,
                apply_mask=apply_mask, apply_weight=apply_weight,
                max_workers=max_workers, outlier_thresh=outlier_thresh,
                ignore_list=ignore_list, oversample_factor=oversample_factor,
                batch_size=batch_size, offset_regularization=offset_regularization,
                reg_weights=reg_weights, adj_infos=adj_infos,
                poly_constraints_list=poly_constraints_list,
                mean_offsets_list=mean_offsets_list,
                det_groups_list=det_groups_list,
                det_templates=det_templates,
                postprocess_func=postprocess_func, preprocess_func=preprocess_func,
                weighted_damping=weighted_damping, damp_weight=damp_weight)

        # Mirror the layout setup_lsqr computed so parse_x / save_calibration
        # don't have to recompute frame_to_group, col_bases, etc.
        num_frames = len(self.reproj_list)
        num_sky = self.ref_shape[0] * self.ref_shape[1]

        any_det_groups = det_groups_list is not None and any(g is not None for g in det_groups_list)
        self.num_scalar_cols = num_frames if any_det_groups else 0

        frame_to_groups = []
        num_offset_groups_list = []
        num_chunks_list = []
        det_template_arr_list = []
        col_bases = [num_sky]
        for m in range(K):
            cm = chunk_maps[m]
            num_chunks_m = int(cm.max()) + 1
            dgm = det_groups_list[m] if det_groups_list is not None else None
            if dgm is not None:
                _, ftg = np.unique(dgm, return_inverse=True)
                num_offset_groups_m = len(np.unique(dgm))
            else:
                ftg = np.arange(num_frames)
                num_offset_groups_m = num_frames
            tmpl = det_templates[m] if det_templates is not None else None
            if tmpl is not None:
                num_offset_groups_m = num_frames  # one alpha per frame
                num_chunks_m = 1
                block = num_frames
                tmpl = np.asarray(tmpl, dtype=np.float32)
            else:
                block = num_offset_groups_m * num_chunks_m
            frame_to_groups.append(ftg)
            num_offset_groups_list.append(num_offset_groups_m)
            num_chunks_list.append(num_chunks_m)
            det_template_arr_list.append(tmpl)
            col_bases.append(col_bases[-1] + block)

        self.chunk_maps = chunk_maps
        self.frame_to_groups = frame_to_groups
        self.num_offset_groups_list = num_offset_groups_list
        self.num_chunks_list = num_chunks_list
        self.det_templates = det_template_arr_list
        self.col_bases = col_bases

    def apply_lsqr(self, x0=None, atol=1e-06, btol=1e-06, damp=1e-2, iter_lim=300, precondition=True, resume=False,
                   solver='lsmr', use_float32=False, n_threads=32):
        if resume:
            if self.x is None:
                print("No previous solution found. Starting from scratch.")
            else:
                x0 = self.x
                print("Resuming LSQR from previous solution.")
        if self.A is None or self.b is None:
            raise ValueError("LSQR matrix A and vector b must be set up before applying LSQR.")
        with timer("LSQR"):
            self.x = MakeMap.apply_lsqr(self.A, self.b, ref_shape=self.ref_shape,
                                        x0=x0, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim, precondition=precondition,
                                        solver=solver, use_float32=use_float32, n_threads=n_threads)

    def load_calibration(self, cal_path=None):
        """Load a saved calibration (dual schema: legacy top-level ``offset``
        or new ``offsets/map_m`` group)."""
        if cal_path is None:
            cal_path = os.path.join(self.config.cal_dir, 'cal.h5')
        num_frames = len(self.reproj_list)
        num_sky = self.ref_shape[0] * self.ref_shape[1]
        with h5py.File(cal_path, 'r') as f:
            skymap = f['skymap'][:]
            if 'offsets' in f:
                K = int(f.attrs.get('num_maps', len(f['offsets'])))
                offsets = [f['offsets'][f'map_{m}'][:] for m in range(K)]
                chunk_maps = ([f['chunk_maps'][f'map_{m}'][:] for m in range(K)]
                              if 'chunk_maps' in f else [])
                frame_scalar = f['frame_scalar'][:] if 'frame_scalar' in f else None
            else:
                offsets = [f['offset'][:]]
                chunk_maps = []
                frame_scalar = None
        # Rebuild self.x assuming saved offsets are already per-frame expanded
        # (which is what save_calibration writes for both schemas).
        parts = [skymap.flatten()] + [o.flatten() for o in offsets]
        if frame_scalar is not None:
            parts.append(frame_scalar.flatten())
        self.x = np.concatenate(parts)

        K = len(offsets)
        self.chunk_maps = chunk_maps
        self.frame_to_groups = [np.arange(num_frames) for _ in range(K)]
        self.num_offset_groups_list = [num_frames for _ in range(K)]
        self.num_chunks_list = [int(o.shape[1]) for o in offsets]
        self.det_templates = [None] * K
        self.num_scalar_cols = num_frames if frame_scalar is not None else 0
        self.col_bases = [num_sky]
        for nc in self.num_chunks_list:
            self.col_bases.append(self.col_bases[-1] + num_frames * nc)

    def _has_scalars(self):
        """Whether the solution vector includes a per-frame scalar bias block."""
        return self.num_scalar_cols > 0

    def _expand_offset(self, m, det_offset_m, frame_scalar=None):
        """Expand map ``m``'s grouped/template offsets to per-frame
        ``(num_frames, num_chunks_m)``. ``frame_scalar`` is added when
        provided (legacy K=1 in-memory consumers); otherwise it is left out
        and saved separately at the top of the cal file.
        """
        ftg = self.frame_to_groups[m]
        if self.det_templates[m] is not None:
            alpha = det_offset_m.squeeze()  # (num_frames,)
            template = np.asarray(self.det_templates[m])
            offset = alpha[:, np.newaxis] * template[ftg]
        else:
            offset = det_offset_m[ftg]
        if frame_scalar is not None and len(frame_scalar) > 0:
            offset = offset + frame_scalar[:, np.newaxis]
        return offset

    def save_calibration(self, cal_dir=None, cal_file='cal.h5'):
        """Write the calibration in the new ``offsets/map_m`` group schema.

        Each map's per-frame offset is stored under ``offsets/map_m`` after
        expansion through that map's frame_to_group / template (no per-frame
        scalar baked in). When any map uses ``det_groups``, the shared
        per-frame scalar bias is stored at the top level as ``frame_scalar``.
        Per-map ``chunk_maps/map_m`` arrays are also stored so analysis can
        recover the chunk indexing without round-tripping config.
        """
        if cal_dir is None:
            cal_dir = self.config.cal_dir
        os.makedirs(cal_dir, exist_ok=True)
        num_frames = len(self.reproj_list)
        K = len(self.chunk_maps)

        skymap, det_offsets, frame_scalar = MakeMap.parse_x(
            self.x, ref_shape=self.ref_shape,
            num_offset_groups_list=self.num_offset_groups_list,
            num_chunks_list=self.num_chunks_list,
            num_frames=num_frames if self._has_scalars() else None)

        skymap_coverage, offset_coverages_layout, offset_valid_fracs_layout = MakeMap.parse_pixel_counts(
            pixel_counts=self.pixel_counts, ref_shape=self.ref_shape,
            num_offset_groups_list=self.num_offset_groups_list,
            chunk_maps=self.chunk_maps)

        expanded_offsets = []
        map_coverages = []
        map_coverage_fracs = []
        for m in range(K):
            num_chunks_real = int(self.chunk_maps[m].max()) + 1
            offset_m = self._expand_offset(m, det_offsets[m])
            if self.det_templates[m] is not None:
                # Template mode coverage in the layout block is shape (num_frames, 1);
                # expand to (num_frames, num_chunks_real) trivially.
                cov_m = np.ones((num_frames, num_chunks_real), dtype=np.int32)
                frac_m = np.ones((num_frames, num_chunks_real), dtype=np.float32)
            else:
                cov_m = offset_coverages_layout[m][self.frame_to_groups[m]]
                frac_m = offset_valid_fracs_layout[m][self.frame_to_groups[m]]
            expanded_offsets.append(offset_m)
            map_coverages.append(cov_m)
            map_coverage_fracs.append(frac_m)

        cal_path = os.path.join(cal_dir, cal_file)
        with h5py.File(cal_path, 'w') as f:
            f.attrs['num_maps'] = K
            f.create_dataset('skymap', data=skymap, compression='gzip')
            f.create_dataset('skymap_coverage', data=skymap_coverage, compression='gzip')
            f.create_dataset('reproj_list', data=np.array(self.reproj_list, dtype='S'))
            offsets_grp = f.create_group('offsets')
            cov_grp = f.create_group('offset_coverage')
            frac_grp = f.create_group('offset_coverage_frac')
            cm_grp = f.create_group('chunk_maps')
            for m in range(K):
                offsets_grp.create_dataset(f'map_{m}', data=expanded_offsets[m], compression='gzip')
                cov_grp.create_dataset(f'map_{m}', data=map_coverages[m], compression='gzip')
                frac_grp.create_dataset(f'map_{m}', data=map_coverage_fracs[m], compression='gzip')
                cm_grp.create_dataset(f'map_{m}', data=self.chunk_maps[m], compression='gzip')
            if self._has_scalars() and frame_scalar is not None and len(frame_scalar) > 0:
                f.create_dataset('frame_scalar', data=frame_scalar, compression='gzip')
        print(f"Calibration saved to {cal_path}")
        return cal_path

    def get_skymap(self):
        num_frames = len(self.reproj_list)
        skymap, _, _ = MakeMap.parse_x(self.x, ref_shape=self.ref_shape,
            num_offset_groups_list=self.num_offset_groups_list,
            num_chunks_list=self.num_chunks_list,
            num_frames=num_frames if self._has_scalars() else None)
        return skymap

    def get_offsets(self):
        """Return per-frame expanded offsets, one ndarray per chunk map.

        The shared per-frame scalar bias (when present) is added to map 0 only,
        matching the legacy K=1 behavior — analysis code that subtracts a
        single ``offset`` array against the data sees the same total bias.
        """
        num_frames = len(self.reproj_list)
        _, det_offsets, frame_scalar = MakeMap.parse_x(self.x, ref_shape=self.ref_shape,
            num_offset_groups_list=self.num_offset_groups_list,
            num_chunks_list=self.num_chunks_list,
            num_frames=num_frames if self._has_scalars() else None)
        out = []
        for m in range(len(self.chunk_maps)):
            scalar = frame_scalar if m == 0 else None
            out.append(self._expand_offset(m, det_offsets[m], frame_scalar=scalar))
        return out

    def get_offset(self):
        """K=1 convenience: return ``get_offsets()[0]``."""
        return self.get_offsets()[0]

    def get_det_offset(self, m=0):
        """Get grouped detector offsets before per-frame expansion.

        Use as a ``det_templates[m]`` for the template-amplitude step.
        """
        if self.det_templates[m] is not None:
            raise ValueError("get_det_offset() not available in template mode. "
                             "Run in locked-offset mode (det_groups only) first.")
        num_frames = len(self.reproj_list)
        _, det_offsets, _ = MakeMap.parse_x(self.x, ref_shape=self.ref_shape,
            num_offset_groups_list=self.num_offset_groups_list,
            num_chunks_list=self.num_chunks_list,
            num_frames=num_frames if self._has_scalars() else None)
        return det_offsets[m]  # shape (num_groups, num_chunks)

class Mosaicker(Reprojector):
    def __init__(self, config: PipelineConfig, reproj_dir=None):
        super().__init__(config)
        self.get_reproj_files(reproj_dir)
        self.ref_wcs, self.ref_shape = WCSHelper.load_from_fits(self.config.ref_path)
        self.cal_path = None
        self.cached_list = []
        # Multi-chunk-map state — list-form, with K=1 the legacy single-map case.
        self.offsets = []
        self.offset_coverages = []
        self.offset_coverage_fracs = []
        self.cal_chunk_maps = []  # chunk_maps stored in the cal file (new schema only)
        self.skymap = None
        self.skymap_coverage = None
        self.cal_path = None
        self.maps = {'mean_map': {'data': None, 'weight': None, 'aux': None, 'unit': 'MJy/sr'},
                     'std_map': {'data': None, 'weight': None, 'aux': None, 'unit': 'MJy/sr'},
                     'sc_mean_map': {'data': None, 'weight': None, 'aux': None, 'unit': 'MJy/sr'}}
        self.mean_offset = 0.0  # mean of map-0 offsets over the valid mask, used in FITS header

    def load_calibration(self, cal_path):
        """Load a saved calibration (dual schema, multi-map aware).

        Populates ``self.offsets`` / ``self.offset_coverages`` /
        ``self.offset_coverage_fracs`` as length-K lists. For the legacy
        single-map schema, K=1 and ``self.cal_chunk_maps`` stays empty. The
        top-level ``frame_scalar`` (when present) is folded into map 0 so a
        single-map subtractor sees the same total bias the legacy schema
        baked in.
        """
        with h5py.File(cal_path, 'r') as f:
            self.skymap = f['skymap'][:]
            self.reproj_list = [s.decode('utf-8') for s in f['reproj_list'][:]]
            self.skymap_coverage = f['skymap_coverage'][:]
            if 'offsets' in f:
                K = int(f.attrs.get('num_maps', len(f['offsets'])))
                self.offsets = [f['offsets'][f'map_{m}'][:] for m in range(K)]
                self.offset_coverages = [f['offset_coverage'][f'map_{m}'][:] for m in range(K)]
                self.offset_coverage_fracs = [f['offset_coverage_frac'][f'map_{m}'][:] for m in range(K)]
                self.cal_chunk_maps = ([f['chunk_maps'][f'map_{m}'][:] for m in range(K)]
                                       if 'chunk_maps' in f else [])
                if 'frame_scalar' in f:
                    self.offsets[0] = self.offsets[0] + f['frame_scalar'][:][:, np.newaxis]
            else:
                self.offsets = [f['offset'][:]]
                self.offset_coverages = [f['offset_coverage'][:]]
                self.offset_coverage_fracs = [f['offset_coverage_frac'][:]]
                self.cal_chunk_maps = []
        print(f"Calibration loaded from {cal_path} ({len(self.offsets)} map(s))")
        self.cal_path = cal_path

    def make_mosaic(self, chunk_maps, grid_valid_weight, oversample_factor=1, apply_mask=True, apply_weight=True, max_workers=20,
        make_std_map=False, apply_sigma_clipping=False, sigma=2.0, normalize_offset=False, apply_offset=True, ignore_list=[],
        det_offset_funcs=None, cache_batch_size=10, coadd_batch_size=10, cache_dir='cache/',
        cache_intermediate=False, det_aux=None, preprocess_func=None, postprocess_func=None, valid_chunk_thresh=0.01):
        """Build coadded maps applying per-map calibration offsets.

        ``chunk_maps`` is a length-K list of (typically grid-resolution) chunk
        maps; ``det_offset_funcs`` is the matching length-K list of
        ``(chunk_map, chunk_offset) -> grid_offset`` callables. The
        per-frame offsets loaded by ``load_calibration`` (one ``(num_frames,
        num_chunks_m)`` array per map) are zeroed where the per-map
        coverage fraction falls below ``valid_chunk_thresh``; ``mean_offset``
        is reported on map 0 only and embedded in the FITS header by
        ``save_mosaic`` for legacy compatibility.
        """
        assert isinstance(chunk_maps, list) and chunk_maps, \
            "chunk_maps must be a non-empty list of ndarrays"
        K = len(chunk_maps)
        if det_offset_funcs is not None:
            assert len(det_offset_funcs) == K, \
                f"det_offset_funcs length must match chunk_maps ({K})"
        self.chunk_maps = chunk_maps

        offset_lists_param = None
        if apply_offset:
            if self.offsets:
                if len(self.offsets) != K:
                    raise ValueError(
                        f"calibration has {len(self.offsets)} maps but "
                        f"make_mosaic was called with {K} chunk_maps")
                offset_lists_param = []
                for m in range(K):
                    off = self.offsets[m].copy()
                    valid = self.offset_coverage_fracs[m] >= valid_chunk_thresh
                    if m == 0:
                        # Legacy compat: report mean_offset on map 0 only.
                        self.mean_offset = (float(np.mean(off[valid]))
                                            if np.any(valid) else 0.0)
                        if normalize_offset:
                            off[valid] = off[valid] - self.mean_offset
                    off[~valid] = 0.0
                    offset_lists_param.append(off)
            else:
                print("Warning: Calibration offsets not available. No offsets will be applied.")

        # Bundle arguments common to all compute_coadd_map calls
        common_kwargs = {
            'ref_shape': self.ref_shape,
            'file_list': self.reproj_list,
            'offset_lists': offset_lists_param,
            'apply_weight': apply_weight,
            'apply_mask': apply_mask,
            'chunk_maps': chunk_maps,
            'max_workers': max_workers,
            'grid_valid_weight': grid_valid_weight,
            'ignore_list': ignore_list,
            'oversample_factor': oversample_factor,
            'det_offset_funcs': det_offset_funcs,
            'cache_dir': cache_dir,
            'use_cached': False,
            'det_aux': det_aux,
            'preprocess_func': preprocess_func,
            'postprocess_func': postprocess_func
        }

        if cache_intermediate:
            print("Caching intermediate computations...")
            with timer("Cache computation"):
                cached_list = MakeMap.compute_coadd_map(
                    mode='cache',
                    batch_size=cache_batch_size,
                    **common_kwargs
                )
            self.cached_list = cached_list
            common_kwargs['file_list'] = cached_list
            common_kwargs['use_cached'] = True

        print("Computing mean map...")
        with timer("Mean map computation"):
            self.maps['mean_map']['data'], self.maps['mean_map']['weight'], self.maps['mean_map']['aux'] = MakeMap.compute_coadd_map(
                mode='mean', 
                batch_size=coadd_batch_size,
                **common_kwargs
            )
        
        if make_std_map:
            print("Computing std map...")
            with timer("Std map computation"):
                self.maps['std_map']['data'], self.maps['std_map']['weight'], self.maps['std_map']['aux'] = MakeMap.compute_coadd_map(
                    mode='std', 
                    mean_map=self.maps['mean_map']['data'], 
                    batch_size=coadd_batch_size,
                    **common_kwargs
                )

        if make_std_map and apply_sigma_clipping:
            print("Computing sigma-clipped mean map...")
            
            with timer("Sigma-clipped mean map computation"):
                self.maps['sc_mean_map']['data'], self.maps['sc_mean_map']['weight'], self.maps['sc_mean_map']['aux'] = MakeMap.compute_coadd_map(
                    mode='sigma_clip',
                    mean_map=self.maps['mean_map']['data'],
                    std_map=self.maps['std_map']['data'],
                    sigma=sigma,
                    batch_size=coadd_batch_size,
                    **common_kwargs
                    )

        return self.maps
    
    def append_maps(self, new_maps):
        for map_name in new_maps:
            self.maps[map_name] = {'data': None, 'weight': None, 'aux': None, 'unit': None}
            for key in new_maps[map_name]:
                self.maps[map_name][key] = new_maps[map_name][key]

    def save_mosaic(self, mos_dir=None, mos_file='mosaic.fits', overwrite=False):
        '''
        Extension naming convention:
        Coadd Maps: 
            - 'MEAN_MAP': Simple mean coadd
            - 'MEAN_MAP_WEIGHT': Weight map for mean coadd
            - 'STD_MAP': Standard deviation of pixel values per pixel
            - 'STD_MAP_WEIGHT': Weight map for std coadd
            - 'SC_MEAN_MAP': Sigma-clipped mean coadd
            - 'SC_MEAN_MAP_WEIGHT': Weight map for sigma-clipped mean coadd
        Auxiliary Maps:
            - 'WAV_MEAN': Mean wavelength map
            - 'WAV_STD': Standard deviation of wavelength map
        '''
        if mos_dir is None:
            mos_dir = self.config.mos_dir
        if not os.path.exists(mos_dir):
            os.makedirs(mos_dir)

        mos_path = os.path.join(mos_dir, mos_file)

        hdu_list = []
        for m in self.maps:
            if self.maps[m]['data'] is not None:
                hdu = fits.ImageHDU(data=self.maps[m]['data'], header=self.ref_wcs.to_header())
                hdu.header['NAXIS1'] = self.ref_shape[1]
                hdu.header['NAXIS2'] = self.ref_shape[0]
                hdu.header['NAXIS'] = 2
                hdu.header['BUNIT'] = self.maps[m]['unit']
                hdu.header['EXTNAME'] = m.upper()
                hdu.header['MEANOFF'] = float(self.mean_offset) if np.isfinite(self.mean_offset) else 0.0
                hdu_list.append(hdu)
            if self.maps[m]['weight'] is not None:
                hdu = fits.ImageHDU(data=self.maps[m]['weight'], header=self.ref_wcs.to_header())
                hdu.header['NAXIS1'] = self.ref_shape[1]
                hdu.header['NAXIS2'] = self.ref_shape[0]
                hdu.header['NAXIS'] = 2
                hdu.header['BUNIT'] = 'Weight'
                hdu.header['EXTNAME'] = f"{m.upper()}_WEIGHT"
                hdu_list.append(hdu)
            if self.maps[m]['aux'] is not None:
                hdu = fits.ImageHDU(data=self.maps[m]['aux'], header=self.ref_wcs.to_header())
                hdu.header['NAXIS1'] = self.ref_shape[1]
                hdu.header['NAXIS2'] = self.ref_shape[0]
                hdu.header['NAXIS'] = 2
                hdu.header['BUNIT'] = 'Auxiliary'
                hdu.header['EXTNAME'] = f"{m.upper()}_AUX"
                hdu_list.append(hdu)


        primary_hdu = fits.PrimaryHDU()

        hdul = fits.HDUList([primary_hdu] + hdu_list)
        hdul.writeto(mos_path, overwrite=overwrite)
        print(f"Mosaic saved to {mos_path}")
        return mos_path
