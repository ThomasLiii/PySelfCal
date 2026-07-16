import datetime
import glob
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field

import h5py
import numpy as np
from astropy.io import fits
from tqdm import tqdm

import warnings

from ..core import coadd
from ..io.reprojection import batch_reproject
from ..io.reproj import load_reproj_file
from ..core.lsqr import (setup_lsqr, apply_lsqr, parse_pixel_counts_sky,
                         parse_pixel_fisher_sky, apply_line_fisher_mask,
                         parse_line_separability)
from ..core.solution import parse_x_sky
from ..geometry import wcs_helper
from ..core.layout import SystemLayout
from ..models.sky_model import SkyModel

# Manifest schema bump when the JSON layout changes incompatibly.
_REPROJ_MANIFEST_SCHEMA = 1
_REPROJ_MANIFEST_NAME = 'manifest.json'
_REPROJ_FAILED_NAME = 'failed.jsonl'
_REPROJ_QUARANTINE_NAME = 'quarantine'

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

    def define_reference(self, padding_pixels=100, use_ext=[1],
                         source_ref_path=None, verify_projection=True):
        '''Define the smallest WCS oriented north-up, east-left frame that
        contains all exposures.

        Resolution order:
          1. If ``self.config.ref_path`` already exists, load it. When
             ``source_ref_path`` is also given and ``verify_projection`` is
             True, assert the loaded WCS shares the projection of the source
             (same CTYPE/CRVAL/CDELT/PC; only CRPIX/shape may differ). This
             guards against silently picking up an incompatible existing
             ref.fits on rerun.
          2. Else if ``source_ref_path`` is given, derive a new ref WCS from
             it via ``wcs_helper.derive_reference_from`` — same projection,
             new bbox + CRPIX sized to this run's exposures. Use this to
             keep multiple runs on a shared pixel grid even though they
             cover different areas.
          3. Else compute a fresh optimal frame from the exposure list.
        '''
        if os.path.exists(self.config.ref_path):
            self.ref_wcs, self.ref_shape = wcs_helper.load_from_fits(self.config.ref_path)
            if source_ref_path is not None and verify_projection:
                src_wcs, _ = wcs_helper.load_from_fits(source_ref_path)
                if not wcs_helper.projections_match(self.ref_wcs, src_wcs):
                    raise ValueError(
                        f"Existing reference at {self.config.ref_path} does "
                        f"not share projection with source_ref_path "
                        f"{source_ref_path}. Delete the existing ref to "
                        f"re-derive, or pass verify_projection=False to skip.")
        elif source_ref_path is not None:
            print(f"Reference WCS not found at {self.config.ref_path}; "
                  f"deriving from {source_ref_path}.")
            self.ref_wcs, self.ref_shape = wcs_helper.derive_reference_from(
                source_ref_path=source_ref_path,
                exposure_list=self.exposure_list,
                padding_pixels=padding_pixels,
                use_ext=use_ext,
            )
            wcs_helper.save_to_fits(self.ref_wcs, self.ref_shape, self.config.ref_path)
            print(f"Reference WCS saved to {self.config.ref_path}")
        else:
            print(f"Reference WCS not found at {self.config.ref_path}. Creating a new reference frame.")
            self.ref_wcs, self.ref_shape = wcs_helper.find_optimal_frame(
                exposure_list=self.exposure_list,
                resolution_arcsec=self.config.resolution_arcsec,
                padding_pixels=padding_pixels,
                use_ext=use_ext
            )
            wcs_helper.save_to_fits(self.ref_wcs, self.ref_shape, self.config.ref_path)
            print(f"Reference WCS saved to {self.config.ref_path}")
        print(f'Mosaic shape: {self.ref_shape}')
        print(f'Mosaic WCS: {self.ref_wcs}')

    # ---- Manifest / failed-log helpers ----
    #
    # Per-run state lives next to the reprojected files under reproj_dir:
    #   reproj_dir/
    #     manifest.json   -- intended task set for the most recent run_reproject
    #     failed.jsonl    -- append-only worker-failure log
    #     quarantine/     -- broken files moved here by check_reproj_files
    #
    # The manifest lets a re-run (after Ctrl-C / crash) know what was expected
    # without re-globbing FITS headers, and pairs cleanly with status() to
    # report done / pending / failed counts.

    @property
    def manifest_path(self):
        return os.path.join(self.config.reproj_dir, _REPROJ_MANIFEST_NAME)

    @property
    def failed_log_path(self):
        return os.path.join(self.config.reproj_dir, _REPROJ_FAILED_NAME)

    @property
    def quarantine_dir(self):
        return os.path.join(self.config.reproj_dir, _REPROJ_QUARANTINE_NAME)

    def _expected_output(self, exp_idx, det_idx, output_dir=None):
        out = output_dir or self.config.reproj_dir
        return os.path.join(out, f'exp_{exp_idx:04d}_det_{det_idx:02d}.h5')

    @staticmethod
    def _safe_mtime(path):
        try:
            return float(os.path.getmtime(path))
        except OSError:
            return None

    def _build_task_records(self, sci_ext_list, dq_ext_list, exp_idx_list,
                            det_idx_list, output_dir):
        """Compose the per-task records (input fits + extensions + output
        filename) the manifest and resume logic both need. Pure function of
        the inputs — no FITS reads, no disk scans."""
        records = []
        for i, file_path in enumerate(self.exposure_list):
            for j, (sci_ext, dq_ext) in enumerate(zip(sci_ext_list, dq_ext_list)):
                exp_idx = int(exp_idx_list[i]) if exp_idx_list is not None else i
                det_idx = int(det_idx_list[j]) if det_idx_list is not None else j
                records.append({
                    'exp_idx': exp_idx,
                    'det_idx': det_idx,
                    'input_fits': file_path,
                    'input_mtime': self._safe_mtime(file_path),
                    'sci_ext': int(sci_ext),
                    'dq_ext': int(dq_ext),
                    'output_h5': os.path.basename(
                        self._expected_output(exp_idx, det_idx, output_dir)),
                })
        return records

    def _write_manifest(self, records, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            'schema_version': _REPROJ_MANIFEST_SCHEMA,
            'created_iso': datetime.datetime.now().isoformat(timespec='seconds'),
            'run_name': self.config.run_name,
            'reproj_dir': output_dir,
            'ref_path': self.config.ref_path,
            'ref_shape': list(self.ref_shape) if self.ref_shape is not None else None,
            'tasks': records,
        }
        # Atomic write: tmp + rename so a crashed write never leaves
        # half-baked JSON behind.
        tmp = self.manifest_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.manifest_path)

    def load_manifest(self):
        """Return the most recent run's manifest dict, or None if missing /
        unreadable. Schema-version check raises on mismatch."""
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, 'r') as f:
            payload = json.load(f)
        v = payload.get('schema_version')
        if v != _REPROJ_MANIFEST_SCHEMA:
            raise ValueError(
                f'Manifest schema mismatch at {self.manifest_path}: '
                f'expected {_REPROJ_MANIFEST_SCHEMA}, found {v}.')
        return payload

    def _append_failures(self, failures):
        if not failures:
            return
        os.makedirs(os.path.dirname(self.failed_log_path), exist_ok=True)
        now = datetime.datetime.now().isoformat(timespec='seconds')
        with open(self.failed_log_path, 'a') as f:
            for fail in failures:
                f.write(json.dumps({
                    'timestamp_iso': now,
                    'exp_idx': fail.get('exp_idx'),
                    'det_idx': fail.get('det_idx'),
                    'input_fits': fail.get('input_fits'),
                    'output_h5': os.path.basename(fail.get('output_file', '')),
                    'reason': 'worker_error',
                    'error': fail.get('error'),
                }) + '\n')

    def status(self, output_dir=None):
        """Report a snapshot of reprojection state for this run.

        Reads the manifest (expected tasks), scans the output dir for
        completed files, counts failed.jsonl entries and quarantined files.
        Returns the dict it prints, so drivers can consume it too."""
        out = output_dir or self.config.reproj_dir
        manifest = None
        try:
            manifest = self.load_manifest()
        except ValueError as e:
            print(f'WARNING: could not load manifest: {e}')
        expected = len(manifest['tasks']) if manifest else None
        existing = set(os.listdir(out)) if os.path.isdir(out) else set()
        done = pending = None
        if manifest is not None:
            expected_names = [t['output_h5'] for t in manifest['tasks']]
            done = sum(1 for n in expected_names if n in existing)
            pending = expected - done
        failed_logged = 0
        if os.path.exists(self.failed_log_path):
            with open(self.failed_log_path, 'r') as f:
                failed_logged = sum(1 for _ in f if _.strip())
        quarantined = 0
        if os.path.isdir(self.quarantine_dir):
            quarantined = sum(1 for n in os.listdir(self.quarantine_dir)
                              if n.endswith('.h5'))
        report = {
            'expected': expected,
            'done': done,
            'pending': pending,
            'failed_logged': failed_logged,
            'quarantined': quarantined,
            'reproj_dir': out,
        }
        # Compact one-line summary, easy to grep in logs.
        def _s(v):
            return '?' if v is None else str(v)
        print(f"[reproj status] expected={_s(expected)} done={_s(done)} "
              f"pending={_s(pending)} failed_logged={failed_logged} "
              f"quarantined={quarantined} dir={out}")
        return report

    def run_reproject(self, max_workers=50, reproj_func='exact', padding_percentage=0.05,
                      sci_ext_list=None, dq_ext_list=None, exp_idx_list=None, det_idx_list=None,
                      output_dir=None, replace_existing=False, reproject_kwargs={}):
        """Build per-(exposure, extension) reprojection tasks, dispatch the
        pending subset, write the run manifest, and log any worker failures.

        Resume behavior: with ``replace_existing=False`` (the default), tasks
        whose final output already exists on disk are filtered out before
        dispatch (zero worker overhead per skipped task). The worker also
        retains its own existing-file check as a safety net. ``self.reproj_list``
        is set to the sorted union of pre-existing and newly-completed outputs.
        """
        if self.ref_wcs is None or self.ref_shape is None:
            raise ValueError("Reference WCS and shape must be defined before running reprojection. Call define_reference() first.")
        if output_dir is None:
            output_dir = self.config.reproj_dir
        os.makedirs(output_dir, exist_ok=True)

        records = self._build_task_records(
            sci_ext_list=sci_ext_list, dq_ext_list=dq_ext_list,
            exp_idx_list=exp_idx_list, det_idx_list=det_idx_list,
            output_dir=output_dir)
        self._write_manifest(records, output_dir)

        # Partition into already-done vs pending. Pre-filtering pending saves
        # the per-task pool dispatch overhead when most files exist (common
        # on resume).
        existing_paths = []
        pending = []
        for rec in records:
            out_path = os.path.join(output_dir, rec['output_h5'])
            if not replace_existing and os.path.exists(out_path):
                existing_paths.append(out_path)
            else:
                pending.append(rec)

        print(f'[run_reproject] manifest={len(records)} tasks, '
              f'already_done={len(existing_paths)}, pending={len(pending)}')

        new_success = []
        failures = []
        if pending:
            pending_files = [r['input_fits'] for r in pending]
            pending_exp = [r['exp_idx'] for r in pending]
            pending_det = [r['det_idx'] for r in pending]
            # sci/dq lists are per-extension and shared across all exposures.
            # batch_reproject expects per-exposure iteration internally; pass
            # the deduped per-exposure file list (one per pending task) plus
            # length-1 sci/dq lists so the inner zip yields exactly one ext
            # per task.
            with timer("Reprojection"):
                new_success, failures = batch_reproject(
                    num_processes=max_workers,
                    reproj_func=reproj_func,
                    exposure_list=pending_files,
                    ref_wcs=self.ref_wcs,
                    ref_shape=self.ref_shape,
                    output_dir=output_dir,
                    padding_percentage=padding_percentage,
                    sci_ext_list=[r['sci_ext'] for r in pending],
                    dq_ext_list=[r['dq_ext'] for r in pending],
                    exp_idx_list=pending_exp,
                    det_idx_list=pending_det,
                    replace_existing=replace_existing,
                    reproject_kwargs=reproject_kwargs,
                    per_task_extensions=True,
                )
        else:
            print('[run_reproject] nothing to do; all outputs already exist.')

        if failures:
            self._append_failures(failures)
            print(f'[run_reproject] logged {len(failures)} failures to '
                  f'{self.failed_log_path}')

        self.reproj_list = sorted(set(existing_paths) | set(new_success))
        # Mirror get_reproj_files so callers always see consistent idx state.
        self.exp_idx_list = []
        self.det_idx_list = []
        for path in self.reproj_list:
            name = os.path.basename(path)
            self.exp_idx_list.append(int(name.split('_')[1]))
            self.det_idx_list.append(int(name.split('_')[3].removesuffix('.h5')))

    def _check_one(self, path):
        """Read sub_data from a single h5 to check if it loads. Returns
        (path, ok, error_str)."""
        try:
            result = load_reproj_file(path, fields=['sub_data'])
            if result['_is_missing_'] or result.get('sub_data') is None:
                return (path, False, 'load_reproj_file _is_missing_ or sub_data is None')
            return (path, True, None)
        except Exception as e:
            return (path, False, str(e))

    def check_reproj_files(self, quarantine=True, max_workers=8):
        """Verify each reprojected file loads. Broken files are quarantined
        (moved to ``quarantine_dir``) by default, or deleted if
        ``quarantine=False`` (legacy behavior). Failures are appended to
        ``failed.jsonl`` either way."""
        if not self.reproj_list:
            print('check_reproj_files: nothing to check (reproj_list empty)')
            return
        broken = []
        # Reads are I/O bound; a small ThreadPool would also work, but the
        # existing _hdd_io_semaphore is process-local, so use ProcessPool
        # for symmetry with batch_reproject. max_workers small to avoid HDD
        # seek thrash when files live on the RAID.
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._check_one, p): p for p in self.reproj_list}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc='Checking reprojected files'):
                path, ok, err = fut.result()
                if not ok:
                    broken.append((path, err))
        if not broken:
            print(f'check_reproj_files: all {len(self.reproj_list)} files OK')
            return
        if quarantine:
            os.makedirs(self.quarantine_dir, exist_ok=True)
        records = []
        for path, err in broken:
            base = os.path.basename(path)
            try:
                if quarantine:
                    dest = os.path.join(self.quarantine_dir, base)
                    shutil.move(path, dest)
                    action = f'quarantined to {dest}'
                else:
                    os.remove(path)
                    action = 'deleted'
            except OSError as e:
                action = f'remove/move failed: {e}'
            print(f'check_reproj_files: {base}: {err} -> {action}')
            # parse idx from filename (best-effort; quarantined names match
            # the exp_NNNN_det_DD.h5 pattern)
            try:
                exp_idx = int(base.split('_')[1])
                det_idx = int(base.split('_')[3].removesuffix('.h5'))
            except (IndexError, ValueError):
                exp_idx = det_idx = None
            records.append({
                'exp_idx': exp_idx,
                'det_idx': det_idx,
                'input_fits': None,
                'output_file': path,
                'error': f'{err} ({action})',
            })
        self._append_failures(records)
        # Drop the broken paths from reproj_list so downstream stages don't
        # try to consume them.
        broken_set = {p for p, _ in broken}
        self.reproj_list = [p for p in self.reproj_list if p not in broken_set]
        print(f'check_reproj_files: {len(broken)} broken; '
              f'{len(self.reproj_list)} remain in reproj_list')

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
        self.ref_wcs, self.ref_shape = wcs_helper.load_from_fits(self.config.ref_path)
        self.A = None
        self.b = None
        self.x = None
        self.pixel_counts = None
        self.pixel_fisher = None
        # Per-pixel cont x line cross moment (2-block sky models). Enables the
        # separability map I_P = Σw²G² − (Σw²G)²/Σw² saved by save_calibration.
        self.pixel_cross = None
        # When setup_lsqr runs the Top 2 gated path (no template-mode map),
        # it returns a CSR matrix that has already had its zero columns
        # eliminated. ``active_mask`` (length num_cols_full) marks which
        # original columns survived; apply_lsqr uses it to expand the
        # compact solution back to the full layout. Both are None in the
        # legacy (template-mode) path.
        self.active_mask = None
        self.num_cols_full = None
        # If set to a non-None float, save_calibration writes it as an
        # informational ``line_fisher_threshold`` attr on the cal file. This is
        # the *recommended* read-time threshold for analysis; the saved
        # skymap_line is always raw (non-destructive). Apply the mask at read
        # time via selfcal.core.lsqr.apply_line_fisher_mask. Default None disables
        # the attr write.
        self.line_fisher_threshold = None
        # Multi-chunk-map state — always lists, even at K=1.
        self.chunk_maps = []
        self.frame_to_groups = []
        self.num_offset_groups_list = []
        self.num_chunks_list = []
        self.det_templates = []
        self.col_bases = None  # length K+1; col_bases[K] == scalar_col_start
        self.num_scalar_cols = 0
        self.layout = None  # selfcal.layout.SystemLayout, set in setup_lsqr
        self.sky_model = None  # selfcal.sky_model.SkyModel, set in setup_lsqr
        self.sky_component_names = None  # set in load_calibration (v3)

    def setup_lsqr(self, chunk_maps=None, grid_valid_weight=None, oversample_factor=1,
                   apply_mask=True, apply_weight=True, max_workers=20,
                   outlier_thresh=3.0, ignore_list=[], batch_size=10,
                   offset_regularization=False,
                   reg_weights=None, adj_infos=None, poly_constraints_list=None,
                   mean_offsets_list=None, poly_basis_list=None,
                   det_groups_list=None, det_templates=None,
                   use_per_frame_scalar=False,
                   postprocess_func=None, preprocess_func=None,
                   weighted_damping=False, damp_weight=0.1, damp_offset=0.0,
                   det_aux=None,
                   spectral_fit=False, line_center=None, line_sigma=None,
                   damp_weight_line=None,
                   line_sep_floor=None, line_sep_floor_quantile=None,
                   line_spatial_floor=None, line_spatial_floor_quantile=None,
                   offset_line_downweight=0.0,
                   offset_model=None, sky_model=None,
                   top2_compaction_enabled=True):
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
        finite-difference operators. See ``spherex_utility.compute_column_polynomial_chains``
        for the SPHEREx column-linearity helper.

        Set ``use_per_frame_scalar=True`` to add an explicit per-frame scalar
        bias column. Combined with ``mean_offsets_list=[zeros]`` and a zero
        chunk-block x0 init (see ``solution.compute_x0_scalar_only``), this
        pushes per-frame DC into the scalar so the chunk offsets only carry
        within-frame structure. This is the recommended setup for narrow
        channel-mask calibrations on H2RG detectors where ``compute_x0_from_Ab``
        alone leaves low-coverage chunks under-constrained.
        """
        # An OffsetModel bundles the per-map offset config; expanding it here to
        # the parallel-list kwargs keeps a single downstream code path that is
        # numerically identical to the equivalent flat-kwarg call. The flat
        # kwargs remain the (deprecated) transitional API.
        if offset_model is not None:
            om = offset_model.to_setup_kwargs()
            chunk_maps = om['chunk_maps']
            det_groups_list = om['det_groups_list']
            det_templates = om['det_templates']
            reg_weights = om['reg_weights']
            adj_infos = om['adj_infos']
            poly_constraints_list = om['poly_constraints_list']
            mean_offsets_list = om['mean_offsets_list']
            poly_basis_list = om['poly_basis_list']
            use_per_frame_scalar = om['use_per_frame_scalar']

        assert isinstance(chunk_maps, list) and len(chunk_maps) >= 1, \
            "chunk_maps must be a non-empty list of ndarrays (or pass offset_model=)"
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

        # Resolve the sky model. sky_model= is the forward-looking API; the
        # legacy spectral_fit flag is a deprecated shim that builds the
        # equivalent SkyModel. Passed through to setup_lsqr, which derives
        # num_sky_blocks / line damping / det_aux requirements from it.
        if sky_model is not None:
            if spectral_fit:
                warnings.warn(
                    "spectral_fit is ignored when sky_model is given; drop spectral_fit.",
                    DeprecationWarning, stacklevel=2)
            self.sky_model = sky_model
        elif spectral_fit:
            warnings.warn(
                "spectral_fit=True is deprecated; pass "
                "sky_model=SkyModel.continuum_plus_pah_gaussian(line_center, line_sigma).",
                DeprecationWarning, stacklevel=2)
            self.sky_model = SkyModel.continuum_plus_pah_gaussian(line_center, line_sigma)
        else:
            self.sky_model = SkyModel.continuum_only()

        with timer("Setup LSQR"):
            _setup_result = setup_lsqr(
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
                poly_basis_list=poly_basis_list,
                det_groups_list=det_groups_list,
                det_templates=det_templates,
                use_per_frame_scalar=use_per_frame_scalar,
                postprocess_func=postprocess_func, preprocess_func=preprocess_func,
                weighted_damping=weighted_damping, damp_weight=damp_weight,
                damp_offset=damp_offset, det_aux=det_aux,
                spectral_fit=spectral_fit, line_center=line_center,
                line_sigma=line_sigma, damp_weight_line=damp_weight_line,
                line_sep_floor=line_sep_floor,
                line_sep_floor_quantile=line_sep_floor_quantile,
                line_spatial_floor=line_spatial_floor,
                line_spatial_floor_quantile=line_spatial_floor_quantile,
                offset_line_downweight=offset_line_downweight,
                sky_model=self.sky_model,
                top2_compaction_enabled=top2_compaction_enabled)
            # setup_lsqr returns a 5-tuple in legacy / template-mode runs and
            # a 6-tuple when the Top 2 inline column compaction fires. The
            # last element is always pixel_cross (per-pixel cont x line cross
            # moment; None for non-2-block sky models).
            if _setup_result is None or _setup_result[0] is None:
                self.A, self.b = None, None
                self.pixel_counts, self.pixel_fisher = None, None
                self.active_mask, self.num_cols_full = None, None
                self.pixel_cross = None
            elif len(_setup_result) == 6:
                (self.A, self.b, self.pixel_counts,
                 self.pixel_fisher, self.active_mask,
                 self.pixel_cross) = _setup_result
                self.num_cols_full = int(self.active_mask.size)
            else:
                (self.A, self.b, self.pixel_counts,
                 self.pixel_fisher, self.pixel_cross) = _setup_result
                self.active_mask = None
                self.num_cols_full = None

        # Track sky-block count so parse_x / save_calibration / get_skymap
        # all know the sky layout. Derived from the resolved sky model.
        self.num_sky_blocks = self.sky_model.n_blocks

        # Mirror the column layout setup_lsqr computed via the SAME SystemLayout
        # so parse_x / save_calibration don't recompute (and can't drift from)
        # frame_to_group, col_bases, the scalar block, etc.
        num_frames = len(self.reproj_list)
        self.layout = SystemLayout.build(
            self.ref_shape, chunk_maps, num_sky_blocks=self.num_sky_blocks,
            num_frames=num_frames, det_groups_list=det_groups_list,
            det_templates=det_templates, use_per_frame_scalar=use_per_frame_scalar,
            poly_basis_list=poly_basis_list)
        self.chunk_maps = chunk_maps
        self.frame_to_groups = self.layout.frame_to_group_list
        self.num_offset_groups_list = self.layout.num_offset_groups_list
        self.num_chunks_list = self.layout.num_chunks_list
        self.det_templates = self.layout.det_template_arr_list
        self.num_scalar_cols = self.layout.num_scalar_cols
        self.col_bases = self.layout.col_bases

    def apply_lsqr(self, x0=None, atol=1e-06, btol=1e-06, damp=1e-2, iter_lim=300, precondition=True, resume=False,
                   solver='lsmr', use_float32=False, n_threads=32, keep_state=False):
        if resume:
            if self.x is None:
                print("No previous solution found. Starting from scratch.")
            else:
                x0 = self.x
                print("Resuming LSQR from previous solution.")
        if self.A is None or self.b is None:
            raise ValueError("LSQR matrix A and vector b must be set up before applying LSQR.")
        with timer("LSQR"):
            # Release self.A / self.b refs before calling apply_lsqr so the COO arrays
            # (~140 GB at no-srcmask region-10k scale) and the f64 b (~80 GB) can be
            # freed inside apply_lsqr after the float32 / CSR conversions complete.
            # When keep_state=True, retain self.A/self.b/self.active_mask/self.num_cols_full
            # so a caller can re-solve (e.g., iter_lim sweep) without rebuilding the system.
            A_local = self.A
            b_local = self.b
            active_mask_local = getattr(self, "active_mask", None)
            num_cols_full_local = getattr(self, "num_cols_full", None)
            if not keep_state:
                self.A = None
                self.b = None
                self.active_mask = None
            self.x = apply_lsqr(A_local, b_local, ref_shape=self.ref_shape,
                                        x0=x0, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim, precondition=precondition,
                                        solver=solver, use_float32=use_float32, n_threads=n_threads,
                                        active_mask=active_mask_local,
                                        num_cols_full=num_cols_full_local)
            if not keep_state:
                self.num_cols_full = None
                del A_local, b_local, active_mask_local

    def load_calibration(self, cal_path=None):
        """Load a saved calibration (tri-generation schema).

        v3: per-component ``sky/<name>`` groups (any number of sky blocks).
        v2: top-level ``skymap`` + ``offsets/map_m`` (+ optional ``skymap_line``).
        v1: legacy top-level ``offset``.
        """
        if cal_path is None:
            cal_path = os.path.join(self.config.cal_dir, 'cal.h5')
        num_frames = len(self.reproj_list)
        num_sky = self.ref_shape[0] * self.ref_shape[1]
        with h5py.File(cal_path, 'r') as f:
            if 'sky' in f:
                # v3: named per-component sky blocks, in declared order.
                names = [n.decode() if isinstance(n, bytes) else n
                         for n in f.attrs['sky_components']]
                sky_maps = [f['sky'][n][:] for n in names]
                self.sky_component_names = names
            else:
                # v2/v1: continuum (+ optional single line via legacy names).
                sky_maps = [f['skymap'][:]]
                if int(f.attrs.get('num_sky_blocks', 1)) == 2 and 'skymap_line' in f:
                    sky_maps.append(f['skymap_line'][:])
                self.sky_component_names = (['continuum', 'line'] if len(sky_maps) == 2
                                            else ['continuum'])
            self.num_sky_blocks = len(sky_maps)
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
        # (which is what save_calibration writes for all schemas).
        parts = [m.flatten() for m in sky_maps] + [o.flatten() for o in offsets]
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
        self.col_bases = [self.num_sky_blocks * num_sky]
        for nc in self.num_chunks_list:
            self.col_bases.append(self.col_bases[-1] + num_frames * nc)

    def _has_scalars(self):
        """Whether the solution vector includes a per-frame scalar bias block."""
        return self.num_scalar_cols > 0

    def _poly_basis_for(self, m):
        """Return the hard-poly-basis spec for map ``m`` if this solve used one
        (only set on the setup/save side; ``None`` on the load/analysis side,
        where the saved cal already holds per-chunk offsets)."""
        L = getattr(self, 'layout', None)
        pbl = getattr(L, 'poly_basis_list', None) if L is not None else None
        return pbl[m] if pbl is not None else None

    def _expand_offset(self, m, det_offset_m, frame_scalar=None):
        """Expand map ``m``'s grouped/template offsets to per-frame
        ``(num_frames, num_chunks_m)``. ``frame_scalar`` is added when
        provided (legacy K=1 in-memory consumers); otherwise it is left out
        and saved separately at the top of the cal file.

        Hard poly-basis maps: ``det_offset_m`` holds the per-frame Chebyshev
        coefficients ``a[frame, col*D + d]`` — reconstruct the per-chunk offset
        ``Σ_d a[frame, col, d]·B_d(subch)`` (chunk = subch*num_col + col) so the
        saved cal + all downstream apply/analysis stay in the standard schema.
        """
        pb = self._poly_basis_for(m)
        if pb is not None:
            from ..models.offset_basis import eval_offset_basis, n_coef
            ng = int(pb['num_groups']); ncf = n_coef(pb)
            coeffs = np.asarray(det_offset_m).reshape(-1, ng, ncf)         # (nf, num_groups, ncf)
            chunk_group = np.asarray(pb['chunk_group'])                    # (num_chunks,)
            B = eval_offset_basis(np.asarray(pb['chunk_coord']), pb)       # (num_chunks, ncf)
            # offset[frame, chunk] = Σ_k coeffs[frame, group(chunk), k] · B[chunk, k]
            sel = coeffs[:, chunk_group, :]                                # (nf, num_chunks, ncf)
            offset = np.einsum('fck,ck->fc', sel, B)                       # (nf, num_chunks)
            if frame_scalar is not None and len(frame_scalar) > 0:
                offset = offset + frame_scalar[:, np.newaxis]
            return offset
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

        # Generic per-component parse (N sky blocks; block 0 = continuum).
        sky_maps, det_offsets, frame_scalar = parse_x_sky(
            self.x, ref_shape=self.ref_shape,
            num_offset_groups_list=self.num_offset_groups_list,
            num_chunks_list=self.num_chunks_list,
            num_frames=num_frames if self._has_scalars() else None,
            num_sky_blocks=self.num_sky_blocks)

        sky_coverages, offset_coverages_layout, offset_valid_fracs_layout = (
            parse_pixel_counts_sky(
                pixel_counts=self.pixel_counts, ref_shape=self.ref_shape,
                num_offset_groups_list=self.num_offset_groups_list,
                chunk_maps=self.chunk_maps,
                num_sky_blocks=self.num_sky_blocks,
                num_chunks_list=self.num_chunks_list))

        if self.pixel_fisher is not None:
            sky_fishers = parse_pixel_fisher_sky(
                pixel_fisher=self.pixel_fisher, ref_shape=self.ref_shape,
                num_sky_blocks=self.num_sky_blocks)
        else:
            sky_fishers = [None] * self.num_sky_blocks

        # Sky-component names (block 0 = continuum). From the resolved sky model
        # when available, else synthesized for a loaded/legacy calibration.
        if getattr(self, 'sky_model', None) is not None:
            sky_names = list(self.sky_model.names)
        else:
            sky_names = ['continuum'] + [f'line_{j}' for j in range(1, self.num_sky_blocks)]
        assert len(sky_names) == self.num_sky_blocks

        expanded_offsets = []
        map_coverages = []
        map_coverage_fracs = []
        for m in range(K):
            num_chunks_real = int(self.chunk_maps[m].max()) + 1
            offset_m = self._expand_offset(m, det_offsets[m])
            if self.det_templates[m] is not None or self._poly_basis_for(m) is not None:
                # Template mode has one alpha/frame; hard poly-basis has coeff
                # columns (num_col*D), not per-chunk — the layout coverage block
                # doesn't match num_chunks_real. Use a trivial per-chunk coverage
                # here (v1: the poly offset is a smooth function of subchannel, so
                # per-chunk coverage is a refinement, not load-bearing for the fit).
                cov_m = np.ones((num_frames, num_chunks_real), dtype=np.int32)
                frac_m = np.ones((num_frames, num_chunks_real), dtype=np.float32)
            else:
                cov_m = offset_coverages_layout[m][self.frame_to_groups[m]]
                frac_m = offset_valid_fracs_layout[m][self.frame_to_groups[m]]
            expanded_offsets.append(offset_m)
            map_coverages.append(cov_m)
            map_coverage_fracs.append(frac_m)

        # Phase 6 (non-destructive): skymap_line is saved RAW. The Fisher-info
        # threshold (self.line_fisher_threshold) is saved as an informational
        # attr only; analysis applies the mask at read time via
        # ``selfcal.core.lsqr.apply_line_fisher_mask`` (or
        # ``selfcal.core.lsqr.apply_line_fisher_mask``). This lets analysts sweep
        # the threshold without re-running the ~6-10 hr calibration.

        cal_path = os.path.join(cal_dir, cal_file)
        with h5py.File(cal_path, 'w') as f:
            f.attrs['num_maps'] = K
            f.attrs['num_sky_blocks'] = self.num_sky_blocks
            f.attrs['schema_version'] = 3
            f.attrs['sky_components'] = np.array(sky_names, dtype='S')
            # --- v3: per-component sky blocks under sky/<name> (block 0 =
            # continuum, 1.. = spectral components, each an arbitrary profile's
            # per-pixel amplitude map). Saved RAW; the Fisher attr below is an
            # informational read-time mask threshold, not applied destructively.
            sky_grp = f.create_group('sky')
            skycov_grp = f.create_group('sky_coverage')
            skyfish_grp = f.create_group('sky_fisher')
            for j, name in enumerate(sky_names):
                sky_grp.create_dataset(name, data=sky_maps[j], compression='gzip')
                skycov_grp.create_dataset(name, data=sky_coverages[j], compression='gzip')
                if sky_fishers[j] is not None:
                    skyfish_grp.create_dataset(name, data=sky_fishers[j].astype('float32'),
                                               compression='gzip')
            # Per-pixel SEPARABILITY I_P (each spectral block's Schur
            # complement against all other sky blocks). Unlike the block's
            # Fisher (a magnitude metric), I_P measures wavelength diversity —
            # the quantity that bounds per-pixel amplitude variance and
            # identifies the degenerate pixels that blow up under LSQR
            # semi-convergence. Mask at read time via
            # selfcal.core.lsqr.apply_line_separability_mask. One dataset per
            # spectral block, sky_separability/<name>; for 2-block cals this is
            # the single legacy dataset, byte-identical.
            if (getattr(self, 'pixel_cross', None) is not None
                    and self.num_sky_blocks >= 2 and self.pixel_fisher is not None):
                sep_grp = f.create_group('sky_separability')
                for j in range(1, self.num_sky_blocks):
                    sep = parse_line_separability(
                        self.pixel_cross, self.pixel_fisher, self.ref_shape,
                        num_sky_blocks=self.num_sky_blocks, block=j)
                    sep_grp.create_dataset(
                        sky_names[j], data=sep.astype('float32'), compression='gzip')
            # --- Back-compat hard-link aliases (v2 readers resolve transparently):
            # skymap -> continuum; skymap_line -> the single spectral block when
            # there is exactly one. h5py resolves these on read, so
            # f['skymap'][...] etc. return identical values without duplicating data.
            cont = sky_names[0]
            f['skymap'] = sky_grp[cont]
            f['skymap_coverage'] = skycov_grp[cont]
            if cont in skyfish_grp:
                f['skymap_fisher'] = skyfish_grp[cont]
            extra_names = sky_names[1:]
            if extra_names:
                # Alias the LAST spectral block (the line; earlier extras are
                # nuisance shapes like a continuum slope). Single-extra cals
                # keep the exact v2 aliasing behavior.
                ln = extra_names[-1]
                f['skymap_line'] = sky_grp[ln]
                f['skymap_line_coverage'] = skycov_grp[ln]
                if ln in skyfish_grp:
                    f['skymap_line_fisher'] = skyfish_grp[ln]
            # Informational: recommended Fisher threshold for read-time masking.
            # Not a contract — analysis is free to pick any threshold.
            if self.line_fisher_threshold is not None:
                f.attrs['line_fisher_threshold'] = float(self.line_fisher_threshold)
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

    def _sky_names(self):
        """Ordered sky-component names (block 0 = continuum)."""
        if getattr(self, 'sky_model', None) is not None:
            return list(self.sky_model.names)
        if getattr(self, 'sky_component_names', None) is not None:
            return list(self.sky_component_names)
        return ['continuum'] + [f'line_{j}' for j in range(1, self.num_sky_blocks)]

    def _parse_x_helper(self):
        """Parse self.x into ``(sky_maps, det_offsets, frame_scalar)``.

        ``sky_maps`` is a length-``num_sky_blocks`` list (block 0 = continuum).
        """
        num_frames = len(self.reproj_list)
        return parse_x_sky(self.x, ref_shape=self.ref_shape,
            num_offset_groups_list=self.num_offset_groups_list,
            num_chunks_list=self.num_chunks_list,
            num_frames=num_frames if self._has_scalars() else None,
            num_sky_blocks=self.num_sky_blocks)

    def get_skymap(self):
        """Continuum sky map (block 0)."""
        sky_maps, _o, _s = self._parse_x_helper()
        return sky_maps[0]

    def get_skymap_line(self):
        """Back-compat: the first spectral component's map; None if continuum-only.

        For N>2 components, prefer ``get_sky(name)``.
        """
        sky_maps, _o, _s = self._parse_x_helper()
        return sky_maps[1] if len(sky_maps) > 1 else None

    def get_sky(self, name):
        """Return the sky map for a named component (e.g. 'continuum', 'pah_3p29')."""
        names = self._sky_names()
        if name not in names:
            raise KeyError(f"sky component {name!r} not in {names}")
        sky_maps, _o, _s = self._parse_x_helper()
        return sky_maps[names.index(name)]

    def get_skymaps(self):
        """All sky-component maps as a name -> ndarray dict."""
        sky_maps, _o, _s = self._parse_x_helper()
        return dict(zip(self._sky_names(), sky_maps))

    def get_offsets(self):
        """Return per-frame expanded offsets, one ndarray per chunk map.

        The shared per-frame scalar bias (when present) is added to map 0 only,
        matching the legacy K=1 behavior — analysis code that subtracts a
        single ``offset`` array against the data sees the same total bias.
        """
        _sky, det_offsets, frame_scalar = self._parse_x_helper()
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
        _sky, det_offsets, _s = self._parse_x_helper()
        return det_offsets[m]  # shape (num_groups, num_chunks)

class Mosaicker(Reprojector):
    def __init__(self, config: PipelineConfig, reproj_dir=None):
        super().__init__(config)
        self.get_reproj_files(reproj_dir)
        self.ref_wcs, self.ref_shape = wcs_helper.load_from_fits(self.config.ref_path)
        self.cal_path = None
        self.cached_list = []
        # Multi-chunk-map state — list-form, with K=1 the legacy single-map case.
        self.offsets = []
        self.offset_coverages = []
        self.offset_coverage_fracs = []
        self.cal_chunk_maps = []  # chunk_maps stored in the cal file (new schema only)
        self.skymap = None
        self.skymap_coverage = None
        self.skymap_fisher = None
        self.skymap_line_fisher = None
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
            self.skymap_fisher = f['skymap_fisher'][:] if 'skymap_fisher' in f else None
            self.skymap_line_fisher = f['skymap_line_fisher'][:] if 'skymap_line_fisher' in f else None
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
                cached_list = coadd.compute_coadd_map(
                    mode='cache',
                    batch_size=cache_batch_size,
                    **common_kwargs
                )
            self.cached_list = cached_list
            common_kwargs['file_list'] = cached_list
            common_kwargs['use_cached'] = True

        print("Computing mean map...")
        with timer("Mean map computation"):
            self.maps['mean_map']['data'], self.maps['mean_map']['weight'], self.maps['mean_map']['aux'] = coadd.compute_coadd_map(
                mode='mean', 
                batch_size=coadd_batch_size,
                **common_kwargs
            )
        
        if make_std_map:
            print("Computing std map...")
            with timer("Std map computation"):
                self.maps['std_map']['data'], self.maps['std_map']['weight'], self.maps['std_map']['aux'] = coadd.compute_coadd_map(
                    mode='std', 
                    mean_map=self.maps['mean_map']['data'], 
                    batch_size=coadd_batch_size,
                    **common_kwargs
                )

        if make_std_map and apply_sigma_clipping:
            print("Computing sigma-clipped mean map...")
            
            with timer("Sigma-clipped mean map computation"):
                self.maps['sc_mean_map']['data'], self.maps['sc_mean_map']['weight'], self.maps['sc_mean_map']['aux'] = coadd.compute_coadd_map(
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
