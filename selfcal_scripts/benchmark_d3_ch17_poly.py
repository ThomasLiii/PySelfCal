"""Benchmark calibration + mosaicking pipeline for D3 Ch17.

Tracks per-phase wall time, peak RSS (incl. child workers), system disk I/O,
and system CPU% via a 0.5s sampler thread. Mosaic sub-phases (cache, mean,
std, sigma_clip) are instrumented by monkey-patching MakeMap.compute_coadd_map
so each call from Mosaicker.make_mosaic gets its own phase record.

Config matches production (run_cal_v2.py) with two changes:
  - frame_setting NumCol bumped from 1 -> 10
  - poly constraint added: linear polynomial along columns within each
    subchannel (compute_column_polynomial_chains, weight=0.5)

The HDD->NVMe transfer is timed but explicitly marked as a one-shot
infrastructure cost - if the cache is already populated, the phase is
near-instant; if not, it triggers the full copy.

Outputs (under figures/benchmark/):
  d3_ch17_summary.txt   - phase table (wall, peak RSS, disk I/O, CPU%, workers)
  d3_ch17_samples.json  - raw sampler data + phase records
  d3_ch17_timeline.png  - RSS / CPU% / disk-rate / workers vs time

Cal + mosaic files use FILE_SUFFIX='_bench_d3_ch17_poly_k1' so they don't
collide with production runs in the same calibration directory.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import contextlib
import gc
import glob as glob_module
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(_HERE)
sys.path.append(parent_path)
sys.path.append(_HERE)  # so we can import run_cal_baseline_test helpers

from SelfCal import PipelineWrapper, MakeMap
from SelfCal.MakeMap import set_hdd_io_limit
from SelfCal.solution import compute_x0_scalar_only
from SelfCal.SPHERExUtility import (
    compute_column_polynomial_chains,
    make_spherex_stripped_offset_map,
)
from SelfCal.SPHERExAppendWav import wav_coadd
from run_cal_baseline_test import prepare_detector_inputs, prepare_channel_inputs


# ============================================================
# Phase tracker
# ============================================================

class PhaseTracker:
    """Per-phase wall time, RSS, disk I/O, CPU%.

    Background thread samples (RSS, system mem, system disk cumulative bytes,
    system CPU%, n_children) at sample_interval_s. Each phase records
    start/end timestamps; peak/avg RSS over samples in the window; disk I/O
    from system disk_io_counters deltas; avg system CPU%.
    """

    def __init__(self, sample_interval_s=0.5):
        self.process = psutil.Process()
        self.sample_interval = sample_interval_s
        self.samples = []
        self.phases = []
        self._stop = threading.Event()
        self._sampler = None
        self.t0 = None

    def start(self):
        psutil.cpu_percent(0.0, percpu=False)
        self.t0 = time.perf_counter()
        self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler.start()

    def stop(self):
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=3.0)

    def _sample_loop(self):
        while not self._stop.wait(self.sample_interval):
            t = time.perf_counter() - self.t0
            try:
                rss = self.process.memory_info().rss
                n_children = 0
                for c in self.process.children(recursive=True):
                    try:
                        rss += c.memory_info().rss
                        n_children += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            vmem = psutil.virtual_memory()
            dsk = psutil.disk_io_counters()
            sys_cpu = psutil.cpu_percent(0.0, percpu=False)
            self.samples.append({
                't': t,
                'rss_gb': rss / 1e9,
                'sys_used_gb': (vmem.total - vmem.available) / 1e9,
                'sys_avail_gb': vmem.available / 1e9,
                'sys_disk_read_b': int(dsk.read_bytes),
                'sys_disk_write_b': int(dsk.write_bytes),
                'sys_cpu_pct': sys_cpu,
                'n_children': n_children,
            })

    @contextlib.contextmanager
    def phase(self, name):
        t_start = time.perf_counter() - self.t0
        d0 = psutil.disk_io_counters()
        print(f"[bench] phase {name} starting at t={t_start:.1f}s ...", flush=True)
        try:
            yield
        finally:
            t_end = time.perf_counter() - self.t0
            d1 = psutil.disk_io_counters()
            samples_in = [s for s in self.samples if t_start <= s['t'] <= t_end]
            peak_rss = max((s['rss_gb'] for s in samples_in), default=0.0)
            avg_rss = float(np.mean([s['rss_gb'] for s in samples_in])) if samples_in else 0.0
            avg_cpu = float(np.mean([s['sys_cpu_pct'] for s in samples_in])) if samples_in else 0.0
            max_children = max((s['n_children'] for s in samples_in), default=0)
            avg_children = float(np.mean([s['n_children'] for s in samples_in])) if samples_in else 0.0
            self.phases.append({
                'name': name,
                't_start': t_start,
                't_end': t_end,
                'duration_s': t_end - t_start,
                'peak_rss_gb': peak_rss,
                'avg_rss_gb': avg_rss,
                'disk_read_gb': (d1.read_bytes - d0.read_bytes) / 1e9,
                'disk_write_gb': (d1.write_bytes - d0.write_bytes) / 1e9,
                'avg_sys_cpu_pct': avg_cpu,
                'max_children': max_children,
                'avg_children': avg_children,
                'n_samples': len(samples_in),
            })
            print(
                f"[bench] phase {name} done in {t_end-t_start:.2f}s  "
                f"peak_rss={peak_rss:.2f}GB  "
                f"diskR={d1.read_bytes-d0.read_bytes:.3e}B  "
                f"diskW={d1.write_bytes-d0.write_bytes:.3e}B  "
                f"sys_cpu={avg_cpu:.1f}%  workers~{avg_children:.1f}",
                flush=True,
            )

    def summary_table(self):
        rows = []
        header = (
            f"{'Phase':<32s}  {'Wall(s)':>9s}  {'Peak RSS(GB)':>13s}  "
            f"{'Disk R(GB)':>11s}  {'Disk W(GB)':>11s}  {'Sys CPU%':>9s}  "
            f"{'Workers(max/avg)':>17s}"
        )
        rows.append(header)
        rows.append("-" * len(header))
        total_wall = 0.0
        max_peak = 0.0
        total_read = 0.0
        total_write = 0.0
        for p in self.phases:
            rows.append(
                f"{p['name']:<32s}  {p['duration_s']:>9.2f}  "
                f"{p['peak_rss_gb']:>13.2f}  "
                f"{p['disk_read_gb']:>11.2f}  {p['disk_write_gb']:>11.2f}  "
                f"{p['avg_sys_cpu_pct']:>9.1f}  "
                f"{p['max_children']:>8d}/{p['avg_children']:>7.1f}"
            )
            total_wall += p['duration_s']
            max_peak = max(max_peak, p['peak_rss_gb'])
            total_read += p['disk_read_gb']
            total_write += p['disk_write_gb']
        rows.append("-" * len(header))
        rows.append(
            f"{'TOTAL':<32s}  {total_wall:>9.2f}  {max_peak:>13.2f}  "
            f"{total_read:>11.2f}  {total_write:>11.2f}"
        )
        return "\n".join(rows)

    def save_json(self, path):
        with open(path, 'w') as f:
            json.dump({'phases': self.phases, 'samples': self.samples}, f)

    def plot_timeline(self, path):
        if not self.samples:
            return
        ts = np.array([s['t'] for s in self.samples])
        rss = np.array([s['rss_gb'] for s in self.samples])
        cpu = np.array([s['sys_cpu_pct'] for s in self.samples])
        n_kids = np.array([s['n_children'] for s in self.samples])
        sys_used = np.array([s['sys_used_gb'] for s in self.samples])
        dr = np.array([s['sys_disk_read_b'] for s in self.samples], dtype=np.float64)
        dw = np.array([s['sys_disk_write_b'] for s in self.samples], dtype=np.float64)
        dt = np.diff(ts, prepend=ts[0] - self.sample_interval)
        dt[dt <= 0] = self.sample_interval
        dr_rate = np.diff(dr, prepend=dr[0]) / dt / 1e9
        dw_rate = np.diff(dw, prepend=dw[0]) / dt / 1e9
        # Smooth the rate display a touch (3-sample moving average) so
        # quantization noise doesn't dominate.
        def smooth(x, n=3):
            if len(x) <= n: return x
            k = np.ones(n) / n
            return np.convolve(x, k, mode='same')
        dr_rate = smooth(dr_rate)
        dw_rate = smooth(dw_rate)

        fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True, dpi=110, constrained_layout=True)
        axes[0].plot(ts, rss, lw=0.9, color='C0', label='Process tree RSS')
        axes[0].plot(ts, sys_used, lw=0.9, color='C5', alpha=0.6, label='System used')
        axes[0].set_ylabel('Memory (GB)')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[1].plot(ts, cpu, lw=0.7, color='C1')
        axes[1].set_ylabel('System CPU %')
        axes[1].set_ylim(0, 105)
        axes[2].plot(ts, dr_rate, lw=0.9, color='C2', label='read')
        axes[2].plot(ts, dw_rate, lw=0.9, color='C3', label='write')
        axes[2].set_ylabel('Disk rate (GB/s)')
        axes[2].legend(loc='upper right', fontsize=8)
        axes[3].plot(ts, n_kids, lw=0.7, color='C4', drawstyle='steps-post')
        axes[3].set_ylabel('Active workers')
        axes[3].set_xlabel('Time since benchmark start (s)')
        # Shade phases + label
        for i, p in enumerate(self.phases):
            color = f'C{i % 10}'
            for ax in axes:
                ax.axvspan(p['t_start'], p['t_end'], alpha=0.06, color=color)
                ax.axvline(p['t_start'], color='k', alpha=0.15, lw=0.4)
            mid = (p['t_start'] + p['t_end']) / 2
            ymax = axes[0].get_ylim()[1]
            axes[0].text(mid, ymax * 0.97, p['name'],
                         rotation=90, fontsize=7, va='top', ha='center', alpha=0.7)
        fig.suptitle('Pipeline benchmark — D3 Ch17, NumCol=10 + linear poly constraint', fontsize=11)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    # ============================================================
    # Config — bumped from production run_cal_v2.py
    # ============================================================
    frame_setting = {
        'Detector': 3,
        'NumSub': 10,
        'NumCh': 34,
        'NumCol': 10,  # bumped from production NumCol=1 to exercise poly constraint
    }
    POLY_DEGREE = 1
    POLY_WEIGHT = 0.5

    selfcal_config = PipelineWrapper.PipelineConfig(
        output_dir='/mnt/md124/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_nep_qr2_det{frame_setting["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2,
    )

    calibration_kwargs = {
        'apply_mask': True,
        'apply_weight': False,
        'outlier_thresh': 5.0,
        'ignore_list': [],
        'batch_size': 20,
        'offset_regularization': True,
        'reg_weights': [0.1],
        'weighted_damping': True,
        'damp_weight': 0.1,
        'max_workers': 32,
        'postprocess_func': None,
    }
    lsqr_kwargs = {
        'atol': 1e-06, 'btol': 1e-06, 'damp': 0,
        'iter_lim': 50, 'precondition': True, 'solver': 'lsqr',
    }
    mosaic_kwargs = {
        'apply_mask': True, 'apply_weight': False,
        'make_std_map': True, 'apply_sigma_clipping': True, 'sigma': 2.0,
        'ignore_list': [21],
        'cache_batch_size': 20, 'coadd_batch_size': 30,
        'cache_intermediate': True, 'max_workers': 32,
    }
    mosaic_oversample_factor = 2

    CACHE_DIR = '/home/thomasli/selfcal-project/selfcal/cache/'
    BENCH_DIR = '/home/thomasli/selfcal-project/selfcal/figures/benchmark/'
    os.makedirs(BENCH_DIR, exist_ok=True)
    FILE_SUFFIX = '_bench_d3_ch17_poly_k1'
    chs = [[17]]
    HDD_IO_LIMIT = 20

    set_hdd_io_limit(HDD_IO_LIMIT)

    # ============================================================
    # Start tracker
    # ============================================================
    tracker = PhaseTracker(sample_interval_s=0.5)
    tracker.start()
    print(
        f"[bench] config: D{frame_setting['Detector']} ch={chs[0]} "
        f"NumCol={frame_setting['NumCol']} poly_degree={POLY_DEGREE} "
        f"poly_weight={POLY_WEIGHT} use_per_frame_scalar=True",
        flush=True,
    )

    # ============================================================
    # Phase: HDD->NVMe transfer (one-time; tracked but lower priority)
    # ============================================================
    nvme_reproj_dir = os.path.join(
        CACHE_DIR, f'reproj_nvme_{selfcal_config.run_name}'
    )
    os.makedirs(nvme_reproj_dir, exist_ok=True)
    hdd_reproj_files = sorted(glob_module.glob(os.path.join(selfcal_config.reproj_dir, '*.h5')))

    with tracker.phase('transfer_hdd_to_nvme'):
        n_existing = sum(
            1 for f in hdd_reproj_files
            if os.path.exists(os.path.join(nvme_reproj_dir, os.path.basename(f)))
        )
        if n_existing == len(hdd_reproj_files):
            print(f"[bench] NVMe cache already populated ({len(hdd_reproj_files)} files); skipping copy", flush=True)
        else:
            print(f"[bench] transferring {len(hdd_reproj_files) - n_existing} files HDD->NVMe...", flush=True)
            def copy_to_nvme(src_path):
                dst_path = os.path.join(nvme_reproj_dir, os.path.basename(src_path))
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                return dst_path
            with ThreadPoolExecutor(max_workers=HDD_IO_LIMIT or 20) as executor:
                for _ in tqdm(executor.map(copy_to_nvme, hdd_reproj_files),
                              total=len(hdd_reproj_files), desc="HDD->NVMe", unit="file"):
                    pass

    set_hdd_io_limit(None)

    # ============================================================
    # Inputs
    # ============================================================
    with tracker.phase('detector_inputs'):
        det_inputs = prepare_detector_inputs(frame_setting, mosaic_oversample_factor)

    ch = chs[0]
    with tracker.phase('channel_inputs'):
        ch_inputs = prepare_channel_inputs(
            ch, frame_setting,
            det_inputs['det_chunk_map'], det_inputs['grid_chunk_map'],
        )

    with tracker.phase('build_poly_chains'):
        poly_chains, poly_stencil = compute_column_polynomial_chains(
            det_inputs['det_chunk_map'], frame_setting['NumCol'], degree=POLY_DEGREE,
        )
        poly_constraints_list = [[{
            'chains': poly_chains,
            'stencil': poly_stencil,
            'weight': POLY_WEIGHT,
        }]]
        print(
            f"[bench] poly chains shape={poly_chains.shape}  "
            f"stencil={poly_stencil.tolist()}",
            flush=True,
        )

    frame_setting_str = '_'.join([f'{k}{v}' for k, v in frame_setting.items()])
    job_name = f"Ch{'-'.join(map(str, ch))}"
    job_tag = f'{frame_setting_str}_{job_name}{FILE_SUFFIX}'
    cal_file = f'cal_{job_tag}.h5'
    mos_file = f'mosaic_{job_tag}.fits'
    cache_dir = os.path.join(CACHE_DIR, f'cache_{job_tag}')

    # ============================================================
    # Calibration
    # ============================================================
    cc = PipelineWrapper.Calibrator(selfcal_config, reproj_dir=nvme_reproj_dir)
    num_frames_run = len(cc.reproj_list)
    print(f"[bench] num_frames={num_frames_run}  num_chunks={int(det_inputs['det_chunk_map'].max())+1}", flush=True)

    with tracker.phase('cal_setup_lsqr'):
        cc.setup_lsqr(
            chunk_maps=[det_inputs['det_chunk_map']],
            grid_valid_weight=ch_inputs['det_valid_mask_padded'],
            oversample_factor=1,
            adj_infos=[det_inputs['adj_info']],
            poly_constraints_list=poly_constraints_list,
            mean_offsets_list=[np.zeros(num_frames_run)],
            use_per_frame_scalar=True,
            **calibration_kwargs,
        )

    with tracker.phase('cal_warmstart'):
        x0 = compute_x0_scalar_only(
            cc.A, cc.b, cc.ref_shape,
            scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
        )

    with tracker.phase('cal_apply_lsqr'):
        cc.apply_lsqr(x0=x0, use_float32=True, n_threads=32, **lsqr_kwargs)

    with tracker.phase('cal_save'):
        nvme_list = cc.reproj_list
        cc.reproj_list = [
            os.path.join(selfcal_config.reproj_dir, os.path.basename(f))
            for f in nvme_list
        ]
        cal_path = cc.save_calibration(cal_file=cal_file)
        cc.reproj_list = nvme_list

    del cc
    gc.collect()

    # ============================================================
    # Mosaicking — patch compute_coadd_map to capture sub-phases
    # ============================================================
    _orig_compute_coadd_map = MakeMap.compute_coadd_map

    def _instrumented_compute_coadd_map(mode, *args, **kwargs):
        with tracker.phase(f'mosaic_coadd_{mode}'):
            return _orig_compute_coadd_map(mode, *args, **kwargs)

    MakeMap.compute_coadd_map = _instrumented_compute_coadd_map

    try:
        mm = PipelineWrapper.Mosaicker(selfcal_config, reproj_dir=nvme_reproj_dir)

        with tracker.phase('mosaic_load_cal'):
            mm.load_calibration(cal_path=cal_path)
            mm.reproj_list = [
                os.path.join(nvme_reproj_dir, os.path.basename(f))
                for f in mm.reproj_list
            ]

        partial_make_offset_map = partial(
            make_spherex_stripped_offset_map,
            chunk_valid_mask=ch_inputs['chunk_valid_mask'],
            lvf_params=det_inputs['lvf_params'],
            r_edges=det_inputs['r_edges'],
            x_edges=det_inputs['x_edges'],
            tot_subchannels=frame_setting['NumSub'] * frame_setting['NumCh'] + 2,
            num_columns=frame_setting['NumCol'],
            fill_invalid=True,
        )

        # mosaic_make_mosaic_total contains the four mosaic_coadd_* sub-phases
        with tracker.phase('mosaic_make_mosaic_total'):
            maps = mm.make_mosaic(
                chunk_maps=[det_inputs['grid_chunk_map']],
                grid_valid_weight=ch_inputs['grid_valid_weight'],
                oversample_factor=mosaic_oversample_factor,
                det_offset_funcs=[partial_make_offset_map],
                cache_dir=cache_dir,
                **mosaic_kwargs,
            )
    finally:
        MakeMap.compute_coadd_map = _orig_compute_coadd_map

    with tracker.phase('mosaic_wav_coadd'):
        wav_mean, wav_std = wav_coadd(
            det_inputs['det_BC'], det_inputs['det_BW'],
            mean_map=maps['mean_map']['data'],
            std_map=maps['std_map']['data'],
            reproj_list=mm.reproj_list,
            cache_list=mm.cached_list,
            ref_shape=maps['mean_map']['data'].shape,
            sigma=mosaic_kwargs['sigma'],
            batch_size=40, max_workers=30,
        )

    with tracker.phase('mosaic_append_maps'):
        mm.append_maps({
            'wav_mean_map': {'data': wav_mean, 'unit': 'um'},
            'wav_std_map': {'data': wav_std, 'unit': 'um'},
        })

    with tracker.phase('mosaic_save'):
        mm.save_mosaic(mos_file=mos_file, overwrite=True)

    # Optional cleanup of mosaic cache scratch dir (NOT the NVMe reproj cache)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    tracker.stop()

    # ============================================================
    # Report
    # ============================================================
    summary = tracker.summary_table()
    print()
    print("=" * 120, flush=True)
    print(summary, flush=True)
    print("=" * 120, flush=True)

    txt_path = os.path.join(BENCH_DIR, 'd3_ch17_summary.txt')
    json_path = os.path.join(BENCH_DIR, 'd3_ch17_samples.json')
    png_path = os.path.join(BENCH_DIR, 'd3_ch17_timeline.png')

    with open(txt_path, 'w') as f:
        f.write(summary + '\n')
    tracker.save_json(json_path)
    tracker.plot_timeline(png_path)

    print(f"\n[bench] wrote {txt_path}")
    print(f"[bench] wrote {json_path}")
    print(f"[bench] wrote {png_path}")


if __name__ == '__main__':
    main()
