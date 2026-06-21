"""NVMe staging + the RSS guardrail — generic, instrument/mode-agnostic.

The cal drivers all copy the HDD reproj files to a per-run NVMe scratch dir for
fast parallel reads, remap cal/mosaic file lists onto it, and (optionally) clean
it up. The tiled driver additionally needs the OOM guardrail. Both behaviors live
here so the engine stays readable.
"""
import glob as glob_module
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from selfcal._state import set_hdd_io_limit


def nvme_dir(cache_dir, run_name):
    return os.path.join(cache_dir, f'reproj_nvme_{run_name}')


def _copy_one(src_path, dst_dir):
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    if not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)
    return dst_path


def stage_copy(reproj_dir, nvme_reproj_dir, hdd_io_limit):
    """Copy every ``*.h5`` from the HDD reproj dir to NVMe (idempotent)."""
    os.makedirs(nvme_reproj_dir, exist_ok=True)
    hdd_files = sorted(glob_module.glob(os.path.join(reproj_dir, '*.h5')))
    print(f"Copying {len(hdd_files)} reproj files to NVMe ({nvme_reproj_dir})...")
    t_copy = time.time()
    with ThreadPoolExecutor(max_workers=hdd_io_limit or 20) as ex:
        for _ in tqdm(ex.map(lambda p: _copy_one(p, nvme_reproj_dir), hdd_files),
                      total=len(hdd_files), desc="HDD->NVMe", unit="file"):
            pass
    print(f"Reproj file copy complete in {time.time() - t_copy:.2f} seconds.")


def stage_files(files, nvme_reproj_dir, hdd_io_limit):
    """Copy a specific list of files to NVMe (tiled per-tile staging)."""
    os.makedirs(nvme_reproj_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=hdd_io_limit or 20) as ex:
        for _ in tqdm(ex.map(lambda p: _copy_one(p, nvme_reproj_dir), files),
                      total=len(files), desc="HDD->NVMe", unit="file"):
            pass


def remap_to_nvme(file_list, nvme_reproj_dir):
    """Replace each path's directory with the NVMe dir, keeping basenames."""
    return [os.path.join(nvme_reproj_dir, os.path.basename(f)) for f in file_list]


def prepare_nvme(cfg, reproj_dir, run_name):
    """Resolve + populate the NVMe scratch dir per the config staging strategy.

    Returns the nvme dir. ``copy`` stages all reproj files; ``reuse`` asserts a
    previously-staged dir exists (the owning run staged it). Either way the HDD
    I/O throttle is disabled afterward (NVMe handles massively parallel reads).
    """
    nvme = nvme_dir(cfg.cache_dir, run_name)
    if cfg.staging == 'copy':
        set_hdd_io_limit(cfg.hdd_io_limit)
        stage_copy(reproj_dir, nvme, cfg.hdd_io_limit)
    elif cfg.staging == 'reuse':
        if not os.path.isdir(nvme):
            raise RuntimeError(
                f"NVMe cache dir missing: {nvme}. staging='reuse' expects the "
                f"owning run to have created it.")
    else:
        raise ValueError(f"unknown staging strategy {cfg.staging!r}")
    set_hdd_io_limit(None)
    return nvme


def cleanup_nvme(cfg, nvme_reproj_dir):
    """Remove the NVMe scratch dir unless the config opts to keep it (or reuses
    a dir it does not own)."""
    if cfg.staging == 'reuse' or cfg.keep_nvme:
        if os.path.exists(nvme_reproj_dir):
            print(f"NVMe reproj cache preserved at {nvme_reproj_dir}.")
        return
    if os.path.exists(nvme_reproj_dir):
        shutil.rmtree(nvme_reproj_dir)
        print("NVMe reproj cache cleaned up.")


# --------------------------------------------------------------------------
# RSS guardrail — poll VmRSS, force a clean os._exit before the kernel OOM-kills
# mid-allocation with no traceback. Used by the tiled build (large per-tile peak).
# Verbatim behavior from run_cal_tiled_NEP.py.
# --------------------------------------------------------------------------
RSS_POLL_SEC = 15.0
RSS_ABORT_FRACTION = 0.85
_RSS_STATE = {'peak_kb': 0, 'aborting': False}


def _read_meminfo_kb(field='MemTotal'):
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith(field + ':'):
                return int(line.split()[1])
    return 0


def _read_self_rss_kb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def _rss_guardrail_loop(mem_total_kb, abort_threshold_kb):
    while True:
        try:
            rss_kb = _read_self_rss_kb()
            if rss_kb > _RSS_STATE['peak_kb']:
                _RSS_STATE['peak_kb'] = rss_kb
            rss_gb = rss_kb / 1024 / 1024
            peak_gb = _RSS_STATE['peak_kb'] / 1024 / 1024
            total_gb = mem_total_kb / 1024 / 1024
            pct = 100.0 * rss_kb / mem_total_kb if mem_total_kb else 0
            print(f'[RSS] {time.strftime("%H:%M:%S")}  RSS={rss_gb:6.1f} GB  '
                  f'peak={peak_gb:6.1f} GB  ({pct:5.1f}% of {total_gb:.0f} GB)',
                  file=sys.stderr, flush=True)
            if rss_kb >= abort_threshold_kb and not _RSS_STATE['aborting']:
                _RSS_STATE['aborting'] = True
                print(f'\n*** RSS GUARDRAIL TRIPPED ***\n'
                      f'    RSS={rss_gb:.1f} GB >= abort threshold '
                      f'{abort_threshold_kb/1024/1024:.0f} GB '
                      f'({100*RSS_ABORT_FRACTION:.0f}% of {total_gb:.0f} GB).\n'
                      f'    Forcing clean exit before kernel OOM-kill.\n',
                      file=sys.stderr, flush=True)
                os._exit(2)
        except Exception as e:
            print(f'[RSS] poll error: {e}', file=sys.stderr, flush=True)
        time.sleep(RSS_POLL_SEC)


def start_rss_guardrail():
    mem_total_kb = _read_meminfo_kb('MemTotal')
    abort_threshold_kb = int(mem_total_kb * RSS_ABORT_FRACTION)
    total_gb = mem_total_kb / 1024 / 1024
    abort_gb = abort_threshold_kb / 1024 / 1024
    print(f'[RSS] starting guardrail: poll={RSS_POLL_SEC:.0f}s  '
          f'abort_threshold={abort_gb:.0f} GB ({100*RSS_ABORT_FRACTION:.0f}% of {total_gb:.0f} GB)',
          flush=True)
    t = threading.Thread(target=_rss_guardrail_loop,
                         args=(mem_total_kb, abort_threshold_kb),
                         daemon=True, name='rss-guardrail')
    t.start()


def rss_checkpoint(label):
    rss_kb = _read_self_rss_kb()
    peak_kb = max(rss_kb, _RSS_STATE['peak_kb'])
    print(f'[RSS] checkpoint {label!r}: RSS={rss_kb/1024/1024:.1f} GB  '
          f'peak so far={peak_kb/1024/1024:.1f} GB', flush=True)
