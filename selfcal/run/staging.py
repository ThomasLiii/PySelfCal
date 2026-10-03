"""NVMe staging + the RSS guardrail — generic, instrument/mode-agnostic.

Calibration runs copy the HDD reproj files to a per-run NVMe scratch dir for
fast parallel reads, remap the cal/mosaic file lists onto it, and (optionally)
clean it up afterward. Tiled runs additionally use the RSS guardrail below
(their per-tile peak can approach machine memory). Both behaviors live here so
the engine stays readable.

A staging directory belongs to the pipeline only when it carries the marker file
:data:`STAGE_MARKER`, written when the directory is created (:func:`claim`). Frames
are copied into it atomically (a temporary name, renamed when complete), so an
interrupted copy never leaves a truncated frame behind. A directory that holds
files but no marker was made by something else (frames linked by hand, a test
fixture, another tool's copy): staging refuses to copy into it, and cleanup never
deletes it.
"""
import glob as glob_module
import json
import os
import shutil
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from tqdm import tqdm

from selfcal._state import set_hdd_io_limit


@contextmanager
def hdd_throttle(limit):
    """Throttle concurrent HDD reads to ``limit`` inside the block; unthrottled after."""
    set_hdd_io_limit(limit)
    try:
        yield
    finally:
        set_hdd_io_limit(None)


def nvme_dir(cache_dir, run_name):
    """Return the run's staging directory for its frames, ``<cache_dir>/reproj_nvme_<run_name>``.

    Only the path is formed here; :func:`prepare_nvme` creates and fills it.
    """
    return os.path.join(cache_dir, f'reproj_nvme_{run_name}')


STAGE_MARKER = '.selfcal-staging.json'


def is_staging_dir(path):
    """Whether ``path`` is a staging directory the pipeline made (it holds :data:`STAGE_MARKER`)."""
    return os.path.isfile(os.path.join(path, STAGE_MARKER))


def claim(stage_dir, source_dir):
    """Make ``stage_dir`` a staging directory for frames from ``source_dir``, or confirm it is one.

    A missing or empty directory is created and marked with :data:`STAGE_MARKER` (a small JSON
    file naming the source, host, process and time); a marked one is used as it is, so an
    interrupted staging resumes. Raises ``RuntimeError`` for a directory that holds files but no
    marker: it was not made by the pipeline, so copying frames into it could mix data sets and
    cleaning it up would delete it. Returns ``stage_dir``.
    """
    if is_staging_dir(stage_dir):
        return stage_dir
    if os.path.isdir(stage_dir) and os.listdir(stage_dir):
        raise RuntimeError(
            f"{stage_dir} holds files but was not made by selfcal staging (it has no {STAGE_MARKER}): "
            f"refusing to copy frames into it, and it would never be cleaned up. Read the frames there "
            f"in place (reproj_override = \"{stage_dir}\", or staging = \"reuse\"), move it away, or, if "
            f"it is a staged copy of {source_dir} made before staging directories were marked, mark it: "
            f"echo '{{}}' > {os.path.join(stage_dir, STAGE_MARKER)}")
    os.makedirs(stage_dir, exist_ok=True)
    marker = os.path.join(stage_dir, STAGE_MARKER)
    tmp = f'{marker}.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump({'source': os.path.abspath(source_dir), 'host': socket.gethostname(), 'pid': os.getpid(),
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S')}, f)
    os.replace(tmp, marker)
    return stage_dir


def _copy_one(src_path, dst_dir):
    """Copy one frame into ``dst_dir``, atomically. A copy already there is kept when its size
    matches the source (a staging that resumes); anything else is copied again."""
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    try:
        if os.path.getsize(dst_path) == os.path.getsize(src_path):
            return dst_path
    except FileNotFoundError:
        pass
    tmp = f'{dst_path}.part-{os.getpid()}-{threading.get_ident()}'
    try:
        shutil.copy2(src_path, tmp)
        os.replace(tmp, dst_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dst_path


def stage_copy(reproj_dir, nvme_reproj_dir, hdd_io_limit):
    """Copy every ``*.h5`` from the HDD reproj dir to NVMe (idempotent; see :func:`claim`)."""
    claim(nvme_reproj_dir, reproj_dir)
    hdd_files = sorted(glob_module.glob(os.path.join(reproj_dir, '*.h5')))
    print(f"Copying {len(hdd_files)} reproj files to NVMe ({nvme_reproj_dir})...")
    t_copy = time.time()
    with ThreadPoolExecutor(max_workers=hdd_io_limit or 20) as ex:
        for _ in tqdm(ex.map(lambda p: _copy_one(p, nvme_reproj_dir), hdd_files),
                      total=len(hdd_files), desc="HDD->NVMe", unit="file"):
            pass
    print(f"Reproj file copy complete in {time.time() - t_copy:.2f} seconds.")


def stage_files(files, nvme_reproj_dir, hdd_io_limit):
    """Copy a specific list of files to NVMe (tiled per-tile staging; see :func:`claim`)."""
    if files:
        claim(nvme_reproj_dir, os.path.dirname(files[0]))
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
    nvme = getattr(cfg, 'stage_dir', None) or nvme_dir(cfg.cache_dir, run_name)
    if cfg.staging == 'copy':
        with hdd_throttle(cfg.hdd_io_limit):
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
    a dir it does not own). A directory without :data:`STAGE_MARKER` was not made
    by the pipeline and is never removed."""
    if cfg.staging == 'reuse' or cfg.keep_nvme:
        if os.path.exists(nvme_reproj_dir):
            print(f"NVMe reproj cache preserved at {nvme_reproj_dir}.")
        return
    if not os.path.exists(nvme_reproj_dir):
        return
    if not is_staging_dir(nvme_reproj_dir):
        print(f"NVMe reproj cache {nvme_reproj_dir} kept: it was not made by selfcal staging "
              f"(no {STAGE_MARKER}).")
        return
    shutil.rmtree(nvme_reproj_dir)
    print("NVMe reproj cache cleaned up.")


# --------------------------------------------------------------------------
# RSS guardrail — polls the HARD RSS (RssAnon + RssShmem; see _read_self_rss_kb)
# NOTE: this thread prints to stderr every poll. Forking a process pool from
# a threaded parent can hand a child the stderr lock in a locked state, and
# multiprocessing children flush stderr at exit -> the child hangs and the
# parent joins it forever (seen in production 2026-09-09). All pipeline
# pools therefore use the forkserver context (selfcal.core.shmbuf).
# every RSS_POLL_SEC (15 s) and forces os._exit(2) once it reaches RSS_ABORT_FRACTION (85%) of MemTotal: a clean, logged exit
# instead of a kernel OOM-kill mid-allocation with no traceback. Used by the
# tiled build (large per-tile peak).
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
    """(hard_kb, vmrss_kb): hard = RssAnon + RssShmem — the part that cannot be
    reclaimed without swapping and is what an OOM-kill is about. VmRSS also
    counts RssFile: clean page-cache pages of the memmapped batch spill files
    (up to ~12 B per matrix entry during assembly), which the kernel drops on
    demand. Guarding on VmRSS would abort a large tile on reclaimable cache.
    """
    anon = shmem = vmrss = 0
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    vmrss = int(line.split()[1])
                elif line.startswith('RssAnon:'):
                    anon = int(line.split()[1])
                elif line.startswith('RssShmem:'):
                    shmem = int(line.split()[1])
    except Exception:
        pass
    hard = anon + shmem if anon else vmrss
    return hard, vmrss


def _rss_guardrail_loop(mem_total_kb, abort_threshold_kb):
    while True:
        try:
            rss_kb, vmrss_kb = _read_self_rss_kb()
            if rss_kb > _RSS_STATE['peak_kb']:
                _RSS_STATE['peak_kb'] = rss_kb
            rss_gb = rss_kb / 1024 / 1024
            peak_gb = _RSS_STATE['peak_kb'] / 1024 / 1024
            total_gb = mem_total_kb / 1024 / 1024
            pct = 100.0 * rss_kb / mem_total_kb if mem_total_kb else 0
            print(f'[RSS] {time.strftime("%H:%M:%S")}  hard={rss_gb:6.1f} GB  '
                  f'peak={peak_gb:6.1f} GB  ({pct:5.1f}% of {total_gb:.0f} GB)'
                  f'  VmRSS={vmrss_kb/1024/1024:6.1f} GB',
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
    """Start a daemon thread that ends the process before it exhausts the machine's memory.

    Every :data:`RSS_POLL_SEC` (15 s) the thread prints this process's hard RSS
    (``RssAnon`` + ``RssShmem`` from ``/proc/self/status``), its peak and ``VmRSS`` to stderr.
    Once the hard RSS reaches :data:`RSS_ABORT_FRACTION` (85%) of ``MemTotal`` it prints a
    message and exits at once with ``os._exit(2)``: a logged exit instead of a kernel OOM kill
    with no traceback. Worker processes are not counted, and each call starts another thread.
    The tiled ``cal`` task starts one unless ``[tiling].rss_guardrail = false``.
    """
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
    """Print one line with this process's hard RSS, ``VmRSS`` and peak, tagged ``label``.

    The peak is the larger of the current hard RSS and the highest value the guardrail thread
    has recorded (only the current value when no guardrail runs). The tiled ``cal`` task calls
    it at start-up when it starts the guardrail, and before and after each tile's
    ``setup_lsqr`` and ``apply_lsqr``.
    """
    rss_kb, vmrss_kb = _read_self_rss_kb()
    peak_kb = max(rss_kb, _RSS_STATE['peak_kb'])
    print(f'[RSS] checkpoint {label!r}: hard={rss_kb/1024/1024:.1f} GB  '
          f'VmRSS={vmrss_kb/1024/1024:.1f} GB  '
          f'peak so far={peak_kb/1024/1024:.1f} GB', flush=True)
