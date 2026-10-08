"""Staging (selfcal/run/staging.py): frames are copied atomically, and the pipeline only stages
into, or deletes, a directory it made (one holding the STAGE_MARKER file); a tiled run takes every
frame of its directory."""
import os

import pytest

from selfcal.run import staging
from selfcal.run.engine import tiling_frames
from selfcal.run.runspec import FrameSource


def _frames(d, names, size=64):
    os.makedirs(d, exist_ok=True)
    for n in names:
        with open(os.path.join(d, n), 'wb') as f:
            f.write(os.urandom(size))
    return [os.path.join(d, n) for n in names]


def test_claim_marks_a_new_or_empty_directory_and_refuses_a_foreign_one(tmp_path):
    new = tmp_path / 'stage_new'
    assert staging.claim(str(new), '/src') == str(new)
    assert staging.is_staging_dir(str(new))
    staging.claim(str(new), '/src')                          # a marked directory: reused
    empty = tmp_path / 'stage_empty'
    empty.mkdir()
    staging.claim(str(empty), '/src')
    assert staging.is_staging_dir(str(empty))
    foreign = tmp_path / 'fixture'
    _frames(str(foreign), ['exp_0000_det_00.h5'])
    with pytest.raises(RuntimeError, match='not made by selfcal staging'):
        staging.claim(str(foreign), '/src')
    assert not staging.is_staging_dir(str(foreign))


def test_copies_are_atomic_and_resume(tmp_path):
    src = _frames(str(tmp_path / 'hdd'), [f'exp_{i:04d}_det_00.h5' for i in range(4)], size=1000)
    dst = tmp_path / 'nvme'
    staging.claim(str(dst), str(tmp_path / 'hdd'))
    with open(dst / os.path.basename(src[0]), 'wb') as f:      # an interrupted copy of an old version
        f.write(b'x' * 10)
    staging.stage_files(src, str(dst), hdd_io_limit=2)
    for p in src:
        assert open(p, 'rb').read() == open(dst / os.path.basename(p), 'rb').read()
    assert sorted(os.listdir(dst)) == sorted([staging.STAGE_MARKER] + [os.path.basename(p) for p in src])
    mtime = os.path.getmtime(dst / os.path.basename(src[1]))
    staging.stage_files(src, str(dst), hdd_io_limit=2)       # complete copies are kept
    assert os.path.getmtime(dst / os.path.basename(src[1])) == mtime


def test_stage_copy_refuses_an_unmarked_directory(tmp_path):
    _frames(str(tmp_path / 'hdd'), ['exp_0000_det_00.h5'])
    fixture = tmp_path / 'reproj_nvme_run'
    os.makedirs(fixture)
    os.symlink(tmp_path / 'hdd' / 'exp_0000_det_00.h5', fixture / 'exp_0000_det_00.h5')
    with pytest.raises(RuntimeError, match='not made by selfcal staging'):
        staging.stage_copy(str(tmp_path / 'hdd'), str(fixture), 2)


@pytest.mark.parametrize('marked, staging_mode, keep, removed', [
    (True, 'copy', False, True),
    (False, 'copy', False, False),          # not made by the pipeline: never deleted
    (True, 'copy', True, False),
    (True, 'reuse', False, False),
])
def test_cleanup_only_removes_an_owned_directory(tmp_path, marked, staging_mode, keep, removed):
    d = tmp_path / 'reproj_nvme_run'
    _frames(str(d), ['exp_0000_det_00.h5'])
    if marked:
        open(d / staging.STAGE_MARKER, 'w').write('{}')
    staging.cleanup_nvme(FrameSource(stage=staging_mode, keep=keep), str(d))
    assert d.exists() != removed


def test_a_tiled_run_takes_every_frame_in_exposure_order(tmp_path):
    names = ['exp_0010_det_01.h5', 'exp_0002_det_00.h5', 'exp_0010_det_00.h5', 'exp_0002_det_15.h5']
    _frames(str(tmp_path), names + ['notes.txt'])
    got = [os.path.basename(p) for p in tiling_frames(str(tmp_path))]
    assert got == ['exp_0002_det_00.h5', 'exp_0002_det_15.h5', 'exp_0010_det_00.h5', 'exp_0010_det_01.h5']
