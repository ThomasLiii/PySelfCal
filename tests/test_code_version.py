"""The code version written into every run record and run log (selfcal/run/records.py, ``code_version``).

The commit is read from git only when the directory that holds the package is the top of a git checkout. A copy of
the package inside another repository's checkout (a ``git archive`` export kept in a worktree, for example) would
otherwise be given the enclosing checkout's HEAD; its commit is recorded as unknown instead.
"""
import subprocess

import pytest

from selfcal.run.records import code_version
from selfcal.run.runlog import _git_description


def _git(cwd, *args):
    return subprocess.run(['git', '-C', str(cwd), '-c', 'user.name=selfcal-test', '-c', 'user.email=test@example.invalid',
                           *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q')
    (repo / 'a.txt').write_text('a\n')
    _git(repo, 'add', 'a.txt')
    _git(repo, 'commit', '-q', '-m', 'first')
    return repo


def test_a_checkout_records_its_commit(checkout):
    sha = _git(checkout, 'rev-parse', 'HEAD')
    code = code_version(str(checkout))
    assert code['commit'] == sha and code['modified'] == [] and code['repository'] == str(checkout)
    assert _git_description(str(checkout)).startswith(sha[:7] + ' on ')
    (checkout / 'a.txt').write_text('b\n')
    assert code_version(str(checkout))['modified'] == ['a.txt']


def test_a_copy_inside_a_checkout_records_no_commit(checkout):
    copy = checkout / 'cache' / 'export' / 'tree'
    copy.mkdir(parents=True)
    assert code_version(str(copy)) == {'commit': None, 'branch': None, 'modified': [], 'repository': str(copy)}
    assert _git_description(str(copy)).startswith('unknown')


def test_outside_git_records_no_commit(tmp_path):
    assert code_version(str(tmp_path))['commit'] is None
