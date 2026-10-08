"""Kaggle payload contracts, exercised without network or a GPU."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import types
import zipfile

import pytest


def module():
    assert importlib.util.find_spec('scripts.v32_mot20_kaggle') is not None, 'MOT20 payload builder is missing'
    return importlib.import_module('scripts.v32_mot20_kaggle')


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


@pytest.fixture
def source_repo(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    (root / 'adaptive_vcm' / 'v32').mkdir(parents=True)
    (root / 'configs').mkdir()
    (root / 'adaptive_vcm' / '__init__.py').write_text('')
    (root / 'adaptive_vcm' / 'v32' / '__init__.py').write_text('')
    # The fixture executes as a real subprocess and exposes the notebook's CLI boundary.
    (root / 'adaptive_vcm' / 'v32' / 'tracking.py').write_text(
        'import json, pathlib, sys\n'
        'out = pathlib.Path(sys.argv[sys.argv.index("--out") + 1])\n'
        'if out.exists() and any(out.iterdir()):\n'
        '    raise FileExistsError("tracking output must be empty")\n'
        'out.mkdir(parents=True, exist_ok=True)\n'
        '(out / "tracking-arguments.json").write_text(json.dumps(sys.argv[1:]))\n'
        'print("fixture tracking complete", flush=True)\n')
    (root / 'configs' / 'v32_roi.json').write_text(json.dumps(
        dict(seed=303201, qps=[30, 35, 40, 45, 50], codecs=['h264', 'h265'])))
    (root / '.gitignore').write_text('pool.json\n*.pth\n')
    (root / 'pool.json').write_text('{"secret": "fixture-do-not-embed"}')
    (root / 'checkpoint.pth').write_bytes(b'fixture-private-model')
    git(root, 'init', '--quiet')
    git(root, 'add', '.')
    git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '--quiet', '--no-gpg-sign', '-m', 'source fixture')
    monkeypatch.setattr(module(), 'ROOT', root)
    return root


def arguments(tmp_path, **updates):
    values = dict(account='fixtureowner', slug='v32-mot20-fixture', out=tmp_path / 'payload',
                  sequences=['MOT20-01'], max_frames=2, size=640)
    values.update(updates)
    return argparse.Namespace(**values)


def source_code(directory):
    notebook = json.loads((directory / 'notebook.ipynb').read_text())
    return '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')


def test_prepare_emits_private_payload_with_exact_committed_bundle(source_repo, tmp_path):
    a = arguments(tmp_path)
    job = module().prepare(a)
    metadata = json.loads((a.out / 'kernel-metadata.json').read_text())
    assert metadata['id'] == 'fixtureowner/v32-mot20-fixture'
    assert metadata['is_private'] is True and metadata['enable_gpu'] is True and metadata['enable_internet'] is True
    assert metadata['dataset_sources'] == metadata['kernel_sources'] == metadata['competition_sources'] == []
    assert metadata['code_file'] == 'notebook.ipynb'
    assert job == json.loads((a.out / 'job.json').read_text())
    assert job['origin_commit'] == git(source_repo, 'rev-parse', 'HEAD')
    assert job['qps'] == [30, 35, 40, 45, 50] and job['codecs'] == ['h264', 'h265']
    assert job['arms'] == ['anchor', 'controls', 'simplification', 'retarget', 'v32']
    assert job['mode'] == 'fixed' and job['seed'] == 303201
    assert job['dataset']['repo_id'] == 'Lekim89/MOT20'
    assert job['dataset']['revision'] == '5fcaa0ef66906b63195a44672ec260d033545ef4'
    assert job['trackeval_commit'] == '12c8791b303e0a0b50f753af204249e622d0281a'
    assert job['dataset']['files'] == ['train/MOT20-01/seqinfo.ini', 'train/MOT20-01/gt/gt.txt',
                                     'train/MOT20-01/img1/000001.jpg', 'train/MOT20-01/img1/000002.jpg']
    assert job['scope']['partial_sequence'] is True and job['scope']['official_rank'] is False
    assert job['scope']['learned_ot'] is False and job['scope']['quality_claim'] is False
    namespace = {}
    # Payload literals are independent from the runtime cell and can be inspected offline.
    notebook = json.loads((a.out / 'notebook.ipynb').read_text())
    exec(compile(''.join(notebook['cells'][0]['source']), 'payload constants', 'exec'), namespace)
    bundle = base64.b64decode(namespace['BUNDLE_BASE64'])
    origin = json.loads((a.out / 'origin-manifest.json').read_text())
    assert hashlib.sha256(bundle).hexdigest() == origin['bundle_sha256'] == job['bundle_sha256']
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert 'pool.json' not in archive.namelist() and 'checkpoint.pth' not in archive.namelist()
        expected = hashlib.sha256(archive.read('adaptive_vcm/v32/tracking.py')).hexdigest()
    assert origin['source_files_sha256']['adaptive_vcm/v32/tracking.py'] == expected
    assert origin['source_files_sha256']['configs/v32_roi.json'] == hashlib.sha256(
        (source_repo / 'configs' / 'v32_roi.json').read_bytes()).hexdigest()
    assert 'fixture-do-not-embed' not in source_code(a.out)
    compile(source_code(a.out), 'complete notebook', 'exec')


def test_prepare_refuses_overwrite_before_modifying_payload(source_repo, tmp_path):
    a = arguments(tmp_path)
    a.out.mkdir()
    marker = a.out / 'keep.txt'
    marker.write_text('keep me')
    with pytest.raises(FileExistsError, match='empty|exist'):
        module().prepare(a)
    assert marker.read_text() == 'keep me' and list(a.out.iterdir()) == [marker]


@pytest.mark.parametrize('kind', ['tracked', 'untracked'])
def test_source_changes_cannot_silently_disappear_from_bundle(source_repo, tmp_path, kind):
    name = 'tracking.py' if kind == 'tracked' else 'forgotten.py'
    (source_repo / 'adaptive_vcm' / 'v32' / name).write_text('# uncommitted source\n')
    with pytest.raises(ValueError, match='dirty|uncommitted|untracked'):
        module().prepare(arguments(tmp_path))
    assert not (tmp_path / 'payload').exists()


@pytest.mark.parametrize('updates', [dict(account='../owner'), dict(slug='../escape'),
    dict(sequences=['MOT20-04']), dict(sequences=['MOT20-01', 'MOT20-01']),
    dict(max_frames=0), dict(size=0)])
def test_invalid_selection_is_rejected_without_creating_payload(source_repo, tmp_path, updates):
    with pytest.raises(ValueError):
        module().prepare(arguments(tmp_path, **updates))
    assert not (tmp_path / 'payload').exists()


@pytest.mark.parametrize('member', ['../escape.py', '/absolute.py', 'C:/escape.py', 'a/../../escape.py',
                                   'a\\escape.py', 'a/./escape.py', 'other.py\x00evil'])
def test_zip_path_guards_reject_malicious_members_before_any_extraction(tmp_path, member):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('safe.py', 'safe')
        # ZipInfo normalizes Windows backslashes during construction. Preserve
        # the actual malicious wire name after constructing the fixture.
        info = zipfile.ZipInfo('placeholder')
        info.filename = info.orig_filename = member
        archive.writestr(info, 'malicious')
    bundle = stream.getvalue()
    target = tmp_path / 'extract'
    with pytest.raises(ValueError, match='unsafe|path'):
        module().safe_extract(bundle, target, hashlib.sha256(bundle).hexdigest())
    assert not target.exists() or not list(target.iterdir())


def test_zip_integrity_and_symlink_members_are_rejected(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        member = zipfile.ZipInfo('link')
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, '/outside')
    bundle = stream.getvalue()
    with pytest.raises(ValueError, match='SHA|hash|digest'):
        module().safe_extract(bundle, tmp_path / 'bad-hash', '0' * 64)
    with pytest.raises(ValueError, match='symlink|unsafe'):
        module().safe_extract(bundle, tmp_path / 'bad-link', hashlib.sha256(bundle).hexdigest())


def runtime_harness(tmp_path, monkeypatch, *, fail_pip=False, seq_length=429):
    working = tmp_path / 'working'
    downloaded = []
    hub = types.ModuleType('huggingface_hub')
    def download(*, repo_id, filename, repo_type, revision, local_dir):
        assert (repo_id, repo_type, revision) == ('Lekim89/MOT20', 'dataset', '5fcaa0ef66906b63195a44672ec260d033545ef4')
        downloaded.append(filename)
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith('seqinfo.ini'):
            path.write_text('[Sequence]\nname=MOT20-01\nimDir=img1\nimExt=.jpg\nframeRate=25\n'
                            f'seqLength={seq_length}\nimWidth=1920\nimHeight=1080\n')
        else:
            path.write_bytes(b'fixture input')
        return str(path)
    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    torch = types.ModuleType('torch')
    torch.set_num_threads = lambda _: None
    torch.manual_seed = lambda _: None
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda *_: 'fixture GPU',
                                     manual_seed_all=lambda _: None)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    real_popen = subprocess.Popen
    real_check_output = subprocess.check_output
    class ExternalProcess:
        def __init__(self, code=0):
            self.stdout = io.StringIO('fixture external setup\n')
            self.returncode = code
        def wait(self):
            return self.returncode
    def popen(command, **kwargs):
        if command[:3] == [sys.executable, '-m', 'pip']:
            return ExternalProcess(1 if fail_pip else 0)
        if command[:2] == ['git', 'clone']:
            Path(command[-1]).mkdir(parents=True)
            return ExternalProcess()
        if command[:3] == ['git', '-C', str(working / 'TrackEval')] and 'checkout' in command:
            return ExternalProcess()
        return real_popen(command, **kwargs)
    def check_output(command, **kwargs):
        if command == ['git', '-C', str(working / 'TrackEval'), 'rev-parse', 'HEAD']:
            return '12c8791b303e0a0b50f753af204249e622d0281a\n'
        if command[:3] == [sys.executable, '-m', 'pip'] and 'freeze' in command:
            return 'fixture-package==1.0\n'
        return real_check_output(command, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    monkeypatch.setattr(subprocess, 'check_output', check_output)
    monkeypatch.setattr('shutil.which', lambda _: '/fixture/ffmpeg')
    return working, downloaded


def execute_notebook(directory, working):
    code = source_code(directory).replace('/kaggle/working', working.as_posix())
    exec(compile(code, 'offline notebook', 'exec'), {'__name__': '__main__'})


def test_notebook_downloads_only_prefix_and_runs_tracking_with_saved_origin(source_repo, tmp_path, monkeypatch):
    a = arguments(tmp_path)
    job = module().prepare(a)
    working, downloaded = runtime_harness(tmp_path, monkeypatch)
    execute_notebook(a.out, working)
    assert downloaded == ['train/MOT20-01/seqinfo.ini', 'train/MOT20-01/gt/gt.txt',
                          'train/MOT20-01/img1/000001.jpg', 'train/MOT20-01/img1/000002.jpg']
    out = working / 'outputs' / a.slug
    tracking_out = out / 'tracking'
    invocation = json.loads((tracking_out / 'tracking-arguments.json').read_text())
    assert invocation == ['--root', str(working / 'mot20' / 'train'), '--sequences', 'MOT20-01',
                          '--max-frames', '2', '--size', '640', '--trackeval-root', str(working / 'TrackEval'),
                          '--out', str(tracking_out)]
    assert json.loads((out / 'job.json').read_text()) == job
    assert json.loads((out / 'origin-manifest.json').read_text())['origin_commit'] == job['origin_commit']
    runtime = json.loads((out / 'runtime-provenance.json').read_text())
    assert runtime['origin_commit'] == job['origin_commit'] and len(runtime['runtime_commit']) == 40
    assert json.loads((out / 'download-manifest.json').read_text())['MOT20-01']['fps'] == '25'
    assert json.loads((out / 'status.json').read_text())['status'] == 'completed'
    with tarfile.open(working / f'{a.slug}.tar.gz') as archive:
        assert f'{a.slug}/origin-manifest.json' in archive.getnames()
        assert f'{a.slug}/tracking/tracking-arguments.json' in archive.getnames()


@pytest.mark.parametrize('failure', ['setup', 'length'])
def test_notebook_packages_diagnostics_even_when_setup_or_metadata_fails(source_repo, tmp_path, monkeypatch, failure):
    a = arguments(tmp_path)
    module().prepare(a)
    working, downloaded = runtime_harness(tmp_path, monkeypatch, fail_pip=failure == 'setup',
                                         seq_length=1 if failure == 'length' else 429)
    with pytest.raises((RuntimeError, ValueError, subprocess.CalledProcessError)):
        execute_notebook(a.out, working)
    out = working / 'outputs' / a.slug
    assert json.loads((out / 'status.json').read_text())['status'] == 'failed'
    assert (out / 'error.txt').read_text() and (out / 'progress.log').is_file()
    assert (out / 'job.json').is_file() and (out / 'origin-manifest.json').is_file()
    assert (working / f'{a.slug}.tar.gz').is_file()
    assert not (out / 'tracking-arguments.json').exists()
    if failure == 'length':
        assert downloaded == ['train/MOT20-01/seqinfo.ini']
