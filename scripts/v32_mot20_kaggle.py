"""Prepare a private Kaggle MOT20 screen from a clean, committed local snapshot.

This command only writes a reviewable payload. Submission and authentication belong
to the caller; all dataset and TrackEval network access occurs inside Kaggle.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import io
import json
from pathlib import Path
import re
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HF_REPO = 'Lekim89/MOT20'
HF_REVISION = '5fcaa0ef66906b63195a44672ec260d033545ef4'
TRACKEVAL_REPO = 'https://github.com/JonathonLuiten/TrackEval.git'
TRACKEVAL_COMMIT = '12c8791b303e0a0b50f753af204249e622d0281a'
SEQUENCES = ('MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05')
SOURCE_SUFFIXES = {'.py', '.ipynb', '.json', '.toml', '.yaml', '.yml', '.ini', '.cfg',
                   '.sh', '.ps1', '.js', '.jsx', '.ts', '.tsx', '.c', '.cpp', '.h', '.rs', '.go'}


def safe_extract(bundle, target, expected_sha256):
    """Verify the entire ZIP and reject unsafe members before writing any file."""
    if hashlib.sha256(bundle).hexdigest() != expected_sha256:
        raise ValueError('source bundle SHA256 mismatch')
    target = Path(target)
    if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
        raise ValueError('unsafe extraction target symlink')
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError('source extraction target must be empty')
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        seen = set()
        for member in archive.infolist():
            # ZipInfo may normalize backslashes or truncate at NUL before
            # exposing filename. Validate the original wire name as well.
            name = member.orig_filename
            parts = name.rstrip('/').split('/')
            mode = member.external_attr >> 16
            if (not name or name != member.filename or '\x00' in name
                    or name.startswith('/') or '\\' in name or ':' in name
                    or any(part in ('', '.', '..') for part in parts)
                    or stat.S_ISLNK(mode) or name in seen):
                raise ValueError(f'unsafe ZIP path or symlink: {name!r}')
            seen.add(name)
            destination = target.joinpath(*parts)
            if any(parent.is_symlink() for parent in destination.parents):
                raise ValueError('unsafe ZIP destination symlink')
        target.mkdir(parents=True, exist_ok=True)
        for member in archive.infolist():
            destination = target.joinpath(*member.filename.rstrip('/').split('/'))
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
                mode = member.external_attr >> 16
                if mode & 0o111:
                    destination.chmod(destination.stat().st_mode | (mode & 0o111))


def source_snapshot():
    """Archive HEAD only, refusing changes that would make local code disappear."""
    def git(*args, text=True):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=text)
    dirty = git('diff', '--name-only', 'HEAD', '--').strip()
    if dirty:
        raise ValueError('dirty tracked source must be committed before preparing: ' + dirty.replace('\n', ', '))
    untracked = [name for name in git('ls-files', '--others', '--exclude-standard', '-z').split('\0')
                 if name and Path(name).suffix.lower() in SOURCE_SUFFIXES]
    if untracked:
        raise ValueError('untracked code/config would be omitted from HEAD: ' + ', '.join(untracked))
    commit = git('rev-parse', 'HEAD').strip()
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('source needs an immutable full Git commit SHA')
    bundle = git('archive', '--format=zip', 'HEAD', text=False)
    files = {}
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            path = Path(member.filename)
            if (path.name.lower() in {'pool.json', 'kaggle.json', 'credentials.json'}
                    or path.name.lower().startswith('.env')
                    or any(part.lower() in {'.aws', '.ssh'} for part in path.parts)
                    or path.suffix.lower() in {'.pth', '.pt', '.pkl', '.pem', '.key'}):
                raise ValueError(f'private credential/model file cannot enter source payload: {member.filename}')
            if path.suffix.lower() == '.py' or path.parts[0] == 'configs' or path.name == 'pyproject.toml':
                files[member.filename] = hashlib.sha256(archive.read(member)).hexdigest()
        config = json.loads(archive.read('configs/v32_roi.json'))
    if type(config.get('seed')) is not int:
        raise ValueError('committed V32 config must contain an integer seed')
    if config.get('qps') != [30, 35, 40, 45, 50] or config.get('codecs') != ['h264', 'h265']:
        raise ValueError('committed V32 config differs from the fixed codec/QP grid')
    origin = dict(schema='adaptive-vcm-source-bundle-1', origin_commit=commit,
                  archive_command='git archive --format=zip HEAD', bundle_sha256=hashlib.sha256(bundle).hexdigest(),
                  bundle_bytes=len(bundle), source_files_sha256=dict(sorted(files.items())),
                  file_hash_semantics='SHA256 of exact committed archive member bytes')
    return bundle, origin, config


RUNTIME_SOURCE = r'''
import configparser
from fractions import Fraction
import os
import random
import shutil
import sys
import tarfile
import traceback
from datetime import datetime, timezone

WORKING = Path('/kaggle/working')
REPO = WORKING / 'adaptive_vcm_snapshot'
DATA = WORKING / 'mot20'
TRACKEVAL = WORKING / 'TrackEval'
OUT = WORKING / 'outputs' / JOB['slug']
OUT.mkdir(parents=True, exist_ok=True)

def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')

save_json('job.json', JOB)
save_json('origin-manifest.json', ORIGIN)

def progress(message):
    line = f'{datetime.now(timezone.utc).isoformat()} {message}'
    print(line, flush=True)
    with (OUT / 'progress.log').open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')

def run(command, cwd=None):
    progress('run ' + ' '.join(map(str, command)))
    process = subprocess.Popen(command, cwd=cwd, env=os.environ.copy(), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    with (OUT / 'progress.log').open('a', encoding='utf-8') as handle:
        for line in process.stdout:
            print(line, end='', flush=True)
            handle.write(line)
            handle.flush()
    status = process.wait()
    if status:
        raise RuntimeError(f'command failed with exit {status}: {command!r}')

status = {'status': 'running', 'stage': 'source', 'origin_commit': JOB['origin_commit']}
save_json('status.json', status)
try:
    bundle = base64.b64decode(BUNDLE_BASE64, validate=True)
    safe_extract(bundle, REPO, ORIGIN['bundle_sha256'])
    for name, expected in ORIGIN['source_files_sha256'].items():
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'extracted source hash mismatch: {name}')
    progress('verified committed source bundle ' + JOB['origin_commit'])
    # protocol.provenance() requires a local Git HEAD. This is an ephemeral
    # runtime snapshot; the original repository commit remains in ORIGIN/JOB.
    run(['git', 'init', '--quiet', str(REPO)])
    run(['git', '-C', str(REPO), 'add', '--all'])
    run(['git', '-C', str(REPO), '-c', 'user.name=Kaggle source snapshot',
         '-c', 'user.email=kaggle-snapshot@example.invalid', 'commit', '--quiet', '--no-gpg-sign',
         '-m', 'Runtime snapshot of ' + JOB['origin_commit']])
    runtime_commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    save_json('runtime-provenance.json', {'origin_commit': JOB['origin_commit'],
              'runtime_commit': runtime_commit, 'bundle_sha256': ORIGIN['bundle_sha256'],
              'source_files_sha256': ORIGIN['source_files_sha256'],
              'runtime_commit_semantics': 'ephemeral extracted snapshot; use origin_commit for original source'})
    os.environ.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', PYTHONHASHSEED=str(JOB['seed']))
    status['stage'] = 'environment'
    save_json('status.json', status)
    assert shutil.which('ffmpeg'), 'FFmpeg is required'
    import torch
    assert torch.cuda.is_available(), 'Kaggle GPU is required'
    torch.set_num_threads(2)
    torch.manual_seed(JOB['seed'])
    torch.cuda.manual_seed_all(JOB['seed'])
    random.seed(JOB['seed'])
    progress('GPU ' + torch.cuda.get_device_name(0))
    run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
         'huggingface_hub', 'pycocotools', 'matplotlib', 'tabulate', 'packaging',
         'scipy', 'pillow', 'opencv-python-headless'])
    import numpy as np
    np.random.seed(JOB['seed'])
    (OUT / 'installed-packages.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    run(['git', 'clone', '--quiet', JOB['trackeval_repo'], str(TRACKEVAL)])
    run(['git', '-C', str(TRACKEVAL), 'checkout', '--quiet', JOB['trackeval_commit']])
    actual = subprocess.check_output(['git', '-C', str(TRACKEVAL), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != JOB['trackeval_commit']:
        raise ValueError('TrackEval commit verification failed')
    status['stage'] = 'download'
    save_json('status.json', status)
    from huggingface_hub import hf_hub_download
    def download(filename):
        path = hf_hub_download(repo_id=JOB['dataset']['repo_id'], filename=filename,
               repo_type='dataset', revision=JOB['dataset']['revision'], local_dir=str(DATA))
        progress('downloaded ' + filename)
        return Path(path)
    manifest = {}
    for sequence in JOB['sequences']:
        prefix = f'train/{sequence}'
        info = download(prefix + '/seqinfo.ini')
        config = configparser.ConfigParser()
        config.read(info)
        section = config['Sequence']
        length = section.getint('seqLength')
        fps = Fraction(section['frameRate'])
        if (section.get('name') != sequence or section.get('imDir') != 'img1'
                or section.get('imExt') != '.jpg' or length < JOB['max_frames'] or fps <= 0):
            raise ValueError('invalid sequence metadata or requested frame count: ' + sequence)
        manifest[sequence] = {'original_frames': length, 'selected_frames': JOB['max_frames'],
                              'first_frame': 1, 'last_frame': JOB['max_frames'], 'fps': str(fps),
                              'seqinfo_sha256': hashlib.sha256(info.read_bytes()).hexdigest()}
        # Ground truth is solely for the tracking runner's final evaluation.
        download(prefix + '/gt/gt.txt')
        for frame in range(1, JOB['max_frames'] + 1):
            download(prefix + f'/img1/{frame:06d}.jpg')
        save_json('download-manifest.json', manifest)
    status['stage'] = 'tracking'
    save_json('status.json', status)
    command = [sys.executable, '-u', '-m', 'adaptive_vcm.v32.tracking',
               '--root', str(DATA / 'train'), '--sequences', *JOB['sequences'],
               '--max-frames', str(JOB['max_frames']), '--size', str(JOB['size']),
               '--trackeval-root', str(TRACKEVAL), '--out', str(OUT / 'tracking')]
    run(command, cwd=str(REPO))
    status.update(status='completed', stage='finished')
    progress('fixed MOT20 prefix screen completed')
except BaseException as error:
    status.update(status='failed', error_type=type(error).__name__, error=str(error))
    (OUT / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    progress('failed: ' + str(error))
    raise
finally:
    save_json('status.json', status)
    archive_path = WORKING / (JOB['slug'] + '.tar.gz')
    with tarfile.open(archive_path, 'w:gz') as archive:
        archive.add(OUT, arcname=JOB['slug'])
    print('saved outputs: ' + str(archive_path), flush=True)
'''


def prepare(args):
    """Write the private notebook and immutable manifests; perform no networking."""
    if (not re.fullmatch(r'[a-zA-Z0-9_-]+', args.account)
            or not re.fullmatch(r'[a-z0-9][a-z0-9-]{5,70}', args.slug)):
        raise ValueError('invalid Kaggle account or slug')
    sequences = list(args.sequences)
    if not sequences or len(sequences) != len(set(sequences)) or any(seq not in SEQUENCES for seq in sequences):
        raise ValueError('select unique MOT20 train sequences 01, 02, 03, or 05')
    if type(args.max_frames) is not int or args.max_frames < 1 or type(args.size) is not int or args.size < 1:
        raise ValueError('max-frames and size must be positive integers')
    directory = Path(args.out)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError('payload output directory must be empty')
    bundle, origin, config = source_snapshot()
    files = []
    for sequence in sequences:
        prefix = f'train/{sequence}'
        files.extend([prefix + '/seqinfo.ini', prefix + '/gt/gt.txt'])
        files.extend(prefix + f'/img1/{frame:06d}.jpg' for frame in range(1, args.max_frames + 1))
    job = dict(schema='adaptive-vcm-v32-mot20-kaggle-1', account=args.account, slug=args.slug,
               handle=f'{args.account}/{args.slug}', task='ot', mode='fixed', sequences=sequences,
               max_frames=args.max_frames, size=args.size, seed=config['seed'],
               origin_commit=origin['origin_commit'], bundle_sha256=origin['bundle_sha256'],
               source_files_sha256=origin['source_files_sha256'],
               qps=[30, 35, 40, 45, 50], codecs=['h264', 'h265'],
               arms=['anchor', 'controls', 'simplification', 'retarget', 'v32'],
               dataset=dict(repo_id=HF_REPO, repo_type='dataset', revision=HF_REVISION, files=files),
               trackeval_repo=TRACKEVAL_REPO, trackeval_commit=TRACKEVAL_COMMIT,
               scope=dict(description=f'First {args.max_frames} frames per selected MOT20 train sequence; exploratory fixed-arm screen',
                          partial_sequence=True, full_sequence=False, official_rank=False,
                          learned_ot=False, quality_claim=False, ground_truth_use='final evaluation only'),
               environment=dict(gpu_required=True, ffmpeg_required=True, threads=2,
                                dependency_versions='Kaggle installed versions saved in installed-packages.txt'),
               private=True)
    constants = ('import base64, hashlib, io, json, stat, subprocess, zipfile\nfrom pathlib import Path\n'
                 f'BUNDLE_BASE64 = {base64.b64encode(bundle).decode("ascii")!r}\n'
                 f'ORIGIN = {origin!r}\nJOB = {job!r}\n')
    cells = []
    for name, source in [('source-bundle', constants),
                         ('run-mot20', inspect.getsource(safe_extract) + '\n' + RUNTIME_SOURCE)]:
        cells.append(dict(cell_type='code', id=name, execution_count=None, metadata={}, outputs=[],
                          source=source.splitlines(keepends=True)))
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    nbformat=4, nbformat_minor=5)
    metadata = dict(id=job['handle'], title=args.slug, code_file='notebook.ipynb', language='python',
                    kernel_type='notebook', is_private=True, enable_gpu=True, enable_internet=True,
                    dataset_sources=[], kernel_sources=[], competition_sources=[], model_sources=[])
    directory.mkdir(parents=True, exist_ok=True)
    for filename, content in [('notebook.ipynb', notebook), ('kernel-metadata.json', metadata),
                              ('job.json', job), ('origin-manifest.json', origin)]:
        (directory / filename).write_text(json.dumps(content, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare', help='prepare a private, reviewable Kaggle notebook')
    prepare_parser.add_argument('--account', required=True)
    prepare_parser.add_argument('--slug', required=True)
    prepare_parser.add_argument('--out', required=True, type=Path)
    prepare_parser.add_argument('--sequences', nargs='+', choices=SEQUENCES, default=['MOT20-01'])
    prepare_parser.add_argument('--max-frames', type=int, default=32)
    prepare_parser.add_argument('--size', type=int, default=640)
    args = parser.parse_args(argv)
    job = prepare(args)
    print(json.dumps({'prepared': str(args.out), 'handle': job['handle'], 'origin_commit': job['origin_commit'],
                      'bundle_sha256': job['bundle_sha256'], 'max_frames': job['max_frames']}))


if __name__ == '__main__':
    main()
