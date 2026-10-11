"""Immutable public, evaluation-only V34-R1 independent validation payloads."""
import argparse
import ast
import base64
import hashlib
import inspect
import io
import json
import lzma
from pathlib import Path
import re
import zipfile

from scripts.v32_mot20_kaggle import source_snapshot, safe_extract
from scripts.v33ap_kaggle import stored_source
from scripts.v32rf_kaggle import serialized_source_bytes
from adaptive_vcm.v34.protocol import digest

REQUIRED = {'source.zip', 'origin.json', 'job.json', 'protocol.json', 'annotations.csv', 'exclusions.json',
            'controller_303401.pth', 'controller_303402.pth', 'controller_303403.pth'}
MAX_SOURCE = 4_000_000


def pack(members):
    if set(members) != REQUIRED:
        raise ValueError('exact independent payload members required')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, raw in sorted(members.items()):
            archive.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), raw)
    if len(stream.getvalue()) > MAX_SOURCE:
        raise ValueError('holdout payload transport budget exceeded')
    return base64.b64encode(lzma.compress(stream.getvalue(), preset=6)).decode()


def unpack(payload):
    try:
        if len(payload) > 2_000_000:
            raise ValueError('holdout payload too large')
        decoder = lzma.LZMADecompressor(memlimit=64*1024*1024)
        raw = decoder.decompress(base64.b64decode(payload, validate=True), max_length=MAX_SOURCE+1)
        if len(raw) > MAX_SOURCE or not decoder.eof or decoder.unused_data:
            raise ValueError('holdout payload exceeds bounds')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != REQUIRED or sum(f.file_size for f in archive.infolist()) > MAX_SOURCE:
                raise ValueError('invalid independent payload members')
            return {name: archive.read(name) for name in names}
    except (ValueError, TypeError, lzma.LZMAError, zipfile.BadZipFile) as e:
        raise ValueError('invalid holdout payload') from e


def job_spec(account, origin, protocol, shard):
    design = protocol['design']
    if (not re.fullmatch(r'[a-zA-Z0-9_-]+', account) or not re.fullmatch(r'[a-f0-9]{40}', origin['origin_commit'])
        or not 0 <= shard < design['shards']):
        raise ValueError('invalid holdout account/origin/shard')
    slug = f'v34-swin-independent-s{shard}-{origin["origin_commit"][:8]}'
    return dict(schema='v34-holdout-kaggle-1', handle=account+'/'+slug, slug=slug, private=False,
        origin_commit=origin['origin_commit'], protocol_sha256=digest(protocol), count=design['count'],
        shards=design['shards'], shard=shard, datasets=['jizeyong/kinetics400-rgb256-f10-val-data'],
        training_performed=False, scope='frozen R1 controller; official K400 validation; global analysis after all shards')


RUNTIME = r'''
import os,random,shutil,sys,tarfile,traceback
from datetime import datetime,timezone
WORKING=Path('/kaggle/working'); REPO=WORKING/'adaptive_vcm_snapshot'
OUT=WORKING/'outputs'/JOB['slug']; OUT.mkdir(parents=True,exist_ok=True)
ASSETS=WORKING/'holdout_assets'; ASSETS.mkdir(exist_ok=True)
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')
def run(command,cwd=None):
    print('run',command,flush=True)
    process=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with (OUT/'progress.log').open('a',encoding='utf-8') as f:
        for line in process.stdout: print(line,end='',flush=True); f.write(line); f.flush()
    if process.wait(): raise RuntimeError('holdout command failed')
status=dict(status='running',stage='source',started_utc=datetime.now(timezone.utc).isoformat()); save('status.json',status)
save('job.json',JOB);save('origin-manifest.json',ORIGIN)
try:
    if JOB['origin_commit']!=ORIGIN['origin_commit']: raise ValueError('source origin mismatch')
    for name,expected in JOB['assets_sha256'].items():
        if hashlib.sha256(MEMBERS[name]).hexdigest()!=expected: raise ValueError('asset mismatch: '+name)
    safe_extract(MEMBERS['source.zip'],REPO,ORIGIN['bundle_sha256'])
    for name,expected in ORIGIN['source_files_sha256'].items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=expected: raise ValueError('source mismatch: '+name)
    for name in REQUIRED-{'source.zip','origin.json','job.json'}: (ASSETS/name).write_bytes(MEMBERS[name])
    if hashlib.sha256((ASSETS/'protocol.json').read_bytes()).hexdigest()!=JOB['assets_sha256']['protocol.json']: raise ValueError('protocol bytes mismatch')
    os.environ.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED='20261011')
    import torch,torchvision
    if not torch.cuda.is_available() or not shutil.which('ffmpeg'): raise RuntimeError('GPU and FFmpeg required')
    torch.set_num_threads(2);torch.manual_seed(20261011);torch.cuda.manual_seed_all(20261011);random.seed(20261011)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    run([sys.executable,'-m','pip','install','--disable-pip-version-check','pycocotools','scipy','pillow','opencv-python-headless','supervision==0.27.0'])
    save('environment.json',dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),python=sys.version,ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True)))
    (OUT/'installed-packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    dataset='kinetics400-rgb256-f10-val-data'
    roots=[Path('/kaggle/input')/dataset,Path('/kaggle/input/datasets/jizeyong')/dataset]
    roots=list(dict.fromkeys(p.resolve() for p in roots if p.is_dir()))
    if len(roots)!=1: raise ValueError('independent validation dataset mount missing/ambiguous')
    save('data-mount.json',dict(root=str(roots[0]),provider='jizeyong/'+dataset,source_kind='predecoded JPEG frames'))
    status['stage']='frozen_independent_evaluation';save('status.json',status)
    run([sys.executable,'-u','-m','adaptive_vcm.v34.holdout','evaluate','--root',str(roots[0]),
        '--protocol',str(ASSETS/'protocol.json'),'--annotations',str(ASSETS/'annotations.csv'),
        '--exclusions',str(ASSETS/'exclusions.json'),'--checkpoints',str(ASSETS),'--shard',str(JOB['shard']),
        '--out',str(OUT/'experiment')],cwd=str(REPO))
    status.update(status='complete',stage='complete')
except Exception:
    (OUT/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8');status.update(status='error');raise
finally:
    status['finished_utc']=datetime.now(timezone.utc).isoformat();save('status.json',status)
    with tarfile.open(WORKING/(JOB['slug']+'-outputs.tar.gz'),'w:gz') as archive:archive.add(OUT,arcname='outputs')
'''


def build_notebook(members, job):
    if set(members) != REQUIRED or json.loads(members['job.json']) != job:
        raise ValueError('holdout job/member mismatch')
    prefix = 'import base64,hashlib,io,json,lzma,stat,subprocess,zipfile\nfrom pathlib import Path\n'
    prefix += f'REQUIRED=set({tuple(sorted(REQUIRED))!r})\nMAX_SOURCE={MAX_SOURCE}\n'+inspect.getsource(unpack)
    prefix += f'PACKED={pack(members)!r}\nMEMBERS=unpack(PACKED)\nORIGIN=json.loads(MEMBERS["origin.json"])\nJOB=json.loads(MEMBERS["job.json"])\n'
    cells = [prefix, inspect.getsource(safe_extract)+'\n'+RUNTIME]
    for source in cells: compile(source, '<holdout notebook>', 'exec')
    notebook = dict(cells=[dict(cell_type='code', id=f'v34-holdout-{i}', execution_count=None, metadata={}, outputs=[],
                               source=s.splitlines(keepends=True)) for i, s in enumerate(cells)],
        metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}}, nbformat=4, nbformat_minor=5)
    if serialized_source_bytes(notebook) >= 1_000_000:
        raise ValueError('Kaggle notebook source budget exceeded')
    metadata = dict(id=job['handle'], title=job['slug'], code_file='notebook.ipynb', language='python', kernel_type='notebook',
                    is_private=False, enable_gpu=True, enable_internet=True, dataset_sources=job['datasets'],
                    kernel_sources=[], competition_sources=[], model_sources=[])
    return notebook, metadata


def prepare(account, out, assets, shard):
    from adaptive_vcm.v34.checkpoint import load_checkpoint
    out, assets = Path(out), Path(assets)
    if out.exists() and any(out.iterdir()): raise FileExistsError('immutable empty payload required')
    protocol = json.loads((assets/'protocol.json').read_text())
    for seed in protocol['settings']['seeds']:
        model = load_checkpoint(assets/f'controller_{seed}.pth')
        if (model.checkpoint_sha256 != protocol['checkpoints'][str(seed)]['sha256']
            or model.metadata['settings'] != protocol['settings'] or model.metadata['source_commit'] != protocol['controller_source_commit']):
            raise ValueError('frozen checkpoint/recipe drift before upload')
    if (hashlib.sha256((assets/'annotations.csv').read_bytes()).hexdigest() != protocol['annotation_csv_sha256']
        or hashlib.sha256((assets/'exclusions.json').read_bytes()).hexdigest() != protocol['exclusions_sha256']):
        raise ValueError('pinned annotation/exclusion manifest drift before upload')
    bundle, origin, _ = source_snapshot(); bundle, origin = stored_source(bundle, origin)
    for name, expected in protocol['original_core_sha256'].items():
        if origin['source_files_sha256'].get(name) != expected: raise ValueError('original R1 core changed: '+name)
    job = job_spec(account, origin, protocol, shard)
    members = {'source.zip': bundle, 'origin.json': json.dumps(origin).encode()}
    members.update({n: (assets/n).read_bytes() for n in REQUIRED-{'source.zip','origin.json','job.json'}})
    job['assets_sha256'] = {n: hashlib.sha256(raw).hexdigest() for n, raw in members.items()}
    members['job.json'] = json.dumps(job).encode()
    notebook, metadata = build_notebook(members, job)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('notebook.ipynb', notebook), ('kernel-metadata.json', metadata), ('job.json', job), ('origin-manifest.json', origin)]:
        (out/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    receipt = dict(handle=job['handle'], origin_commit=origin['origin_commit'], public=True, training_performed=False,
        serialized_source_bytes=serialized_source_bytes(notebook),
        payload_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    (out/'local_validation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    validate_payload(out)
    return receipt


def validate_payload(directory):
    directory = Path(directory); receipt = json.loads((directory/'local_validation.json').read_text())
    expected_files = {'notebook.ipynb','kernel-metadata.json','job.json','origin-manifest.json'}
    if set(receipt['payload_sha256']) != expected_files or any(hashlib.sha256((directory/n).read_bytes()).hexdigest() != receipt['payload_sha256'][n] for n in expected_files):
        raise ValueError('immutable holdout payload bytes changed')
    notebook = json.loads((directory/'notebook.ipynb').read_text()); job = json.loads((directory/'job.json').read_text())
    origin = json.loads((directory/'origin-manifest.json').read_text())
    values = [ast.literal_eval(n.value) for n in ast.parse(''.join(notebook['cells'][0]['source'])).body
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PACKED' for t in n.targets)]
    if len(values) != 1: raise ValueError('invalid packed notebook constant')
    members = unpack(values[0]); protocol = json.loads(members['protocol.json'])
    expected_job = job_spec(job['handle'].split('/')[0], origin, protocol, job['shard'])
    expected_job['assets_sha256'] = {n: hashlib.sha256(raw).hexdigest() for n, raw in members.items() if n != 'job.json'}
    if (job != expected_job or json.loads(members['job.json']) != job or json.loads(members['origin.json']) != origin
        or hashlib.sha256(members['source.zip']).hexdigest() != origin['bundle_sha256']):
        raise ValueError('holdout asset/job/source drift')
    expected, metadata = build_notebook(members, job)
    if notebook != expected or json.loads((directory/'kernel-metadata.json').read_text()) != metadata:
        raise ValueError('holdout notebook/metadata drift')
    return receipt


def submit(client, account, directory):
    directory = Path(directory); job = json.loads((directory/'job.json').read_text()); receipt = directory/'submission.json'
    if job['handle'].split('/')[0] != account: raise ValueError('holdout account mismatch')
    if receipt.exists(): return json.loads(receipt.read_text())
    validate_payload(directory)
    def save(value): receipt.write_text(json.dumps(value, indent=2), encoding='utf-8'); return value
    save(dict(status='ambiguous', handle=job['handle'], reason='intent saved before push; never blindly resubmit'))
    evidence = client.call(account, ['kernels','push','-p',str(directory)], timeout=60)
    output = evidence['stdout']+'\n'+evidence['stderr']; version = re.search(r'Kernel version (\d+) successfully pushed', output, re.I)
    accepted = evidence['exit_code'] == 0 and version is not None and 'https://www.kaggle.com/code/'+job['handle'] in output
    return save(dict(status='accepted' if accepted else 'ambiguous' if evidence['exit_code'] in ('TIMEOUT',0) else 'rejected',
                     handle=job['handle'], version=int(version[1]) if accepted else None, evidence=evidence))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--account', required=True); p.add_argument('--out', required=True)
    p.add_argument('--assets', required=True); p.add_argument('--shard', type=int, required=True)
    args = p.parse_args(); print(json.dumps(prepare(args.account, args.out, args.assets, args.shard)))


if __name__ == '__main__': main()
