"""Prepare immutable public joint/AR-only/OD-only experiments; submit explicitly."""
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
from scripts.v32_mot20_kaggle import source_snapshot,safe_extract
from scripts.v32vc_kaggle import mount_data
from scripts.v32rf_kaggle import serialized_source_bytes
from scripts.v32at_kaggle import pack as old_pack,unpack as old_unpack
from adaptive_vcm.v33ap.protocol import settings,arm_weights

REQUIRED={'source.zip','origin.json','job.json'}
OPTIONAL=set()


def pack(members):
    if set(members)!=REQUIRED: raise ValueError('only source/origin/job allowed')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,raw in sorted(members.items()): archive.writestr(zipfile.ZipInfo(name,(1980,1,1,0,0,0)),raw)
    if len(stream.getvalue())>4_000_000: raise ValueError('packed source too large')
    return base64.b64encode(lzma.compress(stream.getvalue(),preset=6)).decode()


def unpack(payload):
    members=old_unpack(payload)
    if set(members)!=REQUIRED: raise ValueError('only source/origin/job allowed')
    return members


def stored_source(bundle,origin):
    if hashlib.sha256(bundle).hexdigest()!=origin['bundle_sha256']: raise ValueError('original source bundle mismatch')
    stream=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(bundle)) as source,zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_STORED) as target:
        if sum(m.file_size for m in source.infolist())>3_900_000: raise ValueError('source exceeds transport budget')
        for m in source.infolist():
            member=zipfile.ZipInfo(m.filename,m.date_time); member.external_attr=m.external_attr
            member.create_system=m.create_system; member.compress_type=zipfile.ZIP_STORED
            target.writestr(member,source.read(m))
    raw=stream.getvalue()
    return raw,{**origin,'bundle_sha256':hashlib.sha256(raw).hexdigest(),'bundle_bytes':len(raw),
                'transport_repack':'ZIP_STORED committed member bytes; outer deterministic LZMA; no source filtering'}


def job_spec(arm,account,origin):
    arm_weights(arm)
    if not re.fullmatch('[a-zA-Z0-9_-]+',account) or not re.fullmatch('[0-9a-f]{40}',origin['origin_commit']): raise ValueError('invalid account/origin')
    slug='v33ap-'+arm.replace('_','-')+'-s303251'
    return dict(schema='v33ap-kaggle-1',arm=arm,slug=slug,handle=account+'/'+slug,private=False,
        seed=303251,origin_commit=origin['origin_commit'],protocol=settings(),active_tasks=['ar','od'],
        datasets=['qktttttttttt/kineticscleaned','awsaf49/coco-2017-dataset'],
        counts=dict(train=64,fit=48,cal=12,proxyval=4,dev=16,test=32,transfer=8),
        scope='shared adaptive preprocessing; standard actual codec forward/fixed proxy backward; exploratory historical partitions; LAST; no confirmed gain')


RUNTIME=r'''
import os,random,shutil,sys,tarfile,traceback
from datetime import datetime,timezone
WORKING=Path('/kaggle/working'); REPO=WORKING/'adaptive_vcm_snapshot'
OUT=WORKING/'outputs'/JOB['slug']; OUT.mkdir(parents=True,exist_ok=True)
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')
def run(command,cwd=None):
    print('run',command,flush=True)
    process=subprocess.Popen(command,cwd=cwd,env=os.environ.copy(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with (OUT/'progress.log').open('a',encoding='utf-8') as f:
        for line in process.stdout: print(line,end='',flush=True); f.write(line); f.flush()
    if process.wait(): raise RuntimeError('command failed: '+repr(command))
save('job.json',JOB); save('origin-manifest.json',ORIGIN)
status=dict(status='running',stage='source',started_utc=datetime.now(timezone.utc).isoformat()); save('status.json',status)
try:
    if JOB['origin_commit']!=ORIGIN['origin_commit']: raise ValueError('job/origin mismatch')
    safe_extract(MEMBERS['source.zip'],REPO,ORIGIN['bundle_sha256'])
    for name,expected in ORIGIN['source_files_sha256'].items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=expected: raise ValueError('runtime source mismatch: '+name)
    run(['git','init','--quiet',str(REPO)]); run(['git','-C',str(REPO),'add','--all'])
    run(['git','-C',str(REPO),'-c','user.name=Kaggle snapshot','-c','user.email=snapshot@example.invalid','commit','--quiet','--no-gpg-sign','-m','Snapshot '+JOB['origin_commit']])
    (REPO/'origin-manifest.json').write_text(json.dumps(ORIGIN),encoding='utf-8')
    os.environ.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED=str(JOB['seed']))
    import torch,torchvision
    if not torch.cuda.is_available() or not shutil.which('ffmpeg'): raise RuntimeError('GPU/FFmpeg required')
    torch.set_num_threads(2); torch.manual_seed(JOB['seed']); torch.cuda.manual_seed_all(JOB['seed']); random.seed(JOB['seed'])
    run([sys.executable,'-m','pip','install','--disable-pip-version-check','pycocotools','scipy','pillow','opencv-python-headless'])
    save('environment.json',dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),python=sys.version,ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True)))
    (OUT/'installed-packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    sys.path.insert(0,str(REPO))
    from adaptive_vcm.v33ap.protocol import settings
    if settings()!=JOB['protocol']: raise ValueError('runtime recipe mismatch')
    ar_root,_=mount_data(Path('/kaggle/input'),'ar'); od_root,annotations=mount_data(Path('/kaggle/input'),'od')
    save('data-mount.json',dict(ar=str(ar_root),od=str(od_root),annotations=str(annotations)))
    status['stage']='training'; save('status.json',status)
    run([sys.executable,'-u','-m','adaptive_vcm.v33ap.train','--arm',JOB['arm'],'--ar-root',str(ar_root),'--od-root',str(od_root),'--annotations',str(annotations),'--device','cuda','--out',str(OUT/'training')],cwd=str(REPO))
    checkpoint=OUT/'training/shared_last.pth'
    save('checkpoint-sha256.json',{'shared_last.pth':hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    for task,root in [('ar',ar_root),('od',od_root)]:
        extra=['--annotations',str(annotations)] if task=='od' else []
        for split,count,transfer in [('dev',16,False),('test',32,False),('test',8,True)]:
            name=task+'_'+split+('_fast' if transfer else '')
            status['stage']='evaluate_'+name; save('status.json',status)
            run([sys.executable,'-u','-m','adaptive_vcm.v33ap.evaluate','--task',task,'--root',str(root),*extra,'--split',split,'--count',str(count),'--checkpoint',str(checkpoint),'--device','cuda','--out',str(OUT/name),*(['--transfer'] if transfer else [])],cwd=str(REPO))
    status.update(status='complete',stage='finished')
except BaseException as error:
    status.update(status='failed',error_type=type(error).__name__,error=str(error)); (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf-8'); raise
finally:
    if 'torch' in globals():
        weights=Path(torch.hub.get_dir())/'checkpoints'
        if weights.is_dir(): save('model-files-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in weights.iterdir() if p.is_file()})
    status['ended_utc']=datetime.now(timezone.utc).isoformat(); save('status.json',status)
    save('artifact-sha256.json',{p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()})
    with tarfile.open(WORKING/(JOB['slug']+'.tar.gz'),'w:gz') as archive: archive.add(OUT,arcname=JOB['slug'])
'''


def build_notebook(members,job):
    if set(members)!=REQUIRED or json.loads(members['job.json'])!=job: raise ValueError('payload membership/job mismatch')
    constants='import base64,hashlib,io,json,lzma,stat,subprocess,zipfile\nfrom pathlib import Path\n'
    constants+=f'REQUIRED=set({tuple(sorted(REQUIRED))!r})\nOPTIONAL=set()\n'+inspect.getsource(old_unpack)
    constants+=f'PACKED={pack(members)!r}\nMEMBERS=unpack(PACKED)\nORIGIN=json.loads(MEMBERS["origin.json"])\nJOB=json.loads(MEMBERS["job.json"])\n'
    sources=[constants,inspect.getsource(safe_extract)+'\n'+inspect.getsource(mount_data)+'\n'+RUNTIME]
    for s in sources: compile(s,'<notebook>','exec')
    notebook=dict(cells=[dict(cell_type='code',id=f'ap-{i}',execution_count=None,metadata={},outputs=[],source=s.splitlines(keepends=True)) for i,s in enumerate(sources)],
        metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},nbformat=4,nbformat_minor=5)
    if serialized_source_bytes(notebook)>=1_000_000: raise ValueError('serialized Kaggle source too large')
    meta=dict(id=job['handle'],title=job['slug'],code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=False,
        enable_gpu=True,enable_internet=True,dataset_sources=job['datasets'],kernel_sources=[],competition_sources=[],model_sources=[])
    return notebook,meta


def prepare(arm,account,out):
    out=Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError('payload must be empty')
    bundle,origin,_=source_snapshot(); bundle,origin=stored_source(bundle,origin); job=job_spec(arm,account,origin)
    members={'source.zip':bundle,'origin.json':json.dumps(origin).encode(),'job.json':json.dumps(job).encode()}
    notebook,metadata=build_notebook(members,job); out.mkdir(parents=True,exist_ok=True)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',metadata),('job.json',job),('origin-manifest.json',origin)]:
        (out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    receipt=dict(handle=job['handle'],origin_commit=origin['origin_commit'],public=True,serialized_source_bytes=serialized_source_bytes(notebook),
        payload_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    (out/'local_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8'); return receipt


def validate_payload(directory):
    directory=Path(directory)
    receipt=json.loads((directory/'local_validation.json').read_text())
    files={'notebook.ipynb','kernel-metadata.json','job.json','origin-manifest.json'}
    hashes=receipt.get('payload_sha256',{})
    if set(hashes)!=files or any(hashlib.sha256((directory/name).read_bytes()).hexdigest()!=hashes[name] for name in files): raise ValueError('payload bytes differ from local validation')
    nb=json.loads((directory/'notebook.ipynb').read_text()); job=json.loads((directory/'job.json').read_text()); origin=json.loads((directory/'origin-manifest.json').read_text())
    tree=ast.parse(''.join(nb['cells'][0]['source']))
    values=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PACKED' for t in n.targets)]
    if len(values)!=1: raise ValueError('invalid payload constants')
    members=unpack(values[0])
    if hashlib.sha256(members['source.zip']).hexdigest()!=origin['bundle_sha256'] or json.loads(members['origin.json'])!=origin: raise ValueError('payload source mismatch')
    expected=job_spec(job['arm'],job['handle'].split('/')[0],origin)
    if job!=expected: raise ValueError('payload recipe drift')
    rebuilt,meta=build_notebook(members,job)
    if nb!=rebuilt or json.loads((directory/'kernel-metadata.json').read_text())!=meta: raise ValueError('payload notebook/metadata mismatch')
    return receipt


def submit(client,account,directory):
    directory=Path(directory); job=json.loads((directory/'job.json').read_text()); receipt=directory/'submission.json'
    if job['handle'].split('/')[0]!=account: raise ValueError('account/handle mismatch')
    if receipt.is_file():
        previous=json.loads(receipt.read_text())
        if previous.get('handle')!=job['handle']: raise ValueError('receipt belongs to another job')
        # All recorded intents prevent blind retries, including interrupted calls.
        return previous
    validate_payload(directory)
    def save(value): receipt.write_text(json.dumps(value,indent=2),encoding='utf-8'); return value
    save(dict(status='ambiguous',handle=job['handle'],reason='push intent persisted before network call'))
    evidence=client.call(account,['kernels','push','-p',str(directory)],timeout=60)
    output=evidence['stdout']+'\n'+evidence['stderr']; version=re.search(r'Kernel version (\d+) successfully pushed',output,re.I)
    accepted=evidence['exit_code']==0 and version is not None and ('https://www.kaggle.com/code/'+job['handle']) in output
    value=dict(status='accepted' if accepted else 'ambiguous' if evidence['exit_code']=='TIMEOUT' or evidence['exit_code']==0 else 'rejected',
               handle=job['handle'],version=int(version[1]) if accepted else None,evidence=evidence)
    if accepted:
        # One receipt check only; no kernel status, logs or completion monitoring.
        check=client.call(account,['kernels','list','--mine','--search',job['slug'],'--page-size','100','--format','json'])
        value['acceptance_check']=check
    return save(value)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--arm',choices=['joint','ar_only','od_only'],required=True); p.add_argument('--account',required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(prepare(a.arm,a.account,a.out)))


if __name__=='__main__': main()
