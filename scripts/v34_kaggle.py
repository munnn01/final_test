"""Immutable public V34 payloads, isolated credentials, durable submit receipts."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
from scripts.v32_mot20_kaggle import source_snapshot,safe_extract,HF_REPO,HF_REVISION,TRACKEVAL_REPO,TRACKEVAL_COMMIT
from scripts.v32vc_kaggle import mount_data
from scripts.v32rf_kaggle import serialized_source_bytes
from scripts.v33ap_kaggle import pack,unpack,stored_source,REQUIRED,OPTIONAL
from adaptive_vcm.v34.protocol import settings,ACTIONS

def job_spec(experiment,account,origin):
    if experiment not in ('ar_od','ar_ot') or not re.fullmatch('[a-zA-Z0-9_-]+',account) or not re.fullmatch('[0-9a-f]{40}',origin['origin_commit']):
        raise ValueError('invalid job/account/origin')
    slug='v34-'+experiment.replace('_','-')+'-'+origin['origin_commit'][:8]
    return dict(schema='v34-kaggle-1',experiment=experiment,slug=slug,handle=account+'/'+slug,private=False,
        origin_commit=origin['origin_commit'],protocol=settings(),actions=[list(a) for a in ACTIONS],
        datasets=['qktttttttttt/kineticscleaned']+(['awsaf49/coco-2017-dataset'] if experiment=='ar_od' else []),
        hf=dict(repo_id=HF_REPO,revision=HF_REVISION),sequences=['MOT20-02','MOT20-03','MOT20-01'],
        trackeval_repo=TRACKEVAL_REPO,trackeval_commit=TRACKEVAL_COMMIT,
        scope='actual-codec supervised adaptive filtering; three seeds; exploratory domain pilot; no confirmed gain; MOT20 AR has no GT')

RUNTIME=r'''
import os,random,shutil,sys,tarfile,traceback
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
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
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=expected: raise ValueError('source mismatch: '+name)
    (OUT/'source-origin.json').write_text(json.dumps(ORIGIN),encoding='utf-8')
    os.environ.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED='303401')
    import torch,torchvision
    if not torch.cuda.is_available() or not shutil.which('ffmpeg'): raise RuntimeError('GPU and FFmpeg are required')
    torch.set_num_threads(2); torch.manual_seed(303401); torch.cuda.manual_seed_all(303401); random.seed(303401)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    run([sys.executable,'-m','pip','install','--disable-pip-version-check','pycocotools','scipy','pillow','opencv-python-headless','huggingface_hub','supervision==0.27.0','tabulate'])
    save('environment.json',dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),python=sys.version,ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True)))
    (OUT/'installed-packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    sys.path.insert(0,str(REPO))
    from adaptive_vcm.v34.protocol import settings,ACTIONS
    if settings()!=JOB['protocol'] or [list(a) for a in ACTIONS]!=JOB['actions']: raise ValueError('runtime recipe drift')
    ar_root,_=mount_data(Path('/kaggle/input'),'ar')
    command=[sys.executable,'-u','-m','adaptive_vcm.v34.runner','--experiment',JOB['experiment'],'--ar-root',str(ar_root),
        '--origin',str(OUT/'source-origin.json'),'--out',str(OUT/'experiment')]
    if JOB['experiment']=='ar_od':
        od_root,ann=mount_data(Path('/kaggle/input'),'od')
        command+=['--od-root',str(od_root),'--annotations',str(ann)]
        save('data-mount.json',dict(ar=str(ar_root),od=str(od_root),annotations=str(ann)))
    else:
        status['stage']='download_MOT20'; save('status.json',status)
        from huggingface_hub import hf_hub_download
        DATA=WORKING/'mot20'; TRACKEVAL=WORKING/'TrackEval'
        def download(filename):
            path=hf_hub_download(repo_id=JOB['hf']['repo_id'],repo_type='dataset',revision=JOB['hf']['revision'],filename=filename,local_dir=str(DATA))
            return dict(file=filename,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        files=[]
        for sequence in JOB['sequences']:
            prefix='train/'+sequence
            files += [prefix+'/seqinfo.ini',prefix+'/gt/gt.txt']
            files += [prefix+f'/img1/{i:06d}.jpg' for i in range(1,JOB['protocol']['ot_frames']+1)]
        # Public, pinned dataset; bounded parallel download, never account bypass.
        with ThreadPoolExecutor(max_workers=4) as executor: manifest=list(executor.map(download,files))
        save('download-manifest.json',dict(revision=JOB['hf']['revision'],files=manifest))
        run(['git','clone','--quiet',JOB['trackeval_repo'],str(TRACKEVAL)])
        run(['git','-C',str(TRACKEVAL),'checkout','--quiet',JOB['trackeval_commit']])
        command+=['--mot-root',str(DATA/'train'),'--trackeval',str(TRACKEVAL)]
        save('data-mount.json',dict(ar=str(ar_root),mot=str(DATA/'train'),trackeval=str(TRACKEVAL)))
    status['stage']='measure_train_evaluate'; save('status.json',status)
    run(command,cwd=str(REPO)); status.update(status='complete',stage='complete')
except Exception:
    (OUT/'exception.txt').write_text(traceback.format_exc(),encoding='utf-8')
    status.update(status='error',stage=status['stage']); raise
finally:
    status['finished_utc']=datetime.now(timezone.utc).isoformat(); save('status.json',status)
    # Saves partial cache/logs even if a session ends before the full benchmark.
    with tarfile.open(WORKING/(JOB['slug']+'-outputs.tar.gz'),'w:gz') as archive: archive.add(OUT,arcname='outputs')
'''

def build_notebook(members,job):
    if set(members)!=REQUIRED or json.loads(members['job.json'])!=job: raise ValueError('source-only members required')
    constants='import base64,hashlib,io,json,lzma,stat,subprocess,zipfile\nfrom pathlib import Path\n'
    constants+=f'REQUIRED=set({tuple(sorted(REQUIRED))!r})\nOPTIONAL=set({tuple(sorted(OPTIONAL))!r})\n'+inspect.getsource(unpack)
    constants+=f'PACKED={pack(members)!r}\nMEMBERS=unpack(PACKED)\nORIGIN=json.loads(MEMBERS["origin.json"])\nJOB=json.loads(MEMBERS["job.json"])\n'
    sources=[constants,inspect.getsource(safe_extract)+'\n'+inspect.getsource(mount_data)+'\n'+RUNTIME]
    for s in sources: compile(s,'<notebook>','exec')
    nb=dict(cells=[dict(cell_type='code',id=f'v34-{i}',execution_count=None,metadata={},outputs=[],source=s.splitlines(keepends=True)) for i,s in enumerate(sources)],
        metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},nbformat=4,nbformat_minor=5)
    if serialized_source_bytes(nb)>=1_000_000: raise ValueError('Kaggle source limit')
    meta=dict(id=job['handle'],title=job['slug'],code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=False,
        enable_gpu=True,enable_internet=True,dataset_sources=job['datasets'],kernel_sources=[],competition_sources=[],model_sources=[])
    return nb,meta

def prepare(experiment,account,out):
    out=Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError('immutable empty payload required')
    bundle,origin,_=source_snapshot(); bundle,origin=stored_source(bundle,origin)
    job=job_spec(experiment,account,origin)
    members={'source.zip':bundle,'origin.json':json.dumps(origin).encode(),'job.json':json.dumps(job).encode()}
    nb,meta=build_notebook(members,job); out.mkdir(parents=True,exist_ok=True)
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('job.json',job),('origin-manifest.json',origin)]:
        (out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    receipt=dict(handle=job['handle'],origin_commit=origin['origin_commit'],public=True,serialized_source_bytes=serialized_source_bytes(nb),
        payload_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    (out/'local_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8'); return receipt

def validate_payload(directory):
    directory=Path(directory); receipt=json.loads((directory/'local_validation.json').read_text())
    files={'notebook.ipynb','kernel-metadata.json','job.json','origin-manifest.json'}
    if set(receipt['payload_sha256'])!=files or any(hashlib.sha256((directory/n).read_bytes()).hexdigest()!=receipt['payload_sha256'][n] for n in files):
        raise ValueError('payload bytes changed')
    nb=json.loads((directory/'notebook.ipynb').read_text()); job=json.loads((directory/'job.json').read_text()); origin=json.loads((directory/'origin-manifest.json').read_text())
    values=[ast.literal_eval(n.value) for n in ast.parse(''.join(nb['cells'][0]['source'])).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PACKED' for t in n.targets)]
    if len(values)!=1: raise ValueError('invalid notebook constants')
    members=unpack(values[0])
    if set(members)!=REQUIRED or hashlib.sha256(members['source.zip']).hexdigest()!=origin['bundle_sha256'] or json.loads(members['origin.json'])!=origin:
        raise ValueError('source origin mismatch')
    if job!=job_spec(job['experiment'],job['handle'].split('/')[0],origin): raise ValueError('job recipe drift')
    expected,meta=build_notebook(members,job)
    if nb!=expected or json.loads((directory/'kernel-metadata.json').read_text())!=meta: raise ValueError('notebook/metadata drift')
    return receipt

def submit(client,account,directory):
    directory=Path(directory); job=json.loads((directory/'job.json').read_text()); receipt=directory/'submission.json'
    if job['handle'].split('/')[0]!=account: raise ValueError('account mismatch')
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous.get('handle')!=job['handle']: raise ValueError('receipt mismatch')
        return previous
    validate_payload(directory)
    def save(value): receipt.write_text(json.dumps(value,indent=2),encoding='utf-8'); return value
    save(dict(status='ambiguous',handle=job['handle'],reason='intent recorded before push; no blind retry'))
    evidence=client.call(account,['kernels','push','-p',str(directory)],timeout=60)
    output=evidence['stdout']+'\n'+evidence['stderr']; version=re.search(r'Kernel version (\d+) successfully pushed',output,re.I)
    accepted=evidence['exit_code']==0 and version is not None and 'https://www.kaggle.com/code/'+job['handle'] in output
    value=dict(status='accepted' if accepted else 'ambiguous' if evidence['exit_code'] in ('TIMEOUT',0) else 'rejected',handle=job['handle'],version=int(version[1]) if accepted else None,evidence=evidence)
    if accepted: value['acceptance_check']=client.call(account,['kernels','list','--mine','--search',job['slug'],'--page-size','100','--format','json'],timeout=45)
    return save(value)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment',choices=['ar_od','ar_ot'],required=True); p.add_argument('--account',required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(prepare(a.experiment,a.account,a.out)))

if __name__=='__main__': main()
