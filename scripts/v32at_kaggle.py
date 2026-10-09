"""Prepare exactly two public, credential-free V32AT evaluation payloads."""
import argparse
import base64
import hashlib
import inspect
import io
import json
import lzma
from pathlib import Path
import re
import zipfile
from scripts.v32_mot20_kaggle import source_snapshot, safe_extract, HF_REPO, HF_REVISION, TRACKEVAL_REPO, TRACKEVAL_COMMIT
from scripts.v32vc_kaggle import mount_data
from scripts.v32rf_kaggle import serialized_source_bytes
from adaptive_vcm.v32at.protocol import settings, validate_task
from adaptive_vcm.v32at.frozen_rf import BUNDLE_HASH, CHECKPOINT_HASH, verify_runtime

REQUIRED={'source.zip','origin.json','job.json'}
OPTIONAL={'rf-source.zip','rf-origin.json','rf.pth'}


def pack(members):
    if not REQUIRED <= set(members) or set(members)-REQUIRED-OPTIONAL or bool(set(members)&OPTIONAL) and not OPTIONAL<=set(members):
        raise ValueError('unplanned packed payload members')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_STORED) as z:
        for name,raw in sorted(members.items()): z.writestr(name,raw)
    if len(stream.getvalue())>4_000_000: raise ValueError('packed source too large')
    return base64.b64encode(lzma.compress(stream.getvalue(),preset=6)).decode()


def unpack(payload):
    try:
        if len(payload)>2_000_000: raise ValueError('payload too large')
        compressed=base64.b64decode(payload,validate=True)
        decoder=lzma.LZMADecompressor(memlimit=64*1024*1024)
        raw=decoder.decompress(compressed,max_length=4_000_001)
        if len(raw)>4_000_000 or not decoder.eof or decoder.unused_data: raise ValueError('packed source exceeds bounds')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names=z.namelist()
            if len(names)!=len(set(names)) or not REQUIRED<=set(names) or set(names)-REQUIRED-OPTIONAL:
                raise ValueError('unplanned packed payload members')
            if bool(set(names)&OPTIONAL) and not OPTIONAL<=set(names): raise ValueError('incomplete RF bundle')
            if sum(i.file_size for i in z.infolist())>4_000_000: raise ValueError('expanded source too large')
            return {name:z.read(name) for name in names}
    except (lzma.LZMAError,zipfile.BadZipFile,TypeError,ValueError) as error:
        raise ValueError('invalid packed payload') from error


def job_spec(task,account,origin):
    validate_task(task)
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',account): raise ValueError('invalid account')
    slug=f'v32at-{task}-s303241'
    return dict(schema='v32at-kaggle-1', task=task, active_tasks=['ar','ot'], private=False,
        slug=slug,handle=account+'/'+slug, origin_commit=origin['origin_commit'],
        seed=settings()['seed'], protocol=settings(),training=False,
        dataset='qktttttttttt/kineticscleaned' if task=='ar' else dict(repo_id=HF_REPO,revision=HF_REVISION),
        splits=['dev','test'] if task=='ar' else None,count=32 if task=='ar' else None,
        sequences=['MOT20-01'] if task=='ot' else None,max_frames=128 if task=='ot' else None,
        size=640 if task=='ot' else 128,trackeval_repo=TRACKEVAL_REPO,trackeval_commit=TRACKEVAL_COMMIT)


RUNTIME=r'''
import configparser, os, random, shutil, sys, tarfile, traceback
from fractions import Fraction
from datetime import datetime, timezone
WORKING=Path('/kaggle/working'); REPO=WORKING/'v32at_source'; ORIGINAL=WORKING/'rf_original'
OUT=WORKING/'outputs'/JOB['slug']; OUT.mkdir(parents=True,exist_ok=True)
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')
def run(command,cwd=None):
    print('run',command,flush=True)
    process=subprocess.Popen(command,cwd=cwd,env=os.environ.copy(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with (OUT/'progress.log').open('a',encoding='utf-8') as f:
        for line in process.stdout: print(line,end='',flush=True); f.write(line); f.flush()
    if process.wait(): raise RuntimeError('command failed: '+repr(command))
def extract(raw,target,origin):
    safe_extract(raw,target,origin['bundle_sha256'])
    for name,h in origin['source_files_sha256'].items():
        if hashlib.sha256((target/name).read_bytes()).hexdigest()!=h: raise ValueError('source hash mismatch: '+name)
    run(['git','init','--quiet',str(target)])
    run(['git','-C',str(target),'add','--all'])
    run(['git','-C',str(target),'-c','user.name=Kaggle snapshot','-c','user.email=snapshot@example.invalid','commit','--quiet','--no-gpg-sign','-m','Snapshot '+origin['origin_commit']])
    (target/'origin-manifest.json').write_text(json.dumps(origin),encoding='utf-8')
save('job.json',JOB); save('origin-manifest.json',ORIGIN)
status=dict(status='running',stage='source',started_utc=datetime.now(timezone.utc).isoformat())
try:
    extract(MEMBERS['source.zip'],REPO,ORIGIN)
    os.environ.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED=str(JOB['seed']))
    import torch, torchvision
    if not torch.cuda.is_available() or not shutil.which('ffmpeg'): raise RuntimeError('GPU/FFmpeg required')
    torch.set_num_threads(2); torch.manual_seed(JOB['seed']); torch.cuda.manual_seed_all(JOB['seed']); random.seed(JOB['seed'])
    run([sys.executable,'-m','pip','install','--disable-pip-version-check','pycocotools','scipy','pillow','opencv-python-headless','huggingface_hub','matplotlib','tabulate','packaging'])
    save('environment.json',dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__),gpu=torch.cuda.get_device_name(0),python=sys.version,ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True)))
    (OUT/'installed-packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    if JOB['task']=='ar':
        original=json.loads(MEMBERS['rf-origin.json'])
        extract(MEMBERS['rf-source.zip'],ORIGINAL,original)
        checkpoint=WORKING/'rf_real_ar.pth'; checkpoint.write_bytes(MEMBERS['rf.pth'])
        sys.path.insert(0,str(REPO))
        from adaptive_vcm.v32at.frozen_rf import load_frozen_ar
        validated=load_frozen_ar(checkpoint,ORIGINAL,'cpu'); save('rf-validated.json',validated.manifest)
        del validated
        root,_=mount_data(Path('/kaggle/input'),'ar')
        for split in JOB['splits']:
            status['stage']='evaluate_'+split; save('status.json',status)
            run([sys.executable,'-u','-m','adaptive_vcm.v32at.ar','--root',str(root),'--real-checkpoint',str(checkpoint),'--rf-runtime',str(ORIGINAL),'--split',split,'--count',str(JOB['count']),'--bootstrap','0','--device','cuda','--out',str(OUT/('evaluation_'+split))],cwd=str(REPO))
    else:
        DATA=WORKING/'mot20'; TRACKEVAL=WORKING/'TrackEval'
        run(['git','clone','--quiet',JOB['trackeval_repo'],str(TRACKEVAL)])
        run(['git','-C',str(TRACKEVAL),'checkout','--quiet',JOB['trackeval_commit']])
        from huggingface_hub import hf_hub_download
        download_manifest={}
        for sequence in JOB['sequences']:
            def download(name):
                path=Path(hf_hub_download(repo_id=JOB['dataset']['repo_id'],repo_type='dataset',revision=JOB['dataset']['revision'],filename='train/'+sequence+'/'+name,local_dir=str(DATA)))
                download_manifest['train/'+sequence+'/'+name]=hashlib.sha256(path.read_bytes()).hexdigest()
                return path
            info=download('seqinfo.ini'); config=configparser.ConfigParser(); config.read(info)
            s=config['Sequence']
            if s.get('name')!=sequence or s.get('imDir')!='img1' or s.get('imExt')!='.jpg' or s.getint('seqLength')<JOB['max_frames'] or Fraction(s['frameRate'])<=0: raise ValueError('invalid MOT20 metadata')
            download('gt/gt.txt')
            for number in range(1,JOB['max_frames']+1): download(f'img1/{number:06d}.jpg')
        save('download-manifest.json',download_manifest)
        status['stage']='joint_tracking'; save('status.json',status)
        run([sys.executable,'-u','-m','adaptive_vcm.v32at.tracking','--root',str(DATA/'train'),'--sequences',*JOB['sequences'],'--max-frames',str(JOB['max_frames']),'--size',str(JOB['size']),'--trackeval-root',str(TRACKEVAL),'--device','cuda','--out',str(OUT/'tracking')],cwd=str(REPO))
    status.update(status='complete',stage='finished')
except BaseException as error:
    status.update(status='failed',error_type=type(error).__name__,error=str(error)); (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf-8'); raise
finally:
    if 'torch' in globals():
        weights=Path(torch.hub.get_dir())/'checkpoints'
        if weights.is_dir(): save('model-files-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in weights.iterdir() if p.is_file()})
    status['ended_utc']=datetime.now(timezone.utc).isoformat(); save('status.json',status)
    with tarfile.open(WORKING/(JOB['slug']+'.tar.gz'),'w:gz') as archive: archive.add(OUT,arcname=JOB['slug'])
'''


def build_notebook(members,job):
    constants='import base64,hashlib,io,json,lzma,stat,subprocess,zipfile\nfrom pathlib import Path\n'
    constants+=f'REQUIRED={REQUIRED!r}\nOPTIONAL={OPTIONAL!r}\n'+inspect.getsource(unpack)
    constants+=f'PACKED={pack(members)!r}\nMEMBERS=unpack(PACKED)\nORIGIN=json.loads(MEMBERS["origin.json"])\nJOB=json.loads(MEMBERS["job.json"])\n'
    sources=[constants,inspect.getsource(safe_extract)+'\n'+inspect.getsource(mount_data)+'\n'+RUNTIME]
    for source in sources: compile(source,'<notebook>','exec')
    notebook=dict(cells=[dict(cell_type='code',id=f'at-{i}',metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True)) for i,s in enumerate(sources)],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},nbformat=4,nbformat_minor=5)
    if serialized_source_bytes(notebook)>=1_000_000: raise ValueError('serialized Kaggle source exceeds limit')
    metadata=dict(id=job['handle'],title=job['slug'],code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=False,enable_gpu=True,enable_internet=True,dataset_sources=[job['dataset']] if job['task']=='ar' else [],kernel_sources=[],competition_sources=[],model_sources=[])
    return notebook,metadata


def prepare(args):
    validate_task(args.task)
    directory=Path(args.out)
    if directory.exists() and any(directory.iterdir()): raise FileExistsError('payload output must be empty')
    bundle,origin,cfg=source_snapshot(); job=job_spec(args.task,args.account,origin)
    members={'source.zip':bundle,'origin.json':json.dumps(origin).encode()}
    if args.task=='ar':
        runtime,rf_origin=verify_runtime(args.rf_runtime)
        raw=Path(args.checkpoint).read_bytes(); original=Path(args.rf_bundle).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=CHECKPOINT_HASH or hashlib.sha256(original).hexdigest()!=BUNDLE_HASH: raise ValueError('immutable RF artifacts differ')
        members.update({'rf-source.zip':original,'rf-origin.json':json.dumps(rf_origin).encode(),'rf.pth':raw})
        job['rf_checkpoint_sha256']=CHECKPOINT_HASH; job['rf_bundle_sha256']=BUNDLE_HASH
    members['job.json']=json.dumps(job).encode()
    notebook,metadata=build_notebook(members,job)
    directory.mkdir(parents=True,exist_ok=True)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',metadata),('job.json',job),('origin-manifest.json',origin)]:
        (directory/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    receipt=dict(handle=job['handle'],origin_commit=origin['origin_commit'],public=True,
        serialized_source_bytes=serialized_source_bytes(notebook),payload_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()})
    (directory/'local_validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['ar','ot'],required=True); p.add_argument('--account',required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path); p.add_argument('--rf-runtime',type=Path); p.add_argument('--rf-bundle',type=Path)
    print(json.dumps(prepare(p.parse_args())))
