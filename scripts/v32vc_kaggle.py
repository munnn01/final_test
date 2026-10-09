"""Prepare private V32-VC TRAIN64 + paired DEV32/TEST32 notebooks; never submit here."""
import argparse
import base64
import hashlib
import inspect
import json
from pathlib import Path
from scripts.v32_mot20_kaggle import source_snapshot, safe_extract


def mount_data(input_root,task):
    input_root = Path(input_root)
    owner,slug = ('qktttttttttt','kineticscleaned') if task=='ar' else ('awsaf49','coco-2017-dataset')
    for root in (input_root/slug,input_root/'datasets'/owner/slug):
        if task=='ar' and root.is_dir(): return root,None
        if task=='od':
            for data in (root/'coco2017',root):
                ann = data/'annotations/instances_val2017.json'
                if ann.is_file() and (data/'val2017').is_dir(): return data/'val2017',ann
    raise FileNotFoundError('known dataset mount unavailable for '+task)


def job_spec(task,account,origin,raw):
    if task not in ('ar','od') or not account: raise ValueError('invalid task/account')
    slug = f'v32vc-{task}-train64-test32-s303221'
    return dict(schema='v32vc-kaggle-1',task=task,slug=slug,handle=account+'/'+slug,private=True,
        seed=303221,train_count=64,count=32,splits=['dev','test'],bootstrap_draws=0,
        qps=[30,35,40,45,50],codecs=['h264','h265'],origin_commit=origin['origin_commit'],
        dataset='qktttttttttt/kineticscleaned' if task=='ar' else 'awsaf49/coco-2017-dataset',
        baseline=dict(repository='https://github.com/munnn01/adaptive_preprocessing.git',
            commit='59ebb3dc31fc8edd6e1fbb80ed7de508f979edd3',
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_bytes=len(raw)),
        scope='Experimental supervised FIT48/CAL16, LAST plain/joint CNN, locked DEV32/TEST32, real paid packets; no confirmed target')


RUNTIME = r'''
import os, random, shutil, sys, tarfile, traceback
from datetime import datetime, timezone
WORKING = Path('/kaggle/working')
REPO = WORKING/'adaptive_vcm_snapshot'
ORIGINAL = WORKING/'original_v30b'
OUT = WORKING/'outputs'/JOB['slug']
OUT.mkdir(parents=True,exist_ok=True)

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')

def run(command,cwd=None):
    print('run',command,flush=True)
    process = subprocess.Popen(command,cwd=cwd,env=os.environ.copy(),stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,text=True,bufsize=1)
    with (OUT/'progress.log').open('a',encoding='utf-8') as handle:
        for line in process.stdout:
            print(line,end='',flush=True); handle.write(line); handle.flush()
    if process.wait(): raise RuntimeError('command failed: '+repr(command))

save('job.json',JOB); save('origin-manifest.json',ORIGIN)
status = dict(status='running',stage='source',started_utc=datetime.now(timezone.utc).isoformat())
save('status.json',status)
try:
    safe_extract(base64.b64decode(BUNDLE_BASE64,validate=True),REPO,ORIGIN['bundle_sha256'])
    for name,expected in ORIGIN['source_files_sha256'].items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('runtime source mismatch: '+name)
    run(['git','init','--quiet',str(REPO)])
    run(['git','-C',str(REPO),'add','--all'])
    run(['git','-C',str(REPO),'-c','user.name=Kaggle source snapshot','-c','user.email=snapshot@example.invalid',
         'commit','--quiet','--no-gpg-sign','-m','Frozen source '+JOB['origin_commit']])
    save('runtime-provenance.json',dict(origin_commit=JOB['origin_commit'],
        runtime_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
        bundle_sha256=ORIGIN['bundle_sha256'],source_files_sha256=ORIGIN['source_files_sha256']))
    os.environ.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                      PYTHONHASHSEED=str(JOB['seed']))
    import torch, torchvision
    if not torch.cuda.is_available() or not shutil.which('ffmpeg'): raise RuntimeError('GPU/FFmpeg required')
    torch.set_num_threads(2); torch.manual_seed(JOB['seed']); torch.cuda.manual_seed_all(JOB['seed']); random.seed(JOB['seed'])
    run([sys.executable,'-m','pip','install','--disable-pip-version-check','pycocotools','scipy','pillow','opencv-python-headless'])
    (OUT/'installed-packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    save('environment.json',dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),python=sys.version,ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True)))
    status['stage']='baseline'; save('status.json',status)
    run(['git','clone','--quiet',JOB['baseline']['repository'],str(ORIGINAL)])
    run(['git','-C',str(ORIGINAL),'checkout','--quiet',JOB['baseline']['commit']])
    if subprocess.check_output(['git','-C',str(ORIGINAL),'rev-parse','HEAD'],text=True).strip()!=JOB['baseline']['commit']:
        raise ValueError('original runtime commit mismatch')
    raw = base64.b64decode(CHECKPOINT_BASE64,validate=True)
    if hashlib.sha256(raw).hexdigest()!=JOB['baseline']['checkpoint_sha256']: raise ValueError('baseline checkpoint mismatch')
    checkpoint = WORKING/('v30b_'+JOB['task']+'.pth'); checkpoint.write_bytes(raw)
    sys.path.insert(0,str(REPO))
    from adaptive_vcm.v32.baseline import FrozenV30Baseline
    from adaptive_vcm.v32.protocol import load_config
    baseline = FrozenV30Baseline.load(checkpoint,ORIGINAL,JOB['task'],'cpu'); baseline.validate_config(load_config())
    save('baseline-validated.json',baseline.manifest)
    root,annotations = mount_data(Path('/kaggle/input'),JOB['task'])
    extra = ['--annotations',str(annotations)] if annotations else []
    save('data-mount.json',dict(dataset=JOB['dataset'],root=str(root),annotations=str(annotations) if annotations else None))
    status['stage']='train'; save('status.json',status)
    run([sys.executable,'-u','-m','adaptive_vcm.v32vc.train','--task',JOB['task'],'--root',str(root),*extra,
         '--device','cuda','--out',str(OUT/'training')],cwd=str(REPO))
    for split in ('dev','test'):
        status['stage']='evaluate_'+split; save('status.json',status)
        run([sys.executable,'-u','-m','adaptive_vcm.v32vc.evaluate','--task',JOB['task'],'--root',str(root),*extra,
             '--count',str(JOB['count']),'--split',split,'--bootstrap','0','--device','cuda',
             '--checkpoint',str(checkpoint),'--v30-runtime',str(ORIGINAL),
             '--plain-checkpoint',str(OUT/'training/plain_last.pth'),'--joint-checkpoint',str(OUT/'training/joint_last.pth'),
             '--out',str(OUT/('evaluation_'+split))],cwd=str(REPO))
    status.update(status='complete',stage='finished')
except BaseException as error:
    status.update(status='failed',error_type=type(error).__name__,error=str(error))
    (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
finally:
    if 'torch' in globals():
        weights = Path(torch.hub.get_dir())/'checkpoints'
        if weights.is_dir(): save('model-files-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in weights.iterdir() if p.is_file()})
    status['ended_utc']=datetime.now(timezone.utc).isoformat(); save('status.json',status)
    archive = WORKING/(JOB['slug']+'.tar.gz')
    with tarfile.open(archive,'w:gz') as handle: handle.add(OUT,arcname=JOB['slug'])
    print('saved evidence archive',archive,flush=True)
'''


def build_notebook(job,origin,bundle,raw):
    constants = ('import base64, hashlib, io, json, stat, subprocess, zipfile\nfrom pathlib import Path\n'
        f'BUNDLE_BASE64 = {base64.b64encode(bundle).decode()!r}\n'
        f'CHECKPOINT_BASE64 = {base64.b64encode(raw).decode()!r}\n'
        f'ORIGIN = {origin!r}\nJOB = {job!r}\n')
    sources = [constants,inspect.getsource(safe_extract)+'\n'+inspect.getsource(mount_data)+'\n'+RUNTIME]
    for source in sources: compile(source,'<notebook-cell>','exec')
    notebook = dict(cells=[dict(cell_type='code',id=f'cell-{i}',execution_count=None,metadata={},outputs=[],source=s.splitlines(keepends=True))
                           for i,s in enumerate(sources)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5)
    metadata = dict(id=job['handle'],title=job['slug'],code_file='notebook.ipynb',language='python',kernel_type='notebook',
                    is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[job['dataset']],
                    kernel_sources=[],competition_sources=[],model_sources=[])
    return notebook,metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['ar','od'],required=True); parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--v30-runtime',type=Path,required=True); parser.add_argument('--account',default='vtk269')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()): raise FileExistsError('payload directory must be empty')
    from adaptive_vcm.v32.baseline import FrozenV30Baseline
    from adaptive_vcm.v32.protocol import load_config
    baseline = FrozenV30Baseline.load(args.checkpoint,args.v30_runtime,args.task); baseline.validate_config(load_config())
    bundle,origin,_ = source_snapshot(); raw = args.checkpoint.read_bytes()
    job = job_spec(args.task,args.account,origin,raw); notebook,metadata = build_notebook(job,origin,bundle,raw)
    args.out.mkdir(parents=True,exist_ok=True)
    for name,value in [('job.json',job),('origin-manifest.json',origin),('kernel-metadata.json',metadata),('notebook.ipynb',notebook)]:
        (args.out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(handle=job['handle'],origin_commit=origin['origin_commit'],source_bytes=len(bundle),checkpoint_bytes=len(raw))))


if __name__=='__main__': main()
