"""Build immutable public RF AR/OD and Swin receiver payloads, without submitting."""
import argparse
import base64
import hashlib
import inspect
import json
from pathlib import Path
from scripts.v32_mot20_kaggle import source_snapshot,safe_extract
from scripts.v32vc_kaggle import mount_data


def job_spec(task,account,origin,raw,warm_origin,warm_raw,mode='rf'):
    if (task not in ('ar','od') or not account or mode not in ('rf','post') or mode=='post' and task!='ar'
            or warm_origin.get('origin_commit')!='28cc97db6708c87346c025107a8a7ea485c21163'):
        raise ValueError('invalid RF payload task/mode/warm origin')
    slug=f'v32rf-{task}-'+('swin-post-' if mode=='post' else '')+'fit48-test32-s303231'
    return dict(schema='v32rf-kaggle-1',task=task,mode=mode,slug=slug,handle=account+'/'+slug,private=False,
        seed=303231,count=32,splits=['dev','test'],transfer_count=8,bootstrap_draws=0,
        qps=[30,35,40,45,50],codecs=['h264','h265'],origin_commit=origin['origin_commit'],
        dataset='qktttttttttt/kineticscleaned' if task=='ar' else 'awsaf49/coco-2017-dataset',
        baseline=dict(repository='https://github.com/munnn01/adaptive_preprocessing.git',commit='59ebb3dc31fc8edd6e1fbb80ed7de508f979edd3',
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_bytes=len(raw)),
        warm=dict(origin_commit=warm_origin['origin_commit'],bundle_sha256=warm_origin['bundle_sha256'],
            checkpoint_sha256=hashlib.sha256(warm_raw).hexdigest(),checkpoint_bytes=len(warm_raw)),
        scope='Paired real-forward/proxy-gradient FIT48/CAL12/PROXYVAL4 LAST; frozen Swin AR evaluator; separate receiver mode; no confirmed gain')


RUNTIME=r'''
import os,random,shutil,sys,tarfile,traceback
from datetime import datetime,timezone
WORKING=Path('/kaggle/working')
REPO=WORKING/'adaptive_vcm_snapshot'; WARM=WORKING/'v32vc_warm_snapshot'; ORIGINAL=WORKING/'original_v30b'
OUT=WORKING/'outputs'/JOB['slug']; OUT.mkdir(parents=True,exist_ok=True)
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True),encoding='utf-8')
def run(command,cwd=None):
    print('run',command,flush=True)
    process=subprocess.Popen(command,cwd=cwd,env=os.environ.copy(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with (OUT/'progress.log').open('a',encoding='utf-8') as f:
        for line in process.stdout: print(line,end='',flush=True); f.write(line); f.flush()
    if process.wait(): raise RuntimeError('command failed: '+repr(command))
def unpack(bundle,origin,target):
    safe_extract(base64.b64decode(bundle,validate=True),target,origin['bundle_sha256'])
    for name,expected in origin['source_files_sha256'].items():
        if hashlib.sha256((target/name).read_bytes()).hexdigest()!=expected: raise ValueError('runtime source mismatch: '+name)
    (target/'origin-manifest.json').write_text(json.dumps(origin,sort_keys=True),encoding='utf-8')
    run(['git','init','--quiet',str(target)]); run(['git','-C',str(target),'add','--all'])
    run(['git','-C',str(target),'-c','user.name=Kaggle source snapshot','-c','user.email=snapshot@example.invalid','commit','--quiet','--no-gpg-sign','-m','Frozen source '+origin['origin_commit']])
    return dict(origin_commit=origin['origin_commit'],runtime_commit=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip(),
        bundle_sha256=origin['bundle_sha256'],source_files_sha256=origin['source_files_sha256'])
save('job.json',JOB); save('origin-manifest.json',ORIGIN); save('warm-origin-manifest.json',WARM_ORIGIN)
status=dict(status='running',stage='source',started_utc=datetime.now(timezone.utc).isoformat()); save('status.json',status)
try:
    save('runtime-provenance.json',unpack(BUNDLE_BASE64,ORIGIN,REPO))
    save('warm-runtime-provenance.json',unpack(WARM_BUNDLE_BASE64,WARM_ORIGIN,WARM))
    os.environ.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONHASHSEED=str(JOB['seed']))
    import torch,torchvision
    if not torch.cuda.is_available() or not shutil.which('ffmpeg'): raise RuntimeError('GPU/FFmpeg required')
    torch.set_num_threads(2); torch.manual_seed(JOB['seed']); torch.cuda.manual_seed_all(JOB['seed']); random.seed(JOB['seed'])
    run([sys.executable,'-m','pip','install','--disable-pip-version-check','pycocotools','scipy','pillow','opencv-python-headless'])
    (OUT/'installed-packages.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    save('environment.json',dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        python=sys.version,ffmpeg=subprocess.check_output(['ffmpeg','-version'],text=True)))
    status['stage']='baseline'; save('status.json',status)
    run(['git','clone','--quiet',JOB['baseline']['repository'],str(ORIGINAL)])
    run(['git','-C',str(ORIGINAL),'checkout','--quiet',JOB['baseline']['commit']])
    if subprocess.check_output(['git','-C',str(ORIGINAL),'rev-parse','HEAD'],text=True).strip()!=JOB['baseline']['commit']: raise ValueError('original runtime mismatch')
    checkpoint=WORKING/('v30b_'+JOB['task']+'.pth'); warm_checkpoint=WORKING/('warm_'+JOB['task']+'.pth')
    for encoded,path,spec in ((CHECKPOINT_BASE64,checkpoint,JOB['baseline']),(WARM_CHECKPOINT_BASE64,warm_checkpoint,JOB['warm'])):
        raw=base64.b64decode(encoded,validate=True)
        if hashlib.sha256(raw).hexdigest()!=spec['checkpoint_sha256']: raise ValueError('checkpoint bytes mismatch')
        path.write_bytes(raw)
    sys.path.insert(0,str(REPO))
    from adaptive_vcm.v32.baseline import FrozenV30Baseline
    from adaptive_vcm.v32.protocol import load_config
    from adaptive_vcm.v32rf.warm import load_warm
    baseline=FrozenV30Baseline.load(checkpoint,ORIGINAL,JOB['task'],'cpu'); baseline.validate_config(load_config())
    warm=load_warm(warm_checkpoint,WARM,JOB['task'],'cpu')
    save('baseline-validated.json',baseline.manifest); save('warm-validated.json',warm.rf_warm_manifest)
    del baseline,warm
    root,annotations=mount_data(Path('/kaggle/input'),JOB['task']); extra=['--annotations',str(annotations)] if annotations else []
    save('data-mount.json',dict(dataset=JOB['dataset'],root=str(root),annotations=str(annotations) if annotations else None))
    common=['--task',JOB['task'],'--root',str(root),*extra,'--warm-checkpoint',str(warm_checkpoint),'--warm-runtime',str(WARM),'--device','cuda']
    status['stage']='paired_rf_train'; save('status.json',status)
    run([sys.executable,'-u','-m','adaptive_vcm.v32rf.train',*common,'--out',str(OUT/'training')],cwd=str(REPO))
    rf=['--proxy-checkpoint',str(OUT/'training/proxy_last.pth'),'--real-checkpoint',str(OUT/'training/real_last.pth')]
    if JOB['mode']=='rf':
        for split in ('dev','test'):
            status['stage']='evaluate_'+split; save('status.json',status)
            run([sys.executable,'-u','-m','adaptive_vcm.v32rf.evaluate',*common,*rf,'--checkpoint',str(checkpoint),'--v30-runtime',str(ORIGINAL),
                '--count','32','--split',split,'--bootstrap','0','--out',str(OUT/('evaluation_'+split))],cwd=str(REPO))
        status['stage']='fast_transfer_test8'; save('status.json',status)
        run([sys.executable,'-u','-m','adaptive_vcm.v32rf.evaluate',*common,*rf,'--count','8','--split','test','--bootstrap','0','--transfer','--out',str(OUT/'transfer_test8')],cwd=str(REPO))
    else:
        status['stage']='swin_restore_train'; save('status.json',status)
        run([sys.executable,'-u','-m','adaptive_vcm.v32rf.post_train',*common,'--real-checkpoint',str(OUT/'training/real_last.pth'),'--out',str(OUT/'post_training')],cwd=str(REPO))
        for split in ('dev','test'):
            status['stage']='swin_restore_'+split; save('status.json',status)
            run([sys.executable,'-u','-m','adaptive_vcm.v32rf.post_evaluate','--root',str(root),'--device','cuda','--count','32','--split',split,
                '--real-checkpoint',str(OUT/'training/real_last.pth'),'--post-checkpoint',str(OUT/'post_training/post_last.pth'),'--out',str(OUT/('post_evaluation_'+split))],cwd=str(REPO))
    status.update(status='complete',stage='finished')
except BaseException as error:
    status.update(status='failed',error_type=type(error).__name__,error=str(error)); (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf-8'); raise
finally:
    if 'torch' in globals():
        weights=Path(torch.hub.get_dir())/'checkpoints'
        if weights.is_dir(): save('model-files-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in weights.iterdir() if p.is_file()})
    status['ended_utc']=datetime.now(timezone.utc).isoformat(); save('status.json',status)
    with tarfile.open(WORKING/(JOB['slug']+'.tar.gz'),'w:gz') as handle: handle.add(OUT,arcname=JOB['slug'])
    print('saved evidence archive',JOB['slug'],flush=True)
'''


def build_notebook(job,origin,bundle,raw,warm_origin,warm_bundle,warm_raw):
    constants='import base64,hashlib,io,json,stat,subprocess,zipfile\nfrom pathlib import Path\n'
    for name,value in [('BUNDLE_BASE64',base64.b64encode(bundle).decode()),('CHECKPOINT_BASE64',base64.b64encode(raw).decode()),
            ('WARM_BUNDLE_BASE64',base64.b64encode(warm_bundle).decode()),('WARM_CHECKPOINT_BASE64',base64.b64encode(warm_raw).decode()),
            ('ORIGIN',origin),('WARM_ORIGIN',warm_origin),('JOB',job)]: constants+=f'{name} = {value!r}\n'
    sources=[constants,inspect.getsource(safe_extract)+'\n'+inspect.getsource(mount_data)+'\n'+RUNTIME]
    for source in sources: compile(source,'<notebook-cell>','exec')
    notebook=dict(cells=[dict(cell_type='code',id=f'cell-{i}',execution_count=None,metadata={},outputs=[],source=s.splitlines(keepends=True)) for i,s in enumerate(sources)],
        metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},nbformat=4,nbformat_minor=5)
    metadata=dict(id=job['handle'],title=job['slug'],code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=False,
        enable_gpu=True,enable_internet=True,dataset_sources=[job['dataset']],kernel_sources=[],competition_sources=[],model_sources=[])
    return notebook,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--task',choices=['ar','od'],required=True); p.add_argument('--mode',choices=['rf','post'],default='rf')
    for a in ('checkpoint','v30-runtime','warm-checkpoint','warm-runtime','warm-bundle','out'): p.add_argument('--'+a,type=Path,required=True)
    p.add_argument('--account',required=True); args=p.parse_args()
    if args.out.exists() and any(args.out.iterdir()): raise FileExistsError('payload directory must be empty')
    from adaptive_vcm.v32rf.warm import load_warm,verify_runtime
    from adaptive_vcm.v32.baseline import FrozenV30Baseline
    from adaptive_vcm.v32.protocol import load_config
    warm=load_warm(args.warm_checkpoint,args.warm_runtime,args.task)
    baseline=FrozenV30Baseline.load(args.checkpoint,args.v30_runtime,args.task); baseline.validate_config(load_config())
    _,warm_origin=verify_runtime(args.warm_runtime); warm_bundle=args.warm_bundle.read_bytes()
    if hashlib.sha256(warm_bundle).hexdigest()!=warm_origin['bundle_sha256']: raise ValueError('warm ZIP bytes mismatch')
    bundle,origin,_=source_snapshot(); raw=args.checkpoint.read_bytes(); warm_raw=args.warm_checkpoint.read_bytes()
    job=job_spec(args.task,args.account,origin,raw,warm_origin,warm_raw,args.mode)
    notebook,metadata=build_notebook(job,origin,bundle,raw,warm_origin,warm_bundle,warm_raw)
    args.out.mkdir(parents=True,exist_ok=True)
    for name,value in [('job.json',job),('origin-manifest.json',origin),('warm-origin-manifest.json',warm_origin),('kernel-metadata.json',metadata),('notebook.ipynb',notebook)]:
        (args.out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(handle=job['handle'],origin_commit=origin['origin_commit'],public=True,source_bytes=len(bundle),warm_bytes=len(warm_bundle),checkpoint_bytes=[len(raw),len(warm_raw)])))


if __name__=='__main__': main()
