"""Private source-only payloads with pinned task-specific stage parents."""
import argparse
import ast
import hashlib
import inspect
import io
import json
from pathlib import Path,PurePosixPath
import re
import zipfile
from scripts.v32_mot20_kaggle import source_snapshot,safe_extract,HF_REPO,HF_REVISION,TRACKEVAL_REPO,TRACKEVAL_COMMIT
from scripts.v33ap_kaggle import stored_source,pack,unpack,REQUIRED,OPTIONAL
from scripts.v32rf_kaggle import serialized_source_bytes
from scripts.v32vc_kaggle import mount_data
from scripts.v34_kaggle import RUNTIME as NATIVE_RUNTIME
from adaptive_vcm.v34r3.protocol import ROOT,SCHEMA,settings,baseline_manifest,digest,ACTIONS

ACCOUNTS={'thuha1205','qktttttttttt','huolgggnuyen','baoancut','trmnguyn111'}
STAGES={'measure-identity','train-warm','measure-rollout','train-final','evaluate'}


def _parent(value,task,account,stage):
    value=json.loads(json.dumps(value));ref=value.get('ref','')
    if value.get('kind')!='dataset' or value.get('version')!=1 or not re.fullmatch(re.escape(account)+r'/[a-zA-Z0-9_-]+',ref):
        raise ValueError('private execution-account-owned version-1 parent dataset required')
    if value.get('task')!=task or value.get('schema')!=(SCHEMA if task!='ar' else 'v34-native-ar-reference-1'):
        raise ValueError('foreign task or revision parent')
    allowed={'train-warm':{'measure-identity'},'measure-rollout':{'train-warm'},
        'train-final':{'measure-identity'} if task=='od' else {'measure-rollout'},
        'evaluate':{'train-final','evaluate'} if task!='ar' else {'frozen-reference'},'measure-identity':{'measure-identity'}}
    if value.get('stage') not in allowed[stage]:raise ValueError('wrong parent stage')
    files=value.get('files',{})
    if not files or len(files)>20000:raise ValueError('content-pinned parent files required')
    for name,sha in files.items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or not re.fullmatch('[0-9a-f]{64}',sha):
            raise ValueError('invalid parent path/hash')
    required={'native_manifest.json'} if task=='ar' else {'stage.json'}
    if stage!='measure-identity' and task!='ar':required.add('stage_complete.json')
    if not required<=files.keys():raise ValueError('parent receipt missing')
    return value


def job_spec(task,account,origin,*,stage,parent=None,scope=None,ar_reference=None,cohort_manifest=None):
    if task not in ('ar','od','ot') or account not in ACCOUNTS or stage not in STAGES or task=='ar' and stage!='evaluate':
        raise ValueError('unauthorized task, account or stage')
    if not re.fullmatch('[0-9a-f]{40}',origin.get('origin_commit','')):raise ValueError('immutable origin required')
    cfg=settings()
    if task=='od' and stage in ('train-warm','measure-rollout'):raise ValueError('OD has no rollout training')
    if task=='ar':parent=ar_reference
    if parent is not None:parent=_parent(parent,task,account,stage)
    if stage!='measure-identity' and parent is None:raise ValueError('pinned parent required')
    if task=='ar' and parent is None:raise ValueError('frozen AR reference required')
    if scope is not None:
        if set(scope)-{'codecs','splits'} or not scope.get('codecs') or not set(scope['codecs'])<=set(cfg['codecs']):raise ValueError('invalid scope')
        if stage=='evaluate':
            if not scope.get('splits') or not set(scope['splits'])<={'dev','test'}:raise ValueError('heldout scope required')
            if 'dev' in scope['splits'] and set(scope['codecs'])!=set(cfg['codecs']):raise ValueError('one task policy needs both DEV codecs')
        elif 'splits' in scope:raise ValueError('measurement split scope is fixed')
    if cohort_manifest is not None:
        if cohort_manifest.get('sha256')!=digest({k:v for k,v in cohort_manifest.items() if k!='sha256'}):raise ValueError('cohort manifest digest missing')
        if task=='ot' and any(not s.startswith('MOT20-') for names in cohort_manifest.get('ot_sequences',cfg['ot_sequences']).values() for s in names):
            raise ValueError('Kaggle downloader is pinned to MOT20; use local explicit mounts for other benchmarks')
    label=stage+('-'+digest(scope)[:6] if scope else '')+('-r'+digest(parent)[:6] if parent else '')
    slug=f'v34-r3-{task}-{label}-{origin["origin_commit"][:8]}'
    sequences=(cohort_manifest or {}).get('ot_sequences',cfg['ot_sequences'])
    return dict(schema='v34-r3-kaggle-1',experiment=task,slug=slug,handle=account+'/'+slug,private=True,stage=stage,parent=parent,
        ar_reference=parent if task=='ar' else None,execution_scope=scope,cohort_manifest=cohort_manifest,
        origin_commit=origin['origin_commit'],protocol=cfg,actions=[list(a) for a in ACTIONS],config_path='configs/v34_r3_qp4.json',
        datasets=['awsaf49/coco-2017-dataset/2'] if task=='od' else ['qktttttttttt/kineticscleaned/1'] if task=='ar' else [],
        hf=dict(repo_id=HF_REPO,revision=HF_REVISION),sequences=[s for split in ('train','cal','dev','test') for s in sequences[split]],
        trackeval_repo=TRACKEVAL_REPO,trackeval_commit=TRACKEVAL_COMMIT,fresh_training=stage in ('train-warm','train-final'),
        scope='separate task under shared architecture; frozen AR; historical cohorts exploratory; no confirmed gain')


def find_parent(root,files,archive_sha256=None,workdir=None):
    matches=[];root=Path(root).resolve()
    anchor='native_manifest.json' if 'native_manifest.json' in files else 'stage.json'
    for path in root.rglob(anchor):
        directory=path.parent.resolve()
        if all((directory/name).resolve().is_relative_to(root) and (directory/name).is_file()
            and hashlib.sha256((directory/name).read_bytes()).hexdigest()==sha for name,sha in files.items()):matches.append(directory)
    if not matches and archive_sha256:
        archives=[p for p in root.rglob('state.zip') if hashlib.sha256(p.read_bytes()).hexdigest()==archive_sha256]
        if len(archives)!=1:raise ValueError('parent archive missing or ambiguous')
        raw=archives[0].read_bytes()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if len(z.infolist())>20000 or sum(p.file_size for p in z.infolist())>8*1024**3:raise ValueError('parent archive exceeds bounded budget')
        target=Path(workdir) if workdir is not None else Path('/kaggle/working')/('r3_parent_'+archive_sha256[:16])
        if not target.exists():safe_extract(raw,target,archive_sha256)
        if all((target/name).resolve().is_relative_to(target.resolve()) and (target/name).is_file()
            and hashlib.sha256((target/name).read_bytes()).hexdigest()==sha for name,sha in files.items()):matches.append(target)
    if len(matches)!=1:raise ValueError('parent files missing, changed or ambiguous')
    return matches[0]


RUNTIME=NATIVE_RUNTIME.replace('from adaptive_vcm.v34.protocol import settings,ACTIONS','from adaptive_vcm.v34r3.protocol import settings,ACTIONS')
RUNTIME=RUNTIME.replace('adaptive_vcm.v34.runner','adaptive_vcm.v34r3.runner')
RUNTIME=RUNTIME.replace("    if config: command+=['--config',str(config)]", "    if config: command+=['--config',str(config)]\n"
    "    command+=['--stage',JOB['stage']]\n"
    "    if JOB['parent'] is not None:\n"
    "        parent=find_parent(Path('/kaggle/input'),JOB['parent']['files'],JOB['parent'].get('archive_sha256'))\n"
    "        command+=['--parent',str(parent/'native_manifest.json' if JOB['experiment']=='ar' else parent)]\n"
    "    if JOB['execution_scope'] is not None:\n"
    "        save('scope.json',JOB['execution_scope']);command+=['--scope',str(OUT/'scope.json')]\n"
    "    if JOB['cohort_manifest'] is not None:\n"
    "        save('cohort_manifest.json',JOB['cohort_manifest']);command+=['--cohort-manifest',str(OUT/'cohort_manifest.json')]\n")


def build_notebook(members,job):
    if set(members)!=REQUIRED or json.loads(members['job.json'])!=job:raise ValueError('source-only payload required')
    constants='import base64,hashlib,io,json,lzma,stat,subprocess,zipfile\nfrom pathlib import Path\n'
    constants+=f'REQUIRED=set({tuple(sorted(REQUIRED))!r})\nOPTIONAL=set({tuple(sorted(OPTIONAL))!r})\n'+inspect.getsource(unpack)
    constants+=f'PACKED={pack(members)!r}\nMEMBERS=unpack(PACKED)\nORIGIN=json.loads(MEMBERS["origin.json"])\nJOB=json.loads(MEMBERS["job.json"])\n'
    sources=[constants,inspect.getsource(safe_extract)+'\n'+inspect.getsource(mount_data)+'\n'+inspect.getsource(find_parent)+'\n'+RUNTIME]
    for s in sources:compile(s,'<R3 private notebook>','exec')
    notebook=dict(cells=[dict(cell_type='code',id=f'v34-r3-{i}',execution_count=None,metadata={},outputs=[],source=s.splitlines(keepends=True)) for i,s in enumerate(sources)],
        metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},nbformat=4,nbformat_minor=5)
    metadata=dict(id=job['handle'],title=job['slug'],code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True,enable_internet=True,dataset_sources=job['datasets']+([job['parent']['ref']+'/1'] if job['parent'] else []),
        kernel_sources=[],competition_sources=[],model_sources=[])
    if serialized_source_bytes(notebook)>=1_000_000:raise ValueError('Kaggle source limit')
    return notebook,metadata


def prepare(task,account,out,*,stage,parent=None,scope=None,ar_reference=None,cohort_manifest=None):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('immutable payload directory required')
    bundle,origin,_=source_snapshot();bundle,origin=stored_source(bundle,origin);origin['baseline']=baseline_manifest(ROOT)
    job=job_spec(task,account,origin,stage=stage,parent=parent,scope=scope,ar_reference=ar_reference,cohort_manifest=cohort_manifest)
    members={'source.zip':bundle,'origin.json':json.dumps(origin).encode(),'job.json':json.dumps(job).encode()}
    notebook,metadata=build_notebook(members,job);out.mkdir(parents=True,exist_ok=True)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',metadata),('job.json',job),('origin-manifest.json',origin)]:
        (out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    receipt=dict(handle=job['handle'],submitted=False,private=True,origin_commit=origin['origin_commit'],
        payload_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    (out/'local_validation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');return receipt


def validate_payload(directory):
    directory=Path(directory);read=lambda n:json.loads((directory/n).read_text(encoding='utf-8'));receipt=read('local_validation.json')
    names={'notebook.ipynb','kernel-metadata.json','job.json','origin-manifest.json'}
    if receipt['payload_sha256'].keys()!=names or any(hashlib.sha256((directory/n).read_bytes()).hexdigest()!=receipt['payload_sha256'][n] for n in names):
        raise ValueError('payload bytes changed')
    notebook=read('notebook.ipynb');job=read('job.json');origin=read('origin-manifest.json');tree=ast.parse(''.join(notebook['cells'][0]['source']))
    packed=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PACKED' for t in n.targets)]
    if len(packed)!=1:raise ValueError('one notebook transport required')
    members=unpack(packed[0])
    if json.loads(members['origin.json'])!=origin or hashlib.sha256(members['source.zip']).hexdigest()!=origin['bundle_sha256']:raise ValueError('source origin changed')
    expected=job_spec(job['experiment'],job['handle'].split('/')[0],origin,stage=job['stage'],parent=job['parent'],scope=job['execution_scope'],
        ar_reference=job['ar_reference'],cohort_manifest=job['cohort_manifest'])
    if expected!=job or build_notebook(members,job)!=(notebook,read('kernel-metadata.json')):raise ValueError('runtime/job drift')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['ar','od','ot'],required=True);p.add_argument('--account',required=True)
    p.add_argument('--stage',choices=sorted(STAGES),required=True);p.add_argument('--out',required=True);p.add_argument('--parent');p.add_argument('--scope')
    a=p.parse_args();read=lambda s:json.loads(Path(s).read_text()) if s else None
    print(json.dumps(prepare(a.task,a.account,a.out,stage=a.stage,parent=read(a.parent),scope=read(a.scope))))
