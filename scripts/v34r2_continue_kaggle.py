"""Immutable private Kaggle continuations with content-checked inherited state."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
from scripts import v34r2_kaggle as base
from scripts.v32_mot20_kaggle import source_snapshot
from scripts.v33ap_kaggle import stored_source,unpack,REQUIRED
from scripts.v32rf_kaggle import serialized_source_bytes

ACCOUNTS={'thuha1205','qktttttttttt','huolgggnuyen','baoancut','trmnguyn111'}
ALLOWED_FILES={'measurements.json','measurements.partial.json','origin.json','cohorts.json',
               'controller_303401.pth','controller_303402.pth','controller_303403.pth'}


def find_parent(root,files):
    if not files or any('/' in n or '\\' in n or n not in ALLOWED_FILES for n in files):
        raise ValueError('invalid inherited artifact names')
    root=Path(root).resolve(); anchor='measurements.partial.json' if 'measurements.partial.json' in files else 'measurements.json'
    matches=[]
    for path in root.rglob(anchor):
        folder=path.parent.resolve()
        if folder.is_relative_to(root) and all((folder/n).is_file() and (folder/n).resolve().is_relative_to(root)
            and hashlib.sha256((folder/n).read_bytes()).hexdigest()==sha for n,sha in files.items()):
            matches.append(folder)
    if len(matches)!=1: raise ValueError('inherited artifacts missing, altered or ambiguous')
    return matches[0]


def job_spec(experiment,account,origin,*,parent,stage,codecs=None,splits=None,domains=None):
    if account not in ACCOUNTS or experiment not in ('ot','ar_od_ot') or stage not in ('measure-train','evaluate'):
        raise ValueError('unauthorized account, experiment or stage')
    parent=json.loads(json.dumps(parent)); kind=parent.get('kind'); ref=parent.get('ref','')
    if (kind=='kernel' and not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/[1-9][0-9]*',ref)
        or kind=='dataset' and (not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+',ref) or parent.get('version')!=1)
        or kind not in ('kernel','dataset')):
        raise ValueError('content-pinned parent source required')
    if kind=='dataset' and ref.split('/')[0]!=account: raise ValueError('private inherited dataset must belong to the execution account')
    files=parent.get('files',{})
    if not files or any(n not in ALLOWED_FILES or not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{64}',sha) for n,sha in files.items()):
        raise ValueError('invalid inherited artifact manifest')
    required={'origin.json','cohorts.json'}|({'measurements.partial.json'} if stage=='measure-train' else
        {'measurements.json','controller_303401.pth','controller_303402.pth','controller_303403.pth'})
    if not required.issubset(files): raise ValueError('inherited state is incomplete')
    job=base.job_spec(experiment,account,origin,'combined')
    if stage=='measure-train':
        if experiment!='ar_od_ot' or any(x is not None for x in (codecs,splits,domains)):
            raise ValueError('joint measurement continuation uses the full original training grid')
        scope=dict(codecs=list(job['protocol']['codecs']),splits=['train','cal'],domains=['ar','od','ot'])
        label='joint-resume-train'
    else:
        scope=dict(codecs=list(codecs or []),splits=list(splits or []),domains=list(domains or []))
        if (len(scope['codecs'])!=1 or len(scope['domains'])!=1 or not scope['splits']
            or len(set(scope['splits']))!=len(scope['splits'])
            or not set(scope['codecs']).issubset(job['protocol']['codecs'])
            or not set(scope['domains']).issubset(['ot'] if experiment=='ot' else ['ar','od','ot'])
            or not set(scope['splits']).issubset(['dev','test'])):
            raise ValueError('invalid bounded evaluation execution scope')
        label=('ot' if experiment=='ot' else 'joint-'+scope['domains'][0])+'-'+scope['codecs'][0]+'-'+('heldout' if len(scope['splits'])==2 else scope['splits'][0])
    slug='v34-r2-'+label+'-'+origin['origin_commit'][:8]
    job.update(schema='v34-r2-continuation-kaggle-1',slug=slug,handle=account+'/'+slug,private=True,
               stage=stage,parent=parent,execution_scope=scope,fresh_training=stage=='measure-train',
               scope='continuation of the same interrupted R2 experiment; original scientific settings unchanged')
    return job


RUNTIME=base.RUNTIME.replace("'adaptive_vcm.v34r2.runner'","'adaptive_vcm.v34r2.continuation'")
RUNTIME=RUNTIME.replace("    if JOB['origin_commit']!=ORIGIN['origin_commit']: raise ValueError('job/origin mismatch')",
    "    if JOB['origin_commit']!=ORIGIN['origin_commit']: raise ValueError('job/origin mismatch')\n"
    "    PARENT=find_parent(Path('/kaggle/input'),JOB['parent']['files'])")
RUNTIME=RUNTIME.replace("    command+=['--ablation',JOB['ablation']]",
    "    command+=['--ablation',JOB['ablation'],'--stage',JOB['stage'],'--parent',str(PARENT)]\n"
    "    if JOB['stage']=='evaluate':\n"
    "        for field in ('codecs','splits','domains'): command+=['--'+field,','.join(JOB['execution_scope'][field])]")


def build_notebook(members,job):
    notebook,metadata=base.build_notebook(members,job)
    source=''.join(notebook['cells'][1]['source'])
    if source.count(base.RUNTIME)!=1: raise ValueError('continuation runtime boundary changed')
    helper='ALLOWED_FILES=set('+repr(tuple(sorted(ALLOWED_FILES)))+')\n'+inspect.getsource(find_parent)+'\n'
    source=source.replace(base.RUNTIME,helper+RUNTIME)
    compile(source,'<continuation notebook>','exec')
    notebook['cells'][1]['source']=source.splitlines(keepends=True)
    metadata.update(is_private=True,kernel_sources=[job['parent']['ref']] if job['parent']['kind']=='kernel' else [])
    if job['parent']['kind']=='dataset': metadata['dataset_sources']=[*metadata['dataset_sources'],job['parent']['ref']]
    if serialized_source_bytes(notebook)>=1_000_000: raise ValueError('Kaggle source limit')
    return notebook,metadata


def prepare(experiment,account,out,*,parent,stage,codecs=None,splits=None,domains=None):
    out=Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError('immutable empty payload required')
    bundle,origin,_=source_snapshot(); bundle,origin=stored_source(bundle,origin)
    job=job_spec(experiment,account,origin,parent=parent,stage=stage,codecs=codecs,splits=splits,domains=domains)
    members={'source.zip':bundle,'origin.json':json.dumps(origin).encode(),'job.json':json.dumps(job).encode()}
    notebook,metadata=build_notebook(members,job); out.mkdir(parents=True,exist_ok=True)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',metadata),('job.json',job),('origin-manifest.json',origin)]:
        (out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    receipt=dict(handle=job['handle'],origin_commit=origin['origin_commit'],submitted=False,
        payload_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    (out/'local_validation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


def validate_payload(directory):
    directory=Path(directory); receipt=json.loads((directory/'local_validation.json').read_text())
    files={'notebook.ipynb','kernel-metadata.json','job.json','origin-manifest.json'}
    if set(receipt['payload_sha256'])!=files or any(hashlib.sha256((directory/n).read_bytes()).hexdigest()!=receipt['payload_sha256'][n] for n in files):
        raise ValueError('continuation payload bytes changed')
    notebook=json.loads((directory/'notebook.ipynb').read_text()); job=json.loads((directory/'job.json').read_text())
    origin=json.loads((directory/'origin-manifest.json').read_text())
    tree=ast.parse(''.join(notebook['cells'][0]['source']))
    packed=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PACKED' for t in n.targets))
    members=unpack(packed)
    if set(members)!=REQUIRED or json.loads(members['origin.json'])!=origin or hashlib.sha256(members['source.zip']).hexdigest()!=origin['bundle_sha256']:
        raise ValueError('continuation source origin mismatch')
    args={} if job['stage']=='measure-train' else job['execution_scope']
    expected=job_spec(job['experiment'],job['handle'].split('/')[0],origin,parent=job['parent'],stage=job['stage'],**args)
    if job!=expected: raise ValueError('continuation job drift')
    nb,meta=build_notebook(members,job)
    if nb!=notebook or meta!=json.loads((directory/'kernel-metadata.json').read_text()): raise ValueError('continuation runtime drift')
    return receipt
