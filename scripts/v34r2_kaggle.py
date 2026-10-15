"""Prepare immutable fresh-training R2 jobs. Preparation never submits a run."""
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
from scripts.v34_kaggle import RUNTIME as R1_RUNTIME
from adaptive_vcm.v34r2.protocol import settings,ACTIONS


def job_spec(experiment,account,origin,ablation='combined'):
    if (experiment not in ('od','ot','ar_od_ot') or ablation not in ('baseline','combined','map_only','target_only')
        or not re.fullmatch('[a-zA-Z0-9_-]+',account) or not re.fullmatch('[0-9a-f]{40}',origin['origin_commit'])):
        raise ValueError('invalid R2 job/account/origin')
    cfg=settings(); slug=f'v34-r2-{experiment.replace("_","-")}-{ablation}-qp4-{origin["origin_commit"][:8]}'
    datasets=(['qktttttttttt/kineticscleaned'] if experiment=='ar_od_ot' else [])+(['awsaf49/coco-2017-dataset'] if experiment in ('od','ar_od_ot') else [])
    return dict(schema='v34-r2-kaggle-1',experiment=experiment,ablation=ablation,slug=slug,handle=account+'/'+slug,private=False,
        origin_commit=origin['origin_commit'],protocol=cfg,actions=[list(a) for a in ACTIONS],datasets=datasets,
        config_path='configs/v34_r2_qp4.json',hf=dict(repo_id=HF_REPO,revision=HF_REVISION),
        sequences=[s for split in ('train','cal','dev','test') for s in cfg['ot_sequences'][split]],
        trackeval_repo=TRACKEVAL_REPO,trackeval_commit=TRACKEVAL_COMMIT,fresh_training=True,
        scope='fresh R2 dual-target measurement/training; source-only selection; four QPs; historical exploratory cohorts; no confirmed gain')


RUNTIME=R1_RUNTIME.replace('from adaptive_vcm.v34.protocol import settings,ACTIONS','from adaptive_vcm.v34r2.protocol import settings,ACTIONS')
RUNTIME=RUNTIME.replace('adaptive_vcm.v34.runner','adaptive_vcm.v34r2.runner')
RUNTIME=RUNTIME.replace("in ('ar','ar_od','ar_ot')","in ('ar_od_ot',)")
RUNTIME=RUNTIME.replace("in ('od','ar_od')","in ('od','ar_od_ot')")
RUNTIME=RUNTIME.replace("in ('ot','ar_ot')","in ('ot','ar_od_ot')")
RUNTIME=RUNTIME.replace("if config: command+=['--config',str(config)]","if config: command+=['--config',str(config)]\n    command+=['--ablation',JOB['ablation']]")


def build_notebook(members,job):
    if set(members)!=REQUIRED or json.loads(members['job.json'])!=job: raise ValueError('source-only payload required')
    constants='import base64,hashlib,io,json,lzma,stat,subprocess,zipfile\nfrom pathlib import Path\n'
    constants+=f'REQUIRED=set({tuple(sorted(REQUIRED))!r})\nOPTIONAL=set({tuple(sorted(OPTIONAL))!r})\n'+inspect.getsource(unpack)
    constants+=f'PACKED={pack(members)!r}\nMEMBERS=unpack(PACKED)\nORIGIN=json.loads(MEMBERS["origin.json"])\nJOB=json.loads(MEMBERS["job.json"])\n'
    sources=[constants,inspect.getsource(safe_extract)+'\n'+inspect.getsource(mount_data)+'\n'+RUNTIME]
    for source in sources: compile(source,'<R2 notebook>','exec')
    notebook=dict(cells=[dict(cell_type='code',id=f'v34-r2-{i}',execution_count=None,metadata={},outputs=[],source=s.splitlines(keepends=True)) for i,s in enumerate(sources)],
        metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},nbformat=4,nbformat_minor=5)
    if serialized_source_bytes(notebook)>=1_000_000: raise ValueError('Kaggle source limit')
    metadata=dict(id=job['handle'],title=job['slug'],code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=False,
        enable_gpu=True,enable_internet=True,dataset_sources=job['datasets'],kernel_sources=[],competition_sources=[],model_sources=[])
    return notebook,metadata


def prepare(experiment,account,out,ablation='combined'):
    out=Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError('immutable empty payload required')
    bundle,origin,_=source_snapshot(); bundle,origin=stored_source(bundle,origin)
    job=job_spec(experiment,account,origin,ablation)
    members={'source.zip':bundle,'origin.json':json.dumps(origin).encode(),'job.json':json.dumps(job).encode()}
    notebook,metadata=build_notebook(members,job); out.mkdir(parents=True,exist_ok=True)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',metadata),('job.json',job),('origin-manifest.json',origin)]:
        (out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    receipt=dict(handle=job['handle'],origin_commit=origin['origin_commit'],submitted=False,serialized_source_bytes=serialized_source_bytes(notebook),
        payload_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    (out/'local_validation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


def validate_payload(directory):
    directory=Path(directory); receipt=json.loads((directory/'local_validation.json').read_text())
    files={'notebook.ipynb','kernel-metadata.json','job.json','origin-manifest.json'}
    if set(receipt['payload_sha256'])!=files or any(hashlib.sha256((directory/n).read_bytes()).hexdigest()!=receipt['payload_sha256'][n] for n in files):
        raise ValueError('R2 payload bytes changed')
    notebook=json.loads((directory/'notebook.ipynb').read_text()); job=json.loads((directory/'job.json').read_text()); origin=json.loads((directory/'origin-manifest.json').read_text())
    constants=ast.parse(''.join(notebook['cells'][0]['source']))
    values=[ast.literal_eval(n.value) for n in constants.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PACKED' for t in n.targets)]
    if len(values)!=1: raise ValueError('invalid notebook transport')
    members=unpack(values[0])
    if (set(members)!=REQUIRED or hashlib.sha256(members['source.zip']).hexdigest()!=origin['bundle_sha256']
        or json.loads(members['origin.json'])!=origin): raise ValueError('source origin mismatch')
    if job!=job_spec(job['experiment'],job['handle'].split('/')[0],origin,job['ablation']): raise ValueError('R2 job drift')
    expected,metadata=build_notebook(members,job)
    if notebook!=expected or json.loads((directory/'kernel-metadata.json').read_text())!=metadata: raise ValueError('runtime drift')
    return receipt


def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment',choices=['od','ot','ar_od_ot'],required=True)
    p.add_argument('--account',required=True); p.add_argument('--out',required=True)
    p.add_argument('--ablation',choices=['baseline','combined','map_only','target_only'],default='combined')
    a=p.parse_args(); print(json.dumps(prepare(a.experiment,a.account,a.out,a.ablation)))


if __name__=='__main__': main()
