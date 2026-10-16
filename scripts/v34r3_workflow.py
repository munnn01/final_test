"""Durable private R3 stages. Submission ambiguity is never a blind retry."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile
from scripts import v34r3_kaggle as payload
from scripts.run_v31_jobs import KaggleClient,actual_status
from scripts.v34r2_continue_workflow import owner_capacity
from adaptive_vcm.v34r3.protocol import ROOT,SCHEMA,settings,digest
from adaptive_vcm.v34r3.measurement import validate_cache


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8');temporary.replace(path)


def redact(value,client):
    if isinstance(value,str):return client.safe(value)
    if isinstance(value,dict):return {k:redact(v,client) for k,v in value.items()}
    if isinstance(value,list):return [redact(v,client) for v in value]
    return value


def verify_stage(directory,entry):
    directory=Path(directory).resolve();complete=read(directory/'stage_complete.json')
    if complete.get('schema')!=SCHEMA or complete.get('status')!='complete' or complete.get('task')!=entry['task'] or complete.get('stage')!=entry['stage']:
        raise ValueError('stage completion identity mismatch')
    files=complete.get('files_sha256',{})
    if not files or 'stage.json' not in files:raise ValueError('complete stage file manifest required')
    for name,expected in files.items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha(path)!=expected:raise ValueError('stage file missing or changed: '+name)
    receipt=read(directory/'stage.json')
    if receipt.get('task')!=entry['task'] or receipt.get('stage')!=entry['stage'] or receipt.get('scope')!=entry.get('scope'):
        raise ValueError('stage task/scope receipt mismatch')
    if entry.get('origin_commit') and receipt['origin']['origin_commit']!=entry['origin_commit']:raise ValueError('stage source origin mismatch')
    if entry['stage'] in ('measure-identity','measure-rollout'):
        cache=read(directory/'measurements.json');validate_cache(cache)
        if not cache.get('complete') or cache['task']!=entry['task']:raise ValueError('incomplete task cache')
        expected={(k,c,q,cache['variant']) for k in cache['sources'] for c in cache['execution_scope']['codecs'] for q in cache['recipe']['qps']}
        if expected!={(r['id'],r['codec'],r['qp'],r['prefix_variant']) for r in cache['rows']}:raise ValueError('incomplete measurement grid')
    if entry['stage'] in ('train-warm','train-final','evaluate'):
        for seed in settings()['seeds']:
            if f'controller_{seed}.pth' not in files:raise ValueError('three checkpoint artifacts required')
    if entry['stage']=='evaluate':
        for split in (entry.get('scope') or {}).get('splits',['dev','test']):
            if not {split+'_summary.json',split+'_rows.json'}<=files.keys():raise ValueError('evaluation reports incomplete')
    return complete


def recover_timeout(directory,entry,origin):
    if entry['task'] not in ('od','ot') or entry['stage'] not in ('measure-identity','measure-rollout'):
        raise ValueError('only OD/OT measured-row timeout recovery is supported')
    path=Path(directory)/'measurements.partial.json';cache=read(path);validate_cache(cache)
    if cache['task']!=entry['task'] or cache['origin']!=origin:raise ValueError('partial target origin/task mismatch')
    if cache['variant']!=('identity' if entry['stage']=='measure-identity' else 'rollout'):raise ValueError('partial prefix policy mismatch')
    return dict(path=str(path),cache_sha256=cache['sha256'],file_sha256=sha(path),rows=len(cache['rows']),origin=origin,
        scope='validated complete rows only; runtime reconstructs causal prefixes before continuing')


def submit(client,directory,persist_intent):
    payload.validate_payload(directory);job=read(Path(directory)/'job.json');metadata=read(Path(directory)/'kernel-metadata.json')
    if job['private'] is not True or metadata['is_private'] is not True:raise ValueError('private R3 job required')
    notebook=read(Path(directory)/'notebook.ipynb');ns={};exec(''.join(notebook['cells'][0]['source']),ns)
    blobs=list(ns['MEMBERS'].values())+[(Path(directory)/n).read_bytes() for n in ('notebook.ipynb','kernel-metadata.json')]
    if any(s.encode() in b for s in client.secrets if s for b in blobs):raise ValueError('credential scan failed')
    intent=dict(status='submitted',handle=job['handle'],version=None,directory=str(directory),ambiguous=True)
    persist_intent(intent)
    evidence=client.call(job['handle'].split('/')[0],['kernels','push','-p',str(directory)],timeout=90)
    output=evidence['stdout']+evidence['stderr'];match=re.search(r'Kernel version (\d+) successfully pushed',output,re.I)
    success=evidence['exit_code']==0 and match and 'https://www.kaggle.com/code/'+job['handle'] in output
    bad=any(t in output.lower() for t in ('could not be added','not valid dataset sources','not valid kernel sources'))
    return dict(intent,status='failed_attention' if bad else 'submitted',ambiguous=not bool(success) or bad,
        version=int(match[1]) if success else None,submission_evidence=redact(evidence,client))


def _one(root,name):
    paths=list(Path(root).rglob(name))
    if len(paths)!=1:raise ValueError('missing or ambiguous '+name)
    return paths[0]


def archive_parent(source,files,destination):
    source=Path(source).resolve()
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_STORED) as archive:
        for name,expected in sorted(files.items()):
            path=(source/name).resolve()
            if not path.is_relative_to(source) or not path.is_file() or sha(path)!=expected:raise ValueError('parent artifact changed')
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));archive.writestr(info,path.read_bytes())


def _collect(root,entry,item,client):
    dest=root/'artifacts'/entry['id'];dest.mkdir(parents=True,exist_ok=True)
    evidence=client.call(entry['account'],['kernels','output',item['handle'],'--path',str(dest),'--file-pattern',r'.*/experiment/.*','--page-size','1000'],timeout=240)
    save(root/'receipts'/(entry['id']+'-download.json'),redact(evidence,client))
    if evidence['exit_code']!=0:raise ValueError('stage download incomplete')
    directory=_one(dest,'stage.json').parent
    complete=verify_stage(directory,entry);job=read(Path(item['directory'])/'job.json');receipt=read(directory/'stage.json')
    if receipt['recipe']!=__import__('adaptive_vcm.v34r3.protocol',fromlist=['recipe']).recipe(job['protocol'],entry['task'],entry['stage']):
        raise ValueError('runtime task recipe differs from planned payload')
    if receipt['origin']!=read(Path(item['directory'])/'origin-manifest.json'):raise ValueError('runtime source differs from submitted payload')
    item.update(status='complete_verified',artifacts=str(directory),completion=complete)


def _parent(plan,state,entry,client):
    recovery=state['jobs'].get(entry['id'],{}).get('recoverable_parent')
    if not entry['depends'] and recovery is None:return entry.get('parent')
    if entry['depends']:
        if len(entry['depends'])!=1:raise ValueError('one task-specific parent stage required')
        parent=state['jobs'][entry['depends'][0]];source=Path(parent['artifacts']);source_entry=next(j for j in plan['jobs'] if j['id']==entry['depends'][0])
        complete=verify_stage(source,source_entry);files={**complete['files_sha256'],'stage_complete.json':sha(source/'stage_complete.json')}
    else:
        source=Path(recovery['path']).parent;source_entry=entry;files={'stage.json':sha(source/'stage.json')}
    if recovery is not None:
        partial=Path(recovery['path'])
        if sha(partial)!=recovery['file_sha256']:raise ValueError('recovery rows changed')
        staging=Path(plan['root'])/'recovery_parents'/entry['id']/recovery['cache_sha256'];staging.mkdir(parents=True,exist_ok=True)
        for name,expected in files.items():
            target=staging/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
        name='recovery_measurements.partial.json' if entry['depends'] else 'measurements.partial.json'
        shutil.copyfile(partial,staging/name);files={**files,name:recovery['file_sha256']};source=staging
    key=digest(dict(account=entry['account'],files=files));root=Path(plan['root']);directory=root/'private_states'/key
    ref=entry['account']+'/v34-r3-state-'+key[:20];record=state.setdefault('datasets',{}).get(key)
    if record is None:
        directory.mkdir(parents=True,exist_ok=True)
        archive_parent(source,files,directory/'state.zip')
        if any(s.encode() in p.read_bytes() for s in client.secrets if s for p in directory.rglob('*') if p.is_file()):raise ValueError('private parent credential scan failed')
        save(directory/'dataset-metadata.json',dict(id=ref,title='V34 R3 state '+key[:20],licenses=[dict(name='unknown')],
            description='Private task-specific stage state; no original-resolution source images or video.'))
        record=dict(status='ambiguous',ref=ref,files=files,archive_sha256=sha(directory/'state.zip'));state['datasets'][key]=record;save(root/'state.json',redact(state,client))
        evidence=client.call(entry['account'],['datasets','create','-p',str(directory),'--keep-tabular'],timeout=240)
        record['evidence']=redact(evidence,client)
        if evidence['exit_code']==0 and 'Your private Dataset is being created' in evidence['stdout']:record['status']='creating'
        save(root/'state.json',redact(state,client))
    if record['status']=='ambiguous':raise ValueError('parent upload ambiguous; reconcile before retry')
    evidence=client.call(entry['account'],['datasets','status',ref]);record['status_evidence']=redact(evidence,client)
    if evidence['exit_code']!=0 or 'ready' not in evidence['stdout'].lower():return None
    record['status']='ready'
    return dict(kind='dataset',ref=ref,version=1,task=entry['task'],schema=SCHEMA,stage=source_entry['stage'],files=files,archive_sha256=record['archive_sha256'])


def _timeout(root,entry,item,client):
    dest=root/'recovery'/entry['id']/str(item.get('attempt',1));dest.mkdir(parents=True,exist_ok=True)
    evidence=client.call(entry['account'],['kernels','output',item['handle'],'--path',str(dest),'--file-pattern',
        r'.*/(?:experiment/(?:stage\.json|measurements\.partial\.json)|exception\.txt|status\.json)$','--page-size','100'],timeout=240)
    if evidence['exit_code']!=0:return False
    reason=' '.join(p.read_text(encoding='utf-8',errors='replace') for p in dest.rglob('exception.txt'))
    status=actual_status(item['status_evidence'])
    if status!='TIMEOUT' and not re.search(r'(?:exceed|time.?out|time limit|execution limit)',reason,re.I):return False
    partial=_one(dest,'measurements.partial.json');origin=read(Path(item['directory'])/'origin-manifest.json')
    recovered=recover_timeout(partial.parent,entry,origin)
    previous=item.get('recoverable_parent',{}).get('rows',0)
    if recovered['rows']<=previous:return False
    history=item.setdefault('attempt_history',[]);history.append(dict(handle=item['handle'],rows=recovered['rows'],cache_sha256=recovered['cache_sha256']))
    item.update(status='prepared',attempt=item.get('attempt',1)+1,recoverable_parent=recovered,ambiguous=False)
    return True


def step(plan,state,client):
    state=copy.deepcopy(state);root=Path(plan['root']);root.mkdir(parents=True,exist_ok=True)
    try:
        commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        if commit!=plan['source_commit'] or sha(ROOT/'configs/v34_r3_qp4.json')!=plan['config_sha256']:raise ValueError('workflow source/config changed')
        jobs=plan['jobs']
        if len({j['id'] for j in jobs})!=len(jobs):raise ValueError('duplicate stage identifiers')
        for j in jobs:
            if j['task'] not in ('od','ot') or j['account'] not in payload.ACCOUNTS or j['stage'] not in payload.STAGES:raise ValueError('unsupported task/stage/account')
            if any(d not in {x['id'] for x in jobs if x['task']==j['task']} for d in j['depends']):raise ValueError('foreign-task dependency')
            item=state['jobs'].get(j['id'])
            if item is None or item['status'] in ('complete_verified','failed_attention','prepared'):continue
            evidence=client.call(j['account'],['kernels','status',item['handle']]);item['status_evidence']=redact(evidence,client);status=actual_status(evidence)
            if status in ('RUNNING','QUEUED'):
                item.update(status='running',ambiguous=False)
            elif status=='COMPLETE':_collect(root,{**j,'origin_commit':plan['source_commit']},item,client)
            elif status not in ('UNKNOWN',):
                item.update(status='failed_attention',reason=status,recovery='download measurements.partial.json and validate using recover_timeout; no blind rerun')
                if j['stage'] in ('measure-identity','measure-rollout'):_timeout(root,j,item,client)
            save(root/'state.json',redact(state,client))
        ready=[j for j in jobs if state['jobs'].get(j['id'],{}).get('status','prepared')=='prepared' and all(state['jobs'].get(d,{}).get('status')=='complete_verified' for d in j['depends'])]
        capacities={a:owner_capacity(client,a) for a in {j['account'] for j in ready}}
        for j in ready:
            capacity=capacities[j['account']]
            if len(capacity['active'])>=2 or capacity['hours']<j['estimated_gpu_hours']:continue
            parent=_parent(plan,state,j,client)
            if j['depends'] and parent is None:continue
            previous=state['jobs'].get(j['id'],{});attempt=previous.get('attempt',1)
            directory=root/'payloads'/(j['id'] if attempt==1 else j['id']+'-resume-'+str(attempt))
            if not directory.exists():payload.prepare(j['task'],j['account'],directory,stage=j['stage'],parent=parent,scope=j.get('scope'),cohort_manifest=plan.get('cohort_manifest'))
            def persist(value):
                state['jobs'][j['id']]={**previous,**value};save(root/'state.json',redact(state,client))
            item={**previous,**submit(client,directory,persist)};state['jobs'][j['id']]=item;save(root/'state.json',redact(state,client))
            capacity['active'].append(item['handle']);capacity['hours']-=j['estimated_gpu_hours']
        state['status']='complete_verified' if len(state['jobs'])==len(jobs) and all(v['status']=='complete_verified' for v in state['jobs'].values()) else 'failed_attention' if any(v['status']=='failed_attention' for v in state['jobs'].values()) else 'running'
    except Exception as error:state.update(status='failed_attention',error=client.safe(str(error)))
    state=redact(state,client);save(root/'state.json',state);return state


def initialize(root):
    root=Path(root);path=root/'plan.json'
    if path.is_file():return read(path)
    jobs=[]
    def add(task,label,stage,account,depends,hours,scope=None):
        jobs.append(dict(id=task+'-'+label,task=task,stage=stage,account=account,depends=[task+'-'+d for d in depends],estimated_gpu_hours=hours,scope=scope))
    for task,account in [('od','thuha1205'),('ot','qktttttttttt')]:
        add(task,'identity','measure-identity',account,[],5)
        previous='identity'
        if task=='ot':
            add(task,'warm','train-warm',account,['identity'],.25);add(task,'rollout','measure-rollout',account,['warm'],5);previous='rollout'
        add(task,'final','train-final',account,[previous],.25)
        add(task,'dev','evaluate','huolgggnuyen' if task=='ot' else 'thuha1205',['final'],5,dict(codecs=['h264','h265'],splits=['dev']))
        for c,account_test in [('h264','baoancut'),('h265','trmnguyn111')]:add(task,'test-'+c,'evaluate',account_test,['dev'],3,dict(codecs=[c],splits=['test']))
    plan=dict(root=str(root.resolve()),source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        config_sha256=sha(ROOT/'configs/v34_r3_qp4.json'),private=True,jobs=jobs)
    save(path,plan);save(root/'state.json',dict(jobs={},datasets={},status='prepared'));return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--pool',type=Path,required=True);p.add_argument('--run',action='store_true');p.add_argument('--poll-seconds',type=int,default=180)
    a=p.parse_args();plan=initialize(a.out);client=KaggleClient(a.pool)
    while True:
        state=step(plan,read(a.out/'state.json'),client);print(json.dumps(dict(status=state['status'],jobs={k:dict(status=v['status'],handle=v.get('handle')) for k,v in state['jobs'].items()})),flush=True)
        if not a.run or state['status'] in ('complete_verified','failed_attention'):break
        time.sleep(max(60,a.poll_seconds))


if __name__=='__main__':main()
