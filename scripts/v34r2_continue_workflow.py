"""Durable local dispatch of the authorized, private R2 continuation stages."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from scripts import v34r2_continue_kaggle as payload
from scripts.run_v31_jobs import KaggleClient,actual_status,TERMINAL

SEEDS=[303401,303402,303403]


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8'); temporary.replace(path)


def one(root,name):
    paths=list(Path(root).rglob(name))
    if len(paths)!=1: raise ValueError('missing or ambiguous inherited file: '+name)
    return paths[0]


def ready_jobs(jobs,states,external):
    active={a:set(handles) for a,handles in external.items()}
    for job in jobs:
        value=states.get(job['id'],{})
        if value.get('status') in ('RUNNING','QUEUED','SUBMITTED','UNKNOWN'):
            active.setdefault(job['account'],set()).add(value.get('handle',job['id']))
    ready=[]
    for job in jobs:
        if job['id'] in states or any(states.get(d,{}).get('status')!='COMPLETE' or not states[d].get('stage_verified') for d in job['depends']):
            continue
        slots=active.setdefault(job['account'],set())
        if len(slots)>=2: continue
        ready.append(job); slots.add(job['id'])
    return ready


def verify_stage(directory,entry):
    value=load(one(directory,'stage_complete.json'))
    if value.get('status')!='complete' or value.get('stage')!=entry['stage'] or value.get('experiment')!=entry['experiment']:
        raise ValueError('stage completion manifest mismatch')
    if entry['stage']=='evaluate':
        scope=value.get('scope',{})
        if any(scope.get(k)!=v for k,v in entry['scope'].items()) or len(scope.get('checkpoint_sha256',[]))!=3:
            raise ValueError('stage execution scope/checkpoint set mismatch')
        if scope.get('settings',{}).get('qps')!=[30,35,40,45]: raise ValueError('stage QP recipe mismatch')
        if entry.get('source_ids') and scope.get('source_ids')!=entry['source_ids']: raise ValueError('held-out source coverage mismatch')
    elif len(value.get('checkpoint_sha256',[]))!=3:
        raise ValueError('training stage checkpoint set incomplete')
    return value


def artifact_files(directory,training=False):
    names=['origin.json','cohorts.json','measurements.partial.json' if training else 'measurements.json']
    if not training: names += [f'controller_{s}.pth' for s in SEEDS]
    return {n:one(directory,n) for n in names}


def manifest(files): return {n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}


def initialize(root,results_root):
    root=Path(root); plan_path=root/'plan.json'
    if plan_path.exists(): return load(plan_path)
    repo=Path(__file__).resolve().parents[1]
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    # source_snapshot enforces a clean, fully committed transport before a launch.
    payload.source_snapshot()
    parents={}; cohorts={}
    originals={'ot':'trnhlng/v34-r2-ot-combined-qp4-b37c4937/1',
               'ar_od_ot':'thuha1205/v34-r2-ar-od-ot-combined-qp4-b37c4937/1'}
    for task in originals:
        recovery=Path(results_root)/'recovery'/task
        names=['origin.json','measurements.partial.json' if task=='ar_od_ot' else 'measurements.json']
        if task=='ot': names += [f'controller_{s}.pth' for s in SEEDS]
        files={n:one(recovery,n) for n in names}
        files['cohorts.json']=one(Path(results_root)/'artifacts'/task,'cohorts.json')
        cohorts[task]=load(files['cohorts.json'])
        parents[task]=dict(kind='kernel',ref=originals[task],files=manifest(files))
    jobs=[]
    def add(identifier,account,experiment,stage,scope,depends,hours):
        entry=dict(id=identifier,account=account,experiment=experiment,stage=stage,scope=scope,depends=depends,estimated_gpu_hours=hours)
        if scope:
            entry['source_ids']={s:[k for k,v in cohorts[experiment].items() if v['split']==s and v['domain'] in scope['domains']] for s in scope['splits']}
        jobs.append(entry)
    for split,account in [('test','qktttttttttt'),('dev','huolgggnuyen')]:
        for c in ('h264','h265'):
            add(f'ot-{c}-{split}',account,'ot','evaluate',dict(codecs=[c],splits=[split],domains=['ot']),[],8 if split=='dev' else 4)
    add('joint-train','baoancut','ar_od_ot','measure-train',None,[],4)
    for domain,account in [('ar','baoancut'),('od','thuha1205')]:
        for c in ('h264','h265'):
            add(f'joint-{domain}-{c}',account,'ar_od_ot','evaluate',dict(codecs=[c],splits=['dev','test'],domains=[domain]),['joint-train'],6 if domain=='ar' else 4)
    for split,account in [('test','qktttttttttt'),('dev','trmnguyn111')]:
        for c in ('h264','h265'):
            add(f'joint-ot-{c}-{split}',account,'ar_od_ot','evaluate',dict(codecs=[c],splits=[split],domains=['ot']),['joint-train'],8 if split=='dev' else 4)
    plan=dict(schema='v34-r2-continuation-workflow-1',source_commit=commit,parents=parents,jobs=jobs,
              created_utc=datetime.now(timezone.utc).isoformat(),accounts=sorted(payload.ACCOUNTS),private=True)
    save(plan_path,plan); save(root/'state.json',dict(jobs={},datasets={},status='prepared'))
    return plan


def owner_capacity(client,account):
    quota=client.call(account,['quota','--format','json'])
    inv=client.call(account,['kernels','list','--mine','--user',account,'--page-size','100','--format','json'])
    if quota['exit_code']!=0 or inv['exit_code']!=0 or 'next page token' in (inv['stdout']+inv['stderr']).lower():
        raise ValueError('cannot verify complete account inventory/quota')
    hours=float(next(r for r in json.loads(quota['stdout']) if r['resource']=='GPU')['remaining'].removesuffix('h'))
    rows=[r for r in json.loads(inv['stdout']) if r['ref']]
    if any(not r['ref'].startswith(account+'/') for r in rows): raise ValueError('owner inventory mismatch')
    def check(row):
        evidence=client.call(account,['kernels','status',row['ref']])
        return dict(handle=row['ref'],status=actual_status(evidence),evidence=evidence)
    with ThreadPoolExecutor(max_workers=4) as executor: checks=list(executor.map(check,rows))
    active=[]
    for row in checks:
        evidence=row['evidence']; status=row['status']
        if status in ('RUNNING','QUEUED'): active.append(row['handle'])
        elif status not in TERMINAL and '404 Client Error: Not Found' not in evidence['stderr']:
            raise ValueError('unresolved owner notebook status: '+row['handle'])
    return dict(hours=hours,active=active,quota=quota,inventory=inv,status_checks=checks,checked_utc=datetime.now(timezone.utc).isoformat())


def checked_submit(client,directory):
    directory=Path(directory); payload.validate_payload(directory)
    meta=load(directory/'kernel-metadata.json'); account=meta['id'].split('/')[0]
    if account not in payload.ACCOUNTS or meta['is_private'] is not True: raise ValueError('private authorized account required')
    receipt=directory/'submission.json'
    if receipt.exists():
        previous=load(receipt)
        if previous.get('handle')!=meta['id']: raise ValueError('submission receipt mismatch')
        previous.setdefault('version',None)
        return previous
    nb=load(directory/'notebook.ipynb'); namespace={}; exec(''.join(nb['cells'][0]['source']),namespace)
    import io,zipfile
    with zipfile.ZipFile(io.BytesIO(namespace['MEMBERS']['source.zip'])) as archive:
        blobs=[archive.read(n) for n in archive.namelist() if not n.endswith('/')]
    blobs += [(directory/n).read_bytes() for n in ('job.json','kernel-metadata.json','notebook.ipynb')]
    if any(s.encode() in b for s in client.secrets if s for b in blobs): raise ValueError('credential scan failed')
    save(receipt,dict(status='AMBIGUOUS',handle=meta['id'],version=None,reason='intent persisted before push; no blind retry'))
    evidence=client.call(account,['kernels','push','-p',str(directory)],timeout=90)
    output=evidence['stdout']+evidence['stderr']; match=re.search(r'Kernel version (\d+) successfully pushed',output,re.I)
    accepted=evidence['exit_code']==0 and match and 'https://www.kaggle.com/code/'+meta['id'] in output
    result=dict(status='SUBMITTED' if accepted else 'AMBIGUOUS',handle=meta['id'],version=int(match[1]) if accepted else None,evidence=evidence)
    save(receipt,result); return result


def collect_stage(client,root,entry,state):
    directory=root/'artifacts'/entry['id']; directory.mkdir(parents=True,exist_ok=True)
    pattern=(r'.*/(?:stage_complete\.json|execution_scope\.json|(?:dev|test)_(?:rows|summary)\.json|dev_exhaustive_source_guard\.json)$'
             if entry['stage']=='evaluate' else r'.*/(?:stage_complete\.json|controller_[0-9]+\.pth|measurements\.json|cohorts\.json|origin\.json)$')
    evidence=client.call(entry['account'],['kernels','output',state['handle'],'--path',str(directory),'--file-pattern',pattern,'--page-size','100'],timeout=240)
    save(root/'receipts'/(entry['id']+'-download.json'),evidence)
    if evidence['exit_code']!=0: raise ValueError('stage artifact download failed')
    result=verify_stage(directory,entry)
    job=load(Path(state['directory'])/'job.json')
    if entry['stage']=='evaluate':
        wanted=[job['parent']['files'][f'controller_{s}.pth'] for s in SEEDS]
        if result['scope']['checkpoint_sha256']!=wanted: raise ValueError('evaluated checkpoint bytes differ from planned parent')
    state.update(stage_verified=True,artifacts=str(directory),completion=result)


def private_parent(client,root,plan,state,account):
    files=artifact_files(state['jobs']['joint-train']['artifacts']); hashes=manifest(files)
    trained=state['jobs']['joint-train']
    if account=='baoancut': return dict(kind='kernel',ref=trained['handle']+'/'+str(trained['version']),files=hashes)
    ref=account+'/v34-r2-joint-state-'+plan['source_commit'][:8]
    receipt=state['datasets'].get(account)
    if receipt is None:
        directory=root/'private_states'/account; directory.mkdir(parents=True,exist_ok=True)
        for name,path in files.items(): shutil.copyfile(path,directory/name)
        if any(s.encode() in p.read_bytes() for s in client.secrets if s for p in directory.iterdir() if p.is_file()):
            raise ValueError('credential scan failed for private inherited state')
        save(directory/'dataset-metadata.json',dict(id=ref,title='V34 R2 joint state '+plan['source_commit'][:8],licenses=[dict(name='unknown')],
            description='Private continuation state for the same interrupted experiment; no raw images or video.'))
        receipt=dict(status='AMBIGUOUS',ref=ref,files=hashes); state['datasets'][account]=receipt; save(root/'state.json',state)
        evidence=client.call(account,['datasets','create','-p',str(directory)],timeout=240)
        receipt['evidence']=evidence
        if evidence['exit_code']==0 and 'Your private Dataset is being created' in evidence['stdout']:
            receipt['status']='CREATING'
        save(root/'state.json',state)
    if receipt['status']=='AMBIGUOUS': raise ValueError('private state upload ambiguous; reconcile before retry')
    evidence=client.call(account,['datasets','status',ref]); receipt['status_evidence']=evidence
    if evidence['exit_code']!=0 or 'ready' not in evidence['stdout'].lower(): return None
    receipt['status']='READY'; save(root/'state.json',state)
    return dict(kind='dataset',ref=ref,version=1,files=receipt['files'])


def merge_results(root,plan,state):
    for experiment in ('ot','ar_od_ot'):
        for split in ('dev','test'):
            merged=dict(results={},ot_official_metrics={},ot_curves={},selection_counts={},quality_improvement_confirmed=False,
                        scope='historical exploratory cohorts; original R2 recipe; merged execution scopes')
            for entry in plan['jobs']:
                if entry['stage']!='evaluate' or entry['experiment']!=experiment or split not in entry['scope']['splits']: continue
                value=load(one(state['jobs'][entry['id']]['artifacts'],split+'_summary.json'))
                for field in ('results','ot_official_metrics','ot_curves'):
                    if set(merged[field])&set(value.get(field,{})): raise ValueError('overlapping result scopes')
                    merged[field].update(value.get(field,{}))
                for arm,counts in value['selection_counts'].items():
                    total=merged['selection_counts'].setdefault(arm,{})
                    for action,count in counts.items(): total[action]=total.get(action,0)+count
            save(root/'results'/experiment/(split+'_summary.json'),merged)
    save(root/'complete.json',dict(status='complete',source_commit=plan['source_commit'],jobs=len(plan['jobs']),
         statements='All planned scopes completed and stage manifests verified; no task-quality improvement inferred from job completion.'))


def advance(root,plan,client):
    root=Path(root); state=load(root/'state.json')
    repo=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=plan['source_commit']:
        raise ValueError('continuation source commit changed while workflow was pending')
    for entry in plan['jobs']:
        item=state['jobs'].get(entry['id'])
        if item is None: continue
        if item['status'] in ('SUBMITTED','RUNNING','QUEUED','UNKNOWN'):
            evidence=client.call(entry['account'],['kernels','status',item['handle']])
            item.update(status=actual_status(evidence),status_evidence=evidence)
        if item['status']=='COMPLETE' and not item.get('stage_verified'): collect_stage(client,root,entry,item)
        save(root/'state.json',state)
    if all(state['jobs'].get(j['id'],{}).get('stage_verified') for j in plan['jobs']):
        merge_results(root,plan,state); state['status']='complete'; save(root/'state.json',state); return state
    pending=[j for j in plan['jobs'] if j['id'] not in state['jobs']]
    accounts={j['account'] for j in pending if all(state['jobs'].get(d,{}).get('stage_verified') for d in j['depends'])}
    capacities={}
    for account in accounts:
        capacities[account]=owner_capacity(client,account)
        save(root/'preflight'/(account+'.json'),capacities[account])
    ready=ready_jobs(plan['jobs'],state['jobs'],{a:v['active'] for a,v in capacities.items()})
    for entry in ready:
        if entry['account'] not in capacities or capacities[entry['account']]['hours']<entry['estimated_gpu_hours']: continue
        parent=plan['parents'][entry['experiment']] if not entry['depends'] else private_parent(client,root,plan,state,entry['account'])
        if parent is None: continue
        directory=root/'payloads'/entry['id']
        if not directory.exists():
            payload.prepare(entry['experiment'],entry['account'],directory,parent=parent,stage=entry['stage'],**(entry['scope'] or {}))
        item=checked_submit(client,directory); item['directory']=str(directory)
        state['jobs'][entry['id']]=item; capacities[entry['account']]['hours']-=entry['estimated_gpu_hours']
        save(root/'state.json',state)
        print(json.dumps(dict(job=entry['id'],**{k:item[k] for k in ('status','handle','version')})),flush=True)
    failed=[k for k,v in state['jobs'].items() if v['status'] in (TERMINAL-{'COMPLETE'})|{'AMBIGUOUS'}]
    state['status']='needs_attention' if failed else 'running'; state['failed_jobs']=failed
    state['updated_utc']=datetime.now(timezone.utc).isoformat(); save(root/'state.json',state)
    return state


def main():
    p=argparse.ArgumentParser(); p.add_argument('--pool',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--results-root',type=Path,required=True); p.add_argument('--run',action='store_true')
    p.add_argument('--poll-seconds',type=int,default=180)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    plan=initialize(args.out,args.results_root); client=KaggleClient(args.pool)
    while True:
        try:
            state=advance(args.out,plan,client)
            if state['status'] in ('complete','needs_attention') or not args.run: break
        except Exception as error:
            save(args.out/'attention.json',dict(error=client.safe(str(error)),checked_utc=datetime.now(timezone.utc).isoformat()))
            raise
        time.sleep(max(60,args.poll_seconds))


if __name__=='__main__': main()
