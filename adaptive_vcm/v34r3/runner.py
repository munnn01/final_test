"""Restartable standalone stages of one shared filtering method."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np
import torch
from . import SCHEMA
from .protocol import settings,recipe as task_recipe,digest,component_names,component_mask,verify_baseline,ROOT
from .measurement import measure_stage,validate_cache,prepare_source
from .sources import validate_cohorts,load_groups
from .model import Controller
from .learning import train,calibrate,fixed_shortlist
from .checkpoint import save,load
from .evaluation import evaluate,predict,select_policy,promotion
from ..v34.runner import input_pair,write_json
from ..v32vc.checkpoint import state_hash


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_models(parent,cfg,heads,task,*,role):
    models=[load(Path(parent)/f'controller_{seed}.pth',task=task,recipe=cfg,task_models=heads.hashes(),device=heads.device) for seed in cfg['seeds']]
    if any(m.metadata.get('checkpoint_role')!=role or m.metadata.get('seed')!=seed for m,seed in zip(models,cfg['seeds'])):
        raise ValueError(role+' checkpoints with exact seed order required')
    return models


def _cache(path,cfg,heads,task,origin):
    value=read(path);validate_cache(value)
    if not value.get('complete') or value['recipe']!=cfg or value['task']!=task or value['task_models']!=heads.hashes() or value['origin']!=origin:
        raise ValueError('complete matching stage cache required')
    expected={(k,c,q,value['variant']) for k in value['sources'] for c in value['execution_scope']['codecs'] for q in cfg['qps']}
    actual={(r['id'],r['codec'],r['qp'],r['prefix_variant']) for r in value['rows']}
    if actual!=expected:raise ValueError('incomplete measurement grid')
    return value


def train_models(caches,groups,cfg,out,heads,task,role):
    rows=[r for cache in caches for r in cache['rows']];fit=[r for r in rows if r['split']=='train']
    cal=[r for r in caches[-1]['rows'] if r['split']=='cal'];inputs={}
    if not fit or role=='FINAL' and not cal:raise ValueError('TRAIN and FINAL CAL are required')
    ids={r['id'] for r in rows}
    for split in ('train','cal'):
        clean={};supports={}
        for r in groups[split]:
            if r.id not in ids:continue
            source,maps=prepare_source(r,heads,task,cfg,clean,supports);inputs[r.id]=input_pair(r.rgb,maps)
    torch.save(inputs,out/'controller_inputs.pth');models=[]
    for seed in cfg['seeds']:
        np.random.seed(seed);torch.manual_seed(seed);model=Controller(13,9)
        history,scales=train(model,fit,inputs,cfg,device=heads.device)
        predictions=[predict(model,inputs[r['id']],r['codec'],r['qp'],cfg) for r in cal]
        calibration=calibrate(cal,predictions,cfg['cal_coverage'],recipe=cfg) if role=='FINAL' else None
        if calibration is not None:
            calibration.update(cache_sha256=caches[-1]['sha256'],prefix_policy_sha256=digest(dict(variant=caches[-1]['variant'],
                warm_checkpoint_sha256=caches[-1].get('warm_checkpoint_sha256'),prefixes=[(r['id'],r['prefix_sha256'],r['prefix_after']) for r in cal])))
        metadata=dict(task=task,recipe=cfg,task_models=heads.hashes(),risk_components=list(component_names(task)),component_mask=component_mask(task),
            seed=seed,checkpoint_role=role,calibration=calibration,history=history,scales=scales.tolist(),scales_fit_split='train',
            fixed_shortlist=fixed_shortlist(fit,cfg),cache_sha256=[c['sha256'] for c in caches],origin=caches[0]['origin'],
            inputs_sha256=file_hash(out/'controller_inputs.pth'),codec_backpropagation=False,
            seed_scope='optimization seeds share a frozen seed-303401 warm rollout measurement policy')
        save(out/f'controller_{seed}.pth',model,metadata);write_json(out/f'training_{seed}.json',metadata)
        if cal:write_json(out/f'cal_predictions_{seed}.json',[dict(id=r['id'],group=r['group'],codec=r['codec'],qp=r['qp'],**p) for r,p in zip(cal,predictions)])
    return load_models(out,cfg,heads,task,role=role)


def run_records(groups,meta,cfg,out,origin,heads,task,package=None,*,stage='all',parent=None,scope=None):
    torch.set_num_threads(2)
    if task=='ar':
        if stage!='evaluate':raise ValueError('AR is evaluation-only')
        from .ar import evaluate_reference,load_reference
        frozen,models=load_reference(parent,heads.device)
        return evaluate_reference(groups,heads,models,frozen,meta,Path(out),None)
    if task not in ('od','ot') or cfg['task']!=task or stage not in ('all','measure-identity','train-warm','measure-rollout','train-final','evaluate'):
        raise ValueError('invalid standalone task/stage')
    if task=='od' and stage in ('train-warm','measure-rollout'):raise ValueError('OD uses identity data only')
    if not re.fullmatch('[0-9a-f]{40}',origin.get('origin_commit','')) or not re.fullmatch('[0-9a-f]{64}',origin.get('bundle_sha256','')):
        raise ValueError('source origin commit and bundle digest required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);parent=Path(parent) if parent is not None else None
    audit=validate_cohorts(groups,meta.get('history',{}));meta={**meta,'cohort_audit':meta.get('cohort_audit',audit)}
    receipt=dict(schema=SCHEMA,task=task,stage=stage,recipe=cfg,origin=origin,scope=scope,cohorts_sha256=digest(audit),
        task_models=heads.hashes(),parent_sha256=file_hash(parent/'stage_complete.json') if parent is not None and (parent/'stage_complete.json').is_file() else None)
    if (out/'stage.json').is_file() and read(out/'stage.json')!=receipt:raise ValueError('immutable stage provenance mismatch')
    write_json(out/'stage.json',receipt)
    if (out/'stage_complete.json').is_file():
        complete=read(out/'stage_complete.json')
        for name,sha in complete.get('files_sha256',{}).items():
            if file_hash(out/name)!=sha:raise ValueError('completed stage artifact changed')
        return complete
    if stage=='all':
        run_records(groups,meta,cfg,out/'identity',origin,heads,task,package,stage='measure-identity',scope=scope)
        training_parent=out/'identity'
        if task=='ot':
            run_records(groups,meta,cfg,out/'warm',origin,heads,task,package,stage='train-warm',parent=training_parent,scope=scope)
            run_records(groups,meta,cfg,out/'rollout',origin,heads,task,package,stage='measure-rollout',parent=out/'warm',scope=scope)
            training_parent=out/'rollout'
        run_records(groups,meta,cfg,out/'final',origin,heads,task,package,stage='train-final',parent=training_parent,scope=scope)
        # Evaluate in the run root while retaining the enclosing all-stage receipt.
        _evaluate(groups,meta,cfg,out,heads,load_models(out/'final',cfg,heads,task,role='FINAL'),task,package,scope)
    elif stage=='measure-identity':measure_stage(groups,meta,cfg,out,origin,heads,task,variant='identity',package=package,scope=scope)
    elif stage=='train-warm':
        cache=_cache(parent/'measurements.json',cfg,heads,task,origin)
        if cache['variant']!='identity':raise ValueError('warm requires identity measurements')
        train_models([cache],groups,cfg,out,heads,task,'WARM');write_json(out/'identity_measurements.json',cache)
    elif stage=='measure-rollout':
        identity=_cache(parent/'identity_measurements.json',cfg,heads,task,origin)
        warm=load_models(parent,cfg,heads,task,role='WARM')
        measure_stage(groups,meta,cfg,out,origin,heads,task,variant='rollout',warm_models=warm,package=package,scope=scope)
        write_json(out/'identity_measurements.json',identity)
    elif stage=='train-final':
        cache=_cache(parent/'measurements.json',cfg,heads,task,origin);caches=[cache]
        if task=='ot':
            if cache['variant']!='rollout':raise ValueError('OT FINAL requires measured rollout CAL')
            identity=_cache(parent/'identity_measurements.json',cfg,heads,task,origin);caches=[identity,cache]
        train_models(caches,groups,cfg,out,heads,task,'FINAL')
    else:
        models=load_models(parent,cfg,heads,task,role='FINAL')
        for seed in cfg['seeds']:shutil.copyfile(parent/f'controller_{seed}.pth',out/f'controller_{seed}.pth')
        if (parent/'locked_policy.json').is_file():shutil.copyfile(parent/'locked_policy.json',out/'locked_policy.json')
        _evaluate(groups,meta,cfg,out,heads,models,task,package,scope)
    files={p.relative_to(out).as_posix():file_hash(p) for p in out.rglob('*') if p.is_file() and p.name!='stage_complete.json' and not p.name.endswith('.partial.json')}
    complete=dict(status='complete',schema=SCHEMA,task=task,stage=stage,quality_improvement_confirmed=False,files_sha256=files)
    write_json(out/'stage_complete.json',complete);return complete


def _evaluate(groups,meta,cfg,out,heads,models,task,package,scope):
    policies=[dict(gt_limits=[gt]*3,source_limits=[source]*(4 if task=='od' else 3),risk_gate=gate)
        for gt in cfg['limit_grid'] for source in cfg['limit_grid'] for gate in (True,False)]
    codecs=scope.get('codecs') if scope else None
    splits=scope.get('splits',['dev','test']) if scope else ['dev','test']
    if not splits or not set(splits)<={'dev','test'}:raise ValueError('bounded heldout scope required')
    binding=dict(recipe_sha256=digest(cfg),checkpoint_sha256=[m.checkpoint_sha256 for m in models])
    if 'dev' in splits:
        if codecs is not None and set(codecs)!=set(cfg['codecs']):raise ValueError('DEV must lock one task policy across both codecs')
        dev=evaluate(groups,meta,cfg,out,heads,models,task,package,splits=('dev',),codecs=codecs,policies=policies)
        locked={**select_policy(dev['dev'],policies,task),**binding};write_json(out/'locked_policy.json',locked)
    else:
        locked=read(out/'locked_policy.json')
        if any(locked.get(k)!=v for k,v in binding.items()) or locked['fit_split']!='dev' or locked['policy'] not in policies or locked['policy_sha256']!=digest(locked['policy']):
            raise ValueError('matching DEV-locked task policy required before TEST')
    if 'test' not in splits:return
    test=evaluate(groups,meta,cfg,out,heads,models,task,package,splits=('test',),codecs=codecs,policies=[locked['policy']])
    summary=test['test'];gaps=[];rates=[];reliable=[]
    for result in summary['results'].values():
        a=result['curves']['anchor']
        for arm,b in result['curves'].items():
            if arm.startswith('cnn_'):
                gaps.extend((np.asarray(b['quality'],float)-a['quality'])*(1 if task=='ot' else 100))
                rates.extend(np.asarray(b['bpp'])/a['bpp']);reliable.append(result['comparisons'][arm]['bd_reliable'])
    rows=read(out/'test_rows.json');cnn=[r for r in rows if r['arm'].startswith('cnn_')]
    report=dict(task=task,worst_qp_gap_pp=float(min(gaps)),byte_improvement=bool(np.mean(rates)<1),
        nonidentity_fraction=float(np.mean([r['action']!=0 for r in cnn])),bd_reliable=all(reliable),dev_gate_supported=locked['dev_gate_supported'])
    summary.update(promotion=promotion(report,dict(confirmation_eligible=meta['cohort_audit'].get('independent_confirmation',False)),
        dict(promotion_eligible=all(m.metadata['calibration']['promotion_eligible'] for m in models),source_groups=min(m.metadata['calibration']['source_groups'] for m in models))),
        promotion_evidence=report,locked_policy_sha256=locked['policy_sha256'])
    write_json(out/'test_summary.json',summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['ar','od','ot'],required=True)
    p.add_argument('--stage',choices=['all','measure-identity','train-warm','measure-rollout','train-final','evaluate'],default='all')
    p.add_argument('--config');p.add_argument('--parent');p.add_argument('--scope');p.add_argument('--cohort-manifest')
    for key in ('ar_root','od_root','annotations','mot_root','trackeval','origin','out'):
        p.add_argument('--'+key.replace('_','-'),required=key in ('origin','out'))
    args=p.parse_args();origin=read(args.origin)
    if 'baseline' not in origin:raise ValueError('packaged frozen baseline manifest required')
    verify_baseline(ROOT,origin['baseline'])
    for name,sha in origin.get('source_files_sha256',{}).items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or file_hash(path)!=sha:raise ValueError('R3 source provenance mismatch')
    cfg=task_recipe(settings(args.config),args.experiment,args.stage);package=None
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2)
    if args.experiment=='ar':
        from ..v34.tasks import Heads
        from .ar import load_reference
        frozen,_=load_reference(args.parent,device);heads=Heads(device,'ar',frozen);groups,meta=load_groups(args,frozen)
    else:
        from .tasks import Heads
        if args.experiment=='ot':
            from ..v32.tracking import load_trackeval
            package=load_trackeval(args.trackeval)
        heads=Heads(device,args.experiment,cfg);groups,meta=load_groups(args,cfg)
    run_records(groups,meta,cfg,Path(args.out),origin,heads,args.experiment,package,stage=args.stage,parent=args.parent,scope=read(args.scope) if args.scope else None)


if __name__=='__main__':main()
