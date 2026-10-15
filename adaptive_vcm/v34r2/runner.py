"""Measure fresh detached targets, train R2 models, then evaluate paid packets."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from . import SCHEMA
from .protocol import ACTIONS,TASKS,settings,task_mask,seal,validate_cache,validate_origin,digest,codec_factory
from .preprocessing import Support
from .tracking import Prefix
from .tasks import Heads
from .model import Controller
from .learning import train,calibration,fixed_shortlist
from ..v34.runner import input_pair,write_json,infer
from ..v34.preprocessing import region_maps,filter_clip
from ..v33ap.sources import SourceRecord
from ..v32vc.checkpoint import state_hash


def condition(codec,qp,mask,cfg,domain):
    return [float(codec=='h265'),qp/51,*map(float,mask),cfg['budget_ratio'],*[float(domain==t) for t in TASKS]]


def stream_key(record):
    return record.item.get('sequence',record.id)


def source_copy(record):
    # Whitelist source fields. Annotation dictionaries and labels cannot cross this API.
    item={k:record.item[k] for k in ('sequence','start','width','height','image_id') if k in record.item}
    return SourceRecord(record.id,record.task,item,record.rgb,record.geometry,record.frame_recipe)


def prepare_source(record,heads,experiment,cfg,source_prefixes,supports):
    mask=task_mask(experiment,record.task,len(record.rgb)); key=stream_key(record)
    source=source_copy(record)
    prefix=source_prefixes.setdefault(key,Prefix(context_frames=cfg['context_frames'])) if mask[2] else None
    start=time.perf_counter(); heads.prepare(source,mask,prefix)
    if cfg['support_policy']=='legacy': maps=region_maps(record.rgb,source.source_predictions['od'])
    else:
        support=supports.setdefault(key,Support(**cfg['support']))
        maps=support.build(record.rgb,source.source_predictions['od'],mask,source.source_predictions.get('ot'))
    source.sender_support_seconds=time.perf_counter()-start
    return source,maps


def measured_score(heads,rgb,source,cfg,**kwargs):
    value=heads.score(rgb,source,**kwargs)
    if cfg['target_policy']=='legacy':
        # Controlled target/guard ablation: same heads, profiles and prefix engine;
        # legacy one-way OD matching and per-window local OT cost.
        from ..v34.tasks import detection_distance as legacy_distance
        from ..v34.tracking import tracking_cost,track_frames
        if source.mask[1]: value['source_cost'][1]=legacy_distance(source.source_predictions['od'],value['predictions']['od'])
        if source.mask[2]:
            local=track_frames(value['predictions']['od'])
            value['source_cost'][2]=tracking_cost(local,source.source_predictions['ot'])
            gt=kwargs.get('gt')
            if gt is not None and gt.task=='ot': value['gt_cost'][2]=tracking_cost(local,gt.targets,ground_truth=True)
        if source.mask[0] and kwargs.get('gt') is not None and kwargs['gt'].task=='ar':
            value['gt_cost'][0]+=value['source_cost'][0]
    return value


def measure(record,source,maps,codec,heads,cfg,*,prefix=None,gt_context=None,package=None):
    if getattr(record,'split',None) not in ('train','cal'): raise ValueError('TRAIN/CAL measurements only')
    before=prefix.signature if prefix is not None else digest({'source':record.id})
    packets=[]; scores=[]
    for action in ACTIONS:
        start=time.perf_counter(); pixels=filter_clip(record.rgb,maps,action); preprocessing=time.perf_counter()-start
        packet=codec.roundtrip(pixels,source_shape=record.rgb.shape,task=record.task)
        values=measured_score(heads,packet.decoded,source,cfg,prefix=prefix,gt=record,gt_context=gt_context,package=package)
        scores.append(values)
        packets.append(dict(bytes=packet.coded_bytes,elementary_bytes=packet.elementary_bytes,metadata_bytes=packet.metadata_bytes,
            sha256=packet.packet_sha256,decoded_sha256=hashlib.sha256(packet.decoded.tobytes()).hexdigest(),
            prefix_sha256=before,q8_edits=int(np.count_nonzero(pixels!=record.rgb)),preprocess_seconds=preprocessing,
            **codec.last_times,context_metrics=values['context_metrics']))
    if prefix is not None: prefix.commit(scores[0]['probe'])
    def values(key): return [s[key] for s in scores]
    def delta(key): return [[None if v is None else float(v-scores[0][key][k]) for k,v in enumerate(s[key])] for s in scores]
    return dict(rate=[p['bytes']/packets[0]['bytes'] for p in packets],gt_damage=delta('gt_cost'),source_damage=delta('source_cost'),
        gt_cost=values('gt_cost'),source_cost=values('source_cost'),prefix_sha256=before,packets=packets,
        target_scope='detached frozen task costs; OT finite-context positive-pedestrian HOTA; not full-sequence HOTA',
        prefix_policy='locked identity prefix for all TRAIN/CAL candidates; runtime uses selected prefix')


def save_checkpoint(path,model,metadata):
    meta=json.loads(json.dumps(metadata,allow_nan=False)); state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(dict(schema=SCHEMA,actions=model.actions,conditions=model.conditions,metadata=meta,metadata_sha256=digest(meta),
        state=state,state_sha256=state_hash(state)),path)


def load_checkpoint(path,device='cpu'):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if (value.get('schema')!=SCHEMA or digest(value['metadata'])!=value['metadata_sha256']
        or state_hash(value['state'])!=value['state_sha256']): raise ValueError('R2 checkpoint provenance mismatch')
    model=Controller(value['actions'],value['conditions']); model.load_state_dict(value['state'],strict=True)
    model.to(device).eval(); model.metadata=value['metadata']; model.checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return model


def train_models(cache,inputs,cfg,out,device,heads):
    validate_cache(cache)
    if cache['settings']!=cfg or cache['task_models']!=heads.hashes(): raise ValueError('cache/current recipe or frozen heads mismatch')
    fit=[r for r in cache['rows'] if r['split']=='train']; cal=[r for r in cache['rows'] if r['split']=='cal']
    fixed=fixed_shortlist(fit,cfg); models=[]
    for seed in cfg['seeds']:
        np.random.seed(seed); torch.manual_seed(seed)
        model=Controller(len(ACTIONS)); history,scales=train(model,fit,inputs,cfg,device)
        predictions=[infer(model,inputs[r['id']],r['condition']) for r in cal]
        meta=dict(settings=cfg,experiment=cache['experiment'],seed=seed,cache_sha256=cache['sha256'],origin=cache['origin'],
            sources=cache['sources'],task_models=heads.hashes(),scales=scales,scales_fit_split='train',fixed_shortlist=fixed,
            calibration=calibration(cal,predictions,cfg['cal_coverage']),history=history,codec_backpropagation=False,
            checkpoint_role='LAST',inputs_sha256=hashlib.sha256((out/'controller_inputs.pth').read_bytes()).hexdigest())
        save_checkpoint(out/f'controller_{seed}.pth',model,meta); write_json(out/f'training_{seed}.json',meta)
        write_json(out/f'cal_predictions_{seed}.json',[dict(id=r['id'],group=r['group'],codec=r['codec'],qp=r['qp'],**p) for r,p in zip(cal,predictions)])
        models.append(load_checkpoint(out/f'controller_{seed}.pth',device))
    return models


def run_records(groups,meta,cfg,out,origin,heads,experiment,package=None,transfer_ar=None):
    validate_origin(origin,verify_files=True)
    out=Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError('immutable empty R2 output required')
    out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(2)
    sources={}; pixels=set(); sequence_splits={}
    for split,records in groups.items():
        for r in records:
            sha=hashlib.sha256(r.rgb.tobytes()).hexdigest()
            if r.id in sources or sha in pixels: raise ValueError('duplicate source pixels/ID')
            if r.task=='ot':
                seq=stream_key(r)
                if seq in sequence_splits and sequence_splits[seq]!=split: raise ValueError('OT sequence split leakage')
                sequence_splits[seq]=split
            sources[r.id]=dict(split=split,domain=r.task,frames=len(r.rgb),pixel_sha256=sha,frame_recipe=r.frame_recipe)
            pixels.add(sha)
    write_json(out/'cohorts.json',sources); write_json(out/'origin.json',origin)
    cache=dict(schema=SCHEMA,experiment=experiment,settings=cfg,origin=origin,task_models=heads.hashes(),actions=[list(a) for a in ACTIONS],
        sources={k:v for k,v in sources.items() if v['split'] in ('train','cal')},rows=[])
    inputs={}
    for split in ('train','cal'):
        clean={}; supports={}; prefixes={}; gt_history={}
        for index,r in enumerate(groups[split]):
            source,maps=prepare_source(r,heads,experiment,cfg,clean,supports)
            inputs[r.id]=input_pair(r.rgb,maps); key=stream_key(r)
            context=gt_history.get(key,[])+r.targets if r.task=='ot' else None
            for codec in cfg['codecs']:
                for qp in cfg['qps']:
                    prefix=prefixes.setdefault((key,codec,qp),Prefix(context_frames=cfg['context_frames'])) if source.mask[2] else None
                    table=measure(r,source,maps,codec_factory(r.task,codec,qp,cfg),heads,cfg,prefix=prefix,gt_context=context,package=package)
                    cache['rows'].append(dict(id=r.id,group=key,split=split,domain=r.task,codec=codec,qp=qp,active=source.mask,
                        gt_mask=[int(r.task==t) for t in TASKS],condition=condition(codec,qp,source.mask,cfg,r.task),**table))
            if context is not None: gt_history[key]=context[-cfg['context_frames']:]
            write_json(out/'measurements.partial.json',cache)
            print(f'V34 R2 {experiment} measure {split} {index+1}/{len(groups[split])}',flush=True)
    cache=seal(cache); validate_cache(cache); write_json(out/'measurements.json',cache)
    torch.save(inputs,out/'controller_inputs.pth')
    models=train_models(cache,inputs,cfg,out,heads.device,heads)
    from .evaluation import evaluate
    evaluate(groups,meta,cfg,out,heads,models,experiment,package,transfer_ar)
    write_json(out/'complete.json',dict(status='complete',schema=SCHEMA,experiment=experiment,cache_sha256=cache['sha256'],
        checkpoint_sha256=[m.checkpoint_sha256 for m in models],scope=cfg['holdout_status'],quality_improvement_confirmed=False))


def load_groups(args,cfg):
    from ..v34.sources import cohorts
    groups,meta=cohorts(args.ar_root,args.od_root,args.annotations,cfg,'ar_od' if args.experiment=='ar_od_ot' else args.experiment)
    if args.experiment in ('ot','ar_od_ot'):
        from ..v32.tracking import plan_sequences,read_sequence
        from ..v34.tracking import gt_window
        plans=[]
        for split,names in cfg['ot_sequences'].items():
            for item in plan_sequences(Path(args.mot_root),names,cfg['ot_frames']):
                if item['frames']!=cfg['ot_frames'] or item['frames']%cfg['frames']: raise ValueError('complete MOT windows required')
                plans.append(item); rgb,g=read_sequence(item,cfg['ot_size'])
                for offset in range(0,len(rgb),cfg['frames']):
                    start=offset+1; source=dict(id=f"{item['name']}/{start:06}",sequence=item['name'],start=start,
                        width=item['width'],height=item['height'],path=str(item['path']))
                    r=SourceRecord('ot/'+source['id'],'ot',source,rgb[offset:offset+cfg['frames']].copy(),g,
                        dict(kind='contiguous-MOT20',frame_indices=list(range(start,start+cfg['frames'])),repeated_frames=0))
                    r.split=split
                    if split in ('train','cal'): r.targets=gt_window(item['path']/'gt/gt.txt',start,start+cfg['frames'],g)
                    groups[split].append(r)
        meta['ot']=plans
    return groups,meta


def main():
    p=argparse.ArgumentParser(); p.add_argument('--experiment',choices=['od','ot','ar_od_ot'],required=True)
    p.add_argument('--config',default=str(Path(__file__).resolve().parents[2]/'configs/v34_r2_qp4.json'))
    p.add_argument('--ablation',choices=['baseline','combined','map_only','target_only'],default='combined')
    for key in ('ar_root','od_root','annotations','mot_root','trackeval','origin','out'):
        p.add_argument('--'+key.replace('_','-'),required=key in ('origin','out'))
    args=p.parse_args(); cfg=settings(args.config)
    origin=validate_origin(json.loads(Path(args.origin).read_text()),verify_files=True)
    cfg['ablation']=args.ablation
    if args.ablation in ('baseline','map_only'): cfg['target_policy']='legacy'
    if args.ablation in ('baseline','target_only'): cfg['support_policy']='legacy'
    package=None
    if args.experiment in ('ot','ar_od_ot'):
        from ..v32.tracking import load_trackeval
        package=load_trackeval(args.trackeval)
    groups,meta=load_groups(args,cfg)
    device='cuda' if torch.cuda.is_available() else 'cpu'; heads=Heads(device,args.experiment,cfg)
    transfer_ar=None
    if args.experiment=='ar_od_ot':
        from ..analyzers import ActionAnalyzer
        transfer_ar=ActionAnalyzer('r3d_18',device)
    run_records(groups,meta,cfg,Path(args.out),origin,heads,args.experiment,package,transfer_ar)


if __name__=='__main__': main()
