"""Measured detached component targets with replayable causal prefix lineage."""
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from . import SCHEMA
from .protocol import ACTIONS,digest,codec_factory,task_mask
from .sources import validate_cohorts,group_id
from .mot_targets import read_context
from ..v34r2.preprocessing import Support
from ..v34r2.tracking import Prefix
from ..v34r2.runner import source_copy
from ..v34.preprocessing import filter_clip
from ..v34.runner import input_pair,infer
from ..v33ap.sources import serial


def seal(cache):
    value=serial({k:v for k,v in cache.items() if k!='sha256'})
    return {**value,'sha256':digest(value)}


def write_cache(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(serial(value),allow_nan=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def context_for(record):
    if record.task!='ot':return None
    if getattr(record,'split',None) not in ('train','cal'):raise ValueError('offline TRAIN/CAL context only')
    start=record.item['start']
    return read_context(Path(record.item['path'])/'gt/gt.txt',start,start+len(record.rgb))


def prepare_source(record,heads,task,cfg,clean,supports):
    source=source_copy(record);key=group_id(record)
    prefix=clean.setdefault(key,Prefix(context_frames=cfg['context_frames'])) if task=='ot' else None
    if prefix is not None and record.item.get('start')!=prefix.tracker.frame+1:
        raise ValueError('source tracker must receive contiguous causal windows')
    begin=time.perf_counter();heads.prepare(source,task,prefix)
    support=supports.setdefault(key,Support(**cfg['support']))
    maps=support.build(record.rgb,source.source_predictions['od'],task_mask(task),source.source_predictions.get('ot'))
    source.sender_support_seconds=time.perf_counter()-begin
    return source,maps


def measure(record,source,maps,codec,heads,recipe,*,prefix=None,annotations=None,
            gt_context=None,package=None,prefix_variant='identity',policy=None):
    if getattr(record,'split',None) not in ('train','cal'):raise ValueError('TRAIN/CAL targets only')
    if prefix_variant not in ('identity','rollout'):raise ValueError('unknown prefix policy')
    if prefix_variant=='rollout' and policy is None:raise ValueError('frozen source-only rollout policy required')
    before=prefix.signature if prefix is not None else digest({'source':record.id})
    packets=[];values=[];targets=[]
    for action in ACTIONS:
        begin=time.perf_counter();pixels=filter_clip(record.rgb,maps,action);pre=time.perf_counter()-begin
        packet=codec.roundtrip(pixels,source_shape=record.rgb.shape,task=record.task)
        value=heads.source_score(packet.decoded,source,prefix=prefix,package=package)
        target=heads.gt_score(packet.decoded,source,annotations=annotations,context=gt_context,package=package,
                             source_result=value,legacy_record=record,prefix=prefix)
        values.append(value);targets.append(target)
        packets.append(dict(bytes=packet.coded_bytes,elementary_bytes=packet.elementary_bytes,metadata_bytes=packet.metadata_bytes,
            sha256=packet.packet_sha256,decoded_sha256=hashlib.sha256(packet.decoded.tobytes()).hexdigest(),
            prefix_sha256=before,preprocess_seconds=pre,q8_edits=int(np.count_nonzero(pixels!=record.rgb)),
            context_metrics=target['context_metrics'],**codec.last_times))
    costs=np.array([v['source_cost'] for v in values]);gt=np.array([t['gt_cost'] for t in targets])
    rates=np.array([p['bytes']/packets[0]['bytes'] for p in packets])
    # Only these source-derived arrays cross into the rollout selector.
    choice=0 if policy is None else int(policy(dict(rate=rates.copy(),source_damage=costs-costs[0])))
    if not 0<=choice<len(ACTIONS):raise ValueError('invalid committed rollout action')
    if prefix is not None:prefix.commit(values[choice]['probe'])
    return serial(dict(rate=rates,source_cost=costs,source_damage=costs-costs[0],gt_cost=gt,gt_damage=gt-gt[0],
        legacy_source_cost=[v['legacy_source_cost'] for v in values],legacy_gt_cost=[t['legacy_gt_cost'] for t in targets],
        packets=packets,prefix_sha256=before,prefix_after=prefix.signature if prefix is not None else before,
        map_sha256=hashlib.sha256(maps.tobytes()).hexdigest(),
        selected_action=choice,prefix_variant=prefix_variant,
        target_scope='detached frozen task-component increments; GT separate from source consistency'))


def _key(row):return row['id'],row['codec'],row['qp'],row['prefix_variant']


def validate_cache(cache):
    if cache.get('schema')!=SCHEMA or cache.get('sha256')!=seal(cache)['sha256']:
        raise ValueError('R3 cache version/content mismatch')
    if cache.get('task') not in ('od','ot'):raise ValueError('R3 measurement cannot train AR or joint')
    rows=cache['rows'];keys=[_key(r) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('duplicate/conflicting measurement rows')
    cfg=cache['recipe'];variant=cache['variant'];sources=cache['sources']
    expected={(key,c,q,variant) for key in sources for c in cache['execution_scope']['codecs'] for q in cfg['qps']}
    if not set(keys).issubset(expected) or cache.get('complete') and set(keys)!=expected:
        raise ValueError('incomplete or foreign measurement grid')
    for row in rows:
        if row['split'] not in ('train','cal') or sources[row['id']]['split']!=row['split']:
            raise ValueError('heldout targets forbidden')
        gt=np.asarray(row['gt_damage'],float);source=np.asarray(row['source_damage'],float);rate=np.asarray(row['rate'],float)
        nsource=4 if cache['task']=='od' else 3
        if (gt.shape!=(13,3) or source.shape!=(13,nsource) or rate.shape!=(13,)
            or not np.isfinite(gt).all() or not np.isfinite(source).all() or not np.isfinite(rate).all()
            or (rate<=0).any() or not np.allclose(gt[0],0) or not np.allclose(source[0],0) or rate[0]!=1):
            raise ValueError('invalid detached components or identity anchor')
        for field,delta in [('gt_cost',gt),('source_cost',source)]:
            absolute=np.asarray(row[field],float)
            if (absolute.shape!=delta.shape or not np.isfinite(absolute).all()
                or not np.allclose(delta,absolute-absolute[0],rtol=0,atol=1e-10)):
                raise ValueError('detached target does not match its measured costs')
        packets=row['packets'];sizes=np.array([p['bytes'] for p in packets])
        if len(packets)!=13 or (sizes<=0).any() or not np.allclose(rate,sizes/sizes[0],rtol=0,atol=1e-10):
            raise ValueError('unmeasured packet rate')
        if (any(not re.fullmatch('[0-9a-f]{64}',row.get(k,'')) for k in ('prefix_sha256','prefix_after'))
            or any(p['prefix_sha256']!=row['prefix_sha256'] or any(not re.fullmatch('[0-9a-f]{64}',p.get(k,''))
                for k in ('sha256','decoded_sha256')) or p['bytes']!=p['elementary_bytes']+p['metadata_bytes'] for p in packets)):
            raise ValueError('candidate prefix/packet provenance mismatch')
    return cache


def resume(parent,expected):
    if isinstance(parent,(str,Path)):parent=json.loads(Path(parent).read_text(encoding='utf-8'))
    if parent.get('sha256')!=seal(parent)['sha256']:raise ValueError('partial cache content changed')
    for key in ('schema','task','recipe','origin','sources','variant','warm_checkpoint_sha256','task_models','execution_scope'):
        if parent.get(key)!=expected.get(key):raise ValueError('parent measurement lineage changed: '+key)
    validate_cache(parent)
    return {_key(r):r for r in parent['rows']}


def measure_stage(groups,meta,cfg,out,origin,heads,task,*,variant,warm_models=None,parent=None,package=None,scope=None):
    if task not in ('od','ot') or cfg.get('task')!=task:raise ValueError('standalone task recipe required')
    if variant not in ('identity','rollout') or task=='od' and variant!='identity':raise ValueError('unsupported measurement variant')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    audit=validate_cohorts(groups,meta.get('history',{}));splits=('train',) if task=='ot' and variant=='identity' else ('train','cal')
    sources={key:s for key,s in audit['sources'].items() if s['split'] in splits and s['domain']==task}
    annotation_index={}
    for annotation in meta.get('od',{}).get('annotations',[]):
        annotation_index.setdefault(annotation['image_id'],[]).append(annotation)
    annotation_files={}
    for split in splits:
        for record in groups[split]:
            if record.id not in sources:continue
            if task=='ot':
                path=Path(record.item['path'])/'gt/gt.txt'
                if path not in annotation_files:annotation_files[path]=hashlib.sha256(path.read_bytes()).hexdigest()
                sha=annotation_files[path]
            else:sha=digest(annotation_index.get(record.item['image_id'],[]))
            sources[record.id]['annotation_sha256']=sha
    codecs=list(cfg['codecs'] if scope is None else scope['codecs'])
    if not codecs or not set(codecs).issubset(cfg['codecs']):raise ValueError('invalid codec scope')
    warm=None
    if variant=='rollout':
        if not warm_models:raise ValueError('frozen warm checkpoint required')
        warm=next((m for m in warm_models if m.metadata['seed']==303401),None)
        if (warm is None or warm.metadata.get('checkpoint_role')!='WARM' or warm.metadata.get('task')!=task
            or warm.metadata.get('recipe')!=cfg or not re.fullmatch('[0-9a-f]{64}',getattr(warm,'checkpoint_sha256',''))):
            raise ValueError('locked matching seed-303401 warm policy required')
    cache=dict(schema=SCHEMA,task=task,recipe=cfg,origin=origin,sources=sources,variant=variant,
        warm_checkpoint_sha256=getattr(warm,'checkpoint_sha256',None),task_models=heads.hashes(),
        execution_scope=dict(codecs=codecs,splits=list(splits)),rows=[],complete=False)
    if parent is None and (out/'measurements.partial.json').is_file():parent=out/'measurements.partial.json'
    inherited=resume(parent,cache) if parent is not None else {};reused=0
    write_cache(out/'cohorts.json',audit)
    for split in splits:
        clean={};supports={};prefixes={};history={}
        records=[r for r in groups[split] if r.task==task]
        for index,r in enumerate(records):
            source,maps=prepare_source(r,heads,task,cfg,clean,supports);key=group_id(r)
            context=history.get(key,[])+context_for(r) if task=='ot' else None
            anns=annotation_index.get(r.item.get('image_id'),[]) if task=='od' else None
            for c in codecs:
                for q in cfg['qps']:
                    codec=codec_factory(task,c,q,cfg)
                    prefix=prefixes.setdefault((key,c,q),Prefix(context_frames=cfg['context_frames'])) if task=='ot' else None
                    rowkey=(r.id,c,q,variant);old=inherited.get(rowkey)
                    if old is not None:
                        if old.get('map_sha256')!=hashlib.sha256(maps.tobytes()).hexdigest():
                            raise ValueError('resumed support map changed')
                        if prefix is not None:
                            if prefix.signature!=old['prefix_sha256']:raise ValueError('resumed prefix starts from different history')
                            action=old['selected_action'];packet=codec.roundtrip(filter_clip(r.rgb,maps,ACTIONS[action]),source_shape=r.rgb.shape,task=task)
                            if packet.packet_sha256!=old['packets'][action]['sha256']:raise ValueError('resumed selected packet changed')
                            if hashlib.sha256(packet.decoded.tobytes()).hexdigest()!=old['packets'][action]['decoded_sha256']:
                                raise ValueError('resumed decoded pixels changed')
                            result=heads.source_score(packet.decoded,source,prefix=prefix,package=package);prefix.commit(result['probe'])
                            if prefix.signature!=old['prefix_after']:raise ValueError('resumed tracker commit changed')
                        cache['rows'].append(old);reused+=1;continue
                    policy=None
                    if warm is not None:
                        from .selection import shortlist
                        from ..v34r2.runner import condition
                        prediction=infer(warm,input_pair(r.rgb,maps),condition(c,q,task_mask(task),cfg,task))
                        candidates=shortlist(prediction,cfg,'cnn')
                        def policy(table,candidates=candidates):
                            indices=[0,*candidates];rate=table['rate'];damage=table['source_damage']
                            ok=(rate<=cfg['budget_ratio'])&(damage<=np.asarray(cfg['source_limits'])).all(1)
                            return min((a for a in indices if ok[a]),key=lambda a:(rate[a],a))
                    row=measure(r,source,maps,codec,heads,cfg,prefix=prefix,annotations=anns,gt_context=context,
                                package=package,prefix_variant=variant,policy=policy)
                    cache['rows'].append(dict(id=r.id,group=key,split=split,codec=c,qp=q,**row))
                    write_cache(out/'measurements.partial.json',seal(cache))
            if context is not None:history[key]=context[-cfg['context_frames']:]
            print(f'V34 R3 {task} measure-{variant} {split} {index+1}/{len(records)}',flush=True)
    cache.update(complete=True,reused_rows=reused,new_rows=len(cache['rows'])-reused)
    cache=seal(cache);validate_cache(cache);write_cache(out/'measurements.json',cache)
    return cache
