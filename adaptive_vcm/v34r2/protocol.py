"""Versioned R2 contracts; no R1 recipe is accepted implicitly."""
import json
import hashlib
import re
from pathlib import Path
import numpy as np
from . import SCHEMA
from ..v34.protocol import ACTIONS, TASKS, digest, codec_factory

ROOT=Path(__file__).resolve().parents[2]


def _hash(value,length=64):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{'+str(length)+'}',value) is not None


def validate_origin(origin,verify_files=False):
    if not isinstance(origin,dict) or not _hash(origin.get('origin_commit'),40) or not _hash(origin.get('bundle_sha256')):
        raise ValueError('invalid R2 origin commit/bundle provenance')
    if verify_files:
        files=origin.get('source_files_sha256',{})
        required={p.relative_to(ROOT).as_posix() for p in (ROOT/'adaptive_vcm/v34r2').glob('*.py')}|{'configs/v34_r2_qp4.json'}
        if not isinstance(files,dict) or not required.issubset(files): raise ValueError('R2 origin source files missing')
        for name,expected in files.items():
            path=(ROOT/name).resolve()
            if not path.is_relative_to(ROOT) or not _hash(expected) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('R2 origin source mismatch: '+name)
    return origin


def settings(path=None):
    cfg=json.loads((Path(path) if path else ROOT/'configs/v34_r2_qp4.json').read_text())
    if (cfg['schema']!=SCHEMA or cfg['revision']!='r2-qp4' or cfg['qps']!=[30,35,40,45]
        or cfg['codecs']!=['h264','h265'] or cfg['seeds']!=[303401,303402,303403]
        or cfg['frames']!=16 or cfg['preset']!='medium' or cfg['budget_ratio']<1
        or cfg['source_shortlist']!=3 or cfg['context_frames']<cfg['frames']
        or cfg['support_policy'] not in ('r2','legacy') or cfg['target_policy'] not in ('r2','legacy')):
        raise ValueError('R2 recipe drift')
    for key in ('damage_tolerance','source_guard_limits'):
        a=np.asarray(cfg[key],float)
        if a.shape!=(3,) or not np.isfinite(a).all() or (a<0).any(): raise ValueError('invalid task limits')
    assigned=[s for values in cfg['ot_sequences'].values() for s in values]
    if len(assigned)!=len(set(assigned)) or set(cfg['ot_sequences'])!={'train','cal','dev','test'}:
        raise ValueError('OT sequences must be disjoint by split')
    return cfg


def task_mask(experiment,domain,frames):
    if experiment in TASKS: return [int(t==experiment) for t in TASKS]
    if experiment!='ar_od_ot': raise ValueError('unsupported R2 experiment')
    return [int(frames>1),1,int(frames>1)]


def seal(cache):
    return {**cache,'sha256':digest({k:v for k,v in cache.items() if k!='sha256'})}


def validate_cache(cache):
    if cache.get('schema')!=SCHEMA or cache.get('sha256')!=seal(cache)['sha256']:
        raise ValueError('R2 cache schema/content mismatch')
    if cache.get('experiment') not in ('od','ot','ar_od_ot'): raise ValueError('invalid experiment')
    if cache.get('actions')!=[list(a) for a in ACTIONS]: raise ValueError('action bank provenance mismatch')
    validate_origin(cache.get('origin'))
    models=cache.get('task_models',{})
    if not isinstance(models,dict) or any(not isinstance(models.get(t),dict) or not _hash(models[t].get('state_sha256')) or not models[t].get('name') for t in ('ar','od')):
        raise ValueError('frozen task model provenance missing')
    cfg=cache['settings']; rows=cache['rows']; sources=cache['sources']
    if cfg.get('schema')!=SCHEMA or cfg.get('revision')!='r2-qp4': raise ValueError('cache recipe mismatch')
    keys=[(r['id'],r['codec'],r['qp']) for r in rows]
    expected={(s,c,q) for s in sources for c in cfg['codecs'] for q in cfg['qps']}
    if not rows or len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError('incomplete measured grid')
    for r in rows:
        if r['split'] not in ('train','cal') or sources[r['id']]['split']!=r['split']:
            raise ValueError('held-out supervision forbidden')
        source=sources[r['id']]
        if source.get('domain')!=r['domain'] or not _hash(source.get('pixel_sha256')):
            raise ValueError('source domain/pixels mismatch')
        frames=source.get('frames')
        if frames != (1 if r['domain']=='od' else cfg['frames']) or r['domain'] not in TASKS:
            raise ValueError('invalid source frame recipe')
        recipe=source.get('frame_recipe',{})
        if 'frame_indices' in recipe and len(recipe['frame_indices'])!=frames: raise ValueError('frame count mismatch')
        if r['active']!=task_mask(cache['experiment'],r['domain'],frames): raise ValueError('active task mask disagrees with source frames')
        active=np.asarray(r['active'],bool); gt=np.asarray(r['gt_mask'],bool)
        if active.shape!=(3,) or gt.shape!=(3,) or not active.any() or (gt&~active).any(): raise ValueError('invalid masks')
        if r['gt_mask'] != [int(r['domain']==t) for t in TASKS]: raise ValueError('invented task GT')
        if cache['experiment'] in TASKS and (r['domain']!=cache['experiment'] or not np.array_equal(active,gt)):
            raise ValueError('single-task mask mismatch')
        rate=np.asarray(r['rate'],float); packets=r['packets']
        if rate.shape!=(len(ACTIONS),) or len(packets)!=len(ACTIONS) or not np.isfinite(rate).all() or (rate<=0).any():
            raise ValueError('invalid measured rates')
        sizes=np.array([p['bytes'] for p in packets],float)
        if (sizes<=0).any() or not np.allclose(rate,sizes/sizes[0],rtol=0,atol=1e-10): raise ValueError('unmeasured rate')
        for field,mask in [('gt_damage',gt),('source_damage',active)]:
            a=np.array(r[field],float)
            if (a.shape!=(len(ACTIONS),3) or not np.isfinite(a[:,mask]).all()
                or not np.isnan(a[:,~mask]).all() or not np.allclose(a[0,mask],0)):
                raise ValueError('invalid detached target mask/anchor')
        if not _hash(r.get('prefix_sha256')): raise ValueError('missing common prefix provenance')
        for p in packets:
            if (type(p['bytes']) is not int or not _hash(p.get('sha256')) or not _hash(p.get('decoded_sha256'))
                or p.get('prefix_sha256')!=r['prefix_sha256']): raise ValueError('packet/prefix provenance mismatch')
    return cache
