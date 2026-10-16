"""Continue immutable R2 artifacts without changing the scientific recipe."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from .protocol import ACTIONS,ROOT,digest,seal,settings,validate_cache,validate_origin


def validate_inherited(cache,cfg,experiment,task_models,sources,origin):
    rows=cache.get('rows',[]); keys=[(r['id'],r['codec'],r['qp']) for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError('duplicate inherited measurement condition')
    if (cache.get('settings')!=cfg or cache.get('experiment')!=experiment or cache.get('task_models')!=task_models
        or cache.get('sources')!=sources or cache.get('actions')!=[list(a) for a in ACTIONS]):
        raise ValueError('inherited settings, heads, source pixels or action bank mismatch')
    if 'sha256' in cache and cache['sha256']!=seal(cache)['sha256']:
        raise ValueError('inherited cache digest mismatch')
    validate_origin(cache.get('origin'))
    before=cache['origin'].get('source_files_sha256',{}); after=origin.get('source_files_sha256',{})
    required={f'adaptive_vcm/v34r2/{n}.py' for n in ('tasks','tracking','preprocessing','learning','model','protocol')}
    if not required.issubset(before): raise ValueError('inherited scientific source provenance missing')
    orchestration={'adaptive_vcm/v34r2/runner.py','adaptive_vcm/v34r2/evaluation.py'}
    for name,sha in before.items():
        if name.startswith('adaptive_vcm/') and name not in orchestration and after.get(name)!=sha:
            raise ValueError('inherited scientific source changed: '+name)
    expected={(c,q) for c in cfg['codecs'] for q in cfg['qps']}; completed={}
    for row in rows: completed.setdefault(row['id'],set()).add((row['codec'],row['qp']))
    if not rows or any(value!=expected for value in completed.values()):
        raise ValueError('inherited source has incomplete condition grid')
    subset=copy.deepcopy(cache); subset.pop('sha256',None)
    subset['sources']={k:v for k,v in sources.items() if k in completed}
    validate_cache(seal(subset))
    return {key:copy.deepcopy(row) for key,row in zip(keys,rows)}


def replay_identity(source,maps,codec,heads,cfg,prefix,row):
    from ..v34.preprocessing import filter_clip
    from .runner import measured_score
    if prefix.signature!=row['prefix_sha256']: raise ValueError('inherited identity prefix mismatch')
    packet=codec.roundtrip(filter_clip(source.rgb,maps,ACTIONS[0]),source_shape=source.rgb.shape,task=source.task)
    anchor=row['packets'][0]
    if (packet.packet_sha256!=anchor['sha256'] or packet.coded_bytes!=anchor['bytes']
        or hashlib.sha256(packet.decoded.tobytes()).hexdigest()!=anchor['decoded_sha256']):
        raise ValueError('inherited identity packet mismatch')
    value=measured_score(heads,packet.decoded,source,cfg,prefix=prefix)
    prefix.commit(value['probe'])


def source_manifest(groups,splits=('train','cal')):
    return {r.id:dict(split=split,domain=r.task,frames=len(r.rgb),pixel_sha256=hashlib.sha256(r.rgb.tobytes()).hexdigest(),
                     frame_recipe=r.frame_recipe) for split in splits for r in groups[split]}


def verify_cohorts(groups,expected):
    if source_manifest(groups,('train','cal','dev','test'))!=expected:
        raise ValueError('inherited cohort coverage or source pixels mismatch')


def main():
    from .runner import load_groups,run_records,load_checkpoint,write_json
    from .tasks import Heads
    p=argparse.ArgumentParser(); p.add_argument('--experiment',choices=['ot','ar_od_ot'],required=True)
    p.add_argument('--stage',choices=['measure-train','evaluate'],required=True)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--config',default=str(ROOT/'configs/v34_r2_qp4.json'))
    p.add_argument('--ablation',choices=['combined'],default='combined')
    p.add_argument('--codecs'); p.add_argument('--domains'); p.add_argument('--splits',default='dev,test')
    for key in ('ar_root','od_root','annotations','mot_root','trackeval','origin','out'):
        p.add_argument('--'+key.replace('_','-'),required=key in ('origin','out'))
    args=p.parse_args(); cfg=settings(args.config); cfg['ablation']=args.ablation
    origin=validate_origin(json.loads(Path(args.origin).read_text()),verify_files=True)
    cache_name='measurements.partial.json' if args.stage=='measure-train' else 'measurements.json'
    cache=json.loads((args.parent/cache_name).read_text())
    from ..v32.tracking import load_trackeval
    package=load_trackeval(args.trackeval)
    torch.set_num_threads(2); device='cuda' if torch.cuda.is_available() else 'cpu'
    groups,meta=load_groups(args,cfg)
    verify_cohorts(groups,json.loads((args.parent/'cohorts.json').read_text()))
    heads=Heads(device,args.experiment,cfg)
    validate_inherited(cache,cfg,args.experiment,heads.hashes(),source_manifest(groups),origin)
    if args.stage=='measure-train':
        run_records(groups,meta,cfg,Path(args.out),origin,heads,args.experiment,package,
                    inherited_cache=cache,training_only=True)
        return
    validate_cache(cache)
    models=[load_checkpoint(args.parent/f'controller_{s}.pth',device) for s in cfg['seeds']]
    if any(m.metadata['cache_sha256']!=cache['sha256'] for m in models):
        raise ValueError('checkpoint/inherited measurement cache mismatch')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise ValueError('immutable empty continuation output required')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'execution_origin.json',origin)
    write_json(out/'inherited_origin.json',cache['origin'])
    from .evaluation import evaluate
    transfer=None
    domains=tuple(args.domains.split(',')) if args.domains else None
    if args.experiment=='ar_od_ot' and (domains is None or 'ar' in domains):
        from ..analyzers import ActionAnalyzer
        transfer=ActionAnalyzer('r3d_18',device)
    evaluate(groups,meta,cfg,out,heads,models,args.experiment,package,transfer,
        splits=tuple(args.splits.split(',')),codecs=tuple(args.codecs.split(',')) if args.codecs else None,domains=domains)
    scope=json.loads((out/'execution_scope.json').read_text())
    write_json(out/'stage_complete.json',dict(status='complete',stage='evaluate',experiment=args.experiment,
        cache_sha256=cache['sha256'],execution_origin=origin,inherited_origin=cache['origin'],scope=scope))


if __name__=='__main__': main()
