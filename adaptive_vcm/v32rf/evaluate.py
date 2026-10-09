"""Guarded RF families plus explicit forced-CNN and new-backbone diagnostics."""
import argparse
import json
from pathlib import Path
import time
from .checkpoint import load_checkpoint,validate_pair
from .warm import load_warm
from .protocol import codec_factory,execution
from .diagnostics import selector_diagnostics
from .swin import SwinAnalyzer
from ..v32.selection import Candidate
from ..v32.roi import build_grids,retarget
from ..v32.evaluate import run as evaluate
from ..v32.protocol import load_config
from ..evaluate import write_json

EXTRA_FAMILIES={
    'rf_warm':{'controls','rf_warm_plain'},'rf_proxy':{'controls','rf_proxy_plain'},'rf_real':{'controls','rf_real_plain'},
    'rf_raw_retarget':{'controls','rf_raw_retarget'},
    'rf_proxy_retarget':{'controls','rf_proxy_plain','rf_proxy_retarget'},
    'rf_real_retarget':{'controls','rf_real_plain','rf_real_retarget'},
    **{f'rf_{a}_forced':{'controls',f'rf_{a}_plain'} for a in ('warm','proxy','real')}}
FORCED={f'rf_{a}_forced':f'rf_{a}_rgb' for a in ('warm','proxy','real')}


def prepare_extension(rgb,support,codec,models,cfg):
    start=time.perf_counter(); pixels={a:models[a].pixels(rgb,support,codec) for a in ('warm','proxy','real')}
    candidates=[Candidate(f'rf_{a}_rgb',pixels[a],f'rf_{a}_plain') for a in pixels]
    for factor in cfg['roi']['factors']:
        grids=build_grids(rgb.shape,support,factor,cfg['roi']['margin'])
        if grids:
            candidates.append(Candidate(f'rf_raw_retarget_f{factor}',retarget(rgb,grids),'rf_raw_retarget',grids))
            for a in ('proxy','real'):
                candidates.append(Candidate(f'rf_{a}_retarget_f{factor}',retarget(pixels[a],grids),f'rf_{a}_retarget',grids))
    return dict(extra_candidates=candidates,extra_families=EXTRA_FAMILIES,forced_names=FORCED,unconditional_extra=True),time.perf_counter()-start


def run(args):
    import torch
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.transfer and (args.checkpoint or args.v30_runtime): raise ValueError('fast transfer excludes incompatible historical V30-B')
    if args.transfer and (args.split!='test' or args.count!=8): raise ValueError('locked TEST8 transfer required')
    models={a:load_checkpoint(getattr(args,a+'_checkpoint'),args.task,device) for a in ('proxy','real')}; validate_pair(models)
    models['warm']=load_warm(args.warm_checkpoint,args.warm_runtime,args.task,device)
    if models['warm'].rf_warm_manifest!=models['real'].metadata['warm']: raise ValueError('evaluation warm start differs')
    preset='fast' if args.transfer else 'medium'
    plugin=dict(models=models,families=EXTRA_FAMILIES,prepare=prepare_extension,
        evaluators=(lambda d:{'swin3d_t':SwinAnalyzer(d)}) if args.task=='ar' else None,
        manifest=dict(scope='v32rf_paired_forward',v32rf={a:m.manifest for a,m in models.items()},
            codec_execution=execution(args.task,preset),extra_arms=list(EXTRA_FAMILIES),
            ar_evaluators=['r2plus1d_18','r3d_18','swin3d_t'] if args.task=='ar' else None,
            comparison_limit='Original FIT48 warm CNN plus two RF epochs versus historical V30-B TRAIN32; exploratory',
            swin_protocol='Official K400 spatial transform, same received 16-frame clip; not official 32-frame/12-clip benchmark'))
    summary=evaluate(args,extension=plugin,codec_factory=codec_factory(args.task,preset))
    audits=[json.loads(line) for line in (args.out/'selection_audit.jsonl').read_text().splitlines()]
    write_json(args.out/'selector_diagnostics.json',selector_diagnostics(audits,load_config(),args.task))
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--task',choices=['ar','od'],required=True)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--annotations',type=Path); p.add_argument('--config',type=Path)
    p.add_argument('--split',choices=['dev','test'],default='dev'); p.add_argument('--count',type=int,default=32); p.add_argument('--bootstrap',type=int,default=0)
    p.add_argument('--device',choices=['cpu','cuda']); p.add_argument('--checkpoint',type=Path); p.add_argument('--v30-runtime',type=Path)
    for a in ('proxy','real','warm'): p.add_argument('--'+a+'-checkpoint',type=Path,required=True)
    p.add_argument('--warm-runtime',type=Path,required=True); p.add_argument('--transfer',action='store_true'); p.add_argument('--out',type=Path,required=True)
    return p


if __name__=='__main__': run(parser().parse_args())
