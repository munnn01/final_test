"""FIT48 restoration after actual decoding; CNN/codec/task/backbone stay fixed."""
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from .post import SwinRestorer,RECIPE,make_checkpoint
from .checkpoint import load_checkpoint
from .train import prepare_training,parser
from .bridge import real_forward
from .protocol import codec_factory
from .features import feature_vectors,feature_distance
from ..v32.evaluate import empty_output
from ..v32.protocol import canonical_hash
from ..v32vc.train import tensor,support_tensors,task_loss
from ..evaluate import write_json


def post_step(model,teachers,decoded,source,target,targets):
    restored=model(decoded.detach())
    task=task_loss('ar',teachers,restored,target); distortion=F.mse_loss(restored,source)
    feature=feature_distance(feature_vectors('ar',teachers,restored),targets)
    loss=task+10*distortion+.1*feature
    if not torch.isfinite(loss): raise RuntimeError('nonfinite restoration loss')
    return dict(loss=loss,task_loss=task,distortion=distortion,feature_loss=feature)


def run(args):
    if args.task!='ar': raise ValueError('Video Swin restoration experiment is AR-only')
    out=empty_output(args.out); device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2); torch.manual_seed(RECIPE['seed']); np.random.seed(RECIPE['seed'])
    cfg,warm,teachers,records,features,meta,manifest=prepare_training(args,device,out)
    rf=load_checkpoint(args.real_checkpoint,'ar',device); rf.model.requires_grad_(False)
    if rf.metadata['warm']!=warm.rf_warm_manifest or rf.metadata['feature_targets_sha256']!=manifest['feature_targets_sha256']:
        raise ValueError('restoration RF warm/feature provenance differs')
    model=SwinRestorer().to(device).train(); optimizer=torch.optim.Adam(model.head.parameters(),lr=RECIPE['lr'])
    groups=[(c,q) for c in cfg['codecs'] for q in cfg['qps']]; rng=np.random.default_rng(RECIPE['seed']); history=[]; cache={}; step=0
    write_json(out/'manifest.json',dict(status='training',recipe=RECIPE,rf_real=rf.manifest,training=manifest))
    for epoch in range(2):
        start=time.perf_counter(); totals={k:0. for k in ('loss','task_loss','distortion','feature_loss','gradient_norm')}
        for index in rng.permutation(48):
            identifier=manifest['fit_ids'][int(index)]; r=records[identifier]; source=tensor(r['rgb'],device)
            targets=[v.to(device) for v in features[identifier]]
            for group_index in rng.permutation(10):
                c,q=groups[int(group_index)]; key=(identifier,c,q); codec=codec_factory('ar')(c,q)
                if key not in cache:
                    with torch.no_grad(): y=rf.model(source,q,int(c=='h265'),*support_tensors(r['support'],source))
                    decoded,bpp,seconds=real_forward(y,codec,'ar'); cache[key]=(decoded.cpu(),bpp,seconds)
                decoded,bpp,seconds=cache[key]; terms=post_step(model,teachers,decoded.to(device),source,r['item']['label'],targets)
                optimizer.zero_grad(set_to_none=True); terms['loss'].backward()
                norm=torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.,error_if_nonfinite=True); optimizer.step(); step+=1
                row=dict(step=step,epoch=epoch+1,source_id=identifier,codec=c,qp=q,paid_bpp=bpp,codec_seconds=seconds,
                    gradient_norm=float(norm),**{k:float(v.detach()) for k,v in terms.items()})
                with (out/'post_training.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
                for k in totals: totals[k]+=row[k]
                if step%50==0: print(json.dumps(row),flush=True)
        history.append(dict(epoch=epoch+1,steps=480,seconds=time.perf_counter()-start,**{k:v/480 for k,v in totals.items()}))
    torch.save(make_checkpoint(model,rf.manifest,history),out/'post_last.pth')
    write_json(out/'manifest.json',dict(status='complete',recipe=RECIPE,rf_real=rf.manifest,training=manifest,history=history,
        checkpoint_bytes=(out/'post_last.pth').stat().st_size,rate_scope='same transmitted packets; preinstalled restoration weights reported separately'))


if __name__=='__main__':
    p=parser(); p.add_argument('--real-checkpoint',required=True,type=__import__('pathlib').Path); run(p.parse_args())
