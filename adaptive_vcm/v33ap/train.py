"""Native shared-CNN training: real standard codec forward, fixed proxy backward."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from ..evaluate import write_json
from ..v32.evaluate import empty_output
from ..v32.transport import pack_packet
from ..v32vc.train import tensor,support_tensors
from ..v32vc.virtual import VirtualCodec
from ..v32rf.bridge import real_forward,substitute
from .protocol import settings,arm_weights,codec_factory
from .model import JointPreprocessor
from .sources import prepare_sources
from .tasks import TaskHeads,losses
from .calibration import calibrate_domain,diagnose
from .checkpoint import make_checkpoint


def temporal_loss(y,x,cuts):
    if y.shape[2]<2: return y.sum()*0
    delta=(y-x)[:,:,1:]-(y-x)[:,:,:-1]; mask=(~cuts[:,1:])[:,None,:,None,None].to(y)
    return (delta.square()*mask).sum()/(mask.sum()*y.shape[1]*y.shape[3]*y.shape[4]).clamp_min(1)


def exposure(splits,cfg,epoch=0):
    order=[(i,c,q) for task in ('ar','od') for i in splits[task]['fit'] for c in cfg['codecs'] for q in cfg['qps']]
    rng=np.random.default_rng(cfg['seed']+epoch)
    return [order[n] for n in rng.permutation(len(order))]


def training_step(model,virtual,heads,record,codec,arm,cfg):
    arm_weights(arm); device=next(model.parameters()).device
    x=tensor(record.rgb,device); p,m,cuts=support_tensors(record.support,x)
    y=model(x,codec.qp,int(codec.codec=='h265'),p,m,cuts)
    proxy,estimated=virtual(y,codec.qp,int(codec.codec=='h265'),cuts)
    actual,bpp,seconds=real_forward(y,codec,record.task); z=substitute(proxy,actual)
    header=len(pack_packet(b'x',task=record.task,codec=codec.codec,source_shape=record.rgb.shape,coded_shape=record.rgb.shape,fps=Fraction(cfg['fps'])))-1
    estimated=estimated+8*header/np.prod(record.rgb.shape[:3])
    terms=losses(heads,z,record,arm); distortion=F.mse_loss(z,x); temporal=temporal_loss(y,x,cuts)
    loss=terms['ar']+terms['od']+cfg['feature_weight']*terms['feature']+cfg['distortion_weight']*distortion+cfg['rate_weight']*estimated+cfg['temporal_weight']*temporal
    if not torch.isfinite(loss): raise RuntimeError('nonfinite training objective')
    return dict(loss=loss,ar_loss=terms['ar'],od_loss=terms['od'],feature_loss=terms['feature'],distortion=distortion,
                temporal_loss=temporal,estimated_bpp=estimated,actual_bpp=bpp,actual_codec_seconds=seconds,reconstruction=z)


def run(args):
    out=empty_output(args.out); cfg=settings(); arm_weights(args.arm)
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg['seed'])
    heads=TaskHeads(device); prepared=prepare_sources(args.ar_root,args.od_root,args.annotations,heads,cfg)
    records=prepared.pop('records'); supervised=prepared.pop('supervision')
    manifest=dict(prepared,arm=args.arm,settings=cfg,task_models=heads.hashes(),status='calibration',device=str(device))
    write_json(out/'supervision.json',supervised); write_json(out/'manifest.json',manifest)
    calibration={t:calibrate_domain(records,prepared['splits'][t]['cal'],t,cfg,device,out/t) for t in ('ar','od')}
    torch.manual_seed(cfg['seed']); model=JointPreprocessor(cfg['width']).to(device).train()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']); virtual={t:VirtualCodec(calibration[t]).to(device).train() for t in ('ar','od')}
    history=[]; step=0
    for epoch in range(cfg['epochs']):
        start=time.perf_counter(); totals={}; order=exposure(prepared['splits'],cfg,epoch)
        for identifier,c,qp in order:
            seed=cfg['seed']+step; torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            r=records[identifier]; codec=codec_factory(r.task)(c,qp)
            result=training_step(model,virtual[r.task],heads,r,codec,args.arm,cfg)
            optimizer.zero_grad(set_to_none=True); result['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step(); step+=1
            row={k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in result.items() if k!='reconstruction'}
            row.update(step=step,epoch=epoch+1,source_id=identifier,codec=c,qp=qp,gradient_norm=float(norm))
            with (out/'training.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
            for k in ('loss','ar_loss','od_loss','feature_loss','distortion','temporal_loss','estimated_bpp','actual_bpp','gradient_norm','actual_codec_seconds'): totals[k]=totals.get(k,0.)+row[k]
            if step%10==0: print(f'TRAIN {args.arm} step {step}/1920 loss {row["loss"]:.6f}',flush=True)
            del result
        if torch.cuda.is_available(): torch.cuda.synchronize()
        history.append(dict(epoch=epoch+1,steps=len(order),seconds=time.perf_counter()-start,**{k:v/len(order) for k,v in totals.items()}))
        write_json(out/'history.json',history)
    metadata=dict(prepared,arm=args.arm,settings=cfg,task_models=manifest['task_models'],calibration=calibration,history=history)
    torch.save(make_checkpoint(model,metadata),out/'shared_last.pth')
    model.eval(); diagnose(model,records,[i for t in ('ar','od') for i in prepared['splits'][t]['val']],calibration,heads,cfg,out)
    manifest.update(status='complete',calibration=calibration,history=history); write_json(out/'manifest.json',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--arm',choices=['joint','ar_only','od_only'],required=True)
    for name in ('ar-root','od-root','annotations','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device'); run(p.parse_args())


if __name__=='__main__': main()
