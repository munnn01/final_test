"""Paired proxy/actual-forward fits and held-out TRAIN proxy diagnostics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from .bridge import real_forward,substitute,pixels
from .features import feature_vectors,feature_distance
from .protocol import settings,split_cal,codec_factory,execution
from .checkpoint import make_checkpoint
from .warm import load_warm
from ..analyzers import ActionAnalyzer,DetectionAnalyzer
from ..data import ar_plan,od_plan,read_video,read_image
from ..evaluate import write_json
from ..v32.evaluate import empty_output,prepare_source,SourceAudit
from ..v32.protocol import load_config,canonical_hash,provenance
from ..v32vc.model import ResidualPreprocessor
from ..v32vc.virtual import VirtualCodec
from ..v32vc.train import tensor,support_tensors,od_targets,task_loss,calibrate


def training_step(model,virtual,teachers,task,x,support,target,source_features,codec,arm,recipe):
    if arm not in ('proxy','real'): raise ValueError('RF forward arm required')
    p,m,cuts=support; y=model(x,codec.qp,int(codec.codec=='h265'),p,m,cuts)
    proxy,rate=virtual(y,codec.qp,int(codec.codec=='h265'),cuts)
    # Plain paid packets have a fixed 40-byte header, counted at source resolution.
    rate=rate+320/(x.shape[2]*x.shape[3]*x.shape[4])
    actual_bpp=None; seconds=0.
    if arm=='real':
        actual,actual_bpp,seconds=real_forward(y,codec,task); decoded=substitute(proxy,actual)
    else: decoded=proxy
    supervised=task_loss(task,teachers,decoded,target); distortion=F.mse_loss(decoded,x)
    feature=feature_distance(feature_vectors(task,teachers,decoded),source_features)
    loss=supervised+recipe['rate_weight']*rate+recipe['distortion_weight']*distortion+recipe['feature_weight']*feature
    if not torch.isfinite(loss): raise RuntimeError('nonfinite RF objective')
    return dict(loss=loss,task_loss=supervised,feature_loss=feature,distortion=distortion,
        estimated_bpp=rate,actual_bpp=actual_bpp,actual_codec_seconds=seconds)


def diagnose_point(model,virtual,teachers,task,x,support,codec):
    model_states=[(m,m.training) for m in model.modules()]; virtual_state=virtual.training
    try:
        model.eval(); virtual.eval(); p,m,cuts=support
        with torch.no_grad():
            y=model(x,codec.qp,int(codec.codec=='h265'),p,m,cuts)
            approximate,rate=virtual(y,codec.qp,int(codec.codec=='h265'),cuts)
            actual,actual_rate,seconds=real_forward(y,codec,task)
            mse=float(F.mse_loss(approximate,actual))
            feature=float(feature_distance(feature_vectors(task,teachers,approximate),feature_vectors(task,teachers,actual)))
        # Input-only rate derivative: diagnostic must not accumulate parameter gradients.
        editable=y.detach().requires_grad_(True); virtual.train()
        with torch.random.fork_rng(devices=[x.device.index] if x.is_cuda else []):
            torch.manual_seed(303231)
            _,noisy_rate=virtual(editable,codec.qp,int(codec.codec=='h265'),cuts)
            gradient=torch.autograd.grad(noisy_rate,editable)[0]
        direction=gradient/(gradient.square().mean().sqrt()+1e-8)
        changed=(y-.01*(1-p)*(1-.75*m)*direction).clamp(0,1).detach()
        changed_pixels=not np.array_equal(pixels(y),pixels(changed))
        virtual.eval()
        with torch.no_grad():
            _,new_estimate=virtual(changed,codec.qp,int(codec.codec=='h265'),cuts)
            _,new_actual,_=real_forward(changed,codec,task)
        header=320/(x.shape[2]*x.shape[3]*x.shape[4]); predicted=float(rate)+header
        proxy_delta=float(new_estimate-rate); real_delta=new_actual-actual_rate
        return dict(pixel_mse=mse,psnr_db=-10*np.log10(mse) if mse>0 else None,perfect_pixel_match=mse==0,
            feature_cosine_distance=feature,estimated_bpp=predicted,real_bpp=actual_rate,
            rate_relative_error=abs(predicted-actual_rate)/actual_rate,actual_codec_seconds=seconds,
            perturbation_changed_pixels=changed_pixels,proxy_rate_delta=proxy_delta,real_rate_delta=real_delta,
            rate_descent_sign_agreement=bool(np.sign(proxy_delta)==np.sign(real_delta)) if changed_pixels else None)
    finally:
        virtual.train(virtual_state)
        for module,state in model_states: module.training=state


def prepare_training(args,device,out):
    cfg=load_config(); warm=load_warm(args.warm_checkpoint,args.warm_runtime,args.task,device)
    if args.task=='ar':
        plan,meta=ar_plan(args.root,'train',64); teachers=[ActionAnalyzer(n,device) for n in cfg['ar_teachers']]
    else:
        if not args.annotations: raise ValueError('OD annotations required')
        plan,meta=od_plan(args.root,args.annotations,'train',64); teachers=[DetectionAnalyzer(cfg['od_teacher'],device)]
    records={}; audit=SourceAudit(); supervision={}; features={}; feature_hashes={}
    for index,item in enumerate(plan):
        if args.task=='ar': rgb=read_video(item['path'],cfg['frames'],cfg['ar_size'],cfg['temporal_stride']); geometry=None
        else: rgb,geometry=read_image(item,cfg['od_size'])
        audit.add(item['id'],rgb); _,support,_=prepare_source(rgb,args.task,cfg,teachers)
        records[item['id']]=dict(rgb=rgb,support=support,item=item,geometry=geometry)
        print(f'RF source preparation {index+1}/64',flush=True)
    if audit.hashes!=warm.metadata['source_pixel_sha256']: raise ValueError('RF source pixels differ from warm TRAIN64')
    cal,val=split_cal(warm.metadata['cal_ids'],args.task); fit=warm.metadata['fit_ids']
    for identifier in fit:
        r=records[identifier]
        supervision[identifier]=r['item']['label'] if args.task=='ar' else {k:v.tolist() for k,v in od_targets(meta,r['item'],r['geometry'],'cpu').items()}
    ann=hashlib.sha256(args.annotations.read_bytes()).hexdigest() if args.task=='od' else None
    if canonical_hash(supervision)!=warm.metadata['supervision_sha256'] or ann!=warm.metadata['annotations_sha256']:
        raise ValueError('RF supervised targets differ from original FIT48')
    for identifier in fit+val:
        with torch.no_grad(): targets=feature_vectors(args.task,teachers,tensor(records[identifier]['rgb'],device))
        features[identifier]=[v.detach().cpu() for v in targets]
        feature_hashes[identifier]=[hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in features[identifier]]
    write_json(out/'fit_supervision.json',supervision); write_json(out/'feature-targets.json',feature_hashes)
    manifest=dict(task=args.task,settings=settings(),warm=warm.rf_warm_manifest,fit_ids=fit,cal_ids=cal,proxy_val_ids=val,
        source_pixel_sha256=audit.hashes,supervision_sha256=canonical_hash(supervision),annotations_sha256=ann,
        feature_targets_sha256=canonical_hash(feature_hashes),code=provenance(cfg),protocol=cfg,
        codec_execution=execution(args.task),status='calibration',device=device)
    write_json(out/'manifest.json',manifest)
    return cfg,warm,teachers,records,features,meta,manifest


def diagnostics(model,virtual,teachers,records,features,manifest,device,out,arm):
    rows=[]
    for identifier in manifest['proxy_val_ids']:
        r=records[identifier]; x=tensor(r['rgb'],device); support=support_tensors(r['support'],x)
        for c in ('h264','h265'):
            for q in (30,35,40,45,50):
                codec=codec_factory(manifest['task'])(c,q)
                rows.append(dict(arm=arm,source_id=identifier,codec=c,qp=q,
                    **diagnose_point(model,virtual,teachers,manifest['task'],x,support,codec)))
        print(f'PROXYVAL {arm} {identifier} complete',flush=True)
    write_json(out/f'{arm}_proxy_validation.json',rows)


def run(args):
    out=empty_output(args.out); recipe=settings(args.config); device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2); torch.manual_seed(recipe['seed']); random.seed(recipe['seed']); np.random.seed(recipe['seed'])
    cfg,warm,teachers,records,features,meta,manifest=prepare_training(args,device,out)
    calibration=calibrate(records,manifest['cal_ids'],cfg,recipe,args.task,device,out,codec_factory=codec_factory(args.task))
    manifest['calibration']=calibration; manifest['status']='training'; write_json(out/'manifest.json',manifest)
    diagnostics(warm.model,VirtualCodec(calibration).to(device),teachers,records,features,manifest,device,out,'warm')
    groups=[(c,q) for c in cfg['codecs'] for q in cfg['qps']]
    for arm in ('proxy','real'):
        torch.manual_seed(recipe['seed']); random.seed(recipe['seed']); rng=np.random.default_rng(recipe['seed'])
        model=ResidualPreprocessor(args.task,12).to(device); model.load_state_dict(warm.model.state_dict(),strict=True); model.train()
        virtual=VirtualCodec(calibration).to(device).train(); optimizer=torch.optim.Adam(model.parameters(),lr=recipe['lr'])
        history=[]; step=0
        for epoch in range(recipe['epochs']):
            totals={k:0. for k in ('loss','task_loss','feature_loss','distortion','estimated_bpp','gradient_norm','actual_codec_seconds')}
            start=time.perf_counter(); measured=[]
            for index in rng.permutation(48):
                identifier=manifest['fit_ids'][int(index)]; r=records[identifier]; x=tensor(r['rgb'],device)
                support=support_tensors(r['support'],x); target=r['item']['label'] if args.task=='ar' else od_targets(meta,r['item'],r['geometry'],device)
                targets=[v.to(device) for v in features[identifier]]
                for group_index in rng.permutation(10):
                    c,q=groups[int(group_index)]; terms=training_step(model,virtual,teachers,args.task,x,support,target,targets,codec_factory(args.task)(c,q),arm,recipe)
                    component_norms={}
                    if step%50==0:
                        for term,weight in (('task_loss',1.),('estimated_bpp',.01),('distortion',10.),('feature_loss',.1)):
                            grads=torch.autograd.grad(weight*terms[term],tuple(model.parameters()),retain_graph=True,allow_unused=True)
                            component_norms[term]=float(torch.stack([g.square().sum() for g in grads if g is not None]).sum().sqrt())
                    optimizer.zero_grad(set_to_none=True); terms['loss'].backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step(); step+=1
                    row=dict(step=step,epoch=epoch+1,arm=arm,source_id=identifier,codec=c,qp=q,gradient_norm=float(norm),component_gradient_norms=component_norms,
                        **{k:float(v.detach()) if isinstance(v,torch.Tensor) else v for k,v in terms.items()})
                    with (out/f'{arm}_training.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
                    for k in totals: totals[k]+=row[k]
                    if row['actual_bpp'] is not None: measured.append(row['actual_bpp'])
                    if step%50==0: print(json.dumps(row),flush=True)
            history.append(dict(epoch=epoch+1,steps=480,seconds=time.perf_counter()-start,actual_bpp=float(np.mean(measured)) if measured else None,
                **{k:v/480 for k,v in totals.items()})); print(json.dumps(history[-1]),flush=True)
        metadata={k:manifest[k] for k in ('task','settings','warm','fit_ids','cal_ids','proxy_val_ids','source_pixel_sha256','supervision_sha256',
            'annotations_sha256','feature_targets_sha256','calibration')}
        metadata.update(arm=arm,width=12,seed=303231,epochs=2,history=history,codec_execution=execution(args.task))
        torch.save(make_checkpoint(model,metadata),out/f'{arm}_last.pth')
        diagnostics(model,virtual,teachers,records,features,manifest,device,out,arm)
    manifest['status']='complete'; write_json(out/'manifest.json',manifest)


def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--task',choices=['ar','od'],required=True)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--annotations',type=Path); p.add_argument('--config',type=Path)
    p.add_argument('--warm-checkpoint',type=Path,required=True); p.add_argument('--warm-runtime',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--device',choices=['cpu','cuda']); return p


if __name__=='__main__': run(parser().parse_args())
