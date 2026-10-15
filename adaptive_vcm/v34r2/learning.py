"""TRAIN-only normalization/ranking; source-group CAL residuals, no fitting."""
import math
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from ..v34.learning import masked_loss


def _split(rows,split):
    if not rows or any(r['split']!=split for r in rows): raise ValueError(f'{split.upper()} rows required')


def fit_scales(rows):
    _split(rows,'train'); scales={}
    for name,field,mask_name in [('gt','gt_damage','gt_mask'),('source','source_damage','active')]:
        scales[name]=[]
        for k in range(3):
            values=[abs(v[k]) for r in rows if r[mask_name][k] for v in r[field] if v[k] is not None]
            scales[name].append(max(1e-3,float(np.quantile(values,.75))) if values else 1.)
    return scales


def source_feasible(row,cfg):
    rate=np.asarray(row['rate'],float); ok=np.isfinite(rate)&(rate<=cfg['budget_ratio'])&(rate>0)
    mask=np.asarray(row['active'],bool); values=np.array(row['source_damage'],float)[:,mask]
    ok &= np.isfinite(values).all(1)&(values<=np.array(cfg['source_guard_limits'])[mask]).all(1)
    if not ok[0] or rate[0]!=1: raise ValueError('identity missing from feasible actions')
    return ok


def feasible(row,cfg):
    ok=source_feasible(row,cfg); mask=np.asarray(row['gt_mask'],bool)
    values=np.array(row['gt_damage'],float)[:,mask]
    ok &= np.isfinite(values).all(1)&(values<=np.array(cfg['damage_tolerance'])[mask]).all(1)
    if not ok[0]: raise ValueError('identity GT target must be feasible')
    return ok


def soft_targets(row,cfg):
    ok=feasible(row,cfg); rate=np.asarray(row['rate'])
    logits=-(rate-rate[ok].min())/cfg['ranking_temperature']
    p=np.where(ok,np.exp(np.clip(logits,-700,0)),0); return p/p.sum()


def fixed_shortlist(rows,cfg):
    _split(rows,'train')
    scores=np.mean([np.where(feasible(r,cfg),r['rate'],2.) for r in rows],axis=0)
    return [int(i) for i in np.argsort(scores,kind='stable') if i][:cfg['source_shortlist']]


def calibration(rows,predictions,coverage):
    _split(rows,'cal')
    if len(rows)!=len(predictions) or not 0<coverage<1: raise ValueError('invalid calibration')
    for r,p in zip(rows,predictions):
        if not np.isfinite(p['rate']).all(): raise ValueError('nonfinite calibrated rate')
        for name,field,mask_name in [('gt_risk','gt_damage','gt_mask'),('source_risk','source_damage','active')]:
            mask=np.asarray(r[mask_name],bool)
            for a in (p[name],np.array(r[field],float)):
                if not np.isfinite(np.asarray(a)[:,mask]).all(): raise ValueError('nonfinite active calibration target/prediction')
    domains={}
    for domain in sorted({r['domain'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['domain']==domain]
        groups=sorted({rows[i]['group'] for i in ids}); out={}
        for name,field,mask in [('gt_risk','gt_damage','gt_mask'),('source_risk','source_damage','active')]:
            residual=[]
            for group in groups:
                members=[i for i in ids if rows[i]['group']==group]
                values=[]
                for i in members:
                    value=np.array(rows[i][field],float)-np.array(predictions[i][name],float)
                    value[:,~np.asarray(rows[i][mask],bool)]=np.nan; values.append(value)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    residual.append(np.nanmax(values,axis=0))
            a=np.array(residual); bounds=np.zeros((a.shape[1],3))
            for j in range(a.shape[1]):
                for k in range(3):
                    v=np.sort(a[:,j,k][np.isfinite(a[:,j,k])])
                    bounds[j,k]=max(0,float(v[min(len(v)-1,math.ceil((len(v)+1)*coverage)-1)])) if len(v) else 1e6
            bounds[0]=0; out[name]=bounds.tolist()
        rate=[]
        for group in groups:
            rate.append(np.max([np.array(rows[i]['rate'])-predictions[i]['rate'] for i in ids if rows[i]['group']==group],axis=0))
        rates=np.sort(rate,axis=0)[min(len(groups)-1,math.ceil((len(groups)+1)*coverage)-1)]
        rates=np.maximum(0,rates); rates[0]=0
        domains[domain]=dict(**out,rate=rates.tolist(),source_groups=len(groups),rows=len(ids))
    return dict(domains=domains,coverage=coverage,scope='group-max empirical CAL residuals; no coverage guarantee under shift or few sequences')


def train(model,rows,inputs,cfg,device):
    _split(rows,'train'); scales=fit_scales(rows)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']); model.to(device).train(); history=[]
    def as_tensor(a,dtype=torch.float32): return torch.tensor(a,device=device,dtype=dtype)
    for epoch in range(cfg['epochs']):
        losses=[]
        groups={}
        for i,r in enumerate(rows): groups.setdefault(inputs[r['id']][0].shape[0],[]).append(i)
        for group in groups.values():
            order=np.random.permutation(group)
            for offset in range(0,len(order),16):
                batch=[rows[i] for i in order[offset:offset+16]]
                x=torch.stack([inputs[r['id']][0] for r in batch]).to(device)
                maps=torch.stack([inputs[r['id']][1] for r in batch]).to(device)
                out=model(x,maps,as_tensor([r['condition'] for r in batch]))
                loss=-(as_tensor(np.array([soft_targets(r,cfg) for r in batch]))*out['logits'].log_softmax(-1)).sum(1).mean()
                for key,field,mask_name,scale in [('gt_risk','gt_damage','gt_mask','gt'),('source_risk','source_damage','active','source')]:
                    values=np.array([r[field] for r in batch],float)
                    loss=loss+4*masked_loss(out[key],as_tensor(values),as_tensor([r[mask_name] for r in batch],torch.bool),scales[scale])
                labels=np.array([source_feasible(r,cfg) for r in batch],float)
                loss=loss+F.binary_cross_entropy_with_logits(out['feasible_logits'],as_tensor(labels))
                loss=loss+4*F.mse_loss(out['rate'],as_tensor([r['rate'] for r in batch]))
                if not torch.isfinite(loss): raise ValueError('nonfinite R2 loss')
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
                losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
    model.eval(); return history,scales
