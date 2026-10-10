"""Supervision from measured candidates; gradients never enter the codec."""
import math
import numpy as np
import torch
import torch.nn.functional as F

def targets(rate,damage,mask,tolerance,budget):
    rate=np.asarray(rate,float); damage=np.asarray(damage,float); mask=np.asarray(mask,bool)
    if not mask.any() or damage.shape!=(len(rate),3) or mask.shape!=(3,):
        raise ValueError('active measured task required')
    valid=np.isfinite(damage[:,mask]).all(1)&np.isfinite(rate)&(rate<=budget)
    valid &= (damage[:,mask]<=np.asarray(tolerance)[mask]).all(1)
    if not valid[0] or abs(rate[0]-1)>1e-6 or np.max(np.abs(damage[0,mask]))>1e-6:
        raise ValueError('identity anchor missing')
    score=rate+.25*np.nan_to_num(damage[:,mask]).mean(1)
    return int(np.argmin(np.where(valid,score,np.inf)))

def masked_loss(prediction,target,mask):
    active=mask[:,None,:].expand_as(prediction)
    if not active.any(): raise ValueError('empty supervised task mask')
    if not torch.isfinite(target[active]).all(): raise ValueError('nonfinite active task target')
    return (prediction[active]-target[active]).square().mean()

def calibrate(predicted,actual,rate_pred,rate_actual,mask,coverage):
    p,a=np.asarray(predicted),np.asarray(actual)
    mask=np.asarray(mask,bool)
    if p.shape!=a.shape or p.ndim!=3 or mask.shape!=(len(p),3) or not 0<coverage<1:
        raise ValueError('invalid calibration table')
    def upper(values):
        # Finite-sample one-sided order statistic. This is empirical calibration,
        # not a guarantee under distribution shift or correlated codec conditions.
        if not len(values): return 1e6
        return max(0.,float(np.sort(values)[min(len(values)-1,math.ceil((len(values)+1)*coverage)-1)]))
    risk=[[upper((a[:,j,k]-p[:,j,k])[mask[:,k]]) for k in range(3)] for j in range(p.shape[1])]
    r=np.asarray(rate_actual)-np.asarray(rate_pred)
    if r.shape!=p.shape[:2] or not np.isfinite(r).all(): raise ValueError('invalid calibrated rates')
    for k in range(3):
        if mask[:,k].any() and not np.isfinite((a-p)[mask[:,k],:,k]).all(): raise ValueError('invalid calibrated risks')
    risk[0]=[0.]*3
    rates=[upper(r[:,j]) for j in range(p.shape[1])]; rates[0]=0.
    return dict(risk=risk,rate=rates,coverage=coverage,count=len(p),scope='empirical one-sided CAL residuals')

def select_action(risk,rate,calibration,mask,tolerance,budget):
    risk=np.asarray(risk,float).copy(); rate=np.asarray(rate,float).copy(); mask=np.asarray(mask,bool)
    if risk.shape!=(len(rate),3) or mask.shape!=(3,) or not mask.any(): raise ValueError('invalid selection inputs')
    bounds=np.asarray(calibration['risk'])+risk; rate_bound=rate+np.asarray(calibration['rate'])
    bounds[0]=0; rate_bound[0]=1; risk[0]=0; rate[0]=1
    feasible=(bounds[:,mask]<=np.asarray(tolerance)[mask]).all(1)&(rate_bound<=budget)
    feasible &= np.isfinite(bounds[:,mask]).all(1)&np.isfinite(rate_bound)
    score=rate+.25*risk[:,mask].mean(1)
    return int(np.argmin(np.where(feasible,score,np.inf))) if feasible.any() else 0

def train_controller(model,rows,inputs,cfg,device):
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr']); history=[]
    model.to(device).train()
    groups={}
    for i,r in enumerate(rows):
        groups.setdefault(inputs[r['id']][0].shape[0],[]).append(i)
    c=torch.tensor([r['condition'] for r in rows],device=device,dtype=torch.float32)
    rate=torch.tensor([r['rate'] for r in rows],device=device,dtype=torch.float32)
    damage=torch.tensor([[[float('nan') if v is None else v for v in a] for a in r['damage']] for r in rows],device=device,dtype=torch.float32)
    mask=torch.tensor([r['mask'] for r in rows],device=device,dtype=torch.bool)
    labels=torch.tensor([targets(r['rate'],np.array(r['damage'],float),r['mask'],cfg['damage_tolerance'],cfg['budget_ratio']) for r in rows],device=device)
    for epoch in range(cfg['epochs']):
        batches=[]; values=[]
        for group in groups.values():
            indices=torch.tensor(group)[torch.randperm(len(group))]
            batches.extend(indices.split(16))
        for index in torch.randperm(len(batches)).tolist():
            ids=batches[index].tolist(); batch=batches[index].to(device)
            x=torch.stack([inputs[rows[i]['id']][0] for i in ids]).to(device)
            m=torch.stack([inputs[rows[i]['id']][1] for i in ids]).to(device)
            out=model(x,m,c[batch])
            loss=F.cross_entropy(out['logits'],labels[batch])+4*masked_loss(out['risk'],damage[batch],mask[batch])+4*F.mse_loss(out['rate'],rate[batch])
            if not torch.isfinite(loss): raise ValueError('nonfinite controller loss')
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
            values.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(values))))
    model.eval(); return history
