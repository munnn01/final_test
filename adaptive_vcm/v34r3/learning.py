"""Three component risk slots on the unchanged AR controller topology."""
import math
import re
import numpy as np
import torch
import torch.nn.functional as F
from . import SCHEMA
from .protocol import component_names,digest,task_mask
from ..v34r2.runner import condition


def _split(rows,split):
    if not rows or any(r.get('split')!=split for r in rows):raise ValueError(split+'-only fitting required')


def fit_scales(rows):
    _split(rows,'train');values=np.asarray([r['gt_damage'] for r in rows],float)
    if values.shape[1:]!=(13,3) or not np.isfinite(values).all():raise ValueError('finite component targets required')
    return np.maximum(np.percentile(np.abs(values).reshape(-1,3),75,axis=0),1e-3)


def _feasible(row,recipe):
    rate=np.asarray(row['rate'],float);gt=np.asarray(row['gt_damage'],float);source=np.asarray(row['source_damage'],float)
    ok=(rate<=recipe['budget_ratio'])&(rate>0)&np.isfinite(rate)
    ok&=np.isfinite(gt).all(1)&np.isfinite(source).all(1)
    ok&=(gt<=np.asarray(recipe['gt_limits'])).all(1)&(source<=np.asarray(recipe['source_limits'])).all(1)
    if rate[0]!=1 or not np.allclose(gt[0],0) or not np.allclose(source[0],0) or not ok[0]:
        raise ValueError('identity training anchor missing')
    return ok


def soft_targets(row,recipe):
    ok=_feasible(row,recipe);rate=np.asarray(row['rate'],float)
    logits=-(rate-rate[ok].min())/recipe['ranking_temperature']
    values=np.where(ok,np.exp(np.clip(logits,-700,0)),0)
    return values/values.sum()


def fixed_shortlist(rows,recipe):
    _split(rows,'train')
    scores=np.mean([np.where(_feasible(r,recipe),r['rate'],2.) for r in rows],axis=0)
    return [int(a) for a in np.argsort(scores,kind='stable') if a][:3]


def training_weights(rows):
    variants=[r['prefix_variant'] for r in rows];unique=set(variants)
    if not unique or not unique.issubset({'identity','rollout'}):raise ValueError('unknown training prefix variants')
    return np.array([len(rows)/(len(unique)*variants.count(v)) for v in variants])


def train(model,rows,inputs,recipe,*,device):
    _split(rows,'train')
    if recipe['task'] not in ('od','ot') or model.conditions!=9 or model.actions!=13:raise ValueError('standalone native-topology training only')
    scales=fit_scales(rows);weights=training_weights(rows)
    mass={v:float(sum(w for w,r in zip(weights,rows) if r['prefix_variant']==v)/weights.sum())
          for v in sorted({r['prefix_variant'] for r in rows})}
    optimizer=torch.optim.Adam(model.parameters(),lr=recipe['lr']);model.to(device).train();history=[]
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32,device=device)
    group={}
    for index,r in enumerate(rows):group.setdefault(inputs[r['id']][0].shape[0],[]).append(index)
    scale=tensor(scales)
    for epoch in range(recipe['epochs']):
        losses=[]
        for members in group.values():
            order=np.random.permutation(members)
            for offset in range(0,len(order),16):
                indices=order[offset:offset+16];batch=[rows[i] for i in indices]
                x=torch.stack([inputs[r['id']][0] for r in batch]).to(device)
                maps=torch.stack([inputs[r['id']][1] for r in batch]).to(device)
                cond=[condition(r['codec'],r['qp'],task_mask(recipe['task']),recipe,recipe['task']) for r in batch]
                out=model(x,maps,tensor(cond))
                ce=-(tensor(np.array([soft_targets(r,recipe) for r in batch]))*out['logits'].log_softmax(-1)).sum(1)
                gt=tensor(np.array([r['gt_damage'] for r in batch]))
                risk=F.smooth_l1_loss(out['risk']/scale,gt/scale,reduction='none').mean((1,2))
                rate=(out['rate']-tensor([r['rate'] for r in batch])).square().mean(1)
                loss=((ce+4*risk+4*rate)*tensor(weights[indices])).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite R3 learning objective')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),variant_mass=mass))
    model.eval();return history,scales


def calibrate(rows,predictions,coverage=.9,*,recipe):
    _split(rows,'cal')
    if len(rows)!=len(predictions) or not 0<coverage<1:raise ValueError('invalid CAL predictions')
    hashes={p.get('model_state_sha256') for p in predictions}
    if len(hashes)!=1 or not re.fullmatch('[0-9a-f]{64}',next(iter(hashes)) or ''):
        raise ValueError('CAL predictions must identify one frozen controller state')
    groups=sorted({r['group'] for r in rows});residual=[]
    for group in groups:
        values=[]
        for r,p in zip(rows,predictions):
            if r['group']!=group:continue
            actual=np.asarray(r['gt_damage'],float);predicted=np.asarray(p['risk'],float)
            if actual.shape!=(13,3) or predicted.shape!=(13,3) or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
                raise ValueError('invalid finite CAL component targets')
            values.append(actual-predicted)
        residual.append(np.max(values,axis=0))
    ordered=np.sort(residual,axis=0);index=min(len(groups)-1,math.ceil((len(groups)+1)*coverage)-1)
    bounds=np.maximum(0,ordered[index]);bounds[0]=0
    return dict(schema=SCHEMA,task=recipe['task'],risk_components=list(component_names(recipe['task'])),
        recipe_sha256=digest(recipe),model_state_sha256=next(iter(hashes)),bounds=bounds.tolist(),
        source_groups=len(groups),rows=len(rows),coverage=coverage,
        prefix_policy_sha256=digest(sorted({r.get('prefix_variant','identity') for r in rows})),
        promotion_eligible=recipe['task']!='ot' or len(groups)>=recipe['min_ot_cal_sequences'],
        scope='group-max empirical CAL residuals; no coverage guarantee under shift or few sequences')
