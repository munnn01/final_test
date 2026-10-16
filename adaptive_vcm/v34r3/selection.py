"""Measured source guards and checkpoint-bound empirical task-risk constraints."""
import time
import numpy as np
from . import SCHEMA
from .protocol import component_names,digest


def shortlist(prediction,recipe,kind,*,fixed=None,rng=None):
    if kind=='cnn':
        scores=np.asarray(prediction['logits'],float)
        if scores.shape!=(13,):raise ValueError('13 action logits required')
        scores=np.nan_to_num(scores,nan=-np.inf)
        indices=[int(a) for a in np.argsort(-scores,kind='stable') if a][:3]
    elif kind=='train_fixed':indices=list(fixed or [])
    elif kind=='random':indices=(rng if rng is not None else np.random.default_rng(0)).choice(np.arange(1,13),3,replace=False).tolist()
    else:raise ValueError('unknown shortlist kind')
    if len(indices)!=3 or len(set(indices))!=3 or any(not 0<a<13 for a in indices):raise ValueError('three unique nonidentity actions required')
    return indices


def _bounds(calibration,prediction,task,recipe):
    try:
        if (calibration['schema']!=SCHEMA or calibration['task']!=task
            or calibration['risk_components']!=list(component_names(task))
            or calibration['recipe_sha256']!=digest(recipe)
            or calibration['model_state_sha256']!=prediction['model_state_sha256']):return None
        bounds=np.asarray(calibration['bounds'],float);risk=np.asarray(prediction['risk'],float)
        if bounds.shape!=(13,3) or risk.shape!=(13,3) or not np.isfinite(bounds).all() or not np.isfinite(risk).all():return None
        for key in ('logits','rate'):
            if np.asarray(prediction[key]).shape!=(13,) or not np.isfinite(prediction[key]).all():return None
        out=bounds+risk;out[0]=0;return out
    except (KeyError,TypeError,ValueError):return None


def choose(candidates,encode,source_score,*,task,recipe,prediction,calibration,prefix=None,policy=None):
    if task not in ('od','ot') or recipe['task']!=task:raise ValueError('standalone R3 selection required')
    actions=[0,*[int(a) for a in candidates]]
    if len(actions)!=4 or len(set(actions))!=4 or any(not 0<a<13 for a in actions[1:]):
        raise ValueError('exact identity plus three unique candidate budget required')
    before=prefix.signature if prefix is not None else None
    generation=prefix.generation if prefix is not None else None
    packets=[];values=[];work=[]
    for action in actions:
        packet=encode(action);start=time.perf_counter();value=source_score(packet.decoded)
        work.append(time.perf_counter()-start);packets.append(packet);values.append(value)
        if prefix is not None and (prefix.signature!=before or prefix.generation!=generation):
            raise ValueError('candidate score mutated the shared prefix')
    costs=np.asarray([v['source_cost'] for v in values],float)
    expected=4 if task=='od' else 3
    if costs.shape!=(4,expected) or not np.isfinite(costs[0]).all() or any(p.coded_bytes<=0 for p in packets):
        raise ValueError('invalid source anchor or packet')
    policy=policy or {}
    source_limits=np.asarray(policy.get('source_limits',recipe['source_limits']),float)
    gt_limits=np.asarray(policy.get('gt_limits',recipe['gt_limits']),float)
    if (source_limits.shape!=(expected,) or gt_limits.shape!=(3,) or not np.isfinite(source_limits).all()
        or not np.isfinite(gt_limits).all() or (source_limits<0).any() or (gt_limits<0).any()):
        raise ValueError('invalid selection component limits')
    rates=np.array([p.coded_bytes/packets[0].coded_bytes for p in packets]);excess=costs-costs[0]
    feasible=np.isfinite(excess).all(1)&(excess<=source_limits).all(1)&(rates<=recipe['budget_ratio'])
    risk_gate=policy.get('risk_gate',recipe.get('risk_gate',True));bounds=_bounds(calibration,prediction,task,recipe)
    if risk_gate:
        if bounds is None:feasible[1:]=False
        else:feasible&=(bounds[actions]<=gt_limits).all(1)
    feasible[0]=True
    index=min((i for i in range(4) if feasible[i]),key=lambda i:(rates[i],actions[i]))
    if prefix is not None:prefix.commit(values[index]['probe'])
    clean=lambda a:[[float(v) if np.isfinite(v) else None for v in row] for row in a]
    audit=dict(tried_actions=actions,rates=rates.tolist(),source_excess=clean(excess),feasible=feasible.tolist(),
        risk_bounds=clean(bounds[actions]) if bounds is not None else None,risk_gate=bool(risk_gate),
        calibration_valid=bounds is not None,guard_seconds=sum(work),guard_work_seconds=work,
        prefix_before=before,prefix_after=prefix.signature if prefix is not None else None,
        selection_policy='minimum real paid bytes under component constraints; index tie-break',
        source_guard_scope='source consistency plus empirical predicted risk; no task accuracy guarantee',
        policy_sha256=digest(dict(gt_limits=gt_limits.tolist(),source_limits=source_limits.tolist(),risk_gate=bool(risk_gate))))
    return actions[index],packets[index],values[index],audit
