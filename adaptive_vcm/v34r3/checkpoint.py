"""Shared tensor layout does not imply interchangeable task semantics."""
import hashlib
import json
import numpy as np
from pathlib import Path
import torch
from . import SCHEMA
from .model import Controller
from .protocol import digest, component_names, component_mask
from ..v32vc.checkpoint import state_hash


def save(path, model, metadata):
    meta=json.loads(json.dumps(metadata,allow_nan=False))
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(dict(schema=SCHEMA,actions=model.actions,conditions=model.conditions,
        metadata=meta,metadata_sha256=digest(meta),state=state,state_sha256=state_hash(state)),path)


def load(path, *, task, recipe, task_models, lineage, device='cpu'):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if (task not in ('od','ot') or value.get('schema')!=SCHEMA
        or value.get('actions')!=13 or value.get('conditions')!=9
        or digest(value['metadata'])!=value.get('metadata_sha256')
        or state_hash(value['state'])!=value.get('state_sha256')):
        raise ValueError('R3 checkpoint version/content mismatch')
    meta=value['metadata']
    if (meta.get('task')!=task or meta.get('recipe')!=recipe or meta.get('task_models')!=task_models
        or meta.get('risk_components')!=list(component_names(task))
        or meta.get('component_mask')!=component_mask(task)):
        raise ValueError('task, metric, recipe or frozen model mismatch')
    if (not lineage or any(k not in meta or k not in lineage or meta[k]!=lineage[k]
        for k in ('origin','cache_sha256','inputs_sha256'))
        or meta.get('training_lineage_sha256')!=digest(lineage)):
        raise ValueError('checkpoint origin/cache/input training lineage mismatch')
    if meta.get('checkpoint_role')=='FINAL':
        cal=meta.get('calibration') or {};binding=lineage.get('calibration_binding',{})
        bounds=np.asarray(cal.get('bounds',[]),float)
        if (not binding or any(cal.get(k)!=v for k,v in binding.items())
            or cal.get('schema')!=SCHEMA or cal.get('task')!=task
            or cal.get('recipe_sha256')!=digest(recipe) or cal.get('model_state_sha256')!=value['state_sha256']
            or cal.get('risk_components')!=list(component_names(task))
            or bounds.shape!=(13,3) or not np.isfinite(bounds).all()):
            raise ValueError('FINAL calibration state/cache/prefix lineage mismatch')
    model=Controller(13,conditions=9); model.load_state_dict(value['state'],strict=True)
    model.to(device).eval(); model.metadata=meta
    model.checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return model
