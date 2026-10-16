"""Shared tensor layout does not imply interchangeable task semantics."""
import hashlib
import json
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


def load(path, *, task, recipe, task_models, device='cpu'):
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
    model=Controller(13,conditions=9); model.load_state_dict(value['state'],strict=True)
    model.to(device).eval(); model.metadata=meta
    model.checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return model
