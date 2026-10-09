"""Strict paired LAST checkpoints; all original TRAIN pixels remain excluded."""
import hashlib
from pathlib import Path
import re
import numpy as np
import torch
from . import SCHEMA
from .protocol import settings,split_cal,execution
from ..v32.protocol import canonical_hash,load_config,provenance
from ..v32vc.checkpoint import state_hash,LearnedPreprocessor
from ..v32vc.model import ResidualPreprocessor
from ..v32vc.train import split_train


def make_checkpoint(model,metadata):
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    metadata=dict(metadata,schema=SCHEMA,checkpoint_role='LAST',state_sha256=state_hash(state),
        protocol=load_config(),code=provenance(load_config()))
    return dict(metadata=metadata,metadata_sha256=canonical_hash(metadata),model=state)


def _hash(value): return re.fullmatch('[0-9a-f]{64}',str(value)) is not None


def load_checkpoint(path,task,device='cpu'):
    path=Path(path); payload=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict): raise ValueError('invalid RF checkpoint')
    m=payload.get('metadata',{})
    if (m.get('schema')!=SCHEMA or m.get('checkpoint_role')!='LAST' or m.get('task')!=task
            or m.get('arm') not in ('proxy','real') or m.get('width')!=12 or m.get('seed')!=303231
            or m.get('epochs')!=2 or m.get('settings')!=settings()
            or canonical_hash(m)!=payload.get('metadata_sha256') or state_hash(payload.get('model',{}))!=m.get('state_sha256')):
        raise ValueError('RF task/schema/state/recipe mismatch')
    w=m.get('warm',{}); fit=m.get('fit_ids',[]); cal=m.get('cal_ids',[]); val=m.get('proxy_val_ids',[])
    ids=fit+cal+val; hashes=m.get('source_pixel_sha256',{})
    if (w.get('origin_commit')!=settings()['warm_origin'] or not _hash(w.get('checkpoint_sha256')) or not _hash(w.get('state_sha256'))
            or len(fit)!=48 or len(cal)!=12 or len(val)!=4 or len(set(ids))!=64
            or any(type(i)!=str or not i for i in ids) or set(hashes)!=set(ids)
            or len(set(hashes.values()))!=64 or any(not _hash(h) for h in hashes.values())
            or hashes!=w.get('source_pixel_sha256') or fit!=w.get('fit_ids')
            or (fit,w.get('cal_ids',[]))!=split_train(ids,task,303221)
            or (cal,val)!=split_cal(w.get('cal_ids',[]),task)):
        raise ValueError('RF FIT/CAL/validation/warm provenance mismatch')
    if (not _hash(m.get('supervision_sha256')) or not _hash(m.get('feature_targets_sha256'))
            or task=='od' and not _hash(m.get('annotations_sha256'))):
        raise ValueError('RF target provenance missing')
    expected={f'{c}/{q}' for c in ('h264','h265') for q in (30,35,40,45,50)}
    if (set(m.get('calibration',{}))!=expected or m.get('protocol')!=load_config()
            or m.get('codec_execution')!=execution(task)):
        raise ValueError('RF calibration/protocol mismatch')
    for entry in m['calibration'].values():
        if (any(type(entry.get(k)) not in (int,float) or not np.isfinite(entry[k]) for k in ('step','rate_scale','rate_offset'))
                or entry['step']<=0 or entry['rate_scale']<=0 or entry['rate_offset']<0):
            raise ValueError('invalid RF calibration')
    history=m.get('history',[])
    if len(history)!=2 or [r.get('epoch') for r in history]!=[1,2]: raise ValueError('incomplete RF LAST history')
    for row in history:
        if row.get('steps')!=480 or any(type(row.get(k)) not in (int,float) or not np.isfinite(row[k]) or row[k]<0
                for k in ('seconds','loss','task_loss','feature_loss','distortion','estimated_bpp','gradient_norm','actual_codec_seconds')):
            raise ValueError('incomplete/nonfinite RF history')
        measured=row.get('actual_bpp')
        if (m['arm']=='real' and (type(measured) not in (int,float) or not np.isfinite(measured)
                or measured<=0 or row['actual_codec_seconds']<=0)
                or m['arm']=='proxy' and ('actual_bpp' not in row or measured is not None or row['actual_codec_seconds']!=0)):
            raise ValueError('RF actual codec measurement evidence differs from forward arm')
    if m.get('code',{}).get('code_files_sha256')!=provenance(load_config())['code_files_sha256']:
        raise ValueError('RF runtime source mismatch')
    model=ResidualPreprocessor(task,m['width']).to(device); model.load_state_dict(payload['model'],strict=True)
    return LearnedPreprocessor(model.eval(),m,hashlib.sha256(path.read_bytes()).hexdigest())


def validate_pair(models):
    if set(models)!={'proxy','real'} or any(m.metadata.get('arm')!=a for a,m in models.items()):
        raise ValueError('RF proxy/real pair required')
    keys=('task','width','seed','epochs','settings','warm','fit_ids','cal_ids','proxy_val_ids','source_pixel_sha256',
        'supervision_sha256','feature_targets_sha256','annotations_sha256','calibration','protocol','code','codec_execution')
    if any(models['proxy'].metadata.get(k)!=models['real'].metadata.get(k) for k in keys):
        raise ValueError('RF pair starting state/calibration/provenance differs')
