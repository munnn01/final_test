"""Frozen Video Swin features and a trainable bounded RGB restoration head."""
import hashlib
from pathlib import Path
import re
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .bridge import pixels
from ..data import partition
from ..v32.protocol import canonical_hash,provenance,load_config
from ..v32vc.checkpoint import state_hash

SCHEMA='adaptive-vcm-v32rf-swin-post-1'
RECIPE=dict(seed=303232,epochs=2,lr=1e-4,head_width=24,residual_bound=.1,task_weight=1.,distortion_weight=10.,feature_weight=.1,
    backbone='swin3d_t',weights='KINETICS400_V1',backbone_frozen=True,checkpoint_role='LAST')


class SwinRestorer(nn.Module):
    def __init__(self,backbone=None,pretrained=True):
        super().__init__()
        if backbone is None:
            from torchvision.models.video import swin3d_t,Swin3D_T_Weights
            backbone=swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1 if pretrained else None)
        self.backbone=backbone.eval().requires_grad_(False)
        self.head=nn.Sequential(nn.Conv3d(768,24,3,padding=1),nn.GELU(),nn.Conv3d(24,3,1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1,1))

    def train(self,mode=True):
        super().train(mode); self.backbone.eval(); return self

    def forward(self,decoded):
        if decoded.ndim!=5 or decoded.shape[1]!=3 or not torch.isfinite(decoded).all() or decoded.min()<0 or decoded.max()>1:
            raise ValueError('bounded B3THW decoded RGB required')
        with torch.no_grad():
            features=self.backbone.patch_embed((decoded-self.mean)/self.std)
            features=self.backbone.pos_drop(features)
            features=self.backbone.norm(self.backbone.features(features)).permute(0,4,1,2,3)
        residual=F.interpolate(self.head(features),size=decoded.shape[2:],mode='trilinear',align_corners=False).tanh()
        return (decoded+.1*residual).clamp(0,1)


def _backbone_hash(state): return state_hash({k:v for k,v in state.items() if k.startswith('backbone.')})


def make_checkpoint(model,rf_manifest,history):
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    meta=dict(schema=SCHEMA,recipe=RECIPE,rf_real=rf_manifest,history=history,state_sha256=state_hash(state),
        backbone_sha256=_backbone_hash(state),code=provenance(load_config()))
    return dict(metadata=meta,metadata_sha256=canonical_hash(meta),model=state)


class Restorer:
    def __init__(self,model,metadata,path_hash): self.model=model; self.metadata=metadata; self.manifest=dict(checkpoint_sha256=path_hash,**metadata)
    def assert_source(self,identifier,rgb):
        hashes=self.metadata['rf_real']['source_pixel_sha256']
        if identifier in hashes or hashlib.sha256(rgb.tobytes()).hexdigest() in hashes.values(): raise ValueError('post evaluation overlaps TRAIN64')


def load_checkpoint(path,device='cpu'):
    path=Path(path); payload=torch.load(path,map_location='cpu',weights_only=True); meta=payload.get('metadata',{})
    state=payload.get('model',{}); rf=meta.get('rf_real',{}); ids=rf.get('fit_ids',[])+rf.get('cal_ids',[])+rf.get('proxy_val_ids',[])
    hashes=rf.get('source_pixel_sha256',{})
    if (meta.get('schema')!=SCHEMA or meta.get('recipe')!=RECIPE or canonical_hash(meta)!=payload.get('metadata_sha256')
            or state_hash(state)!=meta.get('state_sha256') or _backbone_hash(state)!=meta.get('backbone_sha256')
            or rf.get('task')!='ar' or rf.get('arm')!='real' or len(rf.get('fit_ids',[]))!=48 or len(set(ids))!=64 or set(hashes)!=set(ids)
            or len(set(hashes.values()))!=64 or any(re.fullmatch('[0-9a-f]{64}',str(h)) is None for h in hashes.values())
            or any(partition(i)!='train' for i in ids) or any(re.fullmatch('[0-9a-f]{64}',str(rf.get(k,''))) is None for k in ('state_sha256','checkpoint_sha256','supervision_sha256'))
            or meta.get('code',{}).get('code_files_sha256')!=provenance(load_config())['code_files_sha256']):
        raise ValueError('invalid Swin restoration state/recipe/provenance')
    history=meta.get('history',[])
    if len(history)!=2 or [r.get('epoch') for r in history]!=[1,2] or any(r.get('steps')!=480 or
            any(type(r.get(k)) not in (float,int) or not np.isfinite(r[k]) or r[k]<0 for k in ('loss','seconds')) for r in history):
        raise ValueError('incomplete Swin restoration LAST')
    model=SwinRestorer(pretrained=False).to(device); model.load_state_dict(state,strict=True)
    return Restorer(model.eval(),meta,hashlib.sha256(path.read_bytes()).hexdigest())


@torch.no_grad()
def receiver_variants(packet,model):
    device=next(model.parameters()).device
    decoded=torch.from_numpy(np.array(packet.decoded,copy=True)).to(device).float().permute(3,0,1,2)[None]/255
    if decoded.is_cuda: torch.cuda.synchronize(device)
    start=time.perf_counter(); restored=model(decoded)
    if decoded.is_cuda: torch.cuda.synchronize(device)
    seconds=time.perf_counter()-start
    return dict(decoded=(packet.decoded,0.),restored=(pixels(restored),seconds))
