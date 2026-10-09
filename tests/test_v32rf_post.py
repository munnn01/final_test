import copy,hashlib
import numpy as np
import pytest
import torch
from adaptive_vcm.v32rf.post import SwinRestorer,make_checkpoint,load_checkpoint,receiver_variants
from adaptive_vcm.v32rf.post_train import post_step
from adaptive_vcm.v32rf.features import feature_vectors
from adaptive_vcm.data import partition
from adaptive_vcm.v32.protocol import provenance,load_config
from tests.test_v32rf_bridge import Action


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.patch_embed=torch.nn.Conv3d(3,768,1); self.pos_drop=torch.nn.Dropout(.5)
        self.features=torch.nn.Identity(); self.norm=torch.nn.Identity()
    def embed(self,x): return self.patch_embed(x).permute(0,2,3,4,1)


def toy(**kwargs):
    b=Backbone(); embed=b.patch_embed
    class Embed(torch.nn.Module):
        def __init__(self): super().__init__(); self.conv=embed
        def forward(self,x): return self.conv(x).permute(0,2,3,4,1)
    b.patch_embed=Embed(); return SwinRestorer(backbone=b)


def test_restore_identity_bounds_shape_and_frozen_backbone():
    model=toy().train(); x=torch.rand(1,3,2,8,8); before=copy.deepcopy(model.backbone.state_dict())
    assert torch.equal(model(x),x) and not model.backbone.training
    with torch.no_grad(): model.head[-1].bias.fill_(5.)
    y=model(x); assert y.shape==x.shape and (y-x).abs().max()<=.100001
    y.mean().backward(); assert model.head[-1].bias.grad.abs().sum()>0
    assert all(torch.equal(before[k],v) for k,v in model.backbone.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters())


def test_real_torchvision_swin_geometry_without_weight_download():
    from torchvision.models.video import swin3d_t
    model=SwinRestorer(backbone=swin3d_t(weights=None)).eval()
    x=torch.rand(1,3,2,32,32)
    assert torch.equal(model(x),x)


def test_post_step_uses_frozen_tasks_and_actual_received_rgb():
    model=toy(); teacher=Action(); x=torch.rand(1,3,2,8,8); decoded=(x*.8).detach()
    before=copy.deepcopy(teacher.model.state_dict())
    targets=feature_vectors('ar',[teacher],x)
    terms=post_step(model,[teacher],decoded,x,1,targets); terms['loss'].backward()
    assert model.head[-1].weight.grad.abs().sum()>0
    assert all(torch.equal(before[k],v) for k,v in teacher.model.state_dict().items())


def rf_manifest():
    ids=[str(i) for i in range(500) if partition(str(i))=='train'][:64]
    return dict(task='ar',arm='real',checkpoint_sha256='a'*64,state_sha256='b'*64,fit_ids=ids[:48],cal_ids=ids[48:60],proxy_val_ids=ids[60:],
        source_pixel_sha256={i:hashlib.sha256(i.encode()).hexdigest() for i in ids},supervision_sha256='c'*64)


def test_checkpoint_integrity_and_all_train_overlap(tmp_path,monkeypatch):
    import adaptive_vcm.v32rf.post as module
    monkeypatch.setattr(module,'SwinRestorer',toy)
    model=toy(); history=[dict(epoch=i,steps=480,loss=1.,seconds=1.) for i in (1,2)]
    payload=make_checkpoint(model,rf_manifest(),history); path=tmp_path/'post.pth'; torch.save(payload,path)
    loaded=load_checkpoint(path)
    with pytest.raises(ValueError): loaded.assert_source(rf_manifest()['proxy_val_ids'][0],np.zeros((2,8,8,3),np.uint8))
    payload['model']['head.2.bias'][0]+=1; torch.save(payload,path)
    with pytest.raises(ValueError): load_checkpoint(path)


def test_receiver_variants_keep_exact_packet_bytes_and_paid_rate():
    rgb=np.full((2,8,8,3),100,np.uint8); packet=type('Packet',(),dict(decoded=rgb,data=b'stream',coded_bytes=6,packet_sha256='a'*64))()
    variants=receiver_variants(packet,toy())
    assert set(variants)=={'decoded','restored'} and np.array_equal(variants['decoded'][0],variants['restored'][0])
    assert packet.data==b'stream' and packet.coded_bytes==6
