import copy
import numpy as np
import pytest
import torch
from torch import nn
from adaptive_vcm.v32rf.bridge import substitute, real_forward
from adaptive_vcm.v32rf.features import feature_vectors, feature_distance
from adaptive_vcm.v32rf.protocol import settings, split_cal, codec_factory
from adaptive_vcm.v32rf.diagnostics import selector_diagnostics
from adaptive_vcm.v32.protocol import load_config
from adaptive_vcm.data import partition


def test_real_values_use_proxy_jacobian_and_block_real_gradients():
    x = torch.tensor([.2,.3], requires_grad=True)
    actual = torch.tensor([.8,.7], requires_grad=True)
    y = substitute(3*x, actual)
    assert torch.equal(y, actual)
    y.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x,3)) and actual.grad is None
    with pytest.raises(ValueError): substitute(x, torch.zeros(3))
    with pytest.raises(ValueError): substitute(x, torch.full_like(x,float('nan')))


@pytest.mark.parametrize('codec',['h264','h265'])
def test_actual_forward_uint8_matches_inference_and_preserves_cnn_gradient(codec):
    from adaptive_vcm.v32vc.model import ResidualPreprocessor
    model=ResidualPreprocessor('od')
    x=torch.rand(1,3,1,16,16)
    cuts=torch.ones(1,1,dtype=torch.bool); support=torch.zeros(1,1,1,16,16)
    y=model(x,40,int(codec=='h265'),support,support,cuts)
    actual=codec_factory('od')(codec,40,'ultrafast')
    z,rate,seconds=real_forward(y,actual,'od')
    rgb=y.detach()[0].permute(1,2,3,0).mul(255).round().clamp(0,255).byte().numpy()
    packet=actual.roundtrip(rgb,source_shape=rgb.shape,task='od')
    assert np.array_equal(z[0].permute(1,2,3,0).mul(255).round().byte().numpy(),packet.decoded)
    assert rate==pytest.approx(8*packet.coded_bytes/256) and seconds>0 and not z.requires_grad
    substitute(y*.7,z).square().mean().backward()
    assert model.head.weight.grad.abs().sum()>0
    assert actual.standard.gop_settings==dict(gop=1,bframes=0,scenecut=False,closed_gop=True)


class Action:
    def __init__(self):
        self.model=nn.Module(); self.model.layer2=nn.Sequential(nn.Conv3d(3,4,1),nn.BatchNorm3d(4)).eval()
        self.model.eval().requires_grad_(False)
    def logits(self,x): return self.model.layer2(x).mean((2,3,4))


def test_feature_targets_detached_and_teacher_state_hooks_preserved():
    teacher=Action(); before=copy.deepcopy(teacher.model.state_dict())
    source=feature_vectors('ar',[teacher],torch.rand(1,3,2,16,16))
    x=torch.rand(1,3,2,16,16,requires_grad=True)
    loss=feature_distance(feature_vectors('ar',[teacher],x),source)
    loss.backward()
    assert x.grad.abs().sum()>0 and 0<=loss<=2
    assert all(torch.equal(before[k],v) for k,v in teacher.model.state_dict().items())
    assert not teacher.model.training and not teacher.model.layer2._forward_hooks
    assert all(p.grad is None for p in teacher.model.parameters())


def test_calibration_validation_partition_is_locked_and_order_independent():
    ids=[str(i) for i in range(200) if partition('coco2017/'+str(i))=='train'][:16]
    cal,val=split_cal(ids,'od'); assert len(cal)==12 and len(val)==4 and not set(cal)&set(val)
    assert (cal,val)==split_cal(list(reversed(ids)),'od')
    with pytest.raises(ValueError): split_cal(ids[:15],'od')
    assert settings()['epochs']==2 and settings()['feature_weight']==.1
    assert codec_factory('ar')('h264',40).standard.gop_settings['gop']==16


def test_selector_rejection_reasons_are_distinct_and_selected_is_counted():
    def c(name,size,distance=0,decision=True):
        return dict(name=name,total_bytes=size,relative_distances=[distance],preserves_decision=[decision])
    cfg=load_config()
    audits=[dict(candidates=[c('identity',1000),c('rf_real_rgb',800,.04)],selected={'rf_real':'identity'}),
            dict(candidates=[c('identity',1000),c('rf_real_rgb',999)],selected={'rf_real':'identity'}),
            dict(candidates=[c('identity',1000),c('background2',700),c('rf_real_rgb',800)],selected={'rf_real':'background2'}),
            dict(candidates=[c('identity',1000),c('rf_real_rgb',600)],selected={'rf_real':'rf_real_rgb'})]
    counts=selector_diagnostics(audits,cfg,'od')['rf_real_rgb']
    assert counts==dict(generated=4,selected=1,guard_failed=1,insufficient_savings=1,other_selected=1)
