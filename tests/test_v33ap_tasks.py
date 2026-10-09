import torch
from v33_fixtures import Action,Detector,record


def test_joint_losses_reach_same_cnn_and_restore():
    from adaptive_vcm.v33ap.tasks import TaskHeads,losses
    from adaptive_vcm.v33ap.model import JointPreprocessor
    from adaptive_vcm.v32vc.train import tensor,support_tensors
    h=TaskHeads('cpu',action=Action(),detector=Detector()); r=record()
    h.source_targets(r)
    modes=[m.training for m in h.detector.model.modules()]
    buffers={k:v.clone() for k,v in h.detector.model.named_buffers()}
    model=JointPreprocessor(); x=tensor(r.rgb,'cpu'); p,m,c=support_tensors(r.support,x)
    for name in ('ar','od'):
        model.zero_grad(); y=model(x,35,0,p,m,c)
        terms=losses(h,y,r,'joint'); terms[name].backward()
        assert model.head.weight.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for teacher in (h.action,h.detector) for p in teacher.model.parameters())
    assert modes==[m.training for m in h.detector.model.modules()]
    assert all(torch.equal(v,buffers[k]) for k,v in h.detector.model.named_buffers())
    assert all(not v.requires_grad for v in r.source_features.values())


def test_masks_and_empty_background():
    from adaptive_vcm.v33ap.tasks import TaskHeads,losses,frame_indices
    from adaptive_vcm.v32vc.train import tensor
    h=TaskHeads('cpu',action=Action(),detector=Detector(empty=True)); r=record(); h.source_targets(r)
    assert frame_indices(1)==(0,) and len(set(frame_indices(16)))==3
    assert all(t['boxes'].shape==(0,4) for t in r.targets['pseudo'].values())
    z=tensor(r.rgb,'cpu').requires_grad_(); l=losses(h,z,r,'od_only')
    assert l['ar']==0 and torch.isfinite(l['od'])
    l['od'].backward(); assert z.grad.abs().sum()>0
    coco=record('od'); calls=h.action.calls; h.source_targets(coco)
    losses(h,tensor(coco.rgb,'cpu'),coco,'joint')
    assert calls==h.action.calls
    r.item.pop('label')
    import pytest
    with pytest.raises(ValueError,match='label'): losses(h,z,r,'joint')
