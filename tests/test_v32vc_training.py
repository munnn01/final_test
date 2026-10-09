import copy
import numpy as np
import pytest
import torch
from torch import nn

from adaptive_vcm.data import partition
from adaptive_vcm.v32vc.train import split_train, task_loss, fit_rate, od_targets, settings
from adaptive_vcm.v32vc.checkpoint import make_checkpoint, load_checkpoint
from adaptive_vcm.v32vc.model import ResidualPreprocessor


class ToyAction:
    def __init__(self):
        self.model = nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4)).eval()
        self.model.requires_grad_(False)
    def logits(self,x): return self.model(x.mean((2,3,4)))


def test_frozen_task_network_passes_image_gradient_without_updates():
    teacher = ToyAction(); before = copy.deepcopy(teacher.model.state_dict())
    x = torch.rand(2,3,2,8,8,requires_grad=True)
    task_loss('ar',[teacher],x,torch.tensor([1,2])).backward()
    assert x.grad.abs().sum()>0 and torch.isfinite(x.grad).all()
    assert all(torch.equal(before[k],v) for k,v in teacher.model.state_dict().items())
    assert all(p.grad is None for p in teacher.model.parameters())


def test_split_is_source_disjoint_train_only_order_independent():
    ids = [str(i) for i in range(300) if partition('coco2017/'+str(i))=='train'][:64]
    a,b = split_train(ids,'od',303221)
    c,d = split_train(list(reversed(ids)),'od',303221)
    assert len(a)==48 and len(b)==16 and not set(a)&set(b)
    assert set(a)==set(c) and set(b)==set(d)
    with pytest.raises(ValueError): split_train(ids+[next(str(i) for i in range(300) if partition('coco2017/'+str(i))=='test')],'od',303221)


def test_positive_rate_mapping_and_invalid_measurements():
    scale,offset = fit_rate([1.,2.,3.],[2.2,4.2,6.2])
    assert scale==pytest.approx(2) and offset==pytest.approx(.2)
    with pytest.raises(ValueError): fit_rate([float('nan')],[1.])


def test_actual_torchvision_od_loss_passes_input_gradients_with_frozen_buffers():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None,weights_backbone=None,num_classes=3,
                                           min_size=64,max_size=64).eval().requires_grad_(False)
    teacher = type('Teacher',(),{'model':model})()
    before = {k:v.clone() for k,v in model.named_buffers()}
    x = torch.rand(1,3,1,64,64,requires_grad=True)
    target = {'boxes':torch.tensor([[12.,12.,40.,40.]]),'labels':torch.tensor([1])}
    loss = task_loss('od',[teacher],x,target); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert not model.training and all(torch.equal(before[k],v) for k,v in model.named_buffers())
    assert all(p.grad is None for p in model.parameters())


def test_od_training_boxes_use_canonical_geometry_and_ignore_crowd():
    item = dict(image_id=2)
    meta = dict(annotations=[dict(image_id=2,bbox=[2,4,6,8],category_id=3,iscrowd=0),
                            dict(image_id=2,bbox=[0,0,1,1],category_id=1,iscrowd=1)])
    targets = od_targets(meta,item,(.5,.5,1,2),'cpu')
    assert targets['boxes'].tolist()==[[2.,4.,5.,8.]] and targets['labels'].tolist()==[3]


def checkpoint_fixture():
    model = ResidualPreprocessor('od')
    rgb = np.zeros((1,16,16,3),np.uint8)
    import hashlib
    ids = [str(i) for i in range(300) if partition('coco2017/'+str(i))=='train'][:64]
    fit,cal = split_train(ids,'od',303221)
    metadata = dict(task='od',arm='plain',width=12,seed=303221,epochs=4,settings=settings(),
                    supervision_sha256='a'*64,annotations_sha256='b'*64,
                    fit_ids=fit,cal_ids=cal,source_pixel_sha256={i:hashlib.sha256((rgb+int(i)).tobytes()).hexdigest() for i in ids},
                    calibration={f'{c}/{q}':dict(step=.03,rate_scale=1.,rate_offset=0.) for c in ('h264','h265') for q in (30,35,40,45,50)},
                    history=[dict(epoch=i+1,steps=480,seconds=1.,loss=1.,task_loss=1.,distortion=.01,estimated_bpp=1.,gradient_norm=1.) for i in range(4)])
    return model,metadata,rgb


def test_checkpoint_integrity_and_both_fit_cal_overlap(tmp_path):
    model,metadata,rgb = checkpoint_fixture()
    fit,cal = metadata['fit_ids'],metadata['cal_ids']
    path = tmp_path/'last.pth'; payload = make_checkpoint(model,metadata); torch.save(payload,path)
    loaded = load_checkpoint(path,'od')
    with pytest.raises(ValueError): loaded.assert_source(cal[0],rgb)
    with pytest.raises(ValueError): loaded.assert_source('fresh',rgb+int(fit[0]))
    payload['model']['head.bias'][0] += 1; torch.save(payload,path)
    with pytest.raises(ValueError): load_checkpoint(path,'od')


def test_plain_joint_pair_requires_identical_training_and_calibration_provenance():
    from adaptive_vcm.v32vc.checkpoint import validate_pair
    _,meta,_=checkpoint_fixture()
    plain=type('Model',(),{'metadata':meta})()
    joint=type('Model',(),{'metadata':copy.deepcopy(meta)})()
    joint.metadata['arm']='joint'
    validate_pair({'plain':plain,'joint':joint})
    joint.metadata['calibration']['h264/30']['step']=.2
    with pytest.raises(ValueError): validate_pair({'plain':plain,'joint':joint})


@pytest.mark.parametrize('alter',['counts','settings','history'])
def test_rehashed_incompatible_checkpoint_is_rejected(tmp_path,alter):
    model,meta,_ = checkpoint_fixture()
    if alter=='counts':
        meta['fit_ids']=meta['fit_ids'][:3]; meta['cal_ids']=meta['cal_ids'][:1]
        meta['source_pixel_sha256']={i:meta['source_pixel_sha256'][i] for i in meta['fit_ids']+meta['cal_ids']}
    elif alter=='settings': meta['settings']['lr']=100.
    else: meta['history']=[dict(epoch=i+1) for i in range(4)]
    path=tmp_path/'bad.pth'; torch.save(make_checkpoint(model,meta),path)
    with pytest.raises(ValueError): load_checkpoint(path,'od')
