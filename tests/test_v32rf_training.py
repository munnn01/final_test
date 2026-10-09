import copy
import hashlib
import json
import numpy as np
import pytest
import torch
from adaptive_vcm.v32rf.warm import load_warm
from adaptive_vcm.v32rf.checkpoint import make_checkpoint, load_checkpoint, validate_pair
from adaptive_vcm.v32rf.train import training_step, diagnose_point
from adaptive_vcm.v32rf.protocol import settings, split_cal, codec_factory, execution
from adaptive_vcm.v32rf.features import feature_vectors
from adaptive_vcm.v32vc.virtual import VirtualCodec
from adaptive_vcm.v32vc.model import ResidualPreprocessor
from adaptive_vcm.v32vc.checkpoint import state_hash
from tests.test_v32vc_training import checkpoint_fixture
from tests.test_v32rf_bridge import Action


def fixture():
    model,old,rgb=checkpoint_fixture(); cal,val=split_cal(old['cal_ids'],'od')
    warm=dict(origin_commit=settings()['warm_origin'],checkpoint_sha256='a'*64,state_sha256=state_hash(model.state_dict()),
        fit_ids=old['fit_ids'],cal_ids=old['cal_ids'],source_pixel_sha256=copy.deepcopy(old['source_pixel_sha256']))
    history=[dict(epoch=i,steps=480,seconds=1.,loss=1.,task_loss=1.,feature_loss=.1,
        distortion=.01,estimated_bpp=.2,gradient_norm=.1,actual_codec_seconds=0.,actual_bpp=None) for i in (1,2)]
    meta=dict(task='od',arm='proxy',width=12,seed=303231,epochs=2,settings=settings(),warm=warm,
        fit_ids=old['fit_ids'],cal_ids=cal,proxy_val_ids=val,source_pixel_sha256=old['source_pixel_sha256'],
        supervision_sha256='b'*64,feature_targets_sha256='c'*64,annotations_sha256='d'*64,
        calibration=old['calibration'],history=history,codec_execution=execution('od'))
    return model,meta,rgb


def test_original_runtime_manifest_cannot_be_rehashed_to_hide_source_changes(tmp_path):
    (tmp_path/'origin-manifest.json').write_text(json.dumps(dict(origin_commit=settings()['warm_origin'],
        source_files_sha256={'adaptive_vcm/__init__.py':'a'*64})))
    path=tmp_path/'warm.pth'; torch.save({},path)
    with pytest.raises(ValueError,match='runtime'): load_warm(path,tmp_path,'od')


def test_checkpoint_roundtrip_and_full_train_overlap_protection(tmp_path):
    model,meta,rgb=fixture(); path=tmp_path/'rf.pth'; torch.save(make_checkpoint(model,meta),path)
    loaded=load_checkpoint(path,'od')
    for i in meta['fit_ids']+meta['cal_ids']+meta['proxy_val_ids']:
        with pytest.raises(ValueError): loaded.assert_source(i,rgb)
    assert loaded.metadata['warm']['state_sha256']==meta['warm']['state_sha256']


@pytest.mark.parametrize('field',['settings','history','validation','warm','state'])
def test_rehashed_incompatible_checkpoint_is_rejected(tmp_path,field):
    model,meta,_=fixture()
    if field=='settings': meta['settings']['feature_weight']=0
    elif field=='history': meta['history'][1]['steps']=1
    elif field=='validation': meta['proxy_val_ids']=meta['cal_ids'][:4]
    elif field=='warm': meta['warm']['source_pixel_sha256'][meta['fit_ids'][0]]='e'*64
    payload=make_checkpoint(model,meta)
    if field=='state': payload['model']['head.bias'][0]+=1
    path=tmp_path/'bad.pth'; torch.save(payload,path)
    with pytest.raises(ValueError): load_checkpoint(path,'od')


def test_proxy_real_pair_requires_identical_start_and_calibration():
    _,meta,_=fixture(); real=copy.deepcopy(meta); real['arm']='real'
    models={a:type('Model',(),{'metadata':m})() for a,m in [('proxy',meta),('real',real)]}
    validate_pair(models)
    real['warm']['state_sha256']='f'*64
    with pytest.raises(ValueError): validate_pair(models)


def test_paired_steps_have_actual_forward_only_for_real_and_frozen_teacher():
    base=ResidualPreprocessor('ar'); teacher=Action(); before=copy.deepcopy(teacher.model.state_dict())
    x=torch.rand(1,3,2,16,16); p=torch.zeros(1,1,2,16,16); cuts=torch.tensor([[True,False]])
    targets=feature_vectors('ar',[teacher],x); codec=codec_factory('ar')('h264',40,'ultrafast')
    initial=state_hash(base.state_dict()); results={}
    for arm in ('proxy','real'):
        model=copy.deepcopy(base); torch.manual_seed(303231)
        assert state_hash(model.state_dict())==initial
        terms=training_step(model,VirtualCodec().train(),[teacher],'ar',x,(p,p,cuts),1,targets,codec,arm,settings())
        terms['loss'].backward()
        assert model.head.weight.grad.abs().sum()>0 and torch.isfinite(model.head.weight.grad).all()
        results[arm]=terms
    assert results['proxy']['actual_bpp'] is None and results['real']['actual_bpp']>0
    assert all(torch.equal(before[k],v) for k,v in teacher.model.state_dict().items())
    assert all(p.grad is None for p in teacher.model.parameters())


def test_proxy_diagnostics_do_not_modify_model_or_teacher():
    model=ResidualPreprocessor('ar').eval(); teacher=Action()
    before=state_hash(model.state_dict()); teacher_before=copy.deepcopy(teacher.model.state_dict())
    x=torch.rand(1,3,2,16,16); p=torch.zeros(1,1,2,16,16); cuts=torch.tensor([[True,False]])
    result=diagnose_point(model,VirtualCodec(),[teacher],'ar',x,(p,p,cuts),codec_factory('ar')('h264',40,'ultrafast'))
    assert result['pixel_mse']>=0 and result['real_bpp']>0 and result['rate_relative_error']>=0
    assert before==state_hash(model.state_dict()) and all(p.grad is None for p in model.parameters())
    assert all(torch.equal(teacher_before[k],v) for k,v in teacher.model.state_dict().items())


def test_actual_detection_feature_and_loss_gradient_through_real_decoder():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    net=fasterrcnn_mobilenet_v3_large_fpn(weights=None,weights_backbone=None,num_classes=3,min_size=64,max_size=64).eval().requires_grad_(False)
    teacher=type('Teacher',(),{'model':net})(); before={k:v.clone() for k,v in net.named_buffers()}
    model=ResidualPreprocessor('od'); x=torch.rand(1,3,1,64,64); p=torch.zeros(1,1,1,64,64); cuts=torch.ones(1,1,dtype=torch.bool)
    with torch.no_grad(): targets=feature_vectors('od',[teacher],x)
    target=dict(boxes=torch.tensor([[12.,12.,40.,40.]]),labels=torch.tensor([1]))
    terms=training_step(model,VirtualCodec().train(),[teacher],'od',x,(p,p,cuts),target,targets,codec_factory('od')('h264',40,'ultrafast'),'real',settings())
    terms['loss'].backward()
    assert model.head.weight.grad.abs().sum()>0 and torch.isfinite(model.head.weight.grad).all()
    assert all(torch.equal(before[k],v) for k,v in net.named_buffers()) and all(p.grad is None for p in net.parameters())


@pytest.mark.parametrize('bad',['missing_execution','fast_preset','wrong_gop','missing_real_rate','nonfinite_real_rate','proxy_measured_rate'])
def test_locked_actual_codec_evidence_cannot_be_rehashed_away(tmp_path,bad):
    model,meta,_=fixture()
    if bad=='missing_execution': meta.pop('codec_execution')
    elif bad=='fast_preset': meta['codec_execution']['preset']='fast'
    elif bad=='wrong_gop': meta['codec_execution']['gop_settings']['gop']=16
    else:
        if bad!='proxy_measured_rate':
            meta['arm']='real'
            for r in meta['history']: r['actual_codec_seconds']=1.
        for r in meta['history']: r['actual_bpp']=None if bad=='missing_real_rate' else -1. if bad=='nonfinite_real_rate' else 1.
    path=tmp_path/'bad_conditions.pth'; torch.save(make_checkpoint(model,meta),path)
    with pytest.raises(ValueError): load_checkpoint(path,'od')


def test_pair_cannot_claim_different_codec_execution():
    _,meta,_=fixture(); real=copy.deepcopy(meta); real['arm']='real'; real['codec_execution']['preset']='fast'
    models={a:type('Model',(),{'metadata':m})() for a,m in [('proxy',meta),('real',real)]}
    with pytest.raises(ValueError): validate_pair(models)
