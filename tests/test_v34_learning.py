import numpy as np
import torch
import pytest
from adaptive_vcm.v34.learning import targets, masked_loss, calibrate, select_action
from adaptive_vcm.v34.checkpoint import save_checkpoint, load_checkpoint
from adaptive_vcm.v34.model import Controller

def test_same_rate_quality_gain_is_not_discarded_and_missing_task_is_masked():
    damage=np.array([[0, np.nan, 0],[-.2,np.nan,0],[.9,np.nan,0]])
    choice=targets(np.array([1,1.002,.7]),damage,[1,0,1],[.05]*3,1.01)
    assert choice==1
    with pytest.raises(ValueError): targets([1],[[0,0,0]],[0,0,0],[.05]*3,1.01)
    pred=torch.tensor([[[1.,2.,3.]]],requires_grad=True)
    truth=torch.tensor([[[0.,float('nan'),0.]]])
    loss=masked_loss(pred,truth,torch.tensor([[True,False,True]]))
    loss.backward()
    assert loss.item()==5 and pred.grad[0,0,1].item()==0

def test_calibration_fallback_and_same_rate_gain():
    p=np.zeros((20,3,3)); actual=p.copy(); actual[:,2,0]=.3
    cal=calibrate(p,actual,np.ones((20,3)),np.ones((20,3)),np.ones((20,3),bool),.9)
    risk=np.array([[0,0,0],[-.1,0,0],[0,0,0]])
    assert select_action(risk,np.array([1,1.001,.8]),cal,[1,0,0],[.05]*3,1.01)==1
    assert select_action(risk,np.array([1,1.2,.8]),cal,[1,0,0],[.05]*3,1.01)==0

def test_checkpoint_is_weights_only_safe_and_detects_metadata_tampering(tmp_path):
    model=Controller(3); path=tmp_path/'model.pth'
    meta={'seed':303401,'source_commit':'a'*40,'protocol':'pilot','calibration':{'risk':[[0]*3]*3,'rate':[0]*3}}
    save_checkpoint(path,model,meta)
    loaded=load_checkpoint(path)
    assert loaded.metadata==meta
    value=torch.load(path,weights_only=True)
    value['metadata']['seed']=0; torch.save(value,path)
    with pytest.raises(ValueError,match='metadata'): load_checkpoint(path)
