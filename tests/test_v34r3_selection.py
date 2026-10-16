import copy
from types import SimpleNamespace
import numpy as np
import pytest
from adaptive_vcm.v34r3.protocol import settings,recipe
from adaptive_vcm.v34r3.learning import calibrate
from adaptive_vcm.v34r3.selection import choose,shortlist
from tests.test_v34r3_learning import row


def setup(task='od'):
    cfg=recipe(settings(),task,'train-final')
    p=dict(risk=np.zeros((13,3)),rate=np.ones(13),logits=np.arange(13,dtype=float),model_state_sha256='b'*64)
    r=row('cal',task=task);r['gt_damage']=[[0.]*3 for _ in range(13)]
    c=calibrate([r],[p],recipe=cfg)
    return cfg,p,c


def run_choice(candidates,cfg,pred,cal,*,task='od',costs=None,rates=None):
    costs=costs or [[0.]*(4 if task=='od' else 3) for _ in range(13)]
    rates=rates or [100,50,70,80]+[90]*9
    def encode(a):return SimpleNamespace(coded_bytes=rates[a],decoded=np.full((1,1,1,3),a,np.uint8))
    def score(rgb):
        a=int(rgb[0,0,0,0]);return dict(source_cost=costs[a],probe=None,predictions={})
    return choose(candidates,encode,score,task=task,recipe=cfg,prediction=pred,calibration=cal)


def test_risk_gate_rejects_source_feasible_but_high_risk_candidate():
    cfg,p,c=setup();p['risk'][1,1]=.1
    selected,packet,result,audit=run_choice([1,2,3],cfg,p,c)
    assert selected==2 and packet.coded_bytes==70
    assert audit['tried_actions']==[0,1,2,3] and audit['feasible']==[True,False,True,True]
    cfg,p,c=setup('ot');p['risk'][1]=[0,0,.1]
    assert run_choice([1,2,3],cfg,p,c,task='ot')[0]==2


@pytest.mark.parametrize('bad',['missing','nan','wrong_metrics','wrong_recipe'])
def test_bad_or_missing_calibration_falls_back_to_identity(bad):
    cfg,p,c=setup()
    if bad=='missing':c=None
    elif bad=='nan':c['bounds'][1][0]=float('nan')
    elif bad=='wrong_metrics':c['risk_components']=['fp','fn','localization']
    else:c['recipe_sha256']='a'*64
    assert run_choice([1,2,3],cfg,p,c)[0]==0


def test_gt_component_semantics_reject_shape_compatible_wrong_metadata():
    cfg,p,c=setup();c['task']='ot'
    assert run_choice([1,2,3],cfg,p,c)[0]==0


def test_ties_probe_permutations_and_all_rejections_are_deterministic():
    cfg,p,c=setup();rates=[100,70,70,80]+[90]*9
    assert run_choice([1,2,3],cfg,p,c,rates=rates)[0]==1
    assert run_choice([3,2,1],cfg,p,c,rates=rates)[0]==1
    p['risk'][1:]=1
    assert run_choice([3,2,1],cfg,p,c)[0]==0
    cfg['risk_gate']=False
    assert run_choice([3,2,1],cfg,p,None)[0]==1
    assert shortlist(p,cfg,'cnn')==[12,11,10]
    assert len(set(shortlist(p,cfg,'random',rng=np.random.default_rng(3))))==3
