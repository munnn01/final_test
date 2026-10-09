import numpy as np
import pytest
from adaptive_vcm.v32rf.evaluate import EXTRA_FAMILIES,FORCED,prepare_extension
from adaptive_vcm.v32rf.protocol import codec_factory
from adaptive_vcm.v32.selection import Candidate,choose_stream
from adaptive_vcm.v32.protocol import load_config
from adaptive_vcm.v32.evaluate import summarize
from tests.test_v32_evaluate import records
from tests.test_v32_roi import pixels,support
from tests.test_v32_selection import EmptyODTeacher


def test_forced_cnn_is_measured_even_when_unknown_foreground_and_guard_rejects():
    rgb,s=pixels(),support(); teacher=EmptyODTeacher(); cfg=load_config()
    changed=rgb.copy(); changed[:,:,:,0]=0
    result=choose_stream(rgb,s['protection'][0],s,'od',codec_factory('od')('h264',40,'ultrafast'),cfg,[teacher],[teacher.predict(rgb)],
        extra_candidates=[Candidate('rf_real_rgb',changed,'rf_real_plain')],extra_families={'rf_real':{'controls','rf_real_plain'},'rf_real_forced':{'controls','rf_real_plain'}},
        forced_names={'rf_real_forced':'rf_real_rgb'},unconditional_extra=True,guard=lambda *a:((1.,),(False,)))
    assert result.arms['rf_real'][1]=='identity' and result.arms['rf_real_forced'][1]=='rf_real_rgb'
    assert len(result.audit)==2 and result.arms['rf_real_forced'][0].decoded.shape==rgb.shape
    with pytest.raises(ValueError,match='forced'):
        choose_stream(rgb,s['protection'][0],s,'od',codec_factory('od')('h264',40,'ultrafast'),cfg,[teacher],[teacher.predict(rgb)],forced_names={'bad':'missing'})


def test_primary_and_transfer_complete_arm_counts_and_missing_rejected():
    for baseline,count in ((False,14),(True,16)):
        rows=records()
        for r in list(rows):
            if r['arm']=='controls':
                rows.extend([{**r,'arm':a} for a in EXTRA_FAMILIES])
                if baseline: rows.extend([{**r,'arm':a} for a in ('v30b','hybrid')])
        assert len({r['arm'] for r in rows})==count
        result=summarize(rows,['a','b'],'ar',load_config(),0,has_baseline=baseline,extra_arms=tuple(EXTRA_FAMILIES))
        assert result['complete_paired_grid']
        with pytest.raises(ValueError): summarize(rows[:-1],['a','b'],'ar',load_config(),0,has_baseline=baseline,extra_arms=tuple(EXTRA_FAMILIES))


def test_extension_has_separate_guarded_and_forced_candidates_with_paid_retarget():
    class Model:
        def pixels(self,rgb,*a): return rgb.copy()
    kwargs,seconds=prepare_extension(pixels(),support(),object(),{a:Model() for a in ('warm','proxy','real')},load_config())
    assert kwargs['forced_names']==FORCED and kwargs['unconditional_extra'] and seconds>=0
    assert len(kwargs['extra_candidates'])==9 and sum(bool(c.grids) for c in kwargs['extra_candidates'])==6


def test_swin_evaluator_uses_official_transform_and_no_trainable_weights(monkeypatch):
    import torch
    from torchvision.models import video as V
    from adaptive_vcm.v32rf.swin import SwinAnalyzer
    class Small(torch.nn.Module):
        def __init__(self): super().__init__(); self.weight=torch.nn.Parameter(torch.ones(1))
        def forward(self,x):
            assert x.shape== (1,3,2,224,224)
            return x.mean((2,3,4))*self.weight
    monkeypatch.setattr(V,'swin3d_t',lambda **kw:Small())
    model=SwinAnalyzer('cpu'); result=model.probabilities(pixels(2))
    assert len(result)==3 and result.sum()==pytest.approx(1.) and not model.model.training
    assert all(not p.requires_grad for p in model.model.parameters())
