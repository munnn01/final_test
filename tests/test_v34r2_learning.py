import copy
import numpy as np
import pytest
import torch
from adaptive_vcm.v34r2.tasks import detection_distance, context_hota
from adaptive_vcm.v34r2.learning import fit_scales, calibration, soft_targets, fixed_shortlist,source_feasible,feasible
from adaptive_vcm.v34r2.model import Controller
from adaptive_vcm.v34r2.protocol import settings, validate_cache, seal
from tests.test_v34r2_support import pred


def row(split='train', source='a', group='a'):
    return dict(id=source,group=group,split=split,domain='od',codec='h264',qp=30,
        active=[0,1,0],gt_mask=[0,1,0],rate=[1,.7]+[1.2]*11,
        gt_damage=[[None,0,None],[None,.02,None]]+[[None,.3,None]]*11,
        source_damage=[[None,0,None],[None,0,None]]+[[None,.5,None]]*11,
        prefix_sha256='a'*64,packets=[dict(bytes=100 if i==0 else 70 if i==1 else 120,
            sha256=str(i)*64) for i in range(13)])


def test_two_way_detection_damage_penalizes_fp_class_and_confidence():
    a=pred()
    assert detection_distance([a],[a])==0
    extra=pred(((8,8,16,16),(30,30,40,40)),(.9,.8))
    assert detection_distance([a],[extra])>0
    wrong=copy.deepcopy(a); wrong['labels'][:]=2
    assert detection_distance([a],[wrong])>detection_distance([a],[pred(scores=(.7,))])>0
    with pytest.raises(ValueError): detection_distance([a],[])


def test_scaling_and_fixed_shortlist_reject_cal_or_test_leakage():
    r=row(); cfg=settings()
    scales=fit_scales([r])
    assert scales['gt'][1]==pytest.approx(.3)
    for split in ('cal','dev','test'):
        with pytest.raises(ValueError): fit_scales([row(split)])
        with pytest.raises(ValueError): fixed_shortlist([row(split)],cfg)
    soft=soft_targets(r,cfg)
    assert soft[1]>soft[0]>0 and np.all(soft[2:]==0)
    assert len(fixed_shortlist([r],cfg))==3


def test_calibration_counts_source_groups_not_qp_rows():
    rows=[row('cal',str(i),'one-sequence') for i in range(8)]
    preds=[dict(gt_risk=np.zeros((13,3)),source_risk=np.zeros((13,3)),rate=np.ones(13)) for _ in rows]
    c=calibration(rows,preds,.9)
    assert c['domains']['od']['source_groups']==1
    assert c['domains']['od']['rows']==8
    assert 'no coverage guarantee' in c['scope']
    with pytest.raises(ValueError): calibration([row()],preds[:1],.9)


def test_controller_has_separate_gt_source_feasibility_and_rate_heads():
    m=Controller(13)
    out=m(torch.rand(2,2,3,16,16),torch.rand(2,2,3,16,16),torch.zeros(2,9))
    assert out['gt_risk'].shape==out['source_risk'].shape==(2,13,3)
    assert out['feasible_logits'].shape==out['rate'].shape==out['logits'].shape==(2,13)
    sum(v.square().mean() for v in out.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_cache_rejects_joint_fake_gt_and_legacy_checkpoint_recipe():
    cfg=settings(); cfg.update(codecs=['h264'],qps=[30])
    cache=seal(dict(schema=cfg['schema'],settings=cfg,experiment='od',rows=[row()],
        sources={'a':dict(split='train',pixel_sha256='b'*64)},origin={'origin_commit':'c'*40,'bundle_sha256':'d'*64}))
    validate_cache(cache)
    bad=copy.deepcopy(cache); bad['rows'][0]['gt_mask']=[1,1,1]
    with pytest.raises(ValueError): validate_cache(seal(bad))
    bad=copy.deepcopy(cache); bad['rows'][0]['packets'][1]['bytes']=101
    with pytest.raises(ValueError): validate_cache(seal(bad))


def test_context_hota_scores_association_over_the_whole_context():
    import os
    if not os.environ.get('V34_TRACKEVAL'): pytest.skip('pinned TrackEval required')
    from adaptive_vcm.v32.tracking import load_trackeval
    package=load_trackeval(os.environ['V34_TRACKEVAL'])
    ref=[dict(boxes=np.array([[0,0,10,10]]),track_ids=np.array([1])) for _ in range(4)]
    good=copy.deepcopy(ref); bad=copy.deepcopy(ref)
    for p in bad[2:]: p['track_ids'][:]=2
    assert context_hota(package,good,ref)['HOTA']==pytest.approx(1)
    assert context_hota(package,bad,ref)['AssA']<1
    assert context_hota(package,bad,ref)['HOTA']<1


def test_source_feasibility_label_is_independent_of_gt_and_cal_rejects_nan():
    r=row(); r['gt_damage'][1][1]=99; cfg=settings()
    assert source_feasible(r,cfg)[1] and not feasible(r,cfg)[1]
    p=dict(gt_risk=np.zeros((13,3)),source_risk=np.zeros((13,3)),rate=np.ones(13))
    p['source_risk'][1,1]=np.nan
    with pytest.raises(ValueError): calibration([row('cal')],[p],.9)
