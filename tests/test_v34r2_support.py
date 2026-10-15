import numpy as np
import pytest
from adaptive_vcm.v34.preprocessing import region_maps as legacy_maps
from adaptive_vcm.v34.tracking import ByteTracker
from adaptive_vcm.v34r2.preprocessing import Support, union_maps
from adaptive_vcm.v34r2.tracking import Prefix


def pred(boxes=((8,8,16,16),), scores=(.9,)):
    return dict(boxes=np.array(boxes,float).reshape(-1,4),scores=np.array(scores),labels=np.ones(len(scores),int))


def test_ar_is_exact_and_joint_protection_never_weakens_a_task():
    rgb=np.random.default_rng(1).integers(0,256,(3,48,48,3),dtype=np.uint8)
    p=[pred() for _ in rgb]
    ar=Support().build(rgb,p,[1,0,0])
    assert np.array_equal(ar,legacy_maps(rgb,p))
    od=Support().build(rgb,p,[0,1,0])
    joint=Support().build(rgb,p,[1,1,0])
    assert np.all(joint[...,0]>=ar[...,0]) and np.all(joint[...,0]>=od[...,0])
    assert np.allclose(joint.sum(-1),1) and joint.min()>=0
    assert od[0,7,12,1]>0 and od[0,8,12,0]==1


def test_ot_low_scores_need_association_and_history_moves_with_track():
    support=Support(history_decay=.7)
    rgb=np.zeros((1,48,48,3),np.uint8)
    first=pred(); tracks=[{**first,'track_ids':np.array([7])}]
    support.build(rgb,[first],[0,0,1],tracks=tracks)
    moved=pred(((24,8,32,16),(.0,30,8,38)),(.2,.15))
    matched=[{**pred(((24,8,32,16),),(.2,)),'track_ids':np.array([7])}]
    m=support.build(rgb,[moved],[0,0,1],tracks=matched)
    assert m[0,10,27,0]>0
    assert m[0,10,10,0]==0  # no fixed-coordinate trail
    assert m[0,33,3,0]==0   # unmatched low-confidence detection
    # A real scene cut must clear historical boxes, including missing tracks.
    bright=np.full_like(rgb,255)
    cleared=support.build(bright,[pred((),())],[0,0,1],tracks=[{**pred((),()),'track_ids':np.array([],int)}])
    assert cleared[...,0].max()==0


def test_prefix_forks_are_isolated_order_independent_and_commit_once():
    prefix=Prefix(ByteTracker())
    a=prefix.probe([pred()]); b=prefix.probe([pred((),())])
    assert prefix.tracker.frame==0 and a.tracker.frame==b.tracker.frame==1
    again=prefix.probe([pred()])
    assert np.array_equal(a.frames[-1]['track_ids'],again.frames[-1]['track_ids'])
    prefix.commit(a)
    assert prefix.tracker.frame==1
    with pytest.raises(ValueError,match='stale'): prefix.commit(b)
    next_probe=prefix.probe([pred(scores=(.2,))])
    assert next_probe.frames[-1]['frame_number']==2
    assert np.array_equal(next_probe.frames[-1]['track_ids'],a.frames[-1]['track_ids'])
    assert prefix.tracker.frame==1


def test_union_maps_rejects_invalid_partition():
    with pytest.raises(ValueError): union_maps([np.zeros((1,4,4,3),np.float32)])


def test_missing_track_support_decays_and_snapshot_does_not_commit_parent():
    support=Support(); rgb=np.zeros((1,48,48,3),np.uint8)
    support.build(rgb,[pred()],[0,0,1],tracks=[{**pred(),'track_ids':np.array([1])}])
    missing={**pred((),()),'track_ids':np.array([],int)}
    m=support.build(rgb,[pred((),())],[0,0,1],tracks=[missing])
    assert 0<m[0,10,10,:2].sum()<1
    prefix=Prefix(); snapshot=prefix.snapshot(); snapshot.commit(snapshot.probe([pred()]))
    assert prefix.tracker.frame==0 and snapshot.tracker.frame==1
