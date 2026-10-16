import copy
import json
import os
import numpy as np
import pytest
from adaptive_vcm.v34r3.coco_targets import components
from adaptive_vcm.v34r3 import mot_targets
from adaptive_vcm.v34r3.tasks import Heads
from adaptive_vcm.v34r2.tracking import Prefix
from adaptive_vcm.v32.tracking import load_trackeval,plan_sequences,stage_ground_truth,write_mot,evaluate_mot
from tests.test_v32_tracking import fixture_sequence
from tests.test_v34_single_task import heads_for,record
from adaptive_vcm.v33ap.sources import SourceRecord


def pred(boxes=((0,0,10,10),), labels=None, scores=None):
    return dict(boxes=np.array(boxes,float).reshape(-1,4),
                labels=np.array(labels if labels is not None else [1]*len(boxes),int),
                scores=np.array(scores if scores is not None else [.9]*len(boxes),float))


def annotation(label=1,crowd=0):
    return dict(id=1,image_id=1,category_id=label,bbox=[0,0,10,10],area=100,iscrowd=crowd)


def cost(p,anns):
    return components(p,anns,dict(image_id=1,width=100,height=100),(1,1,0,0))


def test_coco_components_fn_fp_wrong_class_and_localization():
    a=[annotation()]
    assert cost(pred(),a)==pytest.approx([0,0,0])
    assert cost(pred(()),a)==pytest.approx([1,0,0])
    assert cost(pred(((0,0,10,10),(40,40,50,50))),a)==pytest.approx([0,1,0])
    assert cost(pred(labels=[2]),a)==pytest.approx([1,1,0])
    c=cost(pred(((1,0,11,10),)),a)
    assert c[0]>0 and c[1]>0 and c[2]>0
    assert cost(pred(((0,0,20,20),)),a)==pytest.approx([1,1,0])


def test_coco_crowd_ignore_and_empty_scenes():
    assert cost(pred(),[annotation(crowd=1)])==pytest.approx([0,0,0])
    assert cost(pred(()),[])==pytest.approx([0,0,0])
    assert cost(pred(),[])==pytest.approx([0,1,0])
    assert cost(pred(scores=[.09]),[annotation()])==pytest.approx([1,0,0])


def tracked(ids=(1,1,1,1),distractor=False):
    result=[]
    for frame,key in enumerate(ids,1):
        boxes=[[0,0,10,10]]+([[30,0,40,10]] if distractor else [])
        result.append(dict(boxes=np.array(boxes,float),track_ids=np.array([key]+([9] if distractor else [])),
                           scores=np.ones(len(boxes))*.9,labels=np.ones(len(boxes),int),frame_number=frame))
    return result


def mot_fixture(tmp_path):
    path=fixture_sequence(tmp_path,count=4)
    (path/'gt/gt.txt').write_text(''.join(
        f'{i},1,0,0,10,10,1,1,1\n{i},2,30,0,10,10,1,6,1\n'
        f'{i},3,60,0,10,10,0,1,1\n' for i in range(1,5)))
    return path


def test_mot_distractor_zero_marked_and_association_components(tmp_path):
    package=load_trackeval(os.environ['V34_TRACKEVAL'])
    path=mot_fixture(tmp_path); annotations=mot_targets.read_context(path/'gt/gt.txt',1,5)
    assert annotations[0]['classes'].tolist()==[1,6,1]
    good=mot_targets.components(package,tracked(distractor=True),annotations,(1,1,0,0),benchmark='MOT20',sequence='MOT20-01')
    assert good['cost']==pytest.approx([0,0,0])
    bad=mot_targets.components(package,tracked((1,1,2,2),True),annotations,(1,1,0,0),benchmark='MOT20',sequence='MOT20-01')
    assert bad['cost'][0]>0 and bad['cost'][1]==pytest.approx(0) and bad['cost'][2]>0
    plan=plan_sequences(tmp_path,['MOT20-01'],4);out=tmp_path/'official'
    stage_ground_truth(plan,out/'ground_truth')
    write_mot(out/'trackers/reference/data/MOT20-01.txt',tracked(distractor=True),(1,1,0,0),80,40)
    official=evaluate_mot(package,plan,out,['reference'])['reference']['combined']
    assert official['HOTA']==pytest.approx(good['quality']['HOTA']*100)
    assert official['FP']==official['FN']==0


def test_source_score_never_reads_annotations():
    class NoAnnotations(SourceRecord):
        def __getattribute__(self,name):
            if name=='targets': raise AssertionError('GT crossed the source-only interface')
            return super().__getattribute__(name)
    old=record('od'); source=NoAnnotations(old.id,'od',{},old.rgb,old.geometry,{})
    heads=heads_for('od'); heads.__class__=Heads
    heads.prepare(source,'od')
    value=heads.source_score(old.rgb,source,prefix=None)
    assert np.isfinite(value['source_cost']).all() and len(value['source_cost'])==4
    assert value['probe'] is None


def test_source_tracking_cost_preserves_association_information_without_gt():
    package=load_trackeval(os.environ['V34_TRACKEVAL'])
    reference=tracked()
    assert mot_targets.source_components(package,reference,reference)==pytest.approx([0,0,0])
    value=mot_targets.source_components(package,tracked((1,1,2,2)),reference)
    assert value[0]>0 and value[1]==pytest.approx(0) and value[2]>0
