import copy
from types import SimpleNamespace
import numpy as np
from adaptive_vcm.v32.protocol import load_config
from adaptive_vcm.v32.tracking import TrackingTeacher
from tests.test_v32at_receiver import Action, Detector


def test_joint_guard_rejects_either_task_and_support_union():
    from adaptive_vcm.v32at.tracking import JointTrackingTeacher, JointGuard, source_support
    rgb = np.zeros((3,16,16,3),np.uint8)
    action = Action()
    action.saliency = lambda x: np.pad(np.ones((1,1)), ((12,3),(12,3)))
    teacher = JointTrackingTeacher(Detector(),action)
    source = teacher.predict(rgb)
    anchor = teacher.predict(rgb)
    trial = copy.deepcopy(anchor)
    trial['action_probabilities'] = np.array([.1,.9])
    guard=JointGuard()
    _, decision=guard('ot',[source],[anchor],[trial],load_config())
    assert decision == (False,)
    assert guard.records[-1]['ar_guard']['preserves_decision'] is False
    trial = copy.deepcopy(anchor)
    trial['frames'][1]['track_ids'][0] = 99
    _,decision=guard('ot',[source],[anchor],[trial],load_config())
    assert decision == (False,)
    union,support=source_support(rgb,source,True,action=action)
    assert support['protection'].shape==(3,16,16)
    assert union[12,12]>0 and union[3,3]>0


def test_missing_person_evidence_fails_closed():
    from adaptive_vcm.v32at.tracking import JointTrackingTeacher, JointGuard
    class Empty(Detector):
        def predict(self,rgb):
            return dict(boxes=np.empty((0,4)),scores=np.empty(0),labels=np.empty(0,int))
    teacher=JointTrackingTeacher(Empty(),Action())
    p=teacher.predict(np.zeros((2,16,16,3),np.uint8))
    _,ok=JointGuard()('ot',[p],[p],[p],load_config())
    assert ok==(False,)


def test_joint_runner_real_codec_unpadded_and_official_metrics(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from argparse import Namespace
    import pytest
    from tests.test_v32_tracking import fixture_sequence
    from adaptive_vcm.v32 import tracking as base
    from adaptive_vcm.v32at import tracking as joint
    checkout=Path('D:/STUDY/LAB/bao_1/output/v32_tests/trackeval_official')
    if not checkout.exists(): pytest.skip('official pinned TrackEval unavailable')
    class FullFrameDetector:
        def __init__(self,name,device): self.name=name
        def predict(self,clip):
            return dict(boxes=np.array([[0,0,clip.shape[2],clip.shape[1]]],float),scores=np.array([.9]),labels=np.array([1]))
    action=Action(); action.saliency=lambda rgb:np.zeros(rgb.shape[1:3])
    monkeypatch.setattr(base,'DetectionAnalyzer',FullFrameDetector)
    monkeypatch.setattr(joint,'SwinAnalyzer',lambda device:action)
    fixture_sequence(tmp_path/'source',17)
    summary=joint.run(Namespace(root=tmp_path/'source',out=tmp_path/'out',trackeval_root=checkout,
        sequences=['MOT20-01'],max_frames=17,size=64,config=None,device='cpu',motion_support=True))
    assert summary['complete_paired_grid'] and len(summary['points'])==50
    assert summary['action_accuracy'] is None and summary['action_bd_rate'] is None
    rows=[json.loads(x) for x in (tmp_path/'out/action_predictions.jsonl').read_text().splitlines()]
    assert len(rows)==100
    assert {r['frame_count'] for r in rows}=={16,1}
    for p in summary['points']:
        lines=(tmp_path/'out'/p['prediction_paths']['MOT20-01']).read_text().splitlines()
        assert lines[-1].split(',')[:2]==['17','1']
