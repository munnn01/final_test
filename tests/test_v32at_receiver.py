from types import SimpleNamespace
import numpy as np
import pytest
import torch
from adaptive_vcm.v32.tracking import ReferenceTracker


class Action:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.seen = []
    def probabilities(self, rgb):
        self.seen.append(rgb)
        return np.array([.8, .2])


class Detector(Action):
    def predict(self, rgb):
        self.seen.append(rgb)
        return dict(boxes=np.array([[1., 1., 8., 8.]]), scores=np.array([.9]), labels=np.array([1]))


def packet(n=3):
    rgb = np.zeros((n, 16, 16, 3), np.uint8)
    rgb.setflags(write=False)
    return SimpleNamespace(decoded=rgb, packet_sha256='a'*64, coded_bytes=42)


def test_shared_rgb_frozen_models_and_separate_states():
    from adaptive_vcm.v32at.receiver import SharedReceiver
    action, detector = Action(), Detector()
    receiver = SharedReceiver(action, detector)
    p = packet()
    a = receiver.predict(p, sequence='s', arm='h264/q30/a', frame_start=1, tasks=('ar','ot'))
    b = receiver.predict(p, sequence='s', arm='h264/q30/b', frame_start=1, tasks=('ar','ot'))
    assert action.seen[0] is p.decoded and len(action.seen) == 1
    assert len(detector.seen) == 3
    assert a['ot']['frames'][-1]['frame_number'] == b['ot']['frames'][-1]['frame_number'] == 3
    assert a['ot']['frames'][0] is not b['ot']['frames'][0]
    assert a['packet_sha256'] == p.packet_sha256 and a['frame_count'] == 3
    assert a['ar']['accuracy'] is None and a['ar']['bd_rate'] is None
    assert not any(q.requires_grad for model in (action.model, detector.model) for q in model.parameters())
    c = receiver.predict(p, sequence='s', arm='h264/q30/a', frame_start=4, tasks=('ar','ot'))
    assert c['ot']['frames'][-1]['frame_number'] == 6
    with pytest.raises(ValueError, match='ordered'):
        receiver.predict(p, sequence='s', arm='h264/q30/a', frame_start=4, tasks=('ot',))


def test_receiver_rejects_od_and_requires_detector():
    from adaptive_vcm.v32at.receiver import SharedReceiver
    receiver = SharedReceiver(Action())
    with pytest.raises(ValueError):
        receiver.predict(packet(), sequence='s', arm='a', frame_start=1, tasks=('od',))
    with pytest.raises(ValueError, match='detector'):
        receiver.predict(packet(), sequence='s', arm='a', frame_start=1, tasks=('ot',))
