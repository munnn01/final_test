"""MOT20 joint receiver: tracking metrics and unscored clip action predictions."""
import argparse
from pathlib import Path
import numpy as np
import torch
from ..selection import ar_guard
from ..preprocessing import normalize_map
from ..rateaware import semantic_protection
from ..v32 import tracking as base
from ..v32rf.swin import SwinAnalyzer
from .receiver import SharedReceiver
from .protocol import codec_factory, settings
from ..v32.protocol import ROOT


class JointTrackingTeacher(base.TrackingTeacher):
    def __init__(self, detector, action):
        super().__init__(detector)
        self.action = action

    def predict(self, clip):
        result = super().predict(clip)
        result['action_probabilities'] = self.action.probabilities(clip)
        return result


class JointGuard:
    def __init__(self): self.records=[]

    def __call__(self, task, source, anchor, trial, cfg):
        record = base.tracking_guard_details(task, source, anchor, trial, cfg)
        distance, decision = ar_guard(source[0]['action_probabilities'], anchor[0]['action_probabilities'],
            trial[0]['action_probabilities'], cfg['ar_confidence'],
            require_anchor_decision=cfg['ar_require_anchor_decision'])
        action_ok = np.isfinite(distance) and distance <= cfg['ar_kl_slack'] and decision
        record['ar_guard'] = dict(relative_kl=float(distance), preserves_decision=bool(decision), accepted=bool(action_ok))
        record['joint_accepted'] = bool(record['preserves_decision'] and action_ok)
        record['reconstructed_pixel_sha256'] = trial[0]['rgb_sha256']
        self.records.append(record)
        # Tracking distance uses its own slack; AR distance uses its own slack above.
        return (record['maximum_relative_distance'],), (record['joint_accepted'],)


def source_support(rgb, prediction, motion=False, *, action):
    union, support = base.source_support(rgb, prediction, motion)
    saliency = normalize_map(action.saliency(rgb))
    if saliency.shape != rgb.shape[1:3] or not np.isfinite(saliency).all():
        raise ValueError('action saliency geometry or values invalid')
    protected = semantic_protection(saliency)
    support['protection'] = np.maximum(support['protection'], protected[None])
    union = support['protection'].max(0)
    support['metadata'].update(task='ot', active_tasks=['ar','ot'],
        roi_source='union of predicted person boxes, motion and source Swin saliency',
        action_ground_truth=None)
    return union, support


def run(args):
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    action=SwinAnalyzer(device)
    receiver_holder={}
    def receive(packet, *, sequence, arm, frame_start, detector, tracker):
        if 'receiver' not in receiver_holder:
            receiver_holder['receiver']=SharedReceiver(action,detector)
        return receiver_holder['receiver'].predict(packet,sequence=sequence,arm=arm,
            frame_start=frame_start,tasks=('ar','ot'),tracker=tracker)
    extension=dict(teacher_factory=lambda d:JointTrackingTeacher(d,action), guard_factory=JointGuard,
        support=lambda rgb,p,m:source_support(rgb,p,m,action=action), receiver=receive,
        codec_factory=codec_factory('ot'), manifest=dict(scope='v32at_joint_tracking',active_tasks=['ar','ot'],
            primary_action_head='swin3d_t', protocol_v32at=settings(),
            action_accuracy=None,action_bd_rate=None,
            action_metric_reason='MOT20 has no action labels; clip predictions only',
            action_scope='whole received window; no action per track ID'))
    return base.run(args,extension=extension)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trackeval-root',type=Path,required=True)
    p.add_argument('--sequences',nargs='+',default=['MOT20-01'])
    p.add_argument('--max-frames',type=int,default=128)
    p.add_argument('--size',type=int,default=640)
    p.add_argument('--device',choices=['cpu','cuda'])
    p.set_defaults(config=ROOT/'configs/v32at.json',motion_support=True)
    return p


if __name__=='__main__': run(parser().parse_args())
