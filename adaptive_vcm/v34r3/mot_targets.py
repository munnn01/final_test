"""Causal context metrics; official MOT preprocessing is separate from source consistency."""
from pathlib import Path
import numpy as np
from ..v32.tracking import iou_matrix
from ..v34r2.tasks import context_hota


def read_context(path,start,stop):
    if start<1 or stop<=start: raise ValueError('invalid causal frame interval')
    path=Path(path)
    rows=np.loadtxt(path,delimiter=',',ndmin=2) if path.stat().st_size else np.empty((0,9))
    if (rows.shape[1]<9 or not np.isfinite(rows).all()
        or np.any(rows[:,0]!=np.floor(rows[:,0])) or np.any(rows[:,1]!=np.floor(rows[:,1]))):
        raise ValueError('invalid raw MOT annotations')
    frames=[]
    for number in range(start,stop):
        values=rows[rows[:,0]==number]; boxes=values[:,2:6].copy();boxes[:,2:]+=boxes[:,:2]
        frames.append(dict(boxes=boxes,track_ids=values[:,1].astype(int),classes=values[:,7].astype(int),
                           zero_marked=values[:,6].copy(),frame_number=number))
    return frames


def components(package, tracks, annotations, geometry, *, benchmark, sequence):
    if len(tracks)!=len(annotations) or not tracks or benchmark not in ('MOT17','MOT20'):
        raise ValueError('invalid MOT context alignment/benchmark')
    # Only initialize the attributes used by the pinned official preprocessing
    # function. No filesystem dataset is needed for an already-decoded context.
    dataset=package.datasets.MotChallenge2DBox.__new__(package.datasets.MotChallenge2DBox)
    dataset.benchmark=benchmark;dataset.do_preproc=True
    dataset.class_name_to_class_id=dict(pedestrian=1,person_on_vehicle=2,car=3,bicycle=4,motorbike=5,
        non_mot_vehicle=6,static_person=7,distractor=8,occluder=9,occluder_on_ground=10,
        occluder_full=11,reflection=12,crowd=13)
    dataset.valid_class_numbers=list(dataset.class_name_to_class_id.values())
    raw=dict(num_timesteps=len(tracks),seq=sequence)
    for key in ('gt_ids','tracker_ids','gt_dets','tracker_dets','gt_classes','tracker_classes',
                'gt_extras','tracker_confidences','similarity_scores'): raw[key]=[]
    sx,sy,left,top=map(float,geometry)
    if sx<=0 or sy<=0: raise ValueError('positive source geometry required')
    for tr,gt in zip(tracks,annotations):
        a=np.asarray(gt['boxes'],float).reshape(-1,4)
        b=np.asarray(tr['boxes'],float).reshape(-1,4).copy()
        b[:,[0,2]]=(b[:,[0,2]]-left)/sx;b[:,[1,3]]=(b[:,[1,3]]-top)/sy
        aw=a.copy();aw[:,2:]-=aw[:,:2];bw=b.copy();bw[:,2:]-=bw[:,:2]
        raw['gt_ids'].append(np.asarray(gt['track_ids'],int));raw['tracker_ids'].append(np.asarray(tr['track_ids'],int))
        raw['gt_dets'].append(aw);raw['tracker_dets'].append(bw)
        raw['gt_classes'].append(np.asarray(gt['classes'],int));raw['tracker_classes'].append(np.ones(len(b),int))
        raw['gt_extras'].append(dict(zero_marked=np.asarray(gt['zero_marked'])))
        raw['tracker_confidences'].append(np.asarray(tr['scores'],float));raw['similarity_scores'].append(iou_matrix(a,b))
    data=dataset.get_preprocessed_seq_data(raw,'pedestrian')
    value=package.metrics.HOTA().eval_sequence(data)
    quality={k:float(np.mean(value[k])) for k in ('HOTA','DetA','AssA')}
    return dict(cost=np.array([1-quality[k] for k in ('HOTA','DetA','AssA')]),quality=quality,
                scope='official MOT-preprocessed finite causal context; not full-sequence HOTA')


def source_components(package, tracks, source_tracks):
    quality=context_hota(package,tracks,source_tracks)
    return np.array([1-quality[k] for k in ('HOTA','DetA','AssA')])
