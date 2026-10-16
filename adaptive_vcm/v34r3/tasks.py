"""Source-only scoring and a distinct explicit offline annotation API."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .protocol import task_mask
from . import coco_targets,mot_targets
from ..v34r2.tasks import Heads as R2Heads, detection_distance,context_hota
from ..v34.tracking import tracking_cost
from ..v32.tracking import iou_matrix


def detection_components(reference,current):
    values=[]
    for ref,cur in zip(reference,current):
        a=np.where(np.asarray(ref['scores'])>=.25)[0];b=np.where(np.asarray(cur['scores'])>=.1)[0]
        overlap=iou_matrix(np.asarray(ref['boxes']).reshape(-1,4)[a],np.asarray(cur['boxes']).reshape(-1,4)[b])
        same=np.asarray(ref['labels'])[a,None]==np.asarray(cur['labels'])[None,b]
        pairs=[]
        if overlap.size:
            ii,jj=linear_sum_assignment(np.where(same&(overlap>=.1),1-overlap,1e6))
            pairs=[(i,j) for i,j in zip(ii,jj) if same[i,j] and overlap[i,j]>=.1]
        loc=[1-overlap[i,j] for i,j in pairs]
        confidence=[abs(float(ref['scores'][a[i]])-float(cur['scores'][b[j]])) for i,j in pairs]
        values.append([(len(a)-len(pairs))/max(len(a),1),(len(b)-len(pairs))/max(len(a),1),
                       float(np.mean(loc)) if loc else 0.,float(np.mean(confidence)) if confidence else 0.])
    if not values: raise ValueError('detection context cannot be empty')
    return np.mean(values,axis=0)


class Heads(R2Heads):
    @torch.no_grad()
    def prepare(self,source,task,source_prefix=None):
        if task not in ('od','ot'): raise ValueError('AR uses its native frozen adapter')
        source.task=task
        return super().prepare(source,task_mask(task),source_prefix)

    @torch.no_grad()
    def source_score(self,rgb,source,*,prefix,package=None):
        predictions={'od':self.detections(rgb)};probe=None
        if source.task=='od':
            cost=detection_components(source.source_predictions['od'],predictions['od'])
            legacy=detection_distance(source.source_predictions['od'],predictions['od'])
        elif source.task=='ot':
            if prefix is None or package is None: raise ValueError('OT source score needs prefix and pinned evaluator')
            probe=prefix.probe(predictions['od']);clean=source.source_predictions['ot_context']
            cost=mot_targets.source_components(package,probe.frames,clean)
            legacy=tracking_cost(probe.frames,clean)
        else: raise ValueError('unknown source task')
        return dict(source_cost=cost,predictions=predictions,probe=probe,legacy_source_cost=float(legacy),
                    source_scope='frozen source prediction consistency; no GT accuracy guarantee')

    @torch.no_grad()
    def gt_score(self,rgb,source,*,annotations=None,context=None,package=None,
                 source_result=None,legacy_record=None,prefix=None,benchmark=None):
        value=source_result or self.source_score(rgb,source,prefix=prefix,package=package)
        if source.task=='od':
            if annotations is None: raise ValueError('offline COCO annotations required')
            costs=coco_targets.components(value['predictions']['od'][0],annotations,source.item,source.geometry,
                                           score_cutoff=self.cfg.get('score_cutoff',.1))
            legacy=None
            if legacy_record is not None:
                legacy=super().score(rgb,source,gt=legacy_record)['gt_cost'][1]
            return dict(gt_cost=costs,legacy_gt_cost=legacy,context_metrics={},
                        target_scope='COCO-matched inference error components; not cohort mAP')
        if context is None or package is None: raise ValueError('offline MOT context required')
        sequence=source.item['sequence'];benchmark=benchmark or ('MOT17' if sequence.startswith('MOT17') else 'MOT20')
        metric=mot_targets.components(package,value['probe'].frames,context,source.geometry,
                                     benchmark=benchmark,sequence=sequence)
        sx,sy,left,top=source.geometry;positive=[]
        for frame in context:
            keep=(np.asarray(frame['classes'])==1)&(np.asarray(frame['zero_marked'])!=0)
            boxes=np.asarray(frame['boxes'])[keep].copy()
            boxes[:,[0,2]]=boxes[:,[0,2]]*sx+left;boxes[:,[1,3]]=boxes[:,[1,3]]*sy+top
            positive.append(dict(boxes=boxes,track_ids=np.asarray(frame['track_ids'])[keep]))
        legacy=1-context_hota(package,value['probe'].frames,positive)['HOTA']
        return dict(gt_cost=metric['cost'],legacy_gt_cost=legacy,context_metrics=metric['quality'],target_scope=metric['scope'])
