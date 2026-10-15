"""Independent frozen GT targets and label-free source constraints."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from ..v34.tasks import Heads as R1Heads
from ..v33ap.tasks import ar_forward,od_forward,distance
from ..v32vc.train import tensor
from ..v32.tracking import iou_matrix
from ..v34.tracking import tracking_cost


def detection_distance(reference,current):
    if len(reference)!=len(current) or not reference: raise ValueError('detection frame count mismatch')
    costs=[]
    for ref,cur in zip(reference,current):
        a=np.where(np.asarray(ref['scores'])>=.25)[0]; b=np.where(np.asarray(cur['scores'])>=.1)[0]
        boxes=np.asarray(ref['boxes']).reshape(-1,4)[a]; other=np.asarray(cur['boxes']).reshape(-1,4)[b]
        overlap=iou_matrix(boxes,other)
        same=np.asarray(ref['labels'])[a,None]==np.asarray(cur['labels'])[None,b]
        matches=[]
        if overlap.size:
            ii,jj=linear_sum_assignment(np.where(same&(overlap>=.1),1-overlap,1e6))
            matches=[(i,j) for i,j in zip(ii,jj) if same[i,j] and overlap[i,j]>=.1]
        error=float(len(a)+len(b)-2*len(matches))
        for i,j in matches:
            error+=1-overlap[i,j]+abs(float(np.asarray(ref['scores'])[a[i]])-float(np.asarray(cur['scores'])[b[j]]))
        costs.append(error/max(len(a),1))
    return float(np.mean(costs))


def context_hota(package,frames,reference):
    """Official HOTA on a finite positive-pedestrian context, in [0,1].

    This offline target omits MOT distractor preprocessing and is labelled a
    context target; official full-sequence evaluation still uses the dataset API.
    """
    if len(frames)!=len(reference) or not frames: raise ValueError('HOTA context mismatch')
    def remap(values):
        ids=sorted({int(i) for p in values for i in p['track_ids']}); lookup={v:i for i,v in enumerate(ids)}
        return [np.array([lookup[int(i)] for i in p['track_ids']],int) for p in values],len(ids)
    gt_ids,ng=remap(reference); tr_ids,nt=remap(frames)
    data=dict(gt_ids=gt_ids,tracker_ids=tr_ids,num_gt_ids=ng,num_tracker_ids=nt,
        num_gt_dets=sum(map(len,gt_ids)),num_tracker_dets=sum(map(len,tr_ids)),num_timesteps=len(frames),
        similarity_scores=[iou_matrix(np.asarray(a['boxes']).reshape(-1,4),np.asarray(b['boxes']).reshape(-1,4)) for a,b in zip(reference,frames)])
    value=package.metrics.HOTA().eval_sequence(data)
    return {k:float(np.mean(value[k])) for k in ('HOTA','DetA','AssA')}


class Heads(R1Heads):
    @torch.no_grad()
    def prepare(self,record,mask,source_prefix=None):
        start=time.perf_counter(); record.mask=list(mask)
        record.source_predictions={'od':self.detections(record.rgb)}; record.source_features={}
        record.source_detector_seconds=time.perf_counter()-start
        if mask[0]:
            logits,f=ar_forward(self.action,tensor(record.rgb,self.device))
            record.source_predictions['ar']=logits.softmax(-1)[0].cpu().numpy(); record.source_features['ar']=f.cpu()
        if mask[2]:
            if source_prefix is None: raise ValueError('continuous source prefix required')
            probe=source_prefix.probe(record.source_predictions['od'])
            record.source_predictions['ot']=probe.current
            record.source_predictions['ot_context']=probe.frames
            source_prefix.commit(probe)
        record.source_seconds=time.perf_counter()-start
        return record

    @torch.no_grad()
    def score(self,rgb,reference,*,prefix=None,gt=None,gt_context=None,package=None):
        """GT is an explicit TRAIN/CAL-only argument, absent during selection."""
        source=[None]*3; target=[None]*3; predictions={}; probe=None; diagnostics={}
        x=None
        if reference.mask[0]:
            x=tensor(rgb,self.device); logits,f=ar_forward(self.action,x)
            predictions['ar']=logits.softmax(-1)[0].cpu().numpy()
            clean=torch.as_tensor(reference.source_predictions['ar'],device=self.device)[None]
            source[0]=float(.25*F.kl_div(logits.log_softmax(-1),clean,reduction='batchmean')+.1*distance(f,reference.source_features['ar']))
            if gt is not None and gt.task=='ar': target[0]=float(F.cross_entropy(logits,torch.tensor([gt.item['label']],device=self.device)))
        if reference.mask[1] or reference.mask[2]:
            predictions['od']=self.detections(rgb)
        if reference.mask[1]:
            source[1]=detection_distance(reference.source_predictions['od'],predictions['od'])
            if gt is not None and gt.task=='od':
                devices=[torch.device(self.device).index or 0] if str(self.device).startswith('cuda') else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(303401)
                    loss,_=od_forward(self.detector,tensor(rgb,self.device),(0,),[{k:v.to(self.device) for k,v in gt.targets.items()}])
                target[1]=float(sum(loss.values()))
        if reference.mask[2]:
            if prefix is None: raise ValueError('candidate prefix required')
            probe=prefix.probe(predictions['od'])
            clean=reference.source_predictions['ot_context']
            if len(probe.frames)!=len(clean): raise ValueError('candidate/source context misalignment')
            source[2]=tracking_cost(probe.frames,clean)
            if gt is not None and gt.task=='ot':
                if package is None or gt_context is None: raise ValueError('offline GT HOTA context required')
                diagnostics=context_hota(package,probe.frames,gt_context)
                target[2]=1-diagnostics['HOTA']
        return dict(source_cost=source,gt_cost=target,predictions=predictions,probe=probe,context_metrics=diagnostics)
