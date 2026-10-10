"""Frozen task measurements with separate GT and source-consistency provenance."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from ..v32vc.train import tensor
from ..v33ap.tasks import TaskHeads,ar_forward,od_forward,frame_indices,prediction,distance
from ..v32.tracking import iou_matrix

def detection_distance(reference,current):
    values=[]
    for ref,cur in zip(reference,current):
        a=np.where(np.asarray(ref['scores'])>=.25)[0]; b=np.where(np.asarray(cur['scores'])>=.1)[0]
        if not len(a): values.append(float(np.mean(np.asarray(cur['scores'])[b])) if len(b) else 0.); continue
        if not len(b): values.append(1.); continue
        overlap=iou_matrix(ref['boxes'][a],cur['boxes'][b])
        overlap*=np.asarray(ref['labels'])[a,None]==np.asarray(cur['labels'])[None,b]
        ii,jj=linear_sum_assignment(1-overlap)
        penalty=np.ones(len(a)); penalty[ii]=1-overlap[ii,jj]
        values.append(float(penalty.mean()))
    return float(np.mean(values))

class Heads(TaskHeads):
    def __init__(self,device,experiment):
        super().__init__(device); self.experiment=experiment
        # Lock detector transform as well as weights; avoid expensive upsampling
        # of every low-resolution AR frame to 800px during candidate search.
        self.detector.model.transform.min_size=(320,)
        self.detector.model.transform.max_size=640

    def hashes(self):
        value=super().hashes(); value['od']['transform']={'min_size':[320],'max_size':640}
        return value

    @torch.no_grad()
    def detections(self,rgb):
        x=tensor(rgb,self.device)
        result=[]
        for start in range(0,len(rgb),4):
            values=self.detector.model([x[0,:,i] for i in range(start,min(start+4,len(rgb)))])
            result.extend(prediction(v) for v in values)
        return result

    @torch.no_grad()
    def source(self,r):
        r.source_predictions={'od':self.detections(r.rgb)}; r.source_features={}
        if len(r.rgb)>1:
            logits,f=ar_forward(self.action,tensor(r.rgb,self.device))
            r.source_predictions['ar']=logits.softmax(-1)[0].cpu().numpy(); r.source_features['ar']=f.cpu()
        r.mask=[len(r.rgb)>1,self.experiment=='ar_od',self.experiment=='ar_ot' and len(r.rgb)>1]
        r.provenance=[('K400 label + source KL/feature' if r.task=='ar' else 'source KL/feature only') if r.mask[0] else None,
            ('COCO boxes + source feature' if r.task=='od' else 'source detection/feature consistency only') if r.mask[1] else None,
            ('MOT20 GT boxes/IDs, local window surrogate' if r.task=='ot' else 'source tracks consistency only') if r.mask[2] else None]
        if r.mask[1]:
            _,f=od_forward(self.detector,tensor(r.rgb,self.device),frame_indices(len(r.rgb)))
            r.source_features['od']=f.cpu()
        if r.mask[2]:
            from .tracking import track_frames
            r.source_predictions['ot']=track_frames(r.source_predictions['od'])

    @torch.no_grad()
    def costs(self,rgb,r):
        x=tensor(rgb,self.device); result=[None]*3
        if r.mask[0]:
            logits,f=ar_forward(self.action,x)
            target=torch.as_tensor(r.source_predictions['ar'],device=self.device)[None]
            value=.25*F.kl_div(logits.log_softmax(-1),target,reduction='batchmean')+.1*distance(f,r.source_features['ar'])
            if r.task=='ar': value=value+F.cross_entropy(logits,torch.tensor([r.item['label']],device=self.device))
            result[0]=float(value)
        if r.mask[1]:
            indices=frame_indices(len(rgb))
            if r.task=='od':
                targets=[{k:v.to(self.device) for k,v in r.targets.items()}]
                loss,f=od_forward(self.detector,x,indices,targets)
                value=float(sum(loss.values()))
            else:
                preds,f=od_forward(self.detector,x,indices)
                value=detection_distance([r.source_predictions['od'][i] for i in indices],[prediction(v) for v in preds])
            result[1]=value+.1*float(distance(f,r.source_features['od']))
        if r.mask[2]:
            from .tracking import track_frames, tracking_cost
            frames=track_frames(self.detections(rgb))
            result[2]=tracking_cost(frames,r.targets if r.task=='ot' else r.source_predictions['ot'],ground_truth=r.task=='ot')
        return result
