"""Source-only soft OD context and causal track-coordinate OT protection."""
import cv2
import numpy as np
from ..preprocessing import validate_clip
from ..motion_support import _flow_pair
from ..v34.preprocessing import region_maps as ar_maps, filter_clip


def union_maps(maps):
    if not maps: raise ValueError('at least one task map required')
    for m in maps:
        if (m.shape != maps[0].shape or not np.isfinite(m).all() or m.min()<0
            or not np.allclose(m.sum(-1),1,atol=1e-5)):
            raise ValueError('invalid region partition')
    if len(maps)==1: return maps[0].copy()
    core=np.maximum.reduce([m[...,0] for m in maps])
    protected=np.maximum.reduce([m[...,0]+m[...,1] for m in maps])
    context=np.maximum(0,protected-core)
    return np.stack((core,context,np.maximum(0,1-protected)),-1).astype(np.float32)


def _boxes(pred):
    boxes=np.asarray(pred['boxes'],np.float32).reshape(-1,4)
    scores=np.asarray(pred['scores'],np.float32)
    if scores.shape!=(len(boxes),) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError('invalid source detections')
    return boxes,scores


def _paint(core, box, value):
    h,w=core.shape
    x1,y1=np.maximum(0,np.floor(box[:2])).astype(int)
    x2,y2=np.minimum([w,h],np.ceil(box[2:])).astype(int)
    if x2>x1 and y2>y1: core[y1:y2,x1:x2]=np.maximum(core[y1:y2,x1:x2],value)


def _regions(shape, boxes, scores, margin_fraction, min_margin,weights=None):
    core=np.zeros(shape,np.float32); context=np.zeros_like(core)
    weights=np.ones(len(boxes)) if weights is None else weights
    for box,score,weight in zip(boxes,scores,weights):
        local=np.zeros_like(core); _paint(local,box,1.)
        if not local.any(): continue
        # Small objects retain a minimum pixel halo; uncertain boxes get wider context.
        radius=max(min_margin,margin_fraction*max(box[2:]-box[:2]))*(2-float(np.clip(score,0,1)))
        distance=cv2.distanceTransform((1-local).astype(np.uint8),cv2.DIST_L2,3)
        context=np.maximum(context,np.clip(1-distance/max(radius,1),0,1)*weight)
        core=np.maximum(core,local*weight)
    context=np.maximum(0,context-core)
    return np.stack((core,context,np.maximum(0,1-core-context)),-1)


class Support:
    def __init__(self, margin_fraction=.15, min_margin=4, history_decay=.7, history_frames=8):
        if not 0<history_decay<1 or min_margin<1 or margin_fraction<0 or history_frames<0:
            raise ValueError('invalid support settings')
        self.margin_fraction=margin_fraction; self.min_margin=min_margin
        self.history_decay=history_decay; self.history_frames=history_frames
        self.previous=None; self.tracks={}

    def build(self,rgb,predictions,mask,tracks=None):
        validate_clip(rgb)
        if len(predictions)!=len(rgb) or len(mask)!=3 or not any(mask): raise ValueError('invalid task support')
        if mask[2] and (tracks is None or len(tracks)!=len(rgb)): raise ValueError('OT needs associated source tracks')
        maps=[]
        if mask[0]: maps.append(ar_maps(rgb,predictions))  # exact legacy AR branch
        if mask[1]:
            od=[]
            for p in predictions:
                boxes,scores=_boxes(p); keep=scores>=.25
                od.append(_regions(rgb.shape[1:3],boxes[keep],scores[keep],self.margin_fraction,self.min_margin))
            maps.append(np.stack(od))
        if mask[2]:
            ot=[]
            for frame,p in zip(rgb,tracks):
                gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                if self.previous is not None:
                    if self.previous.shape!=gray.shape or _flow_pair(self.previous,gray)[2]: self.tracks.clear()
                boxes,scores=_boxes(p); ids=np.asarray(p['track_ids'])
                if ids.shape!=scores.shape or len(set(ids.tolist()))!=len(ids): raise ValueError('invalid associated tracks')
                seen=set()
                for key,box,score in zip(ids.tolist(),boxes,scores):
                    seen.add(key); old=self.tracks.get(key)
                    velocity=np.zeros(4,np.float32) if old is None else box-old['box']
                    self.tracks[key]=dict(box=box.copy(),velocity=velocity,score=float(score),age=0)
                for key in list(self.tracks):
                    if key in seen: continue
                    old=self.tracks[key]; old['age']+=1
                    if old['age']>self.history_frames: del self.tracks[key]; continue
                    old['box']=old['box']+old['velocity']; old['score']*=self.history_decay
                # Only ByteTrack-associated detections enter; unmatched low scores never do.
                values=list(self.tracks.values())
                regions=_regions(gray.shape,[v['box'] for v in values],[v['score'] for v in values],self.margin_fraction,self.min_margin,
                    weights=[self.history_decay**v['age'] for v in values])
                ot.append(regions); self.previous=gray
            maps.append(np.stack(ot))
        return union_maps(maps)
