"""Canonical sources, label-free support and namespaced provenance."""
from dataclasses import dataclass,field
import hashlib
from pathlib import Path
import cv2
import numpy as np
import torch
from ..data import ar_plan,od_plan,read_image
from ..motion_support import build_motion_support
from ..preprocessing import normalize_map,boxes_to_mask
from ..rateaware import semantic_protection
from ..v32vc.train import od_targets
from ..v32.protocol import canonical_hash
from .protocol import source_id,split_train


@dataclass
class SourceRecord:
    id:str
    task:str
    item:dict
    rgb:np.ndarray
    geometry:object
    frame_recipe:dict
    support:dict=field(default_factory=dict)
    targets:dict=field(default_factory=dict)
    source_features:dict=field(default_factory=dict)
    source_predictions:dict=field(default_factory=dict)


def read_source(task,item,cfg):
    path=Path(item['path'])
    if not path.is_file(): raise FileNotFoundError(path)
    if task=='od':
        rgb,g=read_image(item,cfg['od_size'])
        return rgb,g,dict(kind='single-image',repeated_frames=0,frame_indices=[0])
    if task!='ar': raise ValueError('task must be ar or od')
    frames,stride,size=(cfg[k] for k in ('frames','temporal_stride','ar_size'))
    if frames<1 or stride<1 or size<2 or size%2: raise ValueError('invalid frame recipe')
    cap=cv2.VideoCapture(str(path)); picked=[]; indices=[]
    try:
        if not cap.isOpened(): raise RuntimeError('cannot open source video')
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); start=max(0,total-frames*stride)//2
        if start: cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        while len(picked)<frames:
            ok,frame=cap.read()
            if not ok: break
            indices.append(start+len(picked)*stride)
            picked.append(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(size,size),interpolation=cv2.INTER_AREA))
            for _ in range(stride-1):
                if not cap.grab(): break
    finally: cap.release()
    if not picked: raise RuntimeError('source decoded zero frames')
    count=len(picked)
    while len(picked)<frames: picked.append(picked[-1].copy()); indices.append(indices[-1])
    return np.stack(picked),None,dict(kind='historical-center-stride',reported_source_frames=total,
        start=start,stride=stride,frame_indices=indices,decoded_unique_frames=count,repeated_frames=frames-count)


def pixel_audit(records):
    hashes={}
    for r in records:
        digest=hashlib.sha256(r.rgb.tobytes()).hexdigest()
        if r.id in hashes or digest in hashes.values(): raise ValueError('duplicate source ID/pixels across namespaces')
        hashes[r.id]=digest
    return hashes


def label_free_support(record,heads,cfg):
    rgb=record.rgb; h,w=rgb.shape[1:3]
    semantic=semantic_protection(normalize_map(heads.action.saliency(rgb))) if record.task=='ar' else np.zeros((h,w),np.float32)
    # Source predicted boxes only; a stable tube protects sampled evidence over the clip.
    for pred in record.source_predictions['od'].values():
        semantic=np.maximum(semantic,boxes_to_mask(h,w,pred['boxes'][pred['scores']>=cfg['od_score_threshold']]))
    return build_motion_support(rgb,semantic,record.task)


def serial(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [serial(v) for v in value]
    return value


def prepare_sources(ar_root,od_root,annotations,heads,cfg):
    records={}; splits={}; availability={}; supervised={}
    for task,root in (('ar',ar_root),('od',od_root)):
        plan,meta=ar_plan(Path(root),'train',cfg['train_count']) if task=='ar' else od_plan(Path(root),Path(annotations),'train',cfg['train_count'])
        if task=='ar' and meta['unmapped_folders']: raise ValueError('unknown K400 source folders: '+str(meta['unmapped_folders']))
        ids=[source_id(task,i['id']) for i in plan]
        fit,cal,val=split_train(ids,task); splits[task]=dict(fit=fit,cal=cal,val=val)
        availability[task]=meta if task=='ar' else dict(available=len(meta['images']))
        for item,identifier in zip(plan,ids):
            rgb,g,recipe=read_source(task,item,cfg)
            # Only FIT labels enter supervised records; CAL/VAL are unlabeled diagnostics.
            clean_item={k:v for k,v in item.items() if k!='label' or identifier in fit}
            target=od_targets(meta,item,g,'cpu') if task=='od' and identifier in fit else {}
            record=SourceRecord(identifier,task,clean_item,rgb,g,recipe,targets=target)
            heads.source_targets(record); record.support=label_free_support(record,heads,cfg)
            records[identifier]=record
            supervised[identifier]=dict(label=clean_item.get('label'),targets=serial(record.targets),
                source_predictions=serial(record.source_predictions),feature_sha256={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in record.source_features.items()})
            print('PREPARE '+identifier,flush=True)
    hashes=pixel_audit(records.values())
    return dict(records=records,splits=splits,source_pixel_sha256=hashes,availability=availability,
        annotations_sha256=hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
        supervision_sha256=canonical_hash(supervised),supervision=supervised,
        frame_recipes={i:r.frame_recipe for i,r in records.items()})
