"""Pinned standalone cohorts and independent source/camera identities."""
import hashlib
import configparser
from fractions import Fraction
import json
from pathlib import Path
import re
from .protocol import digest
from ..v33ap.sources import SourceRecord,read_source
from ..v34.sources import cohorts
from ..v32vc.train import od_targets
from ..v32.tracking import read_sequence


def camera_id(sequence):return re.sub(r'-(DPM|FRCNN|SDP)$','',sequence)


def group_id(record):
    sequence=record.item.get('sequence')
    return camera_id(sequence) if sequence else record.id


def plan_sequence(root,name,frames):
    """R3 local mounts retain official sequence names and original FPS metadata."""
    if not re.fullmatch(r'MOT(?:17|20)-\d{2}(?:-(?:DPM|FRCNN|SDP))?',name) or frames<1:
        raise ValueError('explicit MOT17/MOT20 sequence and positive frame scope required')
    path=Path(root)/name;parser=configparser.ConfigParser()
    if not parser.read(path/'seqinfo.ini'):raise FileNotFoundError('missing MOT seqinfo.ini')
    seq=parser['Sequence'];total,width,height=(int(seq[k]) for k in ('seqLength','imWidth','imHeight'))
    fps=Fraction(seq['frameRate'])
    if (min(total,width,height)<1 or fps<=0 or seq['name']!=name or seq['imDir']!='img1' or seq['imExt']!='.jpg'):
        raise ValueError('invalid MOT sequence metadata')
    images=[path/'img1'/f'{n:06}.jpg' for n in range(1,min(frames,total)+1)]
    for file in [*images,path/'gt/gt.txt']:
        if not file.is_file():raise FileNotFoundError('missing MOT file: '+str(file))
    return dict(name=name,path=path,images=images,frames=len(images),full_frames=total,width=width,height=height,fps=fps)


def validate_cohorts(groups,history):
    sources={};pixels=set();assigned={}
    for split,records in groups.items():
        if split not in ('train','cal','dev','test'):raise ValueError('unknown cohort split')
        for r in records:
            sha=hashlib.sha256(r.rgb.tobytes()).hexdigest();group=group_id(r)
            if r.id in sources or sha in pixels:raise ValueError('duplicate source identity/pixels')
            if group in assigned and assigned[group]!=split:raise ValueError('sequence/camera split leakage')
            assigned[group]=split;pixels.add(sha)
            sources[r.id]=dict(split=split,domain=r.task,group=group,pixel_sha256=sha,
                frames=len(r.rgb),frame_recipe=r.frame_recipe,geometry=list(r.geometry),
                source_metadata={k:r.item[k] for k in ('sequence','start','width','height','image_id') if k in r.item})
    known_ids=set(history.get('ids',[]));known_pixels=set(history.get('pixels',[]));known_groups=set(history.get('groups',[]))
    seen=[key for key,s in sources.items() if s['split']=='test' and
          (key in known_ids or s['pixel_sha256'] in known_pixels or s['group'] in known_groups)]
    confirmation=bool(history.get('audited') and not seen and any(s['split']=='test' for s in sources.values()))
    return dict(sources=sources,history_sha256=digest(history),history_audited=bool(history.get('audited')),
                historical_test_ids=seen,independent_confirmation=confirmation,
                scope='audited unused confirmation cohort' if confirmation else 'exploratory; confirmation independence not established')


def load_groups(args,recipe):
    task=args.experiment;cfg=dict(recipe);manifest={}
    if getattr(args,'cohort_manifest',None):
        manifest=json.loads(Path(args.cohort_manifest).read_text(encoding='utf-8'))
        if manifest.get('sha256')!=digest({k:v for k,v in manifest.items() if k!='sha256'}):
            raise ValueError('cohort manifest is not content-pinned')
        if 'ot_sequences' in manifest:cfg['ot_sequences']=manifest['ot_sequences']
    if task=='od' and 'od_ids' in manifest:
        meta=json.loads(Path(args.annotations).read_text(encoding='utf-8'));images={i['id']:i for i in meta['images']}
        groups={s:[] for s in ('train','cal','dev','test')}
        if set(manifest['od_ids'])!=set(groups):raise ValueError('all OD splits must be explicit')
        for split,ids in manifest['od_ids'].items():
            for key in ids:
                im=images[int(key)];item=dict(id=str(key),image_id=int(key),path=str(Path(args.od_root)/im['file_name']),
                    width=im['width'],height=im['height'])
                rgb,g,frame_recipe=read_source('od',item,cfg)
                r=SourceRecord('od/coco2017/'+str(key),'od',item,rgb,g,frame_recipe);r.split=split
                if split in ('train','cal'):r.targets=od_targets(meta,item,g,'cpu')
                groups[split].append(r)
        metadata={'od':meta}
    else:
        groups,metadata=cohorts(getattr(args,'ar_root',None),getattr(args,'od_root',None),
            getattr(args,'annotations',None),cfg,task)
        if task=='od':
            for split in ('dev','test'):
                for r in groups[split]:r.targets={}
    if task=='ot':
        sequence_splits=cfg['ot_sequences'];seen=set();plans=[]
        if set(sequence_splits)!=set(groups):raise ValueError('all OT sequence splits required')
        for split,names in sequence_splits.items():
            for name in names:
                canonical=camera_id(name)
                if canonical in seen:raise ValueError('duplicate camera in OT splits')
                seen.add(canonical)
                item=plan_sequence(Path(args.mot_root),name,cfg['ot_frames'])
                if item['frames']!=cfg['ot_frames'] or item['frames']%cfg['frames']:raise ValueError('complete contiguous windows required')
                rgb,geometry=read_sequence(item,cfg['ot_size']);plans.append(item)
                for offset in range(0,len(rgb),cfg['frames']):
                    start=offset+1;info=dict(sequence=name,start=start,width=item['width'],height=item['height'],path=str(item['path']))
                    r=SourceRecord(f'ot/{name}/{start:06d}','ot',info,rgb[offset:offset+cfg['frames']].copy(),geometry,
                                   dict(kind='contiguous-MOT',frame_indices=list(range(start,start+cfg['frames']))))
                    r.split=split;groups[split].append(r)
        metadata['ot']=plans
    metadata['cohort_audit']=validate_cohorts(groups,manifest.get('history',{}))
    metadata['cohort_input_sha256']=manifest.get('sha256')
    return groups,metadata
