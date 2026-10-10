import hashlib
import torch
import pytest


def metadata():
    from adaptive_vcm.data import partition
    from adaptive_vcm.v33ap.protocol import settings,source_id,split_train
    splits={}; hashes={}
    for task in ('ar','od'):
        raw=[]; n=0
        while len(raw)<64:
            candidate=f'a/{n}.mp4' if task=='ar' else str(n)
            if partition(candidate if task=='ar' else 'coco2017/'+candidate)=='train': raw.append(source_id(task,candidate))
            n+=1
        a,b,c=split_train(raw,task); splits[task]=dict(fit=a,cal=b,val=c)
        hashes.update({i:hashlib.sha256(i.encode()).hexdigest() for i in raw})
    calibration={t:{f'{c}/{q}':dict(step=.1,rate_scale=1.,rate_offset=0.) for c in ('h264','h265') for q in (30,35,40,45,50)} for t in ('ar','od')}
    held_splits={}; held_hashes={}
    for task in ('ar','od'):
        held_splits[task]={}
        for split,count in (('dev',16),('test',32)):
            group=[]; n=0
            while len(group)<count:
                raw=f'a/{n}.mp4' if task=='ar' else str(n)
                if partition(raw if task=='ar' else 'coco2017/'+raw)==split: group.append(source_id(task,raw))
                n+=1
            held_splits[task][split]=group
            held_hashes.update({i:hashlib.sha256(i.encode()).hexdigest() for i in group})
    return dict(arm='joint',settings=settings(),splits=splits,source_pixel_sha256=hashes,
        held_out=dict(splits=held_splits,source_pixel_sha256=held_hashes),
        annotations_sha256='1'*64,supervision_sha256='2'*64,task_models={t:dict(name='fixture',state_sha256='3'*64) for t in ('ar','od')},
        calibration=calibration,history=[dict(epoch=e,steps=960,loss=1.,seconds=1.,actual_bpp=1.,estimated_bpp=1.,gradient_norm=1.,actual_codec_seconds=1.) for e in (1,2)])


def test_strict_state_and_overlap(tmp_path):
    from adaptive_vcm.v33ap.model import JointPreprocessor
    from adaptive_vcm.v33ap.checkpoint import make_checkpoint,load_checkpoint
    import numpy as np
    p=tmp_path/'m.pth'; value=make_checkpoint(JointPreprocessor(),metadata()); torch.save(value,p)
    loaded=load_checkpoint(p)
    with pytest.raises(ValueError,match='overlap'): loaded.assert_source(next(iter(value['metadata']['source_pixel_sha256'])),np.zeros((1,16,16,3),np.uint8))
    value['model']['head.bias'][0]=1.; torch.save(value,p)
    with pytest.raises(ValueError): load_checkpoint(p)


def test_rehashed_bad_split_rejected(tmp_path):
    from adaptive_vcm.v33ap.model import JointPreprocessor
    from adaptive_vcm.v33ap.checkpoint import make_checkpoint,load_checkpoint
    from adaptive_vcm.v32.protocol import canonical_hash
    value=make_checkpoint(JointPreprocessor(),metadata()); value['metadata']['splits']['ar']['cal'][0]=value['metadata']['splits']['ar']['fit'][0]
    value['metadata_sha256']=canonical_hash(value['metadata']); p=tmp_path/'m.pth'; torch.save(value,p)
    with pytest.raises(ValueError): load_checkpoint(p)
