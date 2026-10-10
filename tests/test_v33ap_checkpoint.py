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


def test_actual_rate_history_roundtrip_without_numpy_globals(tmp_path):
    import numpy as np
    from adaptive_vcm.v33ap.model import JointPreprocessor
    from adaptive_vcm.v33ap.checkpoint import make_checkpoint,load_checkpoint
    # real_forward divides by np.prod(shape); epoch accumulation retains float64.
    m=metadata()
    for row in m['history']: row['actual_bpp']=8*101/np.prod((16,128,128))
    p=tmp_path/'real_history.pth'; value=make_checkpoint(JointPreprocessor(),m); torch.save(value,p)
    raw=torch.load(p,map_location='cpu',weights_only=True)
    assert all(type(row['actual_bpp']) is float for row in raw['metadata']['history'])
    assert load_checkpoint(p).metadata['history'][0]['actual_bpp']==m['history'][0]['actual_bpp']


def test_verified_serialization_recovery_preserves_trained_state(tmp_path):
    import numpy as np
    from scripts.v33ap_recover import recover_checkpoint
    from adaptive_vcm.v33ap.model import JointPreprocessor
    from adaptive_vcm.v33ap.checkpoint import make_checkpoint,load_checkpoint
    from adaptive_vcm.v32.protocol import canonical_hash
    value=make_checkpoint(JointPreprocessor(),metadata())
    value['metadata']['history'][0]['actual_bpp']=np.float64(1.)
    value['metadata']['code']['code_files_sha256']['adaptive_vcm/v33ap/checkpoint.py']='4'*64
    value['metadata_sha256']=canonical_hash(value['metadata'])
    origin=dict(origin_commit='a'*40,source_files_sha256=value['metadata']['code']['code_files_sha256'])
    source=tmp_path/'old.pth'; dest=tmp_path/'recovered.pth'; torch.save(value,source)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='hash'): recover_checkpoint(source,dest,'0'*64,origin)
    receipt=recover_checkpoint(source,dest,digest,origin)
    loaded=load_checkpoint(dest)
    assert receipt['state_sha256']==value['metadata']['state_sha256']
    assert all(torch.equal(v,loaded.model.state_dict()[k]) for k,v in value['model'].items())
    assert loaded.metadata['checkpoint_recovery']['parameter_updates'] is False
    assert loaded.metadata['checkpoint_recovery']['original_checkpoint_sha256']==digest
    assert type(loaded.metadata['history'][0]['actual_bpp']) is float
    value['metadata']['code']['code_files_sha256']['adaptive_vcm/v33ap/model.py']='5'*64
    value['metadata_sha256']=canonical_hash(value['metadata']); torch.save(value,source)
    origin['source_files_sha256']=value['metadata']['code']['code_files_sha256']
    with pytest.raises(ValueError,match='serialization'): recover_checkpoint(source,tmp_path/'reject.pth',hashlib.sha256(source.read_bytes()).hexdigest(),origin)
