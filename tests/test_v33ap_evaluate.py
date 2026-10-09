import numpy as np
from v33_fixtures import record,Action,Detector


def pred(scores=(.9,),boxes=((4,4,8,8),)):
    return dict(scores=np.array(scores),boxes=np.array(boxes).reshape(-1,4),labels=np.ones(len(scores),int))


def test_dual_guards_and_unknown_fallback():
    from adaptive_vcm.v33ap.selection import guard
    from adaptive_vcm.v33ap.protocol import settings
    cfg=settings(); source=dict(ar=np.array([.8,.1,.1]),od={0:pred()}); anchor=source
    assert guard(source,anchor,source,'ar',cfg)['passed']
    assert not guard(source,anchor,{**source,'ar':np.array([.1,.8,.1])},'ar',cfg)['passed']
    assert not guard(source,anchor,{**source,'od':{0:pred(boxes=((0,0,1,1),))}},'ar',cfg)['passed']
    empty=dict(ar=source['ar'],od={0:pred((),())})
    assert not guard(empty,empty,empty,'ar',cfg)['passed']
    assert guard(empty,empty,empty,'ar',cfg)['omitted_frames']==[0]


def test_real_packets_paid_retarget_and_savings():
    from adaptive_vcm.v33ap.selection import choose_stream
    from adaptive_vcm.v33ap.protocol import settings,codec_factory
    from adaptive_vcm.v33ap.checkpoint import LearnedJointPreprocessor
    from adaptive_vcm.v33ap.model import JointPreprocessor
    from adaptive_vcm.v33ap.tasks import TaskHeads
    r=record('od'); r.support['protection'][0,4:8,4:8]=1
    model=LearnedJointPreprocessor(JointPreprocessor(),{},'1'*64); h=TaskHeads('cpu',action=Action(),detector=Detector())
    codec=codec_factory('od','ultrafast')('h264',35)
    bundle=choose_stream(r.rgb,r.support,model,codec,h,settings(),task='od',retarget=True)
    assert bundle['selected_name']=='identity' # zero head has no savings
    assert set(bundle['arms'])=={'anchor','forced','guarded','raw_retarget2','cnn_retarget2','raw_retarget4','cnn_retarget4'}
    for packet,name in bundle['arms'].values():
        assert packet.coded_bytes==packet.elementary_bytes+packet.metadata_bytes
        assert np.array_equal(codec.decode_packet(packet.data).decoded,packet.decoded)
    from adaptive_vcm.v33ap.evaluate import summarize
    from adaptive_vcm.metrics import curve_summary
    assert curve_summary([1,2,3],[.5,.5,.5],[.8,1.6,2.4],[.5,.5,.5])['bd_rate_pct'] is None


def test_full_real_evaluation_grid(tmp_path):
    from types import SimpleNamespace
    from PIL import Image
    from adaptive_vcm.data import partition
    from adaptive_vcm.v33ap.protocol import settings
    from adaptive_vcm.v33ap.tasks import TaskHeads
    from adaptive_vcm.v33ap.model import JointPreprocessor
    from adaptive_vcm.v33ap.checkpoint import LearnedJointPreprocessor
    from adaptive_vcm.v33ap.receiver import SharedReceiver
    from adaptive_vcm.v33ap.evaluate import run
    from adaptive_vcm.v33ap.model import JointPreprocessor
    n=next(i for i in range(500) if partition('coco2017/'+str(i))=='test')
    p=tmp_path/'image.png'; Image.fromarray(record('od').rgb[0]).save(p)
    item=dict(id=str(n),image_id=n,path=str(p),width=16,height=16)
    meta=dict(categories=[dict(id=1,name='object')],annotations=[dict(id=1,image_id=n,category_id=1,bbox=[4,4,4,4],area=16,iscrowd=0)])
    h=TaskHeads('cpu',action=Action(),detector=Detector()); model=LearnedJointPreprocessor(JointPreprocessor(),dict(arm='joint',source_pixel_sha256={},task_models=h.hashes()),'1'*64)
    args=SimpleNamespace(task='od',split='test',count=1,transfer=False,out=tmp_path/'out',root=tmp_path,annotations=None,checkpoint=None,device='cpu')
    cfg={**settings(),'od_size':16,'preset':'ultrafast'}
    result=run(args,heads=h,model=model,receiver=SharedReceiver(h.action,h.detector),plan_override=([item],meta),cfg=cfg)
    assert result['complete_paired_grid'] and result['ar_accuracy'] is None
    import json
    rows=[json.loads(line) for line in (args.out/'rows.jsonl').read_text().splitlines()]
    assert len(rows)==70 and {r['codec'] for r in rows}=={'h264','h265'}
    assert all(r['coded_bytes']==r['elementary_bytes']+r['metadata_bytes'] for r in rows)
