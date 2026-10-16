import copy
import json
from pathlib import Path
import numpy as np
import pytest
from adaptive_vcm.v34r3 import measurement
from adaptive_vcm.v34r3.protocol import settings,recipe,digest,codec_factory
from adaptive_vcm.v34r3.sources import validate_cohorts
from adaptive_vcm.v34r3.tasks import Heads
from adaptive_vcm.v34r2.tracking import Prefix
from adaptive_vcm.v32.tracking import load_trackeval
from tests.test_v34_single_task import heads_for,record
from adaptive_vcm.v33ap.sources import serial
import os


def make_heads(task,cfg):
    h=heads_for(task);h.__class__=Heads;h.cfg=cfg
    h.action.model.eval().requires_grad_(False);h.detector.model.eval().requires_grad_(False)
    return h


def tiny_recipe(task='ot'):
    cfg=recipe(settings(),task,'measure-identity')
    cfg.update(codecs=['h264'],qps=[35],epochs=1,bootstrap_draws=0)
    return cfg


def records(tmp_path):
    path=tmp_path/'MOT20-02';(path/'gt').mkdir(parents=True)
    (path/'gt/gt.txt').write_text(''.join(f'{i},1,4,4,12,12,1,1,1\n' for i in range(1,33)))
    result=[]
    for index in range(2):
        r=record('ot');r.id=f'ot/MOT20-02/{1+16*index:06d}'
        r.rgb=r.rgb.copy()^(index*17)
        r.item=dict(sequence='MOT20-02',start=1+16*index,width=24,height=24,path=str(path))
        r.frame_recipe=dict(frame_indices=list(range(1+16*index,17+16*index)))
        result.append(r)
    return result


def origin(): return dict(origin_commit='a'*40,bundle_sha256='b'*64)


@pytest.mark.codec
def test_candidates_share_frozen_prefix_and_probe_order_does_not_change_scores(tmp_path):
    cfg=tiny_recipe();h=make_heads('ot',cfg);r=records(tmp_path)[0]
    source,maps=measurement.prepare_source(r,h,'ot',cfg,{},{});prefix=Prefix()
    package=load_trackeval(os.environ['V34_TRACKEVAL'])
    context=measurement.context_for(r)
    before=prefix.signature
    table=measurement.measure(r,source,maps,codec_factory('ot','h264',35,cfg),h,cfg,
        prefix=prefix,gt_context=context,package=package,prefix_variant='identity')
    assert len(table['packets'])==13 and all(p['prefix_sha256']==before for p in table['packets'])
    assert prefix.generation==1 and prefix.tracker.frame==16
    assert table['selected_action']==0 and table['prefix_after']==prefix.signature
    assert table['gt_damage'][0]==pytest.approx([0,0,0])
    frozen=prefix.snapshot()
    # Independent source references at the same context length; score order cannot commit a probe.
    source.source_predictions['ot_context']=frozen.frames+source.source_predictions['ot']
    first=h.source_score(r.rgb,source,prefix=frozen,package=package)
    second=h.source_score(r.rgb^1,source,prefix=frozen,package=package)
    again=h.source_score(r.rgb,source,prefix=frozen,package=package)
    assert np.array_equal(first['source_cost'],again['source_cost'])
    assert first['probe'].signature==again['probe'].signature and frozen.generation==1


@pytest.mark.codec
def test_resume_replays_selected_stream_and_matches_uninterrupted_packet_hashes(tmp_path):
    cfg=tiny_recipe();rs=records(tmp_path/'data');groups=dict(train=rs,cal=[],dev=[],test=[])
    package=load_trackeval(os.environ['V34_TRACKEVAL'])
    full=measurement.measure_stage(groups,{},cfg,tmp_path/'full',origin(),make_heads('ot',cfg),'ot',
                                   variant='identity',package=package)
    partial=copy.deepcopy(full);partial['rows']=partial['rows'][:1];partial['complete']=False
    partial=measurement.seal(partial)
    restarted=measurement.measure_stage(groups,{},cfg,tmp_path/'restart',origin(),make_heads('ot',cfg),'ot',
                                        variant='identity',parent=partial,package=package)
    for a,b in zip(full['rows'],restarted['rows']):
        assert [p['sha256'] for p in a['packets']]==[p['sha256'] for p in b['packets']]
        assert a['prefix_sha256']==b['prefix_sha256'] and a['prefix_after']==b['prefix_after']
    assert restarted['reused_rows']==1 and restarted['new_rows']==1
    stale=copy.deepcopy(full);stale['rows'][0]['map_sha256']='f'*64
    with pytest.raises(ValueError):
        measurement.measure_stage(groups,{},cfg,tmp_path/'changed-map',origin(),make_heads('ot',cfg),'ot',
            variant='identity',parent=measurement.seal(stale),package=package)
    gtfile=Path(rs[0].item['path'])/'gt/gt.txt'
    gtfile.write_text(gtfile.read_text()+'\n')
    with pytest.raises(ValueError):
        measurement.measure_stage(groups,{},cfg,tmp_path/'changed-gt',origin(),make_heads('ot',cfg),'ot',
            variant='identity',parent=full,package=package)


def test_stale_parent_and_conflicting_row_are_rejected(tmp_path):
    cfg=tiny_recipe('od');r=record('od');r.item=dict(image_id=1,width=24,height=24)
    groups=dict(train=[r],cal=[],dev=[],test=[])
    expected=dict(task='od',recipe=cfg,origin=origin(),sources={'a':dict(pixel_sha256='c'*64)})
    parent=measurement.seal({**expected,'schema':'adaptive-vcm-v34-r3-1','rows':[],'complete':False})
    changed=copy.deepcopy(expected);changed['task']='ot'
    with pytest.raises(ValueError):measurement.resume(parent,changed)
    tampered=copy.deepcopy(parent);tampered['recipe']['qps']=[30]
    with pytest.raises(ValueError):measurement.resume(tampered,expected)


def test_sequence_camera_and_pixel_split_leakage_is_rejected(tmp_path):
    a,b=records(tmp_path);a.item['sequence']='MOT17-02-FRCNN';b.item['sequence']='MOT17-02-DPM'
    b.split='test'
    with pytest.raises(ValueError):validate_cohorts(dict(train=[a],cal=[],dev=[],test=[b]),{})
    a=record('od');b=copy.deepcopy(a);b.id='other';b.split='test'
    with pytest.raises(ValueError):validate_cohorts(dict(train=[a],cal=[],dev=[],test=[b]),{})


def test_only_train_cal_can_measure_gt_targets(tmp_path):
    cfg=tiny_recipe('od');h=make_heads('od',cfg);r=record('od','test')
    source,maps=measurement.prepare_source(r,h,'od',cfg,{}, {})
    with pytest.raises(ValueError):measurement.measure(r,source,maps,codec_factory('od','h264',35,cfg),h,cfg)


@pytest.mark.parametrize('corruption',['decoded_hash','detached_delta'])
def test_resumption_rejects_invalid_decoded_provenance_and_inconsistent_targets(corruption):
    cfg=tiny_recipe('od')
    row=dict(id='a',group='a',split='train',codec='h264',qp=35,prefix_variant='identity',selected_action=0,
        prefix_sha256='c'*64,prefix_after='d'*64,rate=[(100+i)/100 for i in range(13)],
        source_cost=[[0.]*4 for _ in range(13)],source_damage=[[0.]*4 for _ in range(13)],
        gt_cost=[[0.]*3 for _ in range(13)],gt_damage=[[0.]*3 for _ in range(13)],
        packets=[dict(bytes=100+i,elementary_bytes=90+i,metadata_bytes=10,sha256='a'*64,
                      decoded_sha256='b'*64,prefix_sha256='c'*64) for i in range(13)])
    cache=measurement.seal(dict(schema='adaptive-vcm-v34-r3-1',task='od',recipe=cfg,origin=origin(),
        sources={'a':dict(split='train',pixel_sha256='e'*64)},variant='identity',
        execution_scope=dict(codecs=['h264'],splits=['train']),rows=[row],complete=True))
    measurement.validate_cache(cache)
    if corruption=='decoded_hash':cache['rows'][0]['packets'][1]['decoded_sha256']='invalid'
    else:cache['rows'][0]['gt_damage'][2][0]=.1
    with pytest.raises(ValueError):measurement.validate_cache(measurement.seal(cache))


def test_support_scene_cut_context_cap_and_reentry_use_causal_history(tmp_path):
    cfg=tiny_recipe();rs=records(tmp_path);h=make_heads('ot',cfg)
    clean={};supports={}
    a,_=measurement.prepare_source(rs[0],h,'ot',cfg,clean,supports)
    b,_=measurement.prepare_source(rs[1],h,'ot',cfg,clean,supports)
    assert len(b.source_predictions['ot_context'])==32
    assert b.source_predictions['ot'][0]['frame_number']==17
    assert clean['MOT20-02'].generation==2
    supports['MOT20-02'].tracks[999]=dict(box=np.array([0,0,2,2],float),velocity=np.zeros(4),score=1.,age=0)
    # A source-size discontinuity clears the support history before painting the new frame.
    supports['MOT20-02'].previous=np.zeros((2,2),np.uint8)
    third=copy.deepcopy(rs[1]);third.item['start']=33;third.id='ot/MOT20-02/000033'
    measurement.prepare_source(third,h,'ot',cfg,clean,supports)
    assert 999 not in supports['MOT20-02'].tracks


@pytest.mark.codec
def test_replay_rejects_decoder_drift_even_with_identical_packet_and_predictions(tmp_path,monkeypatch):
    from dataclasses import replace
    cfg=tiny_recipe();rs=records(tmp_path/'data');groups=dict(train=rs,cal=[],dev=[],test=[])
    package=load_trackeval(os.environ['V34_TRACKEVAL']);h=make_heads('ot',cfg)
    cache=measurement.measure_stage(groups,{},cfg,tmp_path/'first',origin(),h,'ot',variant='identity',package=package)
    factory=measurement.codec_factory
    class Drift:
        def __init__(self,codec):self.codec=codec
        def roundtrip(self,*a,**k):
            packet=self.codec.roundtrip(*a,**k)
            return replace(packet,decoded=packet.decoded^1)
        def __getattr__(self,key):return getattr(self.codec,key)
    monkeypatch.setattr(measurement,'codec_factory',lambda *a,**k:Drift(factory(*a,**k)))
    with pytest.raises(ValueError,match='decoded|decoder'):
        measurement.measure_stage(groups,{},cfg,tmp_path/'resume',origin(),h,'ot',variant='identity',package=package,parent=cache)
