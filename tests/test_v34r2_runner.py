import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from adaptive_vcm.v34r2.protocol import settings,ACTIONS,codec_factory,validate_cache
from adaptive_vcm.v34r2.runner import measure,run_records,save_checkpoint,load_checkpoint
from adaptive_vcm.v34r2.evaluation import choose
from adaptive_vcm.v34r2.tracking import Prefix
from adaptive_vcm.v34r2.preprocessing import Support
from adaptive_vcm.v34r2.tasks import Heads
from tests.test_v34_single_task import heads_for,record,TinyAction
from tests.test_v34r2_support import pred


def heads():
    h=heads_for('od'); h.__class__=Heads; h.cfg=settings()
    for obj in (h.action,h.detector): obj.model.eval().requires_grad_(False)
    return h


def fixture_origin():
    from adaptive_vcm.v34r2.protocol import ROOT
    paths=[*(ROOT/'adaptive_vcm/v34r2').glob('*.py'),ROOT/'configs/v34_r2_qp4.json']
    return dict(origin_commit='a'*40,bundle_sha256='b'*64,
        source_files_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def test_runtime_uses_all_active_source_constraints_and_commits_only_winner():
    from types import SimpleNamespace
    prefix=Prefix(); before=prefix.signature
    def encode(a): return SimpleNamespace(coded_bytes=[100,60,70,80][a],decoded=np.full((1,1,1,3),a))
    def score(rgb):
        a=int(rgb[0,0,0,0]); probe=prefix.probe([pred()])
        return dict(source_cost=[[0,0,0],[0,.1,0],[0,0,.1],[0,0,0]][a],probe=probe,predictions={})
    cfg=settings(); cfg['source_guard_limits']=[0,0,0]
    selected,packet,result,audit=choose([1,2,3],encode,score,[1,1,1],cfg,prefix)
    assert selected==3 and prefix.tracker.frame==1 and prefix.signature!=before
    assert audit['tried_actions']==[0,1,2,3] and packet.coded_bytes==80
    assert audit['prefix_before']==before


@pytest.mark.codec
def test_real_codecs_measure_dual_targets_from_identical_prefix(tmp_path):
    h=heads(); cfg=settings(); r=record('ot')
    from adaptive_vcm.v33ap.sources import SourceRecord
    source=SourceRecord(r.id,r.task,{},r.rgb,r.geometry,r.frame_recipe)
    h.prepare(source,[0,0,1],Prefix())
    maps=Support().build(r.rgb,source.source_predictions['od'],source.mask,source.source_predictions['ot'])
    from adaptive_vcm.v32.tracking import load_trackeval
    import os
    package=load_trackeval(os.environ['V34_TRACKEVAL'])
    for c in ('h264','h265'):
        prefix=Prefix(); old=prefix.signature
        table=measure(r,source,maps,codec_factory('ot',c,35,cfg),h,cfg,prefix=prefix,gt_context=r.targets,package=package)
        assert table['prefix_sha256']==old and prefix.tracker.frame==16
        assert len(table['packets'])==13 and table['rate'][0]==1
        assert table['gt_damage'][0]==[None,None,0]
        assert table['source_damage'][0]==[None,None,0]
        assert len({p['prefix_sha256'] for p in table['packets']})==1


@pytest.mark.codec
def test_joint_end_to_end_keeps_heldout_gt_out_of_guard(tmp_path):
    import os
    from adaptive_vcm.v32.tracking import load_trackeval
    cfg=settings(); cfg.update(epochs=1,seeds=[303401],qps=[35],codecs=['h264'],dev_exhaustive=False,bootstrap_draws=0)
    groups={s:[record('ar',s),record('od',s)] for s in ('train','cal','dev','test')}
    # Detection GT retained only outside the source/selection record.
    meta={'od':dict(categories=[dict(id=1,name='person')],annotations=[])}
    for i,(split,records) in enumerate(groups.items(),1):
        r=records[1]; r.item=dict(image_id=i,width=24,height=24)
        meta['od']['annotations'].append(dict(id=i,image_id=i,category_id=1,bbox=[4,4,12,12],area=144,iscrowd=0))
        if split in ('dev','test'): r.targets={'must_not_enter_source_guard':True}
    origin=fixture_origin()
    run_records(groups,meta,cfg,tmp_path,origin,heads(),'ar_od_ot',load_trackeval(os.environ['V34_TRACKEVAL']),transfer_ar=TinyAction())
    cache=json.loads((tmp_path/'measurements.json').read_text()); validate_cache(cache)
    assert all(r['gt_mask']==([1,0,0] if r['domain']=='ar' else [0,1,0]) for r in cache['rows'])
    rows=json.loads((tmp_path/'test_rows.json').read_text())
    shared=[r for r in rows if r['task']=='ar' and r['arm'].startswith('cnn_')]
    assert shared and all(r['active']==[1,1,1] and r['shared_packet'] for r in shared)
    assert all(r['deployed_encode_count']==4 for r in rows if r['arm'].startswith(('cnn_','random_','train_fixed_')))
    model=load_checkpoint(tmp_path/'controller_303401.pth')
    assert model.metadata['codec_backpropagation'] is False
    assert model.metadata['experiment']=='ar_od_ot'
    assert model.metadata['scales_fit_split']=='train'
    assert (tmp_path/'complete.json').exists()


@pytest.mark.codec
def test_ot_two_window_end_to_end_preserves_prefix_and_official_metrics(tmp_path):
    import os
    from tests.test_v32_tracking import fixture_sequence
    from adaptive_vcm.v32.tracking import plan_sequences,load_trackeval
    cfg=settings(); cfg.update(epochs=1,seeds=[303401],qps=[35],codecs=['h264'],dev_exhaustive=True,bootstrap_draws=0)
    groups={}; plans=[]
    for split in ('train','cal','dev','test'):
        name=cfg['ot_sequences'][split][0]
        seq=fixture_sequence(tmp_path/split,count=32)
        target=seq.with_name(name)
        if target!=seq: seq.rename(target)
        (target/'seqinfo.ini').write_text((target/'seqinfo.ini').read_text().replace('MOT20-01',name))
        (target/'gt/gt.txt').write_text(''.join(f'{i},1,4,4,12,12,1,1,1\n' for i in range(1,33)))
        plans+=plan_sequences(tmp_path/split,[name],32); groups[split]=[]
        for i in range(2):
            r=record('ot',split); r.id+=f'/{i}'
            r.rgb=r.rgb.copy()^(17*i)
            r.item=dict(sequence=name,start=1+16*i,width=80,height=40,path=str(target))
            r.frame_recipe['frame_indices']=list(range(1+16*i,17+16*i))
            if split in ('dev','test'): r.targets={'heldout_GT_must_not_enter_guard':True}
            groups[split].append(r)
    out=tmp_path/'experiment'
    run_records(groups,{'ot':plans},cfg,out,fixture_origin(),heads(),'ot',load_trackeval(os.environ['V34_TRACKEVAL']))
    rows=json.loads((out/'test_rows.json').read_text())
    cnn=[r for r in rows if r['arm']=='cnn_303401']
    assert len(cnn)==2 and cnn[1]['prefix_before']==cnn[0]['prefix_after']
    summary=json.loads((out/'test_summary.json').read_text())
    values=summary['ot_official_metrics']['h264_35_cnn_303401']['combined']
    assert values['HOTA']==values['DetA']==values['AssA']==100
    lines=(out/'test/trackers/h264_35_cnn_303401/data/MOT20-01.txt').read_text().splitlines()
    assert [int(x.split(',')[0]) for x in lines]==list(range(1,33))
    exhaustive=json.loads((out/'dev_exhaustive_source_guard.json').read_text())
    assert len(exhaustive)==2 and all('gt_action' in r for r in exhaustive)
    assert not (out/'test_exhaustive_source_guard.json').exists()
