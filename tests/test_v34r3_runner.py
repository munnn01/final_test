import copy
import json
import os
from pathlib import Path
import pytest
from adaptive_vcm.v34r3.runner import run_records,load_models
from adaptive_vcm.v34r3 import evaluation
from adaptive_vcm.v34r3.protocol import digest
from adaptive_vcm.v34r3.checkpoint import save
from tests.test_v34r3_measurement import make_heads,tiny_recipe,records,origin
from tests.test_v34_single_task import record


def od_groups():
    groups={s:[record('od',s)] for s in ('train','cal','dev','test')}
    anns=[]
    for i,(split,rs) in enumerate(groups.items(),1):
        rs[0].item=dict(image_id=i,width=24,height=24)
        anns.append(dict(id=i,image_id=i,category_id=1,bbox=[4,4,12,12],area=144,iscrowd=0))
    return groups,dict(od=dict(annotations=anns,categories=[dict(id=1,name='person')],images=[dict(id=i,width=24,height=24) for i in range(1,5)]))


@pytest.mark.codec
def test_od_full_grid_three_seed_fixture_smoke(tmp_path):
    cfg=tiny_recipe('od');cfg.update(codecs=['h264','h265'],qps=[30,35,40,45],limit_grid=[.025])
    groups,meta=od_groups();h=make_heads('od',cfg)
    report=run_records(groups,meta,cfg,tmp_path/'run',origin(),h,'od')
    cache=json.loads((tmp_path/'run/identity/measurements.json').read_text())
    assert len(cache['rows'])==16 and all(len(r['packets'])==13 for r in cache['rows'])
    assert len(list((tmp_path/'run/final').glob('controller_3034*.pth')))==3
    assert (tmp_path/'run/test_summary.json').is_file()
    rows=json.loads((tmp_path/'run/test_rows.json').read_text())
    assert len(rows)==8*10 and all(r['deployed_encode_count']==(1 if r['arm']=='anchor' else 4) for r in rows)
    assert report['status']=='complete' and not report['quality_improvement_confirmed']
    dev_scope=dict(codecs=cfg['codecs'],splits=['dev'])
    run_records(groups,meta,cfg,tmp_path/'dev-only',origin(),h,'od',stage='evaluate',parent=tmp_path/'run/final',scope=dev_scope)
    assert not (tmp_path/'dev-only/test_rows.json').exists()
    run_records(groups,meta,cfg,tmp_path/'test-only',origin(),h,'od',stage='evaluate',parent=tmp_path/'dev-only',scope=dict(codecs=['h264'],splits=['test']))
    assert not (tmp_path/'test-only/dev_rows.json').exists()
    locked=json.loads((tmp_path/'test-only/locked_policy.json').read_text())
    assert locked['fit_split']=='dev' and locked['recipe_sha256']==digest(cfg)


@pytest.mark.codec
def test_ot_two_window_rollout_resume_and_official_metrics(tmp_path):
    from adaptive_vcm.v32.tracking import load_trackeval
    cfg=tiny_recipe();cfg.update(limit_grid=[.005])
    groups={s:records(tmp_path/s) for s in ('train','cal','dev','test')}
    plans=[]
    for i,(split,rs) in enumerate(groups.items()):
        name=f'MOT20-{i+1:02d}';path=Path(rs[0].item['path'])
        (path/'seqinfo.ini').write_text(f'[Sequence]\nname={name}\nimDir=img1\nframeRate=25\nseqLength=32\nimWidth=24\nimHeight=24\nimExt=.jpg\n')
        for r in rs:
            r.split=split;r.id=r.id.replace('MOT20-02',name);r.item['sequence']=name;r.rgb=r.rgb.copy()^(i*31)
        plans.append(dict(name=name,path=path,frames=32,full_frames=32,width=24,height=24))
    meta=dict(ot=plans);package=load_trackeval(os.environ['V34_TRACKEVAL']);h=make_heads('ot',cfg)
    run_records(groups,meta,cfg,tmp_path/'run',origin(),h,'ot',package)
    cache=json.loads((tmp_path/'run/rollout/measurements.json').read_text())
    from adaptive_vcm.v34r3.measurement import measure_stage
    warm=load_models(tmp_path/'run/warm',cfg,h,'ot',role='WARM',origin=origin())
    restarted=measure_stage(groups,meta,cfg,tmp_path/'resume',origin(),h,'ot',variant='rollout',warm_models=warm,parent=cache,package=package)
    assert restarted['reused_rows']==4
    summary=json.loads((tmp_path/'run/test_summary.json').read_text())
    point=summary['official_metrics']['h264_35_anchor']['combined']
    assert {'HOTA','DetA','AssA','IDF1','MOTA','FN','FP','IDs'}<=point.keys()
    assert not summary['promotion']['confirmed']


def test_final_evaluation_rejects_warm_checkpoint(tmp_path):
    from adaptive_vcm.v34r3.model import Controller
    from adaptive_vcm.v34r3.protocol import component_names,component_mask
    cfg=tiny_recipe('od');h=make_heads('od',cfg);tmp_path.mkdir(exist_ok=True)
    for seed in cfg['seeds']:
        save(tmp_path/f'controller_{seed}.pth',Controller(13,9),dict(task='od',recipe=cfg,task_models=h.hashes(),risk_components=list(component_names('od')),component_mask=component_mask('od'),seed=seed,checkpoint_role='WARM'))
    with pytest.raises(ValueError,match='FINAL'):load_models(tmp_path,cfg,h,'od',role='FINAL',origin=origin())


@pytest.mark.parametrize('change',['pixels','annotations','geometry'])
@pytest.mark.codec
def test_training_rejects_changed_sources_or_annotations(tmp_path,change):
    cfg=tiny_recipe('od');groups,meta=od_groups();h=make_heads('od',cfg)
    run_records(groups,meta,cfg,tmp_path/'measure',origin(),h,'od',stage='measure-identity')
    if change=='pixels':groups['train'][0].rgb^=1
    elif change=='annotations':meta['od']['annotations'][0]['bbox'][0]+=1
    else:groups['train'][0].geometry=(2,2,0,0)
    with pytest.raises(ValueError,match='source|annotation|lineage'):
        run_records(groups,meta,cfg,tmp_path/'final',origin(),h,'od',stage='train-final',parent=tmp_path/'measure')


@pytest.mark.codec
def test_heldout_gt_sentinel_is_read_only_after_selection_commit(tmp_path,monkeypatch):
    cfg=tiny_recipe('od');groups,meta=od_groups();h=make_heads('od',cfg)
    run_records(groups,meta,cfg,tmp_path/'fit',origin(),h,'od',stage='measure-identity')
    run_records(groups,meta,cfg,tmp_path/'final',origin(),h,'od',stage='train-final',parent=tmp_path/'fit')
    selected=[];native=evaluation.choose
    def choice(*a,**k):
        value=native(*a,**k);selected.append(True);return value
    monkeypatch.setattr(evaluation,'choose',choice)
    class Guarded(dict):
        def __getitem__(self,key):
            if key=='annotations':assert selected,'held-out GT read before selection'
            return super().__getitem__(key)
    meta['od']=Guarded(meta['od']);models=load_models(tmp_path/'final',cfg,h,'od',role='FINAL',origin=origin())
    evaluation.evaluate(groups,meta,cfg,tmp_path/'eval',h,models,'od',None,splits=('dev',),policies=[dict(risk_gate=True)])


@pytest.mark.codec
def test_ar_common_entrypoint_matches_native_actions_packets_predictions(tmp_path,monkeypatch):
    import hashlib
    import numpy as np
    import torch
    from adaptive_vcm.v34 import evaluation as native
    from adaptive_vcm.v34.checkpoint import save_checkpoint
    from adaptive_vcm.v34.model import Controller
    from tests.test_v34_single_task import heads_for,qp4,TinyAction
    cfg=qp4();h=heads_for('ar');groups={s:[record('ar',s)] for s in ('dev','test')}
    cfgpath=tmp_path/'native.json';cfgpath.write_text(json.dumps(cfg))
    manifest=dict(config=dict(path=str(cfgpath),sha256=hashlib.sha256(cfgpath.read_bytes()).hexdigest()),task_models=h.hashes(),checkpoints=[])
    models=[]
    for seed in cfg['seeds']:
        torch.manual_seed(seed);m=Controller(13,9).eval()
        m.metadata=dict(settings=cfg,experiment='ar',seed=seed,task_models=h.hashes(),calibration={'domains':{'ar':dict(risk=np.zeros((13,3)).tolist(),rate=[0.]*13)}})
        path=tmp_path/f'{seed}.pth';save_checkpoint(path,m,m.metadata);models.append(m)
        manifest['checkpoints'].append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    mpath=tmp_path/'manifest.json';mpath.write_text(json.dumps(manifest));monkeypatch.setattr(native,'ActionAnalyzer',TinyAction)
    native.evaluate(groups,h,models,cfg,{},tmp_path/'native',None)
    run_records(groups,{},cfg,tmp_path/'shared',origin(),h,'ar',stage='evaluate',parent=mpath)
    for split in ('dev','test'):
        a=json.loads((tmp_path/f'native/{split}_rows.json').read_text());b=json.loads((tmp_path/f'shared/{split}_rows.json').read_text())
        assert [(r['arm'],r['action'],r['packet_sha256'],r['decoded_sha256'],r['correct']) for r in a]==[(r['arm'],r['action'],r['packet_sha256'],r['decoded_sha256'],r['correct']) for r in b]
