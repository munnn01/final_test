"""Regressions from the single final R3 review."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from scripts import v34r3_workflow as flow
from tests.test_v34r3_workflow import Client, plan
from adaptive_vcm.v34r3.protocol import digest


def test_identity_recovery_waits_for_private_parent_and_rejects_stale_payload(tmp_path,monkeypatch):
    p=plan(tmp_path,monkeypatch);source=tmp_path/'partial';source.mkdir()
    (source/'stage.json').write_text('{}');partial=source/'measurements.partial.json';partial.write_text('{}')
    recovery=dict(path=str(partial),file_sha256=flow.sha(partial),cache_sha256='c'*64,rows=1)
    state=dict(jobs={'od-identity':dict(status='prepared',attempt=2,recoverable_parent=recovery)},datasets={})
    class DatasetClient(Client):
        ready=False
        def call(self,account,argv,timeout=45):
            if argv[:2]==['datasets','create']:
                return dict(exit_code=0,stdout='Your private Dataset is being created',stderr='')
            if argv[:2]==['datasets','status']:
                return dict(exit_code=0,stdout='ready' if self.ready else 'processing',stderr='')
            return super().call(account,argv,timeout)
    client=DatasetClient();waiting=flow.step(p,state,client)
    assert not any(a[:2]==['kernels','push'] for a in client.calls)
    directory=tmp_path/'payloads/od-identity-resume-2'
    assert not directory.exists()
    client.ready=True;submitted=flow.step(p,waiting,client)
    assert submitted['jobs']['od-identity']['status']=='submitted'
    job=flow.read(directory/'job.json');assert job['parent']['archive_sha256']
    assert job['parent']['files']['measurements.partial.json']==recovery['file_sha256']
    # A stale reusable directory must never submit a payload without its exact recovery parent.
    job['parent']=None;flow.save(directory/'job.json',job)
    count=len(client.calls);rejected=flow.step(p,waiting,client)
    assert rejected['status']=='failed_attention'
    assert 'parent' in rejected.get('error','').lower()
    assert not any(a[:2]==['kernels','push'] for a in client.calls[count:])


@pytest.mark.parametrize('field',['origin','cache_sha256','inputs_sha256','missing'])
def test_checkpoint_requires_expected_experiment_lineage(tmp_path,field):
    from adaptive_vcm.v34r3.checkpoint import save,load
    from adaptive_vcm.v34r3.model import Controller
    from tests.test_v34r3_protocol import metadata
    meta=metadata();expected=dict(origin={'origin_commit':'a'*40,'bundle_sha256':'b'*64},cache_sha256=['c'*64],inputs_sha256='d'*64)
    meta.update(expected)
    if field=='missing':meta.pop('origin')
    else:meta[field]={'origin_commit':'f'*40,'bundle_sha256':'b'*64} if field=='origin' else ['f'*64] if field=='cache_sha256' else 'f'*64
    path=tmp_path/'model.pth';save(path,Controller(13,9),meta)
    with pytest.raises(ValueError,match='lineage|origin|cache|input'):
        load(path,task='od',recipe=meta['recipe'],task_models=meta['task_models'],lineage=expected)


def local_mot(tmp_path,names):
    for i,name in enumerate(names):
        root=tmp_path/name;(root/'img1').mkdir(parents=True);(root/'gt').mkdir()
        (root/'seqinfo.ini').write_text(f'[Sequence]\nname={name}\nimDir=img1\nframeRate=25\nseqLength=16\nimWidth=24\nimHeight=24\nimExt=.jpg\n')
        for f in range(1,17):Image.fromarray(np.full((24,24,3),i*30+f,np.uint8)).save(root/'img1'/f'{f:06}.jpg')
        (root/'gt/gt.txt').write_text(''.join(f'{f},1,4,4,12,12,1,1,1\n' for f in range(1,17)))


def load_local_mot(tmp_path,splits):
    from adaptive_vcm.v34r3.sources import load_groups
    from tests.test_v34r3_measurement import tiny_recipe
    cfg=tiny_recipe();cfg.update(ot_sequences=splits,ot_frames=16,ot_size=24)
    manifest=dict(ot_sequences=splits,history=dict(audited=True,ids=[],pixels=[],groups=[]))
    manifest['sha256']=digest(manifest);path=tmp_path/'cohort.json';path.write_text(json.dumps(manifest))
    args=SimpleNamespace(experiment='ot',mot_root=tmp_path,cohort_manifest=path)
    return (*load_groups(args,cfg),cfg)


def test_local_mot17_cal_and_official_heldout_sequence_name(tmp_path):
    from adaptive_vcm.v34r3.sources import group_id
    from adaptive_vcm.v34r3.evaluation import tracking_summary
    from adaptive_vcm.v32.tracking import load_trackeval
    splits=dict(train=['MOT20-02'],cal=['MOT17-02-FRCNN','MOT17-04-FRCNN'],dev=['MOT17-05-FRCNN'],test=['MOT17-09-FRCNN'])
    local_mot(tmp_path,[n for ns in splits.values() for n in ns]);groups,meta,cfg=load_local_mot(tmp_path,splits)
    assert len({group_id(r) for r in groups['cal']})==2
    r=groups['test'][0];frames=[dict(boxes=np.array([[4,4,16,16]]),scores=np.array([1.]),track_ids=np.array([1]),frame_number=f) for f in range(1,17)]
    summary=tracking_summary({('h264',35,'anchor',group_id(r)):frames},[dict(arm='anchor',codec='h264',qp=35,bpp=1.)],groups['test'],meta,cfg,tmp_path/'metrics',load_trackeval(os.environ['V34_TRACKEVAL']))
    assert summary['official_metrics']['h264_35_anchor']['combined']['HOTA']==pytest.approx(100)
    assert (tmp_path/'metrics/trackers/h264_35_anchor/data/MOT17-09-FRCNN.txt').is_file()


def test_local_mot_detector_variants_cannot_cross_splits(tmp_path):
    splits=dict(train=['MOT20-02'],cal=['MOT17-02-FRCNN'],dev=['MOT17-05-FRCNN'],test=['MOT17-02-DPM'])
    local_mot(tmp_path,[n for ns in splits.values() for n in ns])
    with pytest.raises(ValueError,match='camera|leakage'):load_local_mot(tmp_path,splits)


def test_ot_offline_damage_uses_identity_probe_on_own_prefix(tmp_path,monkeypatch):
    from adaptive_vcm.v34r3 import evaluation as ev
    from tests.test_v34r3_measurement import records,tiny_recipe,make_heads
    cfg=tiny_recipe();rs=records(tmp_path/'data')
    for r in rs:r.split='dev'
    h=make_heads('ot',cfg)
    def detections(rgb):
        x=4+float(rgb[0,0,0,0])/10
        return [dict(boxes=np.array([[x,4,x+12,16]]),scores=np.array([.9]),labels=np.array([1])) for _ in rgb]
    h.detections=detections
    h.source_score=lambda rgb,source,*,prefix,package,detections:dict(source_cost=[0.,0.,0.],probe=prefix.probe(detections))
    def gt_score(rgb,source,**kwargs):
        frames=kwargs['source_result']['probe'].frames
        cost=sum(float(f['boxes'][0,0]) for f in frames)/100
        return dict(gt_cost=[cost]*3)
    h.gt_score=gt_score
    class Codec:
        last_times=dict(encode_seconds=0.,decode_seconds=0.,inverse_seconds=0.)
        def roundtrip(self,rgb,**kwargs):
            a=int(rgb[0,0,0,0]);return SimpleNamespace(decoded=rgb,coded_bytes=100 if a==0 else 90,packet_sha256=digest([a]))
    monkeypatch.setattr(ev,'codec_factory',lambda *args:Codec())
    monkeypatch.setattr(ev,'filter_clip',lambda rgb,maps,action:np.full_like(rgb,ev.ACTIONS.index(action)))
    monkeypatch.setattr(ev,'predict',lambda *args:dict(logits=[0]+list(range(12,0,-1)),risk=np.zeros((13,3)).tolist(),rate=[1.]*13))
    def summary(tracks,rows,*args):
        return dict(results={'ot/h264':dict(curves={a:dict(bpp=[1.],quality=[50.]) for a in {r['arm'] for r in rows}})})
    monkeypatch.setattr(ev,'tracking_summary',summary)
    model=SimpleNamespace(metadata=dict(checkpoint_role='FINAL',recipe=cfg,task_models=h.hashes(),seed=303401,fixed_shortlist=[1,2,3]),checkpoint_sha256='a'*64)
    out=tmp_path/'eval'
    ev.evaluate(dict(dev=rs),{},cfg,out,h,[model],'ot',object(),splits=('dev',),policies=[dict(risk_gate=False)])
    rows=json.loads((out/'dev_rows.json').read_text());cnn=[r for r in rows if r['arm']=='cnn_303401']
    assert [r['action'] for r in cnn]==[1,1]
    # The second identity probe inherits the first selected nonidentity GOP.
    assert cnn[1]['offline_gt_damage']==pytest.approx(cnn[0]['offline_gt_damage'])
    assert cnn[1]['offline_anchor_stream_damage'][0]>cnn[1]['offline_gt_damage'][0]


@pytest.mark.codec
def test_runtime_checkpoint_rejects_foreign_origin_inputs_sources_and_calibration(tmp_path):
    from adaptive_vcm.v34r3.runner import run_records,load_models
    from adaptive_vcm.v34r3.checkpoint import save
    from tests.test_v34r3_measurement import tiny_recipe,make_heads,origin
    from tests.test_v34r3_runner import od_groups
    cfg=tiny_recipe('od');groups,meta=od_groups();h=make_heads('od',cfg)
    run_records(groups,meta,cfg,tmp_path/'identity',origin(),h,'od',stage='measure-identity')
    final=tmp_path/'final';run_records(groups,meta,cfg,final,origin(),h,'od',stage='train-final',parent=tmp_path/'identity')
    def restore(**kwargs):return load_models(final,cfg,h,'od',role='FINAL',origin=kwargs.pop('origin',origin()),**kwargs)
    models=restore();path=final/f'controller_{cfg["seeds"][0]}.pth'
    with pytest.raises(ValueError,match='origin'):restore(origin={**origin(),'bundle_sha256':'e'*64})
    inputs=final/'controller_inputs.pth';raw=inputs.read_bytes();inputs.write_bytes(raw+b'changed')
    with pytest.raises(ValueError,match='input'):restore()
    inputs.write_bytes(raw);groups['train'][0].rgb^=1
    with pytest.raises(ValueError,match='source'):restore(groups=groups,meta=meta)
    groups['train'][0].rgb^=1
    for field in ('model_state_sha256','cache_sha256','prefix_policy_sha256'):
        bad=copy.deepcopy(models[0].metadata);bad['calibration'][field]='f'*64;save(path,models[0],bad)
        with pytest.raises(ValueError,match='calibration'):restore()
    save(path,models[0],models[0].metadata)
    assert len(restore(groups=groups,meta=meta))==3
