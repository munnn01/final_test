import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from adaptive_vcm.v34.model import Controller
from adaptive_vcm.v34.checkpoint import save_checkpoint
from adaptive_vcm.v34.protocol import settings as native_settings
from adaptive_vcm.v34.evaluation import choose_packet as native_choose
from adaptive_vcm.v34r3 import ar


def test_ar_reference_pins_recipe_calibration_and_three_checkpoint_bytes(tmp_path):
    cfg=native_settings(Path(__file__).resolve().parents[1]/'configs/v34_r1_qp4.json')
    path=tmp_path/'config.json'; path.write_text(json.dumps(cfg))
    task_models={'ar':{'name':'fixture','state_sha256':'a'*64}}
    manifest=dict(config=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                  task_models=task_models,checkpoints=[])
    for seed in cfg['seeds']:
        p=tmp_path/f'{seed}.pth'
        save_checkpoint(p,Controller(13,9),dict(seed=seed,settings=cfg,experiment='ar',
            task_models=task_models,calibration={'fixture_value':seed}))
        manifest['checkpoints'].append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    restored,models=ar.load_reference(manifest)
    assert restored==cfg and [m.metadata['seed'] for m in models]==[303401,303402,303403]
    p=Path(manifest['checkpoints'][0]['path'])
    value=torch.load(p,weights_only=True); value['metadata']['calibration']={'fixture_value':0};torch.save(value,p)
    with pytest.raises(ValueError): ar.load_reference(manifest)
    path.write_text(json.dumps({**cfg,'source_shortlist':4}))
    with pytest.raises(ValueError): ar.load_reference(manifest)


def test_ar_adapter_preserves_native_selection_and_decoded_prediction():
    cfg=native_settings(Path(__file__).resolve().parents[1]/'configs/v34_r1_qp4.json')
    model=Controller(13,9).eval()
    model.metadata=dict(calibration={'domains':{'ar':dict(risk=np.zeros((13,3)).tolist(),rate=[0.]*13)}})
    pair=(torch.zeros(16,3,48,48),torch.zeros(16,3,48,48))
    def encode(action):
        return SimpleNamespace(coded_bytes=1000-action,decoded=np.full((16,4,4,3),action,np.uint8))
    def score(rgb): return [float(rgb[0,0,0,0])*.0001,None,None]
    def call(fn):
        return fn(model,pair,'h264',35,[1,0,0],cfg,encode(0),encode,domain='ar',source_score=score)
    old,new=call(native_choose),call(ar.choose_packet)
    assert old[:2]==new[:2]
    assert old[3]['tried_actions']==new[3]['tried_actions']
    assert old[3]['source_guard']==new[3]['source_guard']
    assert old[2].coded_bytes==new[2].coded_bytes
    assert np.array_equal(old[2].decoded,new[2].decoded)
    assert int(old[2].decoded.mean())==int(new[2].decoded.mean())


def test_imported_ar_reports_are_labelled_and_bound_to_frozen_checkpoint_bytes(tmp_path):
    cfg=native_settings(Path(__file__).resolve().parents[1]/'configs/v34_r1_qp4.json')
    cp=tmp_path/'config.json';cp.write_text(json.dumps(cfg));models={'fixture':True}
    pin=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    manifest=dict(config=pin(cp),task_models=models,checkpoints=[])
    for seed in cfg['seeds']:
        p=tmp_path/f'{seed}.pth';save_checkpoint(p,Controller(13,9),dict(seed=seed,settings=cfg,experiment='ar',task_models=models,calibration={}))
        manifest['checkpoints'].append(pin(p))
    reports={}
    for name,value in [('complete.json',dict(status='complete',experiment='ar',cache_sha256='a'*64,checkpoint_sha256=[p['sha256'] for p in manifest['checkpoints']])),
        ('dev_summary.json',dict(results={'ar/h264':{}})),('test_summary.json',dict(results={'ar/h264':{}})),
        ('dev_rows.json',[]),('test_rows.json',[]),('cohorts.json',{})]:
        p=tmp_path/name;p.write_text(json.dumps(value));reports[name]=pin(p)
    receipt=ar.import_reference(manifest,reports,tmp_path/'imported')
    assert receipt['execution']=='reused_frozen_reference' and receipt['retrained'] is False
    _,loaded=ar.load_reference(tmp_path/'imported/native_manifest.json')
    assert [m.checkpoint_sha256 for m in loaded]==[p['sha256'] for p in manifest['checkpoints']]
    reports['test_summary.json']['sha256']='b'*64
    with pytest.raises(ValueError):ar.import_reference(manifest,reports,tmp_path/'bad')
