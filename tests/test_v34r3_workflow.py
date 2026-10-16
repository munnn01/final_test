import copy
import hashlib
import json
import subprocess
import uuid
from pathlib import Path
import pytest
from scripts import v34r3_workflow as flow
from adaptive_vcm.v34r3.protocol import ROOT,settings,digest,SCHEMA
from tests.test_v34r3_payload import fixture_source


class Client:
    secrets=[uuid.uuid4().hex];calls=[]
    def __init__(self):self.calls=[]
    def safe(self,value):return str(value).replace(self.secrets[0],'[REDACTED]')
    def call(self,account,argv,timeout=45):
        self.calls.append(argv)
        if argv[:2]==['kernels','push']:
            handle=json.loads((Path(argv[-1])/'kernel-metadata.json').read_text())['id']
            return dict(exit_code=0,stdout='Kernel version 1 successfully pushed. https://www.kaggle.com/code/'+handle,stderr='')
        if argv[:2]==['kernels','status']:return dict(exit_code=0,stdout='KernelWorkerStatus.RUNNING',stderr='')
        raise AssertionError(argv)


def plan(tmp_path,monkeypatch):
    from scripts import v34r3_kaggle as payload
    monkeypatch.setattr(payload,'source_snapshot',fixture_source)
    monkeypatch.setattr(flow,'owner_capacity',lambda client,account:dict(hours=30,active=['r2/existing']))
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    return dict(root=str(tmp_path),source_commit=commit,config_sha256=hashlib.sha256((ROOT/'configs/v34_r3_qp4.json').read_bytes()).hexdigest(),
        jobs=[dict(id='od-identity',task='od',account='thuha1205',stage='measure-identity',depends=[],scope=None,estimated_gpu_hours=4)])


def test_dispatch_is_idempotent_and_does_not_cancel_r2(tmp_path,monkeypatch):
    p=plan(tmp_path,monkeypatch);client=Client();state=dict(jobs={},datasets={},status='prepared')
    first=flow.step(p,state,client);second=flow.step(p,first,client)
    assert second['jobs']['od-identity']['status']=='running'
    assert sum(a[:2]==['kernels','push'] for a in client.calls)==1
    assert not any('cancel' in a or 'delete' in a for a in client.calls)
    saved=json.loads((tmp_path/'state.json').read_text());assert saved['jobs']['od-identity']['handle'].startswith('thuha1205/')


def test_wrong_or_incomplete_stage_receipt_is_not_complete(tmp_path):
    entry=dict(task='od',stage='train-final')
    (tmp_path/'stage_complete.json').write_text(json.dumps(dict(schema=SCHEMA,status='complete',task='ot',stage='train-final',files_sha256={})))
    with pytest.raises(ValueError):flow.verify_stage(tmp_path,entry)
    (tmp_path/'stage_complete.json').write_text(json.dumps(dict(schema=SCHEMA,status='complete',task='od',stage='train-final',files_sha256={})))
    with pytest.raises(ValueError):flow.verify_stage(tmp_path,entry)


def test_state_and_logs_redact_credentials(tmp_path,monkeypatch):
    p=plan(tmp_path,monkeypatch);client=Client()
    def failure(*a,**k):raise RuntimeError('broken '+client.secrets[0])
    monkeypatch.setattr(flow,'owner_capacity',failure)
    state=flow.step(p,dict(jobs={},datasets={},status='prepared'),client)
    assert client.secrets[0] not in (tmp_path/'state.json').read_text()
    assert '[REDACTED]' in json.dumps(state)


def test_timeout_resumes_verified_rows_without_retraining_ar(tmp_path):
    from adaptive_vcm.v34r3 import measurement
    from tests.test_v34r3_measurement import tiny_recipe,make_heads,origin
    from tests.test_v34_single_task import record
    cfg=tiny_recipe('od');r=record('od');r.item=dict(image_id=1,width=24,height=24)
    cache=measurement.measure_stage(dict(train=[r],cal=[],dev=[],test=[]),{},cfg,tmp_path/'partial',origin(),make_heads('od',cfg),'od',variant='identity')
    cache['complete']=False;cache=measurement.seal(cache);path=tmp_path/'partial/measurements.partial.json';measurement.write_cache(path,cache)
    value=flow.recover_timeout(tmp_path/'partial',dict(task='od',stage='measure-identity'),origin())
    assert value['rows']==1 and value['cache_sha256']==cache['sha256']
    from adaptive_vcm.v34r3.runner import run_records
    run_records(dict(train=[r],cal=[],dev=[],test=[]),{},cfg,tmp_path/'resumed',origin(),make_heads('od',cfg),'od',stage='measure-identity',parent=tmp_path/'partial')
    resumed=json.loads((tmp_path/'resumed/measurements.json').read_text())
    assert resumed['reused_rows']==1 and resumed['new_rows']==0
    with pytest.raises(ValueError):flow.recover_timeout(tmp_path/'partial',dict(task='ar',stage='measure-identity'),origin())


def test_private_parent_archive_restores_nested_files(tmp_path):
    source=tmp_path/'source';(source/'metrics').mkdir(parents=True)
    (source/'stage.json').write_text('{}');(source/'metrics/raw.json').write_text('{"metric":1}')
    files={n:flow.sha(source/n) for n in ('stage.json','metrics/raw.json')}
    mount=tmp_path/'mount';mount.mkdir()
    flow.archive_parent(source,files,mount/'state.zip')
    from scripts.v34r3_kaggle import find_parent
    restored=find_parent(mount,files,flow.sha(mount/'state.zip'),workdir=tmp_path/'unpacked')
    assert restored.name=='unpacked' and flow.sha(restored/'metrics/raw.json')==files['metrics/raw.json']
