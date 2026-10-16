import copy
import hashlib
import io
import json
import subprocess
import zipfile
import pytest
from scripts import v34r3_kaggle as payload
from adaptive_vcm.v34r3.protocol import ROOT,SCHEMA,digest


def fixture_source():
    bundle=subprocess.check_output(['git','-C',str(ROOT),'archive','--format=zip','HEAD'])
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        files={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.endswith('.py') or n.startswith('configs/') and not n.endswith('/')}
    return bundle,dict(origin_commit='a'*40,bundle_sha256=hashlib.sha256(bundle).hexdigest(),bundle_bytes=len(bundle),source_files_sha256=files),{}


def parent(task='ot',stage='measure-identity'):
    return dict(kind='dataset',ref='thuha1205/r3-test-state',version=1,task=task,schema=SCHEMA,stage=stage,
        files={'stage.json':'a'*64,'stage_complete.json':'b'*64,'measurements.json':'c'*64})


def test_private_payload_and_runtime_are_pinned(tmp_path,monkeypatch):
    monkeypatch.setattr(payload,'source_snapshot',fixture_source)
    receipt=payload.prepare('od','thuha1205',tmp_path,stage='measure-identity')
    assert payload.validate_payload(tmp_path)==receipt
    job=json.loads((tmp_path/'job.json').read_text());metadata=json.loads((tmp_path/'kernel-metadata.json').read_text())
    assert job['private'] is True and metadata['is_private'] is True
    assert metadata['dataset_sources']==['awsaf49/coco-2017-dataset/2']
    assert job['protocol']['qps']==[30,35,40,45] and job['trackeval_commit']=='12c8791b303e0a0b50f753af204249e622d0281a'
    nb=json.loads((tmp_path/'notebook.ipynb').read_text())
    from scripts.v32rf_kaggle import serialized_source_bytes
    assert serialized_source_bytes(nb)<1_000_000
    for cell in nb['cells']:compile(''.join(cell['source']),'<R3 notebook>','exec')
    runtime=''.join(nb['cells'][-1]['source'])
    assert 'adaptive_vcm.v34r3.runner' in runtime and "--stage" in runtime and 'find_parent' in runtime


def test_ar_training_and_joint_submission_are_rejected():
    for task,stage in [('ar','all'),('ar','train-warm'),('ar_od_ot','evaluate')]:
        with pytest.raises(ValueError):payload.job_spec(task,'thuha1205',dict(origin_commit='a'*40),stage=stage)
    with pytest.raises(ValueError):payload.job_spec('ar','thuha1205',dict(origin_commit='a'*40),stage='evaluate')


def test_foreign_task_parent_or_edited_notebook_is_rejected(tmp_path,monkeypatch):
    with pytest.raises(ValueError):payload.job_spec('ot','thuha1205',dict(origin_commit='a'*40),stage='train-warm',parent=parent('od'))
    monkeypatch.setattr(payload,'source_snapshot',fixture_source)
    payload.prepare('od','thuha1205',tmp_path,stage='measure-identity')
    nb=json.loads((tmp_path/'notebook.ipynb').read_text());nb['cells'][-1]['source'].append('\nprint("edited")\n')
    (tmp_path/'notebook.ipynb').write_text(json.dumps(nb))
    with pytest.raises(ValueError):payload.validate_payload(tmp_path)
