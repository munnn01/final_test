import json
import pytest
from scripts import v34r2_kaggle as payload
from tests.test_v34_payload import fixture_source


@pytest.mark.parametrize('task,datasets',[
    ('od',['awsaf49/coco-2017-dataset']),('ot',[]),
    ('ar_od_ot',['qktttttttttt/kineticscleaned','awsaf49/coco-2017-dataset'])])
def test_r2_payload_pins_runtime_and_separate_cohorts(tmp_path,monkeypatch,task,datasets):
    monkeypatch.setattr(payload,'source_snapshot',fixture_source)
    receipt=payload.prepare(task,'fixture',tmp_path)
    assert payload.validate_payload(tmp_path)==receipt
    job=json.loads((tmp_path/'job.json').read_text())
    assert job['protocol']['qps']==[30,35,40,45]
    assert job['protocol']['revision']=='r2-qp4'
    assert job['datasets']==datasets
    assert job['sequences']==['MOT20-02','MOT20-05','MOT20-03','MOT20-01']
    assert job['fresh_training'] and job['ablation']=='combined'
    cells=json.loads((tmp_path/'notebook.ipynb').read_text())['cells']
    for c in cells: compile(''.join(c['source']),'<R2 payload>','exec')
    runtime=''.join(cells[-1]['source'])
    assert 'adaptive_vcm.v34r2.runner' in runtime and 'adaptive_vcm.v34r2.protocol' in runtime
    assert '--ablation' in runtime


def test_ablation_payload_keeps_configuration_locked():
    origin=dict(origin_commit='a'*40)
    a=payload.job_spec('ot','fixture',origin,'map_only')
    assert a['ablation']=='map_only' and a['protocol']['target_policy']=='r2'
    with pytest.raises(ValueError): payload.job_spec('ar','fixture',origin)
