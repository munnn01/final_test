import hashlib
import json
import pytest


def test_private_continuation_payload_locks_recipe_parent_and_artifact_bytes(tmp_path,monkeypatch):
    from scripts import v34r2_continue_kaggle as payload
    from tests.test_v34_payload import fixture_source
    monkeypatch.setattr(payload,'source_snapshot',fixture_source)
    parent=dict(kind='kernel',ref='trnhlng/v34-r2-ot-combined-qp4-b37c4937/1',
                files={'measurements.json':'a'*64,'controller_303401.pth':'b'*64,'controller_303402.pth':'c'*64,
                       'controller_303403.pth':'d'*64,'origin.json':'e'*64,'cohorts.json':'f'*64})
    receipt=payload.prepare('ot','qktttttttttt',tmp_path,parent=parent,stage='evaluate',codecs=['h264'],splits=['test'],domains=['ot'])
    assert payload.validate_payload(tmp_path)==receipt
    job=json.loads((tmp_path/'job.json').read_text())
    meta=json.loads((tmp_path/'kernel-metadata.json').read_text())
    assert job['protocol']['qps']==[30,35,40,45] and job['protocol']['codecs']==['h264','h265']
    assert job['protocol']['seeds']==[303401,303402,303403]
    assert job['execution_scope']==dict(codecs=['h264'],splits=['test'],domains=['ot'])
    assert meta['is_private'] is True and meta['kernel_sources']==[parent['ref']]
    nb=json.loads((tmp_path/'notebook.ipynb').read_text())
    for cell in nb['cells']: compile(''.join(cell['source']),'<continuation notebook>','exec')
    (tmp_path/'job.json').write_text('{}')
    with pytest.raises(ValueError): payload.validate_payload(tmp_path)


def test_parent_resolution_requires_all_hashes_and_rejects_ambiguous_copies(tmp_path):
    from scripts.v34r2_continue_kaggle import find_parent
    first=tmp_path/'one'; first.mkdir(); (first/'measurements.json').write_bytes(b'correct')
    (first/'controller_303401.pth').write_bytes(b'checkpoint')
    hashes={n:hashlib.sha256((first/n).read_bytes()).hexdigest() for n in ('measurements.json','controller_303401.pth')}
    assert find_parent(tmp_path,hashes)==first
    (first/'controller_303401.pth').write_bytes(b'corrupt')
    with pytest.raises(ValueError): find_parent(tmp_path,hashes)
    (first/'controller_303401.pth').write_bytes(b'checkpoint')
    second=tmp_path/'two'; second.mkdir()
    for p in first.iterdir(): (second/p.name).write_bytes(p.read_bytes())
    with pytest.raises(ValueError): find_parent(tmp_path,hashes)


@pytest.mark.parametrize('account,parent,scope',[
    ('not_authorized',dict(kind='kernel',ref='a/valid-kernel/1',files={'measurements.json':'a'*64}),dict()),
    ('baoancut',dict(kind='kernel',ref='a/valid-kernel',files={'measurements.json':'a'*64}),dict()),
    ('baoancut',dict(kind='kernel',ref='a/valid-kernel/1',files={'../pool.json':'a'*64}),dict()),
    ('baoancut',dict(kind='kernel',ref='a/valid-kernel/1',files={'measurements.json':'a'*64}),dict(codecs=['av1'])),
])
def test_job_rejects_unapproved_account_unpinned_parent_or_invalid_scope(account,parent,scope):
    from scripts.v34r2_continue_kaggle import job_spec
    with pytest.raises(ValueError):
        job_spec('ot',account,dict(origin_commit='a'*40),parent=parent,stage='evaluate',**scope)


def test_private_dataset_parent_must_belong_to_execution_account():
    from scripts.v34r2_continue_kaggle import job_spec
    files={n:'a'*64 for n in ['measurements.json','origin.json','cohorts.json',
                             'controller_303401.pth','controller_303402.pth','controller_303403.pth']}
    parent=dict(kind='dataset',ref='baoancut/v34-private-state',version=1,files=files)
    with pytest.raises(ValueError):
        job_spec('ot','qktttttttttt',dict(origin_commit='a'*40),parent=parent,stage='evaluate',
                 codecs=['h264'],splits=['test'],domains=['ot'])
