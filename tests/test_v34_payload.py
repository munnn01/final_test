import hashlib
import io
import json
import zipfile
import pytest
from scripts import v34_kaggle as payload

def fixture_source():
    raw=io.BytesIO()
    config=json.dumps(payload.settings()).encode()
    with zipfile.ZipFile(raw,'w') as z: z.writestr('configs/v34.json',config)
    bundle=raw.getvalue()
    origin=dict(origin_commit='a'*40,bundle_sha256=hashlib.sha256(bundle).hexdigest(),bundle_bytes=len(bundle),
        source_files_sha256={'configs/v34.json':hashlib.sha256(config).hexdigest()})
    return bundle,origin,{}

@pytest.mark.parametrize('experiment',['ar_od','ar_ot'])
def test_public_notebook_is_exact_recipe_and_compiles_with_secret_free_members(tmp_path,monkeypatch,experiment):
    monkeypatch.setattr(payload,'source_snapshot',fixture_source)
    receipt=payload.prepare(experiment,'fixture',tmp_path)
    assert payload.validate_payload(tmp_path)==receipt
    nb=json.loads((tmp_path/'notebook.ipynb').read_text())
    for cell in nb['cells']: compile(''.join(cell['source']),'<notebook>','exec')
    metadata=json.loads((tmp_path/'kernel-metadata.json').read_text())
    assert metadata['is_private'] is False
    ns={}; exec(''.join(nb['cells'][0]['source']),ns)
    assert set(ns['MEMBERS'])=={'source.zip','origin.json','job.json'}
    job=json.loads((tmp_path/'job.json').read_text()); job['protocol']['test_count']=1
    (tmp_path/'job.json').write_text(json.dumps(job))
    with pytest.raises(ValueError): payload.validate_payload(tmp_path)

def test_submission_intent_prevents_duplicate_after_timeout(tmp_path,monkeypatch):
    monkeypatch.setattr(payload,'source_snapshot',fixture_source); payload.prepare('ar_od','fixture',tmp_path)
    class Client:
        calls=0
        def call(self,account,argv,timeout):
            assert (tmp_path/'submission.json').exists()
            assert argv[:2]==['kernels','push']
            self.calls+=1; return {'exit_code':'TIMEOUT','stdout':'','stderr':''}
    client=Client(); first=payload.submit(client,'fixture',tmp_path)
    assert first['status']=='ambiguous' and payload.submit(client,'fixture',tmp_path)==first
    assert client.calls==1
