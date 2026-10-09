import io
import json
import zipfile
from pathlib import Path
import pytest


def origin_bundle():
    import hashlib
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z: z.writestr('a.py','print(1)')
    raw=b.getvalue()
    return dict(origin_commit='a'*40,bundle_sha256=hashlib.sha256(raw).hexdigest(),source_files_sha256={'a.py':hashlib.sha256(b'print(1)').hexdigest()}),raw


def test_three_public_same_source_and_bounded_runtime():
    from scripts.v33ap_kaggle import job_spec,build_notebook,pack,unpack
    from scripts.v32rf_kaggle import serialized_source_bytes
    origin,bundle=origin_bundle(); jobs=[]
    for arm in ('joint','ar_only','od_only'):
        j=job_spec(arm,'owner',origin); jobs.append(j)
        members={'source.zip':bundle,'origin.json':json.dumps(origin).encode(),'job.json':json.dumps(j).encode()}
        assert unpack(pack(members))==members
        nb,meta=build_notebook(members,j)
        assert not meta['is_private'] and len(meta['dataset_sources'])==2
        assert serialized_source_bytes(nb)<1_000_000
        source=''.join(''.join(c['source']) for c in nb['cells'])
        for cell in nb['cells']: compile(''.join(cell['source']),'<cell>','exec')
        assert 'adaptive_vcm.v33ap.train' in source and 'adaptive_vcm.v33ap.evaluate' in source
        assert 'finally:' in source and 'tarfile.open' in source and 'traceback.format_exc' in source
    assert len({j['origin_commit'] for j in jobs})==1 and {j['seed'] for j in jobs}=={303251}
    with pytest.raises(ValueError): unpack('a'*2_000_001)
    with pytest.raises(ValueError): pack({'source.zip':b'a','origin.json':b'b','job.json':b'c','credentials.json':b'd'})


def test_duplicate_and_ambiguous_push(tmp_path):
    from scripts.v33ap_kaggle import submit
    class Client:
        calls=[]
        def call(self,a,args,timeout=45):
            self.calls.append(args)
            return dict(exit_code='TIMEOUT',stdout='',stderr='timeout',checked_utc='now')
    (tmp_path/'job.json').write_text(json.dumps(dict(handle='owner/v33ap-joint-s303251')))
    (tmp_path/'submission.json').write_text(json.dumps(dict(status='accepted',handle='owner/v33ap-joint-s303251',version=1)))
    c=Client(); assert submit(c,'owner',tmp_path)['status']=='accepted' and not c.calls
    (tmp_path/'submission.json').unlink()
    write_payload(tmp_path)
    assert submit(c,'owner',tmp_path)['status']=='ambiguous'
    assert submit(c,'owner',tmp_path)['status']=='ambiguous' and len(c.calls)==1


def test_payload_tamper_rejected(tmp_path):
    from scripts.v33ap_kaggle import validate_payload
    write_payload(tmp_path); validate_payload(tmp_path)
    meta=json.loads((tmp_path/'kernel-metadata.json').read_text()); meta['is_private']=True
    (tmp_path/'kernel-metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='payload'): validate_payload(tmp_path)


def write_payload(tmp_path):
    import hashlib
    from scripts.v33ap_kaggle import job_spec,build_notebook
    origin,bundle=origin_bundle(); job=job_spec('joint','owner',origin)
    members={'source.zip':bundle,'origin.json':json.dumps(origin).encode(),'job.json':json.dumps(job).encode()}
    nb,meta=build_notebook(members,job)
    for name,value in [('notebook.ipynb',nb),('job.json',job),('kernel-metadata.json',meta),('origin-manifest.json',origin)]:
        (tmp_path/name).write_text(json.dumps(value))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    (tmp_path/'local_validation.json').write_text(json.dumps(dict(payload_sha256=hashes)))
