import base64
import io
import zipfile
import pytest


def test_public_plans_are_only_ar_and_ot():
    from scripts.v32at_kaggle import job_spec
    for task in ('ar','ot'):
        job=job_spec(task,'example',{'origin_commit':'a'*40})
        assert job['private'] is False and job['active_tasks']==['ar','ot']
        assert job['training'] is False
        assert 'coco' not in str(job.get('dataset','')).lower()
    with pytest.raises(ValueError): job_spec('od','example',{})


def test_packed_payload_is_bounded_and_roundtrips():
    from scripts.v32at_kaggle import pack, unpack
    members={'source.zip':b'a','origin.json':b'{}','job.json':b'{}'}
    assert unpack(pack(members))==members
    with pytest.raises(ValueError): unpack('not base64')
    with pytest.raises(ValueError): pack({'pool.json':b'credential'})


def test_notebook_source_public_and_compile_without_network():
    import json
    from scripts.v32at_kaggle import build_notebook,job_spec
    from scripts.v32rf_kaggle import serialized_source_bytes
    origin={'origin_commit':'a'*40,'bundle_sha256':'b'*64,'source_files_sha256':{}}
    for task in ('ar','ot'):
        job=job_spec(task,'example',origin)
        members={'source.zip':b'fixture','origin.json':json.dumps(origin).encode(),'job.json':json.dumps(job).encode()}
        nb,meta=build_notebook(members,job)
        assert meta['is_private'] is False and serialized_source_bytes(nb)<1_000_000
        assert meta['dataset_sources']==(['qktttttttttt/kineticscleaned'] if task=='ar' else [])
        for cell in nb['cells']: compile(''.join(cell['source']),'<fixture>','exec')
        source=''.join(nb['cells'][1]['source'])
        assert 'finally:' in source and 'tarfile.open' in source
        assert 'adaptive_vcm.v32at.tracking' in source and 'adaptive_vcm.v32at.ar' in source
