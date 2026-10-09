import base64,inspect,json
import pytest
from scripts.v32rf_kaggle import job_spec,build_notebook
from adaptive_vcm.v32rf.protocol import settings
from adaptive_vcm.v32rf.warm import WARM_BUNDLE_HASH


def spec(mode='rf',task='ar'):
    return job_spec(task,'owner',{'origin_commit':'d'*40},b'baseline',{'origin_commit':settings()['warm_origin'],'bundle_sha256':WARM_BUNDLE_HASH},b'warm',mode)


@pytest.mark.parametrize('task,mode',[('ar','rf'),('od','rf'),('ar','post')])
def test_public_bound_payload_preserves_both_checkpoint_and_runtime_bytes(task,mode):
    job=spec(mode,task); notebook,metadata=build_notebook(job,{'origin_commit':'d'*40},b'new-runtime',b'baseline',{'origin_commit':settings()['warm_origin']},b'old-runtime',b'warm')
    assert metadata['is_private'] is False and metadata['enable_gpu'] and metadata['enable_internet'] and job['private'] is False
    assert metadata['dataset_sources']==[job['dataset']]
    source=''.join(notebook['cells'][0]['source']); ns={}; exec(source,ns)
    assert base64.b64decode(ns['CHECKPOINT_BASE64'])==b'baseline'
    assert base64.b64decode(ns['WARM_CHECKPOINT_BASE64'])==b'warm'
    assert base64.b64decode(ns['WARM_BUNDLE_BASE64'])==b'old-runtime'
    assert base64.b64decode(ns['BUNDLE_BASE64'])==b'new-runtime'
    for cell in notebook['cells']: compile(''.join(cell['source']),'<cell>','exec')
    runtime=''.join(notebook['cells'][1]['source'])
    assert 'finally:' in runtime and 'tarfile.open' in runtime and "status='failed'" in runtime
    assert 'load_warm' in runtime and 'WARM_ORIGIN' in runtime and '--transfer' in runtime and 'post_evaluate' in runtime
    assert 'pool.json' not in runtime and 'KAGGLE_KEY' not in runtime


def test_invalid_modes_and_wrong_warm_origin_rejected():
    with pytest.raises(ValueError): spec('post','od')
    with pytest.raises(ValueError): job_spec('ar','owner',{},b'cp',{'origin_commit':'bad'},b'warm','rf')


def test_generator_validates_warm_and_original_before_snapshot():
    import scripts.v32rf_kaggle as module
    source=inspect.getsource(module.main)
    assert source.index('load_warm(')<source.index('source_snapshot(')
    assert source.index('FrozenV30Baseline.load(')<source.index('source_snapshot(')


def test_complete_payload_fits_kaggle_one_megabyte_source_limit():
    import random
    source=random.Random(303231).randbytes(665000)
    notebook,_=build_notebook(spec(),{'origin_commit':'d'*40},source,b'baseline',{'origin_commit':settings()['warm_origin']},source[:620000],b'warm')
    for cell in notebook['cells']: cell['source']=''.join(cell['source'])
    assert len(json.dumps(notebook).encode('utf-8'))<1_000_000


def test_corrupt_packed_transport_rejected():
    from scripts.v32rf_kaggle import unpack_constants,pack_constants
    packed=pack_constants(spec(),{},b'source',b'cp',{},b'warm',b'warmcp')
    with pytest.raises(ValueError): unpack_constants(packed[:-8])
    with pytest.raises(ValueError): unpack_constants('not-a-valid-payload?')
