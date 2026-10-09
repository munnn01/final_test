import ast
import base64
import hashlib
from pathlib import Path
import pytest
from scripts.v32vc_kaggle import mount_data, build_notebook, job_spec


def test_known_coco_mount_has_bounded_discovery(tmp_path):
    root = tmp_path/'datasets/awsaf49/coco-2017-dataset/coco2017'
    (root/'annotations').mkdir(parents=True); (root/'val2017').mkdir()
    (root/'annotations/instances_val2017.json').write_text('{}')
    images,annotations = mount_data(tmp_path,'od')
    assert images==root/'val2017' and annotations==root/'annotations/instances_val2017.json'
    with pytest.raises(FileNotFoundError): mount_data(tmp_path,'ar')


def test_duplicate_mount_alias_uses_first_known_path(tmp_path):
    (tmp_path/'kineticscleaned').mkdir()
    (tmp_path/'datasets/qktttttttttt/kineticscleaned').mkdir(parents=True)
    assert mount_data(tmp_path,'ar')==(tmp_path/'kineticscleaned',None)


def test_notebook_is_private_and_embeds_exact_separate_baseline():
    raw,bundle = b'checkpoint-allowed',b'committed-source'
    origin = {'origin_commit':'a'*40,'bundle_sha256':hashlib.sha256(bundle).hexdigest(),'source_files_sha256':{}}
    job = job_spec('ar','vtk269',origin,raw)
    notebook,metadata = build_notebook(job,origin,bundle,raw)
    assert metadata['is_private'] and metadata['enable_gpu'] and metadata['dataset_sources']==['qktttttttttt/kineticscleaned']
    constants = ast.parse(''.join(notebook['cells'][0]['source']))
    values = {n.targets[0].id:ast.literal_eval(n.value) for n in constants.body if isinstance(n,ast.Assign)}
    assert base64.b64decode(values['CHECKPOINT_BASE64'])==raw
    assert base64.b64decode(values['BUNDLE_BASE64'])==bundle
    for cell in notebook['cells']: compile(''.join(cell['source']),'<cell>','exec')
    runtime = ''.join(notebook['cells'][1]['source'])
    assert '.rglob(' not in runtime and 'pool.json' not in runtime
    assert 'adaptive_vcm.v32vc.train' in runtime and "('dev','test')" in runtime
