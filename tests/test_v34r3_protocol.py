import copy
import json
import numpy as np
import pytest
import torch
from adaptive_vcm.v34.model import Controller as NativeController
from adaptive_vcm.v34r3.model import Controller
from adaptive_vcm.v34r3.protocol import (settings, recipe, component_names, task_mask,
                                       component_mask, baseline_manifest, verify_baseline)
from adaptive_vcm.v34r3.protocol import digest
from adaptive_vcm.v34r3.checkpoint import save, load


def test_controller_is_native_ar_topology():
    native = NativeController(13, conditions=9).eval()
    common = Controller(13, conditions=9).eval()
    common.load_state_dict(native.state_dict(), strict=True)
    x = torch.rand(2, 2, 3, 48, 48)
    maps = torch.rand_like(x)
    condition = torch.zeros(2, 9)
    expected, actual = native(x, maps, condition), common(x, maps, condition)
    assert {k: tuple(v.shape) for k,v in actual.items()} == {
        'logits': (2,13), 'risk': (2,13,3), 'rate': (2,13)}
    assert common.state_dict().keys() == native.state_dict().keys()
    assert all(torch.equal(expected[k], actual[k]) for k in expected)


def test_task_and_component_masks_are_distinct():
    cfg = settings()
    assert recipe(cfg, 'ar', 'evaluate')['mode'] == 'frozen_reference'
    assert task_mask('od') == [0,1,0]
    assert component_mask('od') == [1,1,1]
    assert component_names('od') == ('fn','fp','localization')
    assert component_names('ot') == ('hota','deta','assa')
    assert recipe(cfg,'od','train-final')['qps'] == [30,35,40,45]


@pytest.mark.parametrize('task,stage', [('ar_od_ot','evaluate'),('ar','train-final'),
                                     ('ar','train-warm'),('ar','measure-identity')])
def test_joint_and_ar_retraining_are_rejected(task, stage):
    with pytest.raises(ValueError): recipe(settings(),task,stage)


def metadata(task='od'):
    return dict(task=task, recipe=recipe(settings(),task,'train-final'),
                risk_components=list(component_names(task)), component_mask=[1,1,1],
                task_models={'od': {'name':'fixture','state_sha256':'a'*64}},
                checkpoint_role='WARM', seed=303401,**lineage(),training_lineage_sha256=digest(lineage()))


def lineage():return dict(origin={'origin_commit':'a'*40,'bundle_sha256':'b'*64},cache_sha256=['c'*64],inputs_sha256='d'*64)


def test_shape_compatible_wrong_task_or_metric_checkpoint_is_rejected(tmp_path):
    m=Controller(13,conditions=9); meta=metadata(); p=tmp_path/'model.pth'
    save(p,m,meta)
    restored=load(p,task='od',recipe=meta['recipe'],task_models=meta['task_models'],lineage=lineage())
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in m.state_dict().items())
    with pytest.raises(ValueError): load(p,task='ot',recipe=meta['recipe'],task_models=meta['task_models'],lineage=lineage())
    bad=copy.deepcopy(meta); bad['risk_components']=['fp','fn','localization']; save(p,m,bad)
    with pytest.raises(ValueError): load(p,task='od',recipe=meta['recipe'],task_models=meta['task_models'],lineage=lineage())
    save(p,m,meta); changed=copy.deepcopy(meta['recipe']); changed['gt_limits']=[.05]*3
    with pytest.raises(ValueError): load(p,task='od',recipe=changed,task_models=meta['task_models'],lineage=lineage())
    value=torch.load(p,weights_only=True); value['metadata']['seed']=0; torch.save(value,p)
    with pytest.raises(ValueError): load(p,task='od',recipe=meta['recipe'],task_models=meta['task_models'],lineage=lineage())


def test_baseline_manifest_checks_transitive_code_without_git_at_runtime(tmp_path):
    import subprocess
    root=tmp_path/'repository'; root.mkdir()
    subprocess.run(['git','init','-q',str(root)],check=True)
    (root/'adaptive_vcm').mkdir(); p=root/'adaptive_vcm/shared.py'; p.write_text('VALUE=1\n')
    subprocess.run(['git','-C',str(root),'add','.'],check=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
                    'commit','-qm','baseline'],check=True)
    sha=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    manifest=baseline_manifest(root,sha); verify_baseline(root,manifest)
    p.write_text('VALUE=2\n')
    with pytest.raises(ValueError): verify_baseline(root,manifest)

