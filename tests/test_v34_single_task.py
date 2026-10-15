"""Independent four-QP supervision, native training and task-only evaluation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adaptive_vcm.v34.protocol import ROOT, ACTIONS, settings, digest, codec_factory
from adaptive_vcm.v34.tasks import Heads
from adaptive_vcm.v33ap.sources import SourceRecord


def qp4():
    return settings(ROOT / 'configs/v34_r1_qp4.json')


def test_four_qp_recipe_preserves_original_and_r1_protocols(tmp_path):
    cfg = qp4()
    assert cfg['qps'] == [30, 35, 40, 45]
    original = settings(ROOT / 'configs/v34_arot_r1.json')
    assert original['qps'] == settings()['qps'] == [30, 35, 40, 45, 50]
    assert {k:v for k,v in cfg.items() if k not in ('qps','revision')} == {
        k:v for k,v in original.items() if k not in ('qps','revision')}
    for key, value in [('qps', [30,35,40,50]), ('revision','arot-r1')]:
        bad = {**cfg, key:value}; path = tmp_path / 'bad.json'
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError): settings(path)


class TinySwin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
    def forward(self, x):
        z = self.features(x.permute(0,2,3,4,1)).mean((1,2,3,4))
        return torch.stack((z,-z), 1)


class TinyAction:
    def __init__(self, name='swin3d_t', device='cpu'):
        self.model, self.name, self.device = TinySwin(), name, device
    def logits(self, x): return self.model(x)
    def probabilities(self, rgb):
        from adaptive_vcm.v32vc.train import tensor
        return self.logits(tensor(rgb, self.device)).softmax(-1)[0].detach().numpy()


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Identity()
        self.transform = SimpleNamespace(min_size=(320,), max_size=640)
        self.roi_heads = SimpleNamespace(detections_per_img=300)
    def forward(self, images, targets=None):
        features = self.backbone(torch.stack(images))
        if targets is not None:
            assert all('boxes' in t and 'labels' in t for t in targets)
            return {'loss': features.square().mean()}
        return [dict(boxes=torch.tensor([[4.,4.,16.,16.]]),
                     scores=torch.tensor([.9]), labels=torch.tensor([1])) for _ in images]


class TinyDetection:
    def __init__(self, name='resnet50', device='cpu'):
        self.model, self.name = TinyDetector(), name
    def predict(self, rgb):
        from adaptive_vcm.v32vc.train import tensor
        values=self.model([tensor(rgb,'cpu')[0,:,0]])[0]
        return {k:v.detach().numpy() for k,v in values.items()}


def heads_for(task):
    heads = Heads.__new__(Heads)
    heads.device, heads.experiment, heads.cfg = 'cpu', task, qp4()
    heads.action = TinyAction()
    heads.detector = SimpleNamespace(model=TinyDetector(), name='resnet50')
    heads.configure_detector()
    return heads


def record(task, split='train'):
    rgb = np.random.default_rng({'train':1,'cal':2,'dev':3,'test':4}[split]).integers(
        0,256,(1 if task=='od' else 16,24,24,3),dtype=np.uint8)
    r = SourceRecord(task+'/'+split, task, {'label':0}, rgb, (1,1,0,0), {})
    r.split = split
    if task=='od':
        r.targets = dict(boxes=torch.tensor([[4.,4.,16.,16.]]), labels=torch.tensor([1]))
    elif task=='ot':
        r.targets = [dict(boxes=np.array([[4.,4.,16.,16.]]),track_ids=np.array([1])) for _ in rgb]
    return r


@pytest.mark.parametrize('task', ['ar','od','ot'])
def test_single_task_measurement_and_guard_have_only_requested_damage(task):
    heads, r = heads_for(task), record(task)
    heads.source(r)
    active = ['ar','od','ot'].index(task)
    assert r.mask == [int(i==active) for i in range(3)]
    cost = heads.costs(r.rgb, r)
    assert np.isfinite(cost[active])
    assert all(v is None for i,v in enumerate(cost) if i != active)
    # Held-out source references contain neither labels nor GT targets.
    source = SourceRecord(r.id,task,{},r.rgb,r.geometry,{})
    heads.source(source)
    assert np.isfinite(heads.costs(r.rgb,source,source_only=True)[active])
    if task != 'ar': assert 'ar' not in source.source_predictions


@pytest.mark.codec
@pytest.mark.parametrize('task', ['ar','od','ot'])
def test_actual_four_qp_cache_trains_only_its_own_domain(tmp_path, task):
    from adaptive_vcm.v34.runner import measure_candidates, input_pair, condition, train_models, validate_cache
    from adaptive_vcm.v34.preprocessing import region_maps
    cfg = qp4(); cfg.update(epochs=1)
    heads = heads_for(task); sources={}; inputs={}; rows=[]
    for split in ('train','cal'):
        r=record(task,split); heads.source(r)
        sources[r.id]=dict(split=split,pixel_sha256=__import__('hashlib').sha256(r.rgb.tobytes()).hexdigest())
        maps=region_maps(r.rgb,r.source_predictions['od']); inputs[r.id]=input_pair(r.rgb,maps)
        for codec in cfg['codecs']:
            for qp in cfg['qps']:
                table=measure_candidates(r.rgb,maps,ACTIONS,codec_factory(task,codec,qp,cfg),task,
                                         lambda pixels:heads.costs(pixels,r))
                rows.append(dict(id=r.id,split=split,domain=task,codec=codec,qp=qp,mask=r.mask,
                                 condition=condition(codec,qp,r.mask,cfg,task),**table))
    cache=dict(schema=cfg['schema'],experiment=task,settings=cfg,actions=[list(a) for a in ACTIONS],
               origin=dict(origin_commit='a'*40,bundle_sha256='b'*64),sources=sources,rows=rows)
    cache['sha256']=digest(cache)
    assert len(rows)==16 and {r['qp'] for r in rows}=={30,35,40,45}
    models=train_models(cache,inputs,cfg,tmp_path,'cpu',heads.hashes())
    assert len(models)==3
    for model in models:
        assert set(model.metadata['calibration']['domains'])=={task}
        assert model.metadata['experiment']==task
        assert not model.metadata['codec_backpropagation']
    bad=copy.deepcopy(cache); bad['rows'][0]['mask']=[1,1,1]
    bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError): validate_cache(bad)


@pytest.mark.parametrize('task,datasets', [
    ('ar',['qktttttttttt/kineticscleaned']),
    ('od',['awsaf49/coco-2017-dataset']), ('ot',[])])
def test_separate_public_jobs_pin_four_qps_and_single_task(tmp_path, monkeypatch, task, datasets):
    from scripts import v34_kaggle as payload
    from tests.test_v34_payload import fixture_source
    monkeypatch.setattr(payload,'source_snapshot',fixture_source)
    receipt=payload.prepare(task,'fixture',tmp_path,recipe='r1-qp4-single-task')
    assert payload.validate_payload(tmp_path)==receipt
    job=json.loads((tmp_path/'job.json').read_text())
    assert job['protocol']['qps']==[30,35,40,45]
    assert job['experiment']==task and job['datasets']==datasets
    assert job['config_path']=='configs/v34_r1_qp4.json'
    assert job['slug'].startswith('v34-'+task+'-r1-qp4-')
    for cell in json.loads((tmp_path/'notebook.ipynb').read_text())['cells']:
        compile(''.join(cell['source']),'<single task notebook>','exec')


@pytest.mark.codec
@pytest.mark.parametrize('task', ['ar','od','ot'])
def test_single_task_heldout_evaluation_uses_no_other_task_gt(tmp_path, monkeypatch, task):
    import os
    from adaptive_vcm.v34 import evaluation
    from adaptive_vcm.v34.model import Controller
    if task=='ot' and not os.environ.get('V34_TRACKEVAL'):
        pytest.skip('requires pinned TrackEval')
    # A bounded condition smoke; the separate training regression covers the
    # full 4-QP x 2-codec x 13-action measurement grid and all three seeds.
    cfg=qp4(); cfg.update(qps=[35],codecs=['h264'],seeds=[303401])
    heads=heads_for(task); groups={s:[record(task,s)] for s in ('dev','test')}; meta={}
    monkeypatch.setattr(evaluation,'ActionAnalyzer',TinyAction)
    monkeypatch.setattr(evaluation,'DetectionAnalyzer',TinyDetection)
    if task=='od':
        meta['od']=dict(categories=[dict(id=1,name='person')],annotations=[])
        for i,records in enumerate(groups.values(),1):
            r=records[0]; r.item=dict(image_id=i,width=24,height=24)
            r.targets={'heldout_GT_must_not_enter_guard':True}
            meta['od']['annotations'].append(dict(id=i,image_id=i,category_id=1,
                bbox=[4,4,12,12],area=144,iscrowd=0))
    elif task=='ot':
        from tests.test_v32_tracking import fixture_sequence
        from adaptive_vcm.v32.tracking import plan_sequences
        seq=fixture_sequence(tmp_path/'mot',count=16)
        (seq/'gt/gt.txt').write_text(''.join(f'{i},1,4,4,12,12,1,1,1\n' for i in range(1,17)))
        meta['ot']=plan_sequences(tmp_path/'mot',['MOT20-01'],16)
        for records in groups.values():
            records[0].item=dict(sequence='MOT20-01',width=80,height=40)
            records[0].targets={'heldout_GT_must_not_enter_guard':True}
    model=Controller(len(ACTIONS),conditions=9).eval()
    model.metadata=dict(seed=303401,calibration=dict(domains={task:dict(risk=[[0.]*3]*13,rate=[0.]*13)}))
    out=tmp_path/'evaluated'; out.mkdir()
    evaluation.evaluate(groups,heads,[model],cfg,meta,out,Path(os.environ['V34_TRACKEVAL']) if task=='ot' else None)
    rows=json.loads((out/'test_rows.json').read_text())
    assert {r['task'] for r in rows}=={task}
    assert {r['arm'] for r in rows}=={'anchor','fixed','forced_303401','selected_303401','calibrated_303401'}
    summary=json.loads((out/'test_summary.json').read_text())
    if task=='ot':
        assert summary['clean_source_tracking']['combined']['HOTA']>99
        assert 'ar_source_kl' not in rows[0]
    else:
        assert set(summary['results'])=={task+'/h264'}
    evaluator=json.loads((out/'evaluators.json').read_text())
    assert (evaluator['transfer_ar'] is not None)==(task=='ar')
