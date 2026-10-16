"""Continuations preserve scientific outputs while changing execution boundaries."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from adaptive_vcm.v34r2 import evaluation
from adaptive_vcm.v34r2.model import Controller
from adaptive_vcm.v34r2.protocol import settings
from tests.test_v34r2_runner import heads
from tests.test_v34_single_task import record


def model_for(h,cfg):
    m=Controller().eval()
    m.metadata=dict(settings=copy.deepcopy(cfg),experiment='ot',task_models=h.hashes(),
                    seed=303401,fixed_shortlist=[1,2,3])
    return m


@pytest.mark.codec
def test_codec_scopes_preserve_packets_continuous_tracking_and_metrics(tmp_path):
    # A reset at the second window, a changed condition, or a missing codec
    # must change the observable rows/official metrics in this test.
    import os
    from tests.test_v32_tracking import fixture_sequence
    from adaptive_vcm.v32.tracking import plan_sequences,load_trackeval
    cfg=settings(); cfg.update(qps=[35],seeds=[303401],bootstrap_draws=0,dev_exhaustive=False)
    sequence=fixture_sequence(tmp_path/'mot',count=32)
    (sequence/'gt/gt.txt').write_text(''.join(f'{i},1,4,4,12,12,1,1,1\n' for i in range(1,33)))
    name=sequence.name; plans=plan_sequences(sequence.parent,[name],32)
    records=[]
    for i in range(2):
        r=record('ot','test'); r.id+=f'/{i}'; r.rgb=r.rgb.copy()^(17*i)
        r.item=dict(sequence=name,start=1+16*i,width=80,height=40,path=str(sequence))
        r.frame_recipe['frame_indices']=list(range(1+16*i,17+16*i))
        r.targets={'heldout_GT_must_not_enter_guard':True}; records.append(r)
    groups=dict(dev=[],test=records); meta=dict(ot=plans)
    h=heads(); m=model_for(h,cfg); package=load_trackeval(os.environ['V34_TRACKEVAL'])
    full=tmp_path/'full'; full.mkdir()
    evaluation.evaluate(groups,meta,cfg,full,h,[m],'ot',package,None)
    baseline=json.loads((full/'test_rows.json').read_text())
    fields=['id','codec','qp','arm','action','packet_sha256','decoded_sha256','prefix_before','prefix_after']
    project=lambda row:{k:row.get(k) for k in fields}
    for c in ('h264','h265'):
        out=tmp_path/c; out.mkdir()
        evaluation.evaluate(groups,meta,cfg,out,h,[m],'ot',package,None,
                            splits=('test',),codecs=(c,),domains=('ot',))
        assert not (out/'dev_summary.json').exists()
        actual=json.loads((out/'test_rows.json').read_text())
        assert [project(r) for r in actual]==[project(r) for r in baseline if r['codec']==c]
        cnn=[r for r in actual if r['arm']=='cnn_303401']
        assert cnn[1]['prefix_before']==cnn[0]['prefix_after']
        result=json.loads((out/'test_summary.json').read_text())
        assert set(result['ot_curves'])=={c}
        assert result['ot_official_metrics'][f'{c}_35_cnn_303401']['combined']['HOTA']==100
    assert m.metadata['settings']==cfg and cfg['codecs']==['h264','h265']


@pytest.mark.parametrize('scope', [dict(codecs=('av1',)),dict(splits=('train',)),dict(domains=('bogus',)),dict(codecs=())])
def test_invalid_execution_scope_is_rejected_before_evaluation(tmp_path,scope):
    h=heads(); cfg=settings(); m=model_for(h,cfg)
    with pytest.raises(ValueError,match='scope'):
        evaluation.evaluate(dict(dev=[],test=[]),{},cfg,tmp_path,h,[m],'ot',None,None,**scope)


@pytest.mark.codec
@pytest.mark.parametrize('task',['od','ot'])
def test_resume_keeps_cached_rows_and_measures_only_missing_source(tmp_path,monkeypatch,task):
    import os
    from adaptive_vcm.v34r2 import runner
    from tests.test_v34r2_runner import fixture_origin
    from adaptive_vcm.v32.tracking import load_trackeval
    cfg=settings(); cfg.update(codecs=['h264'],qps=[35],seeds=[303401],epochs=1,bootstrap_draws=0,dev_exhaustive=False)
    groups={s:[] for s in ('train','cal','dev','test')}
    for split,count in [('train',1),('cal',2)]:
        for i in range(count):
            r=record(task,split); r.id+=f'/{i}'; r.rgb=r.rgb.copy()^(17*i)
            if task=='ot':
                r.item=dict(sequence='sequence-'+split,start=1+16*i,width=24,height=24)
                r.frame_recipe['frame_indices']=list(range(1+16*i,17+16*i))
            groups[split].append(r)
    h=heads(); origin=fixture_origin()
    meta={'od':dict(categories=[dict(id=1,name='person')],annotations=[])}
    package=load_trackeval(os.environ['V34_TRACKEVAL']) if task=='ot' else None
    first=tmp_path/'first'
    runner.run_records(groups,meta,cfg,first,origin,h,task,package)
    cache=json.loads((first/'measurements.json').read_text())
    missing=groups['cal'][-1].id
    original=next(r for r in cache['rows'] if r['id']==missing)
    partial=copy.deepcopy(cache); partial.pop('sha256')
    partial['rows']=[r for r in partial['rows'] if r['id']!=missing]
    called=[]; real=runner.measure
    def measured(record,*args,**kwargs):
        called.append(record.id)
        return real(record,*args,**kwargs)
    monkeypatch.setattr(runner,'measure',measured)
    out=tmp_path/'resumed'
    runner.run_records(groups,meta,cfg,out,origin,h,task,package,inherited_cache=partial,training_only=True)
    result=json.loads((out/'measurements.json').read_text())
    assert called==[missing]
    assert result['rows'][:-1]==partial['rows']
    resumed=result['rows'][-1]
    assert resumed['prefix_sha256']==original['prefix_sha256']
    assert [p['sha256'] for p in resumed['packets']]==[p['sha256'] for p in original['packets']]
    assert not (out/'test_summary.json').exists() and not (out/'complete.json').exists()
    assert (out/'stage_complete.json').exists() and (out/'controller_303401.pth').exists()


def test_inherited_cache_rejects_changed_heads_pixels_and_duplicate_conditions(tmp_path):
    from adaptive_vcm.v34r2.continuation import validate_inherited
    from tests.test_v34r2_review import cache as cached
    from adaptive_vcm.v34r2.protocol import seal
    c=cached(); cfg=c['settings']; src=c['sources']; task_models=c['task_models']; origin=c['origin']
    for bad,models,sources in [(c,{**task_models,'ar':dict(name='other',state_sha256='d'*64)},src),
                              (c,task_models,{'a':{**src['a'],'pixel_sha256':'d'*64}})]:
        with pytest.raises(ValueError): validate_inherited(bad,cfg,'od',models,sources,origin)
    duplicate=copy.deepcopy(c); duplicate['rows']*=2; duplicate.pop('sha256')
    with pytest.raises(ValueError,match='duplicate'):
        validate_inherited(duplicate,cfg,'od',task_models,src,origin)


def test_continuation_rejects_changed_heldout_pixels_and_missing_sources():
    import hashlib
    from adaptive_vcm.v34r2.continuation import verify_cohorts
    r=record('od','test'); groups=dict(train=[],cal=[],dev=[],test=[r])
    saved={r.id:dict(split='test',domain='od',frames=1,pixel_sha256=hashlib.sha256(r.rgb.tobytes()).hexdigest(),frame_recipe=r.frame_recipe)}
    verify_cohorts(groups,saved)
    r.rgb=r.rgb.copy()^1
    with pytest.raises(ValueError,match='cohort'): verify_cohorts(groups,saved)
    with pytest.raises(ValueError,match='cohort'): verify_cohorts(groups,{})
