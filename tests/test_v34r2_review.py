"""Reproductions from the independent whole-branch review."""
import copy
import hashlib
import json
import numpy as np
import pytest
from adaptive_vcm.v34.preprocessing import region_maps
from adaptive_vcm.v34r2.preprocessing import Support
from adaptive_vcm.v34r2 import evaluation,runner
from adaptive_vcm.v34r2.protocol import settings,seal,validate_cache
from tests.test_v34r2_support import pred
from tests.test_v34r2_learning import row


def test_fading_ar_maps_are_bit_exact():
    rgb=np.zeros((16,32,32,3),np.uint8)
    predictions=[pred()]+[pred((),()) for _ in range(15)]
    assert np.array_equal(Support().build(rgb,predictions,[1,0,0]),region_maps(rgb,predictions))


def test_nonmonotone_bd_clears_all_dependent_acceptance_flags():
    a=dict(bpp=[1,2,3,4],quality=[.5,.7,.6,.8]); b={**a,'bpp':[.8,1.6,2.4,3.2]}
    result=evaluation.comparison(a,b)
    assert result['bd_rate_pct'] is None and result['point_passes'] is False
    assert result['screen_passes'] is False
    assert all(not v for k,v in result.get('guards',{}).items() if k.startswith(('canonical_bd','pchip_bd','bd_quality')))


@pytest.mark.parametrize('qualities,valid', [([.5,.7,.6,.8],False),([.5,.5,.8,.8],False),([.5,.6,.7,.8],True)])
def test_bootstrap_uses_identical_curve_validity_rules(qualities,valid):
    table={a:{} for a in ('anchor','cnn')}
    for i,q in enumerate([30,35,40,45]):
        for arm,scale in [('anchor',1),('cnn',.8)]:
            table[arm][('h264',q)]={j:((i+1)*scale,[dict(image_id=j,quality=qualities[i])]) for j in (1,2)}
    def metric(predictions,*_): return predictions[0]['quality']
    value=evaluation.paired_bootstrap_detection_bd(table,codec='h264',qps=[30,35,40,45],
        gt_by_id={1:[],2:[]},image_ids=[1,2],ann_meta={},arms=['cnn'],n_boot=5,seed=1,metric_fn=metric)['cnn']
    assert value['requested_draws']==5 and value['n_draws']==(5 if valid else 0)
    assert value['valid_fraction']==(1 if valid else 0)
    if valid: assert value['lo']==pytest.approx(-20)
    else: assert value['lo'] is value['hi'] is value['p_lt_zero'] is None


def cache():
    cfg=settings(); cfg.update(codecs=['h264'],qps=[30])
    return seal(dict(schema=cfg['schema'],settings=cfg,experiment='od',rows=[row()],
        actions=[list(a) for a in runner.ACTIONS],task_models={
            'ar':dict(name='swin3d_t',state_sha256='e'*64), 'od':dict(name='resnet50',state_sha256='f'*64)},
        sources={'a':dict(split='train',domain='od',frames=1,pixel_sha256='b'*64,frame_recipe={})},
        origin=dict(origin_commit='c'*40,bundle_sha256='d'*64)))


@pytest.mark.parametrize('mutation', ['actions','origin','heads','prefix','packet_hash','joint_mask'])
def test_sealed_cache_still_requires_semantic_measurement_provenance(mutation):
    c=cache()
    if mutation=='actions': c['actions']=[]
    if mutation=='origin': c['origin']={}
    if mutation=='heads': c.pop('task_models')
    if mutation=='prefix': c['rows'][0]['packets'][1]['prefix_sha256']='0'*64
    if mutation=='packet_hash': c['rows'][0]['packets'][1]['sha256']='not-a-hash'
    if mutation=='joint_mask':
        c['experiment']='ar_od_ot'; c['rows'][0]['active']=[1,1,1]
        c['rows'][0]['source_damage']=[[0,v[1],0] for v in c['rows'][0]['source_damage']]
    with pytest.raises(ValueError): validate_cache(seal(c))


def test_local_cli_rejects_origin_before_reading_data_or_models(tmp_path,monkeypatch):
    path=tmp_path/'origin.json'; path.write_text(json.dumps(dict(origin_commit='a'*40,bundle_sha256='b'*64)))
    monkeypatch.setattr('sys.argv',['runner','--experiment','od','--origin',str(path),'--out',str(tmp_path/'out')])
    def touched(*args,**kwargs): raise AssertionError('data touched before origin validation')
    monkeypatch.setattr(runner,'load_groups',touched)
    with pytest.raises(ValueError,match='origin'): runner.main()


def test_evaluation_rejects_stale_checkpoint_settings_and_heads(tmp_path):
    from tests.test_v34r2_runner import heads
    from adaptive_vcm.v34r2.model import Controller
    h=heads(); cfg=settings(); model=Controller(); model.metadata=dict(settings=cfg,experiment='od',task_models=h.hashes())
    model.metadata['settings']={**cfg,'budget_ratio':2}
    with pytest.raises(ValueError,match='checkpoint'):
        evaluation.evaluate({'dev':[],'test':[]},{},cfg,tmp_path,h,[model],'od',None,None)
    model.metadata.update(settings=cfg,task_models={})
    with pytest.raises(ValueError,match='checkpoint'):
        evaluation.evaluate({'dev':[],'test':[]},{},cfg,tmp_path,h,[model],'od',None,None)


def test_empty_detections_preserve_undefined_coco_size_bins():
    meta={'categories':[dict(id=1,name='person')]}
    gt={1:[dict(id=1,image_id=1,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0)]}
    stats=evaluation.coco_stats([],gt,[1],meta)
    assert stats['mAP']==stats['APsmall']==0
    assert stats['APmedium'] is None and stats['APlarge'] is None


def test_source_origin_hashes_are_checked_against_executing_files():
    from adaptive_vcm.v34r2.protocol import validate_origin
    from tests.test_v34r2_runner import fixture_origin
    origin=fixture_origin(); validate_origin(origin,verify_files=True)
    origin['source_files_sha256']['adaptive_vcm/v34r2/tasks.py']='0'*64
    with pytest.raises(ValueError,match='origin source mismatch'):
        validate_origin(origin,verify_files=True)
