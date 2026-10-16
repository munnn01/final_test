import copy
import numpy as np
import pytest
import torch
from adaptive_vcm.v34r3.protocol import settings,recipe
from adaptive_vcm.v34r3.model import Controller
from adaptive_vcm.v34r3 import learning


def row(split='train',identifier='a',group='a',variant='identity',task='od'):
    return dict(id=identifier,group=group,split=split,prefix_variant=variant,codec='h264',qp=35,
        gt_damage=[[0.,0.,0.],[.02,.02,.02]]+[[.3,.3,.3]]*11,
        source_damage=[[0.]*(4 if task=='od' else 3) for _ in range(13)],rate=[1.,.7]+[1.2]*11)


def test_train_scales_and_shortlist_never_fit_cal_or_test():
    cfg=recipe(settings(),'od','train-final');r=row()
    assert learning.fit_scales([r])==pytest.approx([.3,.3,.3])
    for split in ('cal','dev','test'):
        with pytest.raises(ValueError):learning.fit_scales([row(split)])
        with pytest.raises(ValueError):learning.fixed_shortlist([row(split)],cfg)
    assert learning.soft_targets(r,cfg)[1]>learning.soft_targets(r,cfg)[0]>0
    assert np.all(learning.soft_targets(r,cfg)[2:]==0)
    assert len(learning.fixed_shortlist([r],cfg))==3


def test_ot_identity_rollout_have_equal_training_weight():
    rows=[row(identifier='a',task='ot')]+[row(identifier=str(i),variant='rollout',task='ot') for i in range(3)]
    assert learning.training_weights(rows)==pytest.approx([2,2/3,2/3,2/3])
    cfg=recipe(settings(),'ot','train-final');cfg['epochs']=1
    inputs={r['id']:(torch.rand(2,3,48,48),torch.rand(2,3,48,48)) for r in rows}
    m=Controller(13,9);history,scales=learning.train(m,rows,inputs,cfg,device='cpu')
    assert np.isfinite(history[0]['loss']) and not m.training
    gradients=m.head[-1].weight.grad.reshape(13,5,-1)[:,1:4]
    assert (gradients.abs().sum((0,2))>0).all()
    assert history[0]['variant_mass']==pytest.approx({'identity':.5,'rollout':.5})


def test_cal_counts_sequences_and_keeps_weights_unchanged():
    cfg=recipe(settings(),'ot','train-final')
    rows=[row('cal',str(i),'one-sequence',task='ot') for i in range(8)]
    preds=[dict(risk=np.zeros((13,3)),rate=np.ones(13),logits=np.zeros(13),model_state_sha256='b'*64) for _ in rows]
    model=Controller(13,9);before={k:v.clone() for k,v in model.state_dict().items()}
    cal=learning.calibrate(rows,preds,recipe=cfg)
    assert cal['source_groups']==1 and cal['rows']==8 and not cal['promotion_eligible']
    assert 'no coverage guarantee' in cal['scope']
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    rows.append(row('cal','new','second-sequence',task='ot'));preds.append(preds[0])
    assert learning.calibrate(rows,preds,recipe=cfg)['promotion_eligible']
    with pytest.raises(ValueError):learning.calibrate([row()],preds[:1],recipe=cfg)
