import numpy as np
import pytest
from adaptive_vcm.selection import Observation
from adaptive_vcm.v32.selection import select_arms
from adaptive_vcm.v32.evaluate import summarize
from adaptive_vcm.v32.protocol import load_config
from adaptive_vcm.v32vc.evaluate import EXTRA_FAMILIES, extra_candidates
from tests.test_v32_evaluate import records
from tests.test_v32_roi import pixels,support


def test_new_families_are_explicit_and_do_not_enter_old_v32_arm():
    rows = [dict(family='controls',observation=Observation('identity',1000,(0.,),(True,))),
            dict(family='vc_plain',observation=Observation('vc',500,(0.,),(True,))),
            dict(family='vc_joint_retarget',observation=Observation('joint',400,(.04,),(True,)))]
    result = select_arms(rows,.03,.01,False,extra_families=EXTRA_FAMILIES)
    assert result['v32']==0 and result['vc_plain']==1 and result['vc_joint']==0
    assert set(select_arms(rows,.03,.01,False))=={'controls','simplification','retarget','v32'}


def test_additional_paired_grid_is_complete_or_rejected():
    rows = records()
    for row in list(rows):
        if row['arm']=='controls':
            rows.extend([{**row,'arm':arm} for arm in EXTRA_FAMILIES])
    result = summarize(rows,['a','b'],'ar',load_config(),0,extra_arms=tuple(EXTRA_FAMILIES))
    assert set(EXTRA_FAMILIES)<=set(result['results']['h264'])
    with pytest.raises(ValueError): summarize(rows[:-1],['a','b'],'ar',load_config(),0,extra_arms=tuple(EXTRA_FAMILIES))


def test_candidates_keep_same_source_and_paid_grids():
    class FixedModel:
        def pixels(self,rgb,support,codec): return rgb.copy()
    rgb,s = pixels(),support()
    generated = extra_candidates(rgb,s,object(),{'plain':FixedModel(),'joint':FixedModel()},load_config())
    assert sum(c.family=='vc_plain' for c in generated)==1
    assert sum(c.family=='vc_joint_plain' for c in generated)==1
    assert sum(c.family=='vc_retarget_only' for c in generated)==2
    assert sum(c.family=='vc_joint_retarget' for c in generated)==2
    assert all(c.grids and c.rgb.shape[1]<rgb.shape[1] for c in generated if 'retarget' in c.family)


@pytest.mark.parametrize('codec',['h264','h265'])
def test_new_cnn_real_codec_receiver_plain_and_retarget(codec):
    import torch
    from adaptive_vcm.v32vc.model import ResidualPreprocessor
    from adaptive_vcm.v32vc.checkpoint import LearnedPreprocessor
    from adaptive_vcm.v32.codec import V32Codec
    model = ResidualPreprocessor('od').eval()
    with torch.no_grad(): model.head.bias.fill_(.2)
    wrapper = LearnedPreprocessor(model,{},'synthetic')
    rgb,s = pixels(),support()
    actual = V32Codec(codec,40,preset='ultrafast')
    candidates = extra_candidates(rgb,s,actual,{'plain':wrapper,'joint':wrapper},load_config())
    for c in (candidates[0],next(c for c in candidates if c.family=='vc_joint_retarget')):
        packet = actual.roundtrip(c.rgb,source_shape=rgb.shape,task='od',grids=c.grids)
        independent = actual.decode_packet(packet.data)
        assert np.array_equal(packet.decoded,independent.decoded)
        assert independent.decoded.shape==rgb.shape and packet.coded_bytes==packet.elementary_bytes+packet.metadata_bytes
