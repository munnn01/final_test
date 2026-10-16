import numpy as np
from adaptive_vcm.v34r3.metrics import compare
from adaptive_vcm.v34r3.evaluation import promotion


def test_nonmonotone_and_cubic_unstable_curves_clear_reliability():
    a=dict(bpp=[1,2,3,4],quality=[.4,.6,.5,.8])
    result=compare(a,a,task='ot')
    assert not result['bd_reliable'] and result['bd_rate_pct'] is None
    assert result['raw_points']['anchor']==a
    a=dict(bpp=[1,2,3,4],quality=[.1,.11,.12,.9])
    result=compare(a,dict(bpp=[.9,1.8,2.7,3.6],quality=a['quality']),task='od')
    assert result['interpolation_sensitive'] and not result['bd_reliable']
    a=dict(bpp=[1,2,3,4],quality=[.2,.4,.6,.8])
    result=compare(a,dict(bpp=[.9,1.8,2.7,3.6],quality=a['quality']),task='od')
    assert result['bd_reliable'] and abs(result['bd_rate_pct']+10)<1e-6


def test_one_cal_sequence_or_identity_only_cannot_confirm_improvement():
    report=dict(task='ot',byte_improvement=True,worst_qp_gap_pp=0,dev_gate_supported=True,nonidentity_fraction=.2,bd_reliable=True)
    cohorts=dict(confirmation_eligible=True)
    assert not promotion(report,cohorts,dict(source_groups=1,promotion_eligible=False))['confirmed']
    report.update(task='od',nonidentity_fraction=0)
    assert not promotion(report,cohorts,dict(source_groups=64,promotion_eligible=True))['confirmed']


def test_nonfinite_or_missing_quality_is_reported_without_fitting():
    for q in (float('nan'),None,float('inf')):
        a=dict(bpp=[1,2,3,4],quality=[.1,.2,q,.4])
        result=compare(a,a,task='od')
        assert not result['bd_reliable'] and result['bd_rate_pct'] is None
