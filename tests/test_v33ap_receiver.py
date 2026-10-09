import hashlib
from types import SimpleNamespace
import numpy as np
import pytest
from v33_fixtures import Action,Detector,record


def packet(rgb): return SimpleNamespace(decoded=rgb,packet_sha256='1'*64,coded_bytes=100)


def test_shared_rgb_scope_and_singleton():
    from adaptive_vcm.v33ap.receiver import SharedReceiver
    a=Action(); d=Detector(); r=SharedReceiver(a,d); rgb=record().rgb
    p=packet(rgb); got=r.predict(p,tasks=('ar','od'))
    assert got['ar_packet_sha256']==got['od_packet_sha256'] and got['coded_bytes']==100
    assert got['od_map'] is None and got['ar_accuracy'] is None
    got2=r.predict(p,tasks=('od',),indices=(0,)); assert list(got2['od'])==[0]
    calls=a.calls
    single=r.predict(packet(rgb[:1]),tasks=('od',)); assert a.calls==calls and len(single['od'])==1
    with pytest.raises(ValueError): r.predict(packet(rgb[:1]),tasks=('ar',))
    reshaped=rgb.reshape(1,48,16,3); r.predict(packet(reshaped),tasks=('od',))
    assert r._cache['key'][0]==reshaped.shape


def test_backbone_latencies():
    from adaptive_vcm.v33ap.receiver import SharedReceiver
    r=SharedReceiver({'swin3d_t':Action(),'r3d_18':Action()},Detector())
    got=r.predict(packet(record().rgb),tasks=('ar','od'),indices=(0,2))
    assert set(got['latency_seconds'])=={'ar/swin3d_t','ar/r3d_18','od'}
    assert got['total_prediction_seconds']>=sum(got['latency_seconds'].values())
