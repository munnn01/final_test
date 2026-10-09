import numpy as np
import torch
from v33_fixtures import record,Action,Detector


def test_actual_forward_proxy_backward():
    from adaptive_vcm.v33ap.train import training_step,temporal_loss,exposure
    from adaptive_vcm.v33ap.protocol import settings,codec_factory
    from adaptive_vcm.v33ap.tasks import TaskHeads
    from adaptive_vcm.v33ap.model import JointPreprocessor
    from adaptive_vcm.v32vc.virtual import VirtualCodec
    from adaptive_vcm.v32rf.bridge import pixels
    r=record('od'); heads=TaskHeads('cpu',action=Action(),detector=Detector()); heads.source_targets(r)
    model=JointPreprocessor(); cfg=settings()
    for name in ('h264','h265'):
        codec=codec_factory('od','ultrafast')(name,35)
        step=training_step(model,VirtualCodec().train(),heads,r,codec,'joint',cfg)
        packet=codec.roundtrip(r.rgb,source_shape=r.rgb.shape,task='od')
        assert np.array_equal(pixels(step['reconstruction']),packet.decoded)
        assert step['actual_bpp']==8*packet.coded_bytes/256
        model.zero_grad(); step['loss'].backward(); assert model.head.weight.grad.abs().sum()>0
    x=torch.rand(1,3,3,8,8); y=x+.1
    assert temporal_loss(y,x,torch.ones(1,3,dtype=torch.bool))==0
    splits={k:dict(fit=[f'{k}/{i}' for i in range(48)]) for k in ('ar','od')}
    order=exposure(splits,cfg)
    assert len(order)==960 and order==exposure(splits,cfg)
    assert len({(i,c,q) for i,c,q in order})==960


def test_diagnostics_no_updates(tmp_path):
    from adaptive_vcm.v33ap.calibration import diagnose
    from adaptive_vcm.v33ap.model import JointPreprocessor
    from adaptive_vcm.v33ap.protocol import settings
    from adaptive_vcm.v33ap.tasks import TaskHeads
    r=record('od'); h=TaskHeads('cpu',action=Action(),detector=Detector()); h.source_targets(r)
    m=JointPreprocessor(); state={k:v.clone() for k,v in m.state_dict().items()}
    cfg={**settings(),'codecs':['h264'],'qps':[35],'preset':'ultrafast'}
    d=diagnose(m,{r.id:r},[r.id],{'od':{}},h,cfg,tmp_path)
    assert len(d['rows'])==1 and all(torch.equal(v,state[k]) for k,v in m.state_dict().items())
    assert np.isfinite(d['rows'][0]['rate_relative_error'])
