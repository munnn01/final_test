from types import SimpleNamespace
import numpy as np


def test_ar_extra_candidates_never_retarget():
    from adaptive_vcm.v32at.ar import prepare_extension
    rgb = np.zeros((3,16,16,3), np.uint8)
    model = SimpleNamespace(pixels=lambda x,s,c: x.copy())
    options, seconds = prepare_extension(rgb, {}, object(), {'real':model}, {})
    assert options['enable_retarget'] is False
    assert all(not c.grids for c in options['extra_candidates'])
    assert set(options['extra_families']) == {'rf_real', 'rf_real_forced'}
    assert options['forced_names'] == {'rf_real_forced':'rf_real_rgb'}


def test_ar_adapter_scores_same_packets_complete_grid_without_retarget(tmp_path,monkeypatch):
    import json
    from argparse import Namespace
    from adaptive_vcm.data import partition
    from adaptive_vcm.v32 import evaluate as base
    from adaptive_vcm.v32at import ar
    from adaptive_vcm.v32.protocol import load_config
    from adaptive_vcm.v32.transport import unpack_packet
    identifier=next(str(i) for i in range(1000) if partition(str(i))=='test')
    rgb=np.random.default_rng(1).integers(0,256,(2,64,64,3),np.uint8)
    seen=[]
    class Analyzer:
        def __init__(self,name,device='cpu'): self.name=name
        def probabilities(self,clip):
            seen.append((self.name,clip))
            return np.array([.9,.1])
        def saliency(self,clip): return np.zeros(clip.shape[1:3])
    model=SimpleNamespace(pixels=lambda x,s,c:x.copy(),manifest={},assert_source=lambda i,x:None)
    cfg=load_config(); cfg['ar_candidates']=['identity']; cfg['roi']['strengths']=[.5,1.]
    monkeypatch.setattr(base,'load_config',lambda p:cfg)
    monkeypatch.setattr(base,'ar_plan',lambda *a:([dict(id=identifier,path='fixture',label=0)],{}))
    monkeypatch.setattr(base,'read_video',lambda *a:rgb)
    monkeypatch.setattr(base,'ActionAnalyzer',Analyzer)
    monkeypatch.setattr(ar,'SwinAnalyzer',lambda d:Analyzer('swin3d_t'))
    monkeypatch.setattr(ar,'load_frozen_ar',lambda *a:model)
    result=ar.run(Namespace(root=tmp_path,out=tmp_path/'out',real_checkpoint=None,rf_runtime=None,
        config=None,task='ar',split='test',count=1,bootstrap=0,device='cpu',checkpoint=None,v30_runtime=None,annotations=None))
    assert result['complete_paired_grid'] and result['task']=='ar'
    rows=[json.loads(x) for x in (tmp_path/'out/rows.jsonl').read_text().splitlines()]
    assert len(rows)==70
    for row in rows:
        assert set(row['correct'])=={'r3d_18','r2plus1d_18','swin3d_t'}
        assert row['shared_receiver']['packet_sha256']==row['packet_sha256']
        parsed=unpack_packet((tmp_path/'out'/row['packet_path']).read_bytes())
        assert not parsed.grids
