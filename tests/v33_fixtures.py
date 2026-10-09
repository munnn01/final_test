import numpy as np
import torch
from torch import nn


class ActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.features = nn.Sequential(nn.Identity(), nn.Identity())
    def forward(self, x):
        f = self.features(x.permute(0,2,3,4,1))
        v = f.mean((1,2,3)) * self.scale
        return torch.stack((v[:,0],v[:,1],v[:,2]),1)


class Action:
    name = 'fake-swin'
    def __init__(self): self.model = ActionModel(); self.calls = 0
    def logits(self,x): self.calls += 1; return self.model(x)
    def saliency(self,rgb): return np.zeros(rgb.shape[1:3],np.float32)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__(); self.bn = nn.BatchNorm2d(3)
    def forward(self,x): return {'0': self.bn(x)}


class DetectorModel(nn.Module):
    def __init__(self, empty=False):
        super().__init__(); self.backbone = Backbone(); self.empty = empty
    def forward(self,images,targets=None):
        f = self.backbone(torch.stack(images))['0']
        if self.training: return {'loss': (f.mean()-0.15).square()}
        results = []
        for image in images:
            n = 0 if self.empty else 1
            results.append(dict(boxes=image.new_tensor([[4,4,8,8]])[:n],
                scores=image.new_tensor([.9])[:n],labels=torch.ones(n,device=image.device,dtype=torch.long)))
        return results


class Detector:
    name = 'fake-detector'
    def __init__(self,empty=False): self.model = DetectorModel(empty)


def record(task='ar', t=3):
    from adaptive_vcm.v33ap.sources import SourceRecord
    rgb = np.random.default_rng(3).integers(20,220,(t if task=='ar' else 1,16,16,3),dtype=np.uint8)
    support = dict(protection=np.zeros(rgb.shape[:3],np.float32),motion=np.zeros(rgb.shape[:3],np.float32),
                   cuts=np.array([True]+[False]*(len(rgb)-1)))
    target = dict(boxes=torch.tensor([[4.,4.,8.,8.]]), labels=torch.tensor([1])) if task=='od' else {}
    return SourceRecord('ar/a/f.mp4' if task=='ar' else 'od/coco2017/5',task,dict(label=1),rgb,
                        None,{},support, target)
