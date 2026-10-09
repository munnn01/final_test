"""Frozen intermediate features; source targets are detached by the loss."""
import torch
import torch.nn.functional as F


def feature_vectors(task,teachers,video):
    if task not in ('ar','od') or not teachers or task=='od' and len(teachers)!=1:
        raise ValueError('invalid feature teachers/task')
    vectors=[]
    for teacher in teachers:
        states=[(m,m.training) for m in teacher.model.modules()]
        handle=None
        try:
            teacher.model.eval()
            if task=='ar':
                captured=[]
                handle=teacher.model.layer2.register_forward_hook(lambda m,a,out:captured.append(out))
                teacher.logits(video)
                if len(captured)!=1: raise ValueError('AR layer2 feature unavailable')
                feature=F.adaptive_avg_pool3d(captured[0],(2,4,4)).flatten(1)
            else:
                if video.shape[0]!=1 or video.shape[2]!=1: raise ValueError('OD needs one frame')
                transformed,_=teacher.model.transform([video[0,:,0]],None)
                maps=teacher.model.backbone(transformed.tensors)
                if isinstance(maps,torch.Tensor): maps={'0':maps}
                feature=torch.cat([F.adaptive_avg_pool2d(v,(4,4)).flatten(1) for _,v in sorted(maps.items())],1)
            vectors.append(feature)
        finally:
            if handle is not None: handle.remove()
            for module,state in states: module.training=state
    return vectors


def feature_distance(features,targets):
    if not features or len(features)!=len(targets): raise ValueError('matching feature targets required')
    losses=[]
    for current,target in zip(features,targets):
        if current.shape!=target.shape: raise ValueError('feature geometry differs')
        losses.append((1-F.cosine_similarity(current,target.detach(),dim=1,eps=1e-8)).clamp(0,2).mean())
    return torch.stack(losses).mean()
