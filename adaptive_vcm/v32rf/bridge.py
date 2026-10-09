"""Exact actual forward values with a separately supplied proxy Jacobian."""
import numpy as np
import torch


class _Substitute(torch.autograd.Function):
    @staticmethod
    def forward(ctx,proxy,actual): return actual.detach().clone()
    @staticmethod
    def backward(ctx,gradient): return gradient,None


def substitute(proxy,actual):
    if (proxy.shape!=actual.shape or proxy.dtype!=actual.dtype or proxy.device!=actual.device
            or not torch.isfinite(proxy).all() or not torch.isfinite(actual).all()):
        raise ValueError('finite matching reconstruction tensors required')
    return _Substitute.apply(proxy,actual)


def pixels(video):
    if (video.ndim!=5 or video.shape[0]!=1 or video.shape[1]!=3
            or not torch.isfinite(video).all() or video.min()<0 or video.max()>1):
        raise ValueError('one finite bounded B3THW RGB clip required')
    return video.detach()[0].permute(1,2,3,0).mul(255).round().clamp(0,255).byte().cpu().numpy()


def real_forward(video,codec,task):
    rgb=pixels(video)
    packet=codec.roundtrip(rgb,source_shape=rgb.shape,task=task)
    decoded=torch.from_numpy(np.array(packet.decoded,copy=True)).to(video).permute(3,0,1,2)[None]/255
    return decoded.detach(),8*packet.coded_bytes/np.prod(rgb.shape[:3]),packet.seconds
