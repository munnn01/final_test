import numpy as np
import pytest
import torch

from adaptive_vcm.v32.roi import Grid, retarget, inverse_retarget
from adaptive_vcm.v32vc.model import ResidualPreprocessor
from adaptive_vcm.v32vc.virtual import VirtualCodec, dct_basis, color_planes, color_rgb
from adaptive_vcm.v32vc.geometry import transform

torch.set_num_threads(2)


def inputs(t=3):
    x = torch.rand(1, 3, t, 16, 18)
    p = torch.zeros(1, 1, t, 16, 18); p[..., :4, :4] = 1
    m = torch.zeros_like(p)
    cuts = torch.zeros(1, t, dtype=torch.bool); cuts[:, 0] = True
    return x, p, m, cuts


def test_identity_initialization_and_bounded_protected_edits():
    x,p,m,c = inputs()
    model = ResidualPreprocessor('ar')
    assert torch.equal(model(x, 35, 0, p, m, c), x)
    with torch.no_grad(): model.head.bias.fill_(10)
    y = model(x, 35, 0, p, m, c)
    assert torch.equal(y.masked_select(p.expand_as(y)==1), x.masked_select(p.expand_as(x)==1))
    assert (y-x).abs().max() <= .100001 and y.min() >= 0 and y.max() <= 1
    y.mean().backward()
    assert model.head.weight.grad is not None


def test_scene_cut_resets_temporal_features():
    x,p,m,c = inputs(4); c[:,2] = True
    model = ResidualPreprocessor('ar')
    with torch.no_grad(): model.head.weight.normal_(0,.1)
    a = model(x,35,0,p,m,c)
    b = model(x[:,:,2:],35,0,p[:,:,2:],m[:,:,2:],c[:,2:])
    assert torch.allclose(a[:,:,2:],b,atol=1e-7)


@pytest.mark.parametrize('t',[1,3])
def test_virtual_has_finite_input_gradients_and_no_parameters(t):
    x,p,m,c = inputs(t); x.requires_grad_()
    virtual = VirtualCodec().train()
    y,rate = virtual(x,35,0,c)
    assert list(virtual.parameters()) == []
    assert y.shape == x.shape and rate.ndim == 0 and rate > 0
    (y.square().mean()+.01*rate).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_constant_virtual_entropy_and_cut_reset():
    x,p,m,c = inputs(); x.fill_(.5); x.requires_grad_()
    virtual = VirtualCodec().train()
    y,rate = virtual(x,50,1,c)
    (y.mean()+rate).backward()
    assert torch.isfinite(rate) and torch.isfinite(x.grad).all()
    virtual.eval(); c[:,2] = True
    all_y,_ = virtual(x.detach(),40,0,c)
    cut_y,_ = virtual(x.detach()[:,:,2:],40,0,c[:,2:])
    assert torch.equal(all_y[:,:,2:],cut_y)


def test_orthonormal_dct_and_constant_color_roundtrip():
    d = dct_basis(8)
    assert torch.allclose(d@d.T,torch.eye(8),atol=1e-6)
    x = torch.tensor([.2,.5,.8]).view(1,3,1,1).expand(1,3,16,18)
    assert torch.allclose(color_rgb(color_planes(x)),x,atol=1e-6)


def test_actual_block_dct_inverse_preserves_nonaligned_pixels():
    virtual = VirtualCodec()
    x = torch.rand(2,1,14,18)
    coefficients,shape = virtual._blocks(x)
    assert torch.allclose(virtual._inverse(coefficients,shape),x,atol=4e-6)


def test_fixed_grid_resampling_matches_receiver_and_passes_gradients():
    g = (Grid(0,2,(0,5,18),(0,5,12),(0,4,16),(0,4,10)),)
    pixels = np.random.default_rng(4).integers(0,256,(2,16,18,3),dtype=np.uint8)
    x = torch.from_numpy(pixels.copy()).permute(3,0,1,2)[None].float()/255
    x.requires_grad_()
    coded = transform(x,g)
    expected = torch.from_numpy(retarget(pixels,g)).permute(3,0,1,2)[None].float()/255
    assert (coded-expected).abs().max() <= 1/255+.00001
    decoded = transform(coded,g,inverse=True)
    assert decoded.shape == x.shape
    decoded.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    rounded = coded.detach().mul(255).round().byte()[0].permute(1,2,3,0).numpy()
    rx = torch.from_numpy(inverse_retarget(rounded,g,pixels.shape)).permute(3,0,1,2)[None].float()/255
    inverse_float = transform(torch.from_numpy(rounded.copy()).permute(3,0,1,2)[None].float()/255,g,inverse=True)
    assert (rx-inverse_float).abs().max() <= 1/255+.00001


@pytest.mark.parametrize('level',[0.,1.])
@pytest.mark.parametrize('shape',[(110,154,56,78),(320,320,162,166)])
def test_retarget_extreme_rgb_stays_bounded_for_virtual_codec(level,shape):
    h,w,dh,dw = shape
    grid = (Grid(0,1,(0,w),(0,dw),(0,h),(0,dh)),)
    x = torch.full((1,3,1,h,w),level)
    coded = transform(x,grid)
    assert coded.min() >= 0 and coded.max() <= 1
    assert torch.allclose(coded,x.new_full(coded.shape,level),atol=2e-7)
    decoded,rate = VirtualCodec().eval()(coded,35,0)
    restored = transform(decoded,grid,inverse=True)
    assert restored.min() >= 0 and restored.max() <= 1
    assert torch.isfinite(rate)


@pytest.mark.parametrize('bad',[float('nan'),-1,52])
def test_invalid_qp_fails(bad):
    x,p,m,c = inputs()
    with pytest.raises(ValueError): VirtualCodec()(x,bad,0,c)


def test_od_rejects_temporal_input():
    x,p,m,c = inputs()
    with pytest.raises(ValueError): ResidualPreprocessor('od')(x,35,0,p,m,c)
