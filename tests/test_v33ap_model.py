import pytest
import torch

torch.set_num_threads(2)


def inputs(t):
    x = torch.rand(1, 3, t, 16, 18)
    p = torch.zeros(1, 1, t, 16, 18); p[..., :4, :4] = 1
    m = torch.zeros_like(p)
    c = torch.zeros(1, t, dtype=torch.bool); c[:, 0] = True
    return x, p, m, c


def test_one_state_serves_image_and_clip():
    from adaptive_vcm.v33ap.model import JointPreprocessor
    model = JointPreprocessor()
    ids = [id(p) for p in model.parameters()]
    for t in (1, 3):
        x, p, m, cuts = inputs(t)
        assert torch.equal(model(x, 35, 0, p, m, cuts), x)
        assert ids == [id(p) for p in model.parameters()]


def test_identity_and_exact_protection():
    from adaptive_vcm.v33ap.model import JointPreprocessor
    model = JointPreprocessor(); x, p, m, cuts = inputs(3)
    with torch.no_grad(): model.head.bias.fill_(10)
    y = model(x, 35, 1, p, m, cuts)
    protected = p.expand_as(x) == 1
    assert torch.equal(y[protected], x[protected])
    assert (y-x).abs().max() <= .100001 and y.min() >= 0 and y.max() <= 1
    # Extreme bias checks the bound; use an unsaturated value to check gradients.
    with torch.no_grad(): model.head.bias.fill_(.5)
    model(x, 35, 1, p, m, cuts).mean().backward()
    assert model.head.weight.grad.abs().sum() > 0


def test_temporal_cut_isolation():
    from adaptive_vcm.v33ap.model import JointPreprocessor
    model = JointPreprocessor(); x, p, m, cuts = inputs(4); cuts[:, 2] = True
    with torch.no_grad(): model.head.weight.normal_(0, .1)
    whole = model(x, 35, 0, p, m, cuts)
    suffix = model(x[:, :, 2:], 35, 0, p[:, :, 2:], m[:, :, 2:], cuts[:, 2:])
    assert torch.allclose(whole[:, :, 2:], suffix, atol=1e-7)


@pytest.mark.parametrize('qp', [float('nan'), -1, 52])
def test_invalid_conditions(qp):
    from adaptive_vcm.v33ap.model import JointPreprocessor
    x, p, m, cuts = inputs(1)
    with pytest.raises(ValueError): JointPreprocessor()(x, qp, 0, p, m, cuts)
