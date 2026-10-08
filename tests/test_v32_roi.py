"""ROI decisions must preserve cores and geometry, not just array shapes."""
import importlib

import numpy as np
import pytest


def roi():
    return importlib.import_module('adaptive_vcm.v32.roi')


def support(frames=1, size=64):
    p = np.zeros((frames, size, size), np.float32)
    p[:, 24:40, 24:40] = 1
    cuts = np.zeros(frames, bool)
    cuts[0] = True
    return dict(protection=p, motion=np.zeros_like(p), cuts=cuts)


def pixels(frames=1, size=64):
    rng = np.random.default_rng(32)
    return rng.integers(0, 256, (frames, size, size, 3), dtype=np.uint8)


def test_hand_derived_grid_reduces_background_but_keeps_roi_samples():
    rgb = pixels()
    grids = roi().build_grids(rgb.shape, support(), factor=4, margin=0)
    assert grids[0].x_source == (0, 24, 40, 64)
    assert grids[0].x_coded == (0, 6, 22, 28)
    coded = roi().retarget(rgb, grids)
    assert coded.shape == (1, 28, 28, 3)
    np.testing.assert_array_equal(coded[:, 6:22, 6:22], rgb[:, 24:40, 24:40])
    restored = roi().inverse_retarget(coded, grids, rgb.shape)
    np.testing.assert_array_equal(restored[:, 24:40, 24:40], rgb[:, 24:40, 24:40])


def test_scene_segments_have_stable_mapping_and_shared_coded_resolution():
    rgb = pixels(4)
    s = support(4)
    s['protection'][1] = np.roll(s['protection'][1], 8, axis=1)
    s['cuts'][2] = True
    grids = roi().build_grids(rgb.shape, s, factor=2, margin=0)
    assert [(g.start, g.stop) for g in grids] == [(0, 2), (2, 4)]
    assert len({(g.x_coded[-1], g.y_coded[-1]) for g in grids}) == 1
    decoded = roi().inverse_retarget(roi().retarget(rgb, grids), grids, rgb.shape)
    assert decoded.shape == rgb.shape


def test_motion_alone_protects_roi_and_empty_or_full_support_skips_retarget():
    rgb = pixels()
    s = support()
    s['motion'][:] = s['protection']
    s['protection'][:] = 0
    assert roi().build_grids(rgb.shape, s, factor=4, margin=0)
    s['motion'][:] = 0
    assert roi().build_grids(rgb.shape, s, factor=4, margin=0) is None
    s['protection'][:] = 1
    assert roi().build_grids(rgb.shape, s, factor=4, margin=0) is None


def test_simplification_preserves_hard_core_and_does_not_cross_scene_cuts():
    rgb = pixels(2)
    s = support(2)
    s['cuts'][:] = True
    result = roi().simplify(rgb, s, strength=1, margin=0)
    np.testing.assert_array_equal(result[:, 24:40, 24:40], rgb[:, 24:40, 24:40])
    assert np.all(result[:, :8, :8] == 128)
    changed = rgb.copy()
    changed[1] = 0
    result2 = roi().simplify(changed, s, strength=1, margin=0)
    np.testing.assert_array_equal(result2[0], result[0])


def test_strength_zero_and_empty_roi_return_exact_identity():
    rgb = pixels()
    s = support()
    np.testing.assert_array_equal(roi().simplify(rgb, s, strength=0), rgb)
    s['protection'][:] = 0
    np.testing.assert_array_equal(roi().simplify(rgb, s, strength=1), rgb)


@pytest.mark.parametrize('field,value', [('protection', np.nan), ('motion', 2)])
def test_invalid_support_fails_closed(field, value):
    s = support()
    s[field][0, 0, 0] = value
    with pytest.raises(ValueError):
        roi().build_grids(pixels().shape, s, factor=2)


def test_grid_rejects_duplicate_edges_and_wrong_segment_coverage():
    with pytest.raises(ValueError):
        roi().Grid(0, 1, (0, 24, 24, 64), (0, 6, 22, 28), (0, 64), (0, 28))
    grids = roi().build_grids(pixels().shape, support(), factor=4, margin=0)
    with pytest.raises(ValueError):
        roi().retarget(pixels(2), grids)


@pytest.mark.parametrize('factor', [True, 0, 1.5])
def test_invalid_downsampling_factor_is_rejected(factor):
    with pytest.raises(ValueError):
        roi().build_grids(pixels().shape, support(), factor=factor)
