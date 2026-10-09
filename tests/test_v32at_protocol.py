import pytest
from tests.test_v32_roi import pixels, support
from adaptive_vcm.v32.protocol import load_config


def test_new_campaign_accepts_only_ar_and_tracking_and_locks_video_codec():
    from adaptive_vcm.v32at.protocol import validate_task, settings, codec_factory
    assert [validate_task(t) for t in ('ar', 'ot')] == ['ar', 'ot']
    with pytest.raises(ValueError): validate_task('od')
    assert settings()['tasks'] == ['ar', 'ot']
    for task in ('ar', 'ot'):
        codec = codec_factory(task)('h264', 40)
        assert codec.standard.gop_settings == dict(gop=16, bframes=0, scenecut=False, closed_gop=True)


def test_ar_switch_never_constructs_retarget_grids(monkeypatch):
    from adaptive_vcm.v32.selection import build_candidates
    def forbidden(*args, **kwargs): raise AssertionError('AR must not construct a grid')
    monkeypatch.setattr('adaptive_vcm.v32.selection.build_grids', forbidden)
    codec = type('Codec', (), {'qp': 40})()
    rgb, s = pixels(), support()
    candidates = build_candidates(rgb, s['protection'][0], s, 'ar', codec, load_config(), enable_retarget=False)
    assert candidates and all(not c.grids for c in candidates)


def test_tracking_default_still_constructs_retarget():
    from adaptive_vcm.v32.selection import build_candidates
    rgb, s = pixels(), support()
    codec = type('Codec', (), {'qp': 40})()
    candidates = build_candidates(rgb, s['protection'][0], s, 'ot', codec, load_config())
    assert any(c.grids for c in candidates)
