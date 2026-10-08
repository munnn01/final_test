import importlib
from fractions import Fraction

import numpy as np

from adaptive_vcm.selection import Observation
from adaptive_vcm.v32.codec import V32Codec
from tests.test_v32_roi import pixels, support


def selection():
    return importlib.import_module('adaptive_vcm.v32.selection')


def config():
    return importlib.import_module('adaptive_vcm.v32.protocol').load_config()


def test_total_metadata_bytes_can_reverse_the_elementary_stream_winner():
    rows = [dict(family='controls', observation=Observation('identity', 1000, (0.,), (True,))),
            dict(family='retarget', observation=Observation('small-stream-large-grid', 1100, (0.,), (True,))),
            dict(family='simplification', observation=Observation('roi-small', 900, (0.,), (True,)))]
    winners = selection().select_arms(rows, .03, .01, has_baseline=False)
    assert winners['retarget'] == 0
    assert winners['v32'] == 2


def test_teacher_rejection_and_arm_isolation_keep_identity_fallback():
    rows = [dict(family='controls', observation=Observation('identity', 1000, (0.,), (True,))),
            dict(family='learned', observation=Observation('learned', 800, (.02,), (True,))),
            dict(family='hybrid', observation=Observation('hybrid', 500, (.04,), (True,)))]
    winners = selection().select_arms(rows, .03, .01, has_baseline=True)
    assert winners['v30b'] == 1
    assert winners['hybrid'] == 1
    assert winners['v32'] == 1
    assert winners['retarget'] == 0


class EmptyODTeacher:
    """A known-empty synthetic source; no pretrained weight download in contract test."""
    def predict(self, clip):
        return dict(boxes=np.empty((0, 4)), scores=np.empty(0), labels=np.empty(0, int))


def test_unknown_od_foreground_probes_identity_only_with_real_codec():
    rgb = pixels()
    teacher = EmptyODTeacher()
    codec = V32Codec('h264', 40, preset='ultrafast', fps=Fraction(25))
    result = selection().choose_stream(rgb, support()['protection'][0], support(), 'od',
        codec, config(), [teacher], [teacher.predict(rgb)])
    assert len(result.audit) == 1
    assert all(name == 'identity' for _, name in result.arms.values())
    assert result.audit[0]['total_bytes'] == result.anchor.coded_bytes


class StableARTeacher:
    """Analytic frozen teacher fixture to test the codec/selection boundary."""
    def probabilities(self, clip):
        return np.array([.8, .2])


def test_real_fixed_branches_have_source_geometry_and_separate_overhead_audit():
    rgb = pixels(2)
    s = support(2)
    cfg = config()
    cfg['ar_candidates'] = ['identity']
    teacher = StableARTeacher()
    result = selection().choose_stream(rgb, s['protection'].max(0), s, 'ar',
        V32Codec('h264', 40, preset='ultrafast', fps=Fraction(25)), cfg,
        [teacher, teacher], [teacher.probabilities(rgb)] * 2)
    assert set(result.arms) == {'controls', 'simplification', 'retarget', 'v32'}
    assert any(row['family'] == 'retarget' and row['coded_shape'][1] < 64 for row in result.audit)
    for packet, _ in result.arms.values():
        assert packet.decoded.shape == rgb.shape
    for row in result.audit:
        assert row['total_bytes'] == row['elementary_bytes'] + row['metadata_bytes']


def test_hybrid_candidates_use_frozen_rgb_proposals_and_keep_three_baseline_names():
    rgb = pixels()
    class Baseline:
        def proposals(self, rgb, support, codec):
            from adaptive_vcm.preprocessing import Candidate
            return [Candidate(f'learned_motion_s{scale:03}', rgb.copy()) for scale in (50, 100, 150)]
    candidates = selection().build_candidates(rgb, support()['protection'][0], support(),
        'od', V32Codec('h264', 40, preset='ultrafast'), config(), Baseline())
    assert len([c for c in candidates if c.family == 'learned']) == 3
    assert len([c for c in candidates if c.family == 'hybrid']) == 6
    assert all(c.grids for c in candidates if c.family == 'hybrid')


def test_same_encoded_pixels_with_different_mapping_get_separate_receiver_and_teacher(monkeypatch):
    from adaptive_vcm.v32.roi import Grid
    rgb, coded = pixels(), pixels(size=28)
    first = (Grid(0, 1, (0, 24, 40, 64), (0, 6, 22, 28), (0, 24, 40, 64), (0, 6, 22, 28)),)
    second = (Grid(0, 1, (0, 16, 32, 64), (0, 6, 22, 28), (0, 24, 40, 64), (0, 6, 22, 28)),)
    candidates = [selection().Candidate('identity', rgb, 'controls'),
                  selection().Candidate('first', coded, 'retarget', first),
                  selection().Candidate('second', coded, 'retarget', second)]
    monkeypatch.setattr(selection(), 'build_candidates', lambda *args: candidates)
    class CountingTeacher:
        def __init__(self):
            self.calls = 0
        def predict(self, clip):
            self.calls += 1
            return dict(boxes=np.array([[24., 24., 40., 40.]]), scores=np.array([.9]), labels=np.array([1]))
    teacher = CountingTeacher()
    source = teacher.predict(rgb)
    teacher.calls = 0
    result = selection().choose_stream(rgb, support()['protection'][0], support(), 'od',
        V32Codec('h264', 40, preset='ultrafast'), config(), [teacher], [source])
    assert [r['distinct_encode'] for r in result.audit] == [True, True, False]
    assert all(r['distinct_packet_decode'] for r in result.audit)
    assert teacher.calls == 3
    assert result.audit[1]['reconstructed_pixel_sha256'] != result.audit[2]['reconstructed_pixel_sha256']
