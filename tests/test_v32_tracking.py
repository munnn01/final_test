"""Contracts for the reference tracker, label-free guard, and MOT20 adapter."""
import importlib
from argparse import Namespace
from fractions import Fraction

import numpy as np
import pytest
from PIL import Image


def module():
    assert importlib.util.find_spec('adaptive_vcm.v32.tracking') is not None, 'tracking runner is missing'
    return importlib.import_module('adaptive_vcm.v32.tracking')


def detections(boxes, scores=None, labels=None):
    return dict(boxes=np.asarray(boxes, float).reshape(-1, 4),
                scores=np.asarray(scores if scores is not None else [.9] * len(boxes)),
                labels=np.asarray(labels if labels is not None else [1] * len(boxes), int))


def fixture_sequence(root, count=3):
    seq = root / 'MOT20-01'
    (seq / 'img1').mkdir(parents=True)
    (seq / 'gt').mkdir()
    (seq / 'seqinfo.ini').write_text('[Sequence]\nname=MOT20-01\nimDir=img1\nframeRate=25\n'
                                   f'seqLength={count}\nimWidth=80\nimHeight=40\nimExt=.jpg\n')
    for number in range(1, count + 1):
        Image.fromarray(np.full((40, 80, 3), (number * 20) % 256, np.uint8)).save(seq / 'img1' / f'{number:06}.jpg')
    (seq / 'gt' / 'gt.txt').write_text('1,1,10,5,20,20,1,1,1\n1,2,50,5,20,20,1,7,1\n'
                                    '2,1,11,5,20,20,1,1,1\n3,1,12,5,20,20,1,1,1\n')
    return seq


def test_reference_tracker_keeps_ids_across_missing_frames_and_chunk_boundary():
    tracker = module().ReferenceTracker()
    assert tracker.step(detections([[0, 0, 10, 10], [30, 0, 40, 10]]))['track_ids'].tolist() == [1, 2]
    assert tracker.step(detections([[31, 0, 41, 10], [1, 0, 11, 10]]))['track_ids'].tolist() == [2, 1]
    tracker.step(detections([]))
    tracker.step(detections([]))
    assert tracker.step(detections([[2, 0, 12, 10]]))['track_ids'].tolist() == [1]
    for _ in range(3):
        tracker.step(detections([]))
    assert tracker.step(detections([[2, 0, 12, 10]]))['track_ids'].tolist() == [3]


def test_tracker_filters_coco_person_scores_and_uses_global_hungarian_assignment():
    tracker = module().ReferenceTracker()
    # Scores are IoU: A-X=.6, A-Y=.5, B-X=.9, B-Y=.1. Greedy would lose B.
    first = tracker.step(detections([[0, 0, 10, 10], [2, 0, 12, 10], [40, 0, 50, 10]],
                                    [.9, .9, .99], [1, 1, 2]))
    assert first['track_ids'].tolist() == [1, 2]
    second = tracker.step(detections([[2, 0, 12, 10], [-3, 0, 7, 10], [60, 0, 70, 10]], [.9, .9, .24]))
    assert second['track_ids'].tolist() == [2, 1]


def test_teacher_probes_clone_state_and_commit_only_the_selected_anchor():
    class Detector:
        def predict(self, clip):
            return detections([[int(clip[0, 0, 0, 0]), 0, 20, 20]])
    teacher = module().TrackingTeacher(Detector())
    clip = np.zeros((2, 32, 32, 3), np.uint8)
    first = teacher.predict(clip)
    assert first['scores'].tolist() == [.9, .9]
    assert [frame['track_ids'].tolist() for frame in first['frames']] == [[1], [1]]
    assert teacher.predict(clip)['frames'][0]['frame_number'] == 1
    teacher.commit(first)
    assert teacher.predict(clip)['frames'][0]['frame_number'] == 3


def prediction(frames):
    tracker = module().ReferenceTracker()
    tracked = [tracker.step(frame) for frame in frames]
    return dict(frames=tracked, scores=np.concatenate([f['scores'] for f in tracked]),
                associations=[link for f in tracked for link in f['associations']])


def test_guard_rejects_one_bad_frame_even_when_window_average_is_small():
    cfg = importlib.import_module('adaptive_vcm.v32.protocol').load_config()
    source = prediction([detections([[0, 0, 20, 20]])] * 16)
    candidate = prediction([detections([[0, 0, 20, 20]])] * 15 + [detections([[4, 0, 24, 20]])])
    distances, decisions = module().tracking_guard('ot', [source], [source], [candidate], cfg)
    assert distances[0] > .03
    assert decisions[0] is False


def test_guard_preserves_anchor_association_links_despite_identical_frame_boxes():
    cfg = importlib.import_module('adaptive_vcm.v32.protocol').load_config()
    anchor = prediction([detections([[0, 0, 20, 20]])] * 2)
    candidate = prediction([detections([[0, 0, 20, 20]])] * 2)
    candidate['frames'][1]['track_ids'][0] = 99
    distances, decisions = module().tracking_guard('ot', [anchor], [anchor], [candidate], cfg)
    assert distances == (0.,)
    assert decisions == (False,)
    details = module().tracking_guard_details('ot', [anchor], [anchor], [candidate], cfg)
    assert len(details['frames']) == 2 and not details['preserves_associations']


def test_source_support_is_per_frame_and_union_control_has_no_gt_input():
    rgb = np.zeros((2, 128, 128, 3), np.uint8)
    source = prediction([detections([[16, 16, 24, 24]]), detections([[96, 96, 104, 104]])])
    union, support = module().source_support(rgb, source)
    assert support['protection'].shape == (2, 128, 128)
    assert support['protection'][0, 20, 20] == 1 and support['protection'][1, 20, 20] == 0
    assert support['protection'][1, 100, 100] == 1
    assert union[20, 20] == union[100, 100] == 1


def test_mot20_plan_reads_contiguous_prefix_with_exact_inverse_letterbox(tmp_path):
    fixture_sequence(tmp_path)
    plan = module().plan_sequences(tmp_path, ['MOT20-01'], 2)
    assert plan[0]['frames'] == 2 and plan[0]['fps'] == Fraction(25)
    rgb, geometry = module().read_sequence(plan[0], 64)
    assert rgb.shape == (2, 64, 64, 3)
    boxes = module().original_boxes(np.array([[8, 20, 24, 36]], float), geometry, 80, 40)
    np.testing.assert_allclose(boxes, [[10, 5, 30, 25]])


def test_plan_rejects_missing_middle_frame_instead_of_padding(tmp_path):
    seq = fixture_sequence(tmp_path)
    (seq / 'img1' / '000002.jpg').unlink()
    with pytest.raises(FileNotFoundError, match='000002'):
        module().plan_sequences(tmp_path, ['MOT20-01'], 3)


def test_staging_preserves_distractors_and_changes_only_partial_frame_scope(tmp_path):
    root = tmp_path / 'source'
    fixture_sequence(root)
    plan = module().plan_sequences(root, ['MOT20-01'], 2)
    manifest = module().stage_ground_truth(plan, tmp_path / 'staged')
    staged = tmp_path / 'staged' / 'MOT20-01'
    assert (staged / 'gt' / 'gt.txt').read_text().splitlines() == [
        '1,1,10,5,20,20,1,1,1', '1,2,50,5,20,20,1,7,1', '2,1,11,5,20,20,1,1,1']
    assert manifest['MOT20-01']['selected_rows'] == 3
    import configparser
    config = configparser.ConfigParser()
    config.read(staged / 'seqinfo.ini')
    assert config['Sequence']['seqLength'] == '2'


def test_cli_missing_trackeval_fails_before_models_or_candidate_probes(tmp_path):
    with pytest.raises(RuntimeError, match='TrackEval'):
        module().run(Namespace(out=tmp_path / 'out', trackeval_root=tmp_path / 'absent'))


def test_cli_never_overwrites_prior_output(tmp_path):
    (tmp_path / 'evidence').write_text('preserve')
    with pytest.raises(ValueError, match='empty'):
        module().run(Namespace(out=tmp_path))
    assert (tmp_path / 'evidence').read_text() == 'preserve'


def test_mot_writer_returns_original_coordinates_and_preserves_track_ids(tmp_path):
    frame = module().ReferenceTracker().step(detections([[8, 20, 24, 36]]))
    path = tmp_path / 'predictions.txt'
    module().write_mot(path, [frame], (.8, .8, 0, 16), 80, 40)
    values = [float(v) for v in path.read_text().strip().split(',')]
    assert values[:6] == [1, 1, 10, 5, 20, 20]
    assert values[6:] == [.9, -1, -1, -1]


def test_official_trackeval_mot20_preprocessing_ignores_distractor_matches(tmp_path):
    """Real official code; no hand-written substitute for HOTA/IDF1/MOTA."""
    from pathlib import Path
    checkout = Path('D:/STUDY/LAB/bao_1/output/v32_tests/trackeval_official')
    if not checkout.exists():
        pytest.skip('optional pinned official TrackEval checkout unavailable')
    tracking = module()
    package = tracking.load_trackeval(checkout)
    fixture_sequence(tmp_path / 'source')
    plan = tracking.plan_sequences(tmp_path / 'source', ['MOT20-01'], 2)
    tracking.stage_ground_truth(plan, tmp_path / 'ground_truth')
    output = tmp_path / 'trackers' / 'anchor' / 'data'
    output.mkdir(parents=True)
    (output / 'MOT20-01.txt').write_text('1,1,10,5,20,20,.9,-1,-1,-1\n'
        '1,2,50,5,20,20,.9,-1,-1,-1\n2,1,11,5,20,20,.9,-1,-1,-1\n')
    result = tracking.evaluate_mot(package, plan, tmp_path, ['anchor'])
    assert result['anchor']['combined']['HOTA'] == pytest.approx(100)
    assert result['anchor']['combined']['IDF1'] == pytest.approx(100)
    assert result['anchor']['combined']['MOTA'] == pytest.approx(100)
    assert result['anchor']['combined']['IDs'] == 0
    assert result['anchor']['combined']['FP'] == 0


def test_tracking_callback_and_real_codec_are_multi_frame_and_receiver_canonical():
    from adaptive_vcm.v32.codec import V32Codec
    from adaptive_vcm.v32.selection import choose_stream
    from adaptive_vcm.v32.protocol import load_config
    from adaptive_vcm.v32.transport import unpack_packet
    class Detector:
        def predict(self, clip):
            return detections([[24, 24, 40, 40]])
    rgb = np.random.default_rng(32).integers(0, 256, (3, 64, 64, 3), np.uint8)
    teacher = module().TrackingTeacher(Detector())
    source = teacher.predict(rgb)
    union, support = module().source_support(rgb, source)
    cfg = load_config()
    cfg['od_candidates'] = ['identity']
    guard = module().TrackingGuard()
    bundle = choose_stream(rgb, union, support, 'ot', V32Codec('h264', 40, preset='ultrafast'),
                           cfg, [teacher], [source], guard=guard)
    assert set(bundle.arms) == {'controls', 'simplification', 'retarget', 'v32'}
    assert unpack_packet(bundle.anchor.data).task == 'ot'
    assert all(packet.decoded.shape == (3, 64, 64, 3) for packet, _ in bundle.arms.values())
    assert any(row['family'] == 'retarget' for row in bundle.audit)


def test_exploratory_curves_keep_all_five_qps_without_one_sequence_confidence():
    rows = [dict(codec=codec, qp=qp, arm=arm, bpp=(.2 + .1 * index) * (1 if arm == 'anchor' else .9),
                 metrics=dict(HOTA=70 + index, IDF1=75 + index, MOTA=65 + index, IDs=2))
            for codec in ['h264', 'h265'] for index, qp in enumerate([30, 35, 40, 45, 50])
            for arm in ['anchor', 'controls', 'simplification', 'retarget', 'v32']]
    result = module().exploratory_summary(rows)
    assert result['target_confirmed'] is False and result['bootstrap_confidence'] is None
    assert result['results']['h264']['v32']['HOTA']['bd_rate_pct'] == pytest.approx(-10)
    assert len(result['results']['h265']['retarget']['IDF1']['same_qp_gap_pp']) == 5


def test_full_cli_grid_uses_actual_packets_and_never_restarts_ids_at_frame_17(tmp_path, monkeypatch):
    from pathlib import Path
    checkout = Path('D:/STUDY/LAB/bao_1/output/v32_tests/trackeval_official')
    if not checkout.exists():
        pytest.skip('optional pinned official TrackEval checkout unavailable')
    tracking = module()
    class FullFrameDetector:
        def __init__(self, name, device):
            self.name = name
        def predict(self, clip):
            # Full foreground forces identity-only fixed ROI; codecs/metrics remain real.
            return detections([[0, 0, clip.shape[2], clip.shape[1]]])
    monkeypatch.setattr(tracking, 'DetectionAnalyzer', FullFrameDetector)
    fixture_sequence(tmp_path / 'source', 17)
    result = tracking.run(Namespace(out=tmp_path / 'out', trackeval_root=checkout,
        root=tmp_path / 'source', sequences=['MOT20-01'], max_frames=17, size=64,
        config=None, device='cpu', motion_support=False))
    assert result['complete_paired_grid'] and len(result['points']) == 50
    manifest = __import__('json').loads((tmp_path / 'out' / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['scope'] == 'fixed_roi_only_tracking'
    assert manifest['temporal_subsample'] == 1 and manifest['padding_frames'] == 0
    rows = [__import__('json').loads(line) for line in (tmp_path / 'out' / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == 100
    from adaptive_vcm.v32.transport import unpack_packet
    for row in rows:
        wire = (tmp_path / 'out' / row['packet_path']).read_bytes()
        assert len(wire) == row['coded_bytes'] == row['elementary_bytes'] + row['metadata_bytes']
        assert unpack_packet(wire).source_shape[0] == row['frame_end'] - row['frame_start'] + 1
    for point in result['points']:
        predictions = tmp_path / 'out' / point['prediction_paths']['MOT20-01']
        lines = predictions.read_text().splitlines()
        assert len(lines) == 17 and lines[-1].split(',')[:2] == ['17', '1']


def test_model_manifest_hashes_actual_state_and_reports_frozen_status():
    import torch
    from types import SimpleNamespace
    model = torch.nn.Linear(2, 1)
    model.requires_grad_(False)
    analyzer = SimpleNamespace(name='mobilenet', model=model)
    before = module().model_provenance(analyzer)
    assert before['frozen'] and len(before['state_sha256']) == 64
    with torch.no_grad():
        model.weight.add_(1)
    after = module().model_provenance(analyzer)
    assert before['state_sha256'] != after['state_sha256']
