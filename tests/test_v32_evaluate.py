"""Paired evidence completeness and runnable packet-only transport smoke."""
import importlib
from argparse import Namespace
import json

import pytest


def module():
    return importlib.import_module('adaptive_vcm.v32.evaluate')


def records():
    return [dict(id=identifier, codec=codec, qp=qp, arm=arm,
                 bpp=1. + (50-qp)*.01, correct={'r2plus1d_18': int(identifier == 'a')},
                 coded_bytes=1000 if arm == 'anchor' else 900, candidate='identity')
            for codec in ('h264', 'h265') for qp in (30, 35, 40, 45, 50)
            for identifier in ('a', 'b') for arm in ('anchor', 'controls', 'v32')]


@pytest.mark.parametrize('alter', ['missing', 'duplicate', 'unknown', 'one_codec'])
def test_summary_rejects_incomplete_or_unpaired_grid(alter):
    rows = records()
    if alter == 'missing':
        rows.pop()
    elif alter == 'duplicate':
        rows.append(rows[0])
    elif alter == 'unknown':
        rows[-1] = {**rows[-1], 'id': 'unplanned'}
    else:
        rows = [r for r in rows if r['codec'] == 'h264']
    with pytest.raises(ValueError, match='grid'):
        module().summarize(rows, ['a', 'b'], 'ar',
                           importlib.import_module('adaptive_vcm.v32.protocol').load_config(), 0)


def test_summary_keeps_each_arm_and_codec_separate_without_claiming_target():
    result = module().summarize(records(), ['a', 'b'], 'ar',
                               importlib.import_module('adaptive_vcm.v32.protocol').load_config(), 0)
    assert set(result['results']) == {'h264', 'h265'}
    assert set(result['results']['h264']) == {'controls', 'v32'}
    assert result['target_confirmed'] is False
    assert result['results']['h264']['v32']['r2plus1d_18']['bd_rate_pct'] is None


def test_existing_output_is_rejected_before_loading_models_or_data(tmp_path):
    (tmp_path / 'do_not_replace.txt').write_text('evidence')
    with pytest.raises(ValueError, match='empty'):
        module().run(Namespace(out=tmp_path))
    assert (tmp_path / 'do_not_replace.txt').read_text() == 'evidence'


def test_synthetic_smoke_is_explicitly_transport_only_and_received_bytes_redecode(tmp_path):
    smoke = importlib.import_module('adaptive_vcm.v32.smoke')
    result = smoke.run(tmp_path)
    assert result['scope'] == 'synthetic_transport_only'
    assert result['quality_measured'] is False
    assert {r['codec'] for r in result['records']} == {'h264', 'h265'}
    assert all(r['source_shape'] == [1, 64, 64, 3] for r in result['records'])
    assert all(r['total_bytes'] == r['elementary_bytes'] + r['metadata_bytes'] for r in result['records'])
    assert all(r['received_bytes_redecode_matches'] for r in result['records'])
    assert any(r['coded_shape'][1] < 64 for r in result['records'])
    assert json.loads((tmp_path / 'summary.json').read_text()) == result


def test_source_partition_audit_rejects_duplicate_pixels_under_different_ids():
    import numpy as np
    audit = module().SourceAudit()
    pixels = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    audit.add('a', pixels)
    with pytest.raises(ValueError, match='pixel'):
        audit.add('b', pixels)


def test_dataset_cli_contract_runs_all_qps_and_codecs_with_real_paid_packets(tmp_path, monkeypatch):
    """Analytic detection fixture; real images, codec, receiver, plans and COCO metrics."""
    import numpy as np
    from PIL import Image
    from adaptive_vcm.data import partition
    from adaptive_vcm.v32.transport import unpack_packet
    evaluate = module()
    identifier = next(i for i in range(1, 100) if partition(f'coco2017/{i}') == 'dev')
    root = tmp_path / 'images'
    root.mkdir()
    pixels = np.random.default_rng(32).integers(0, 256, (320, 320, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(root / 'sample.png')
    annotation = tmp_path / 'instances.json'
    annotation.write_text(json.dumps(dict(images=[dict(id=identifier, file_name='sample.png', width=320, height=320)],
        categories=[dict(id=1, name='person')], annotations=[dict(id=1, image_id=identifier, category_id=1,
        bbox=[128, 128, 64, 64], area=4096, iscrowd=0)])))
    class AnalyticDetection:
        def __init__(self, name, device):
            self.name = name
        def predict(self, clip):
            return dict(boxes=np.array([[128., 128., 192., 192.]]), scores=np.array([.9]), labels=np.array([1]))
    monkeypatch.setattr(evaluate, 'DetectionAnalyzer', AnalyticDetection)
    cfg_path = importlib.import_module('adaptive_vcm.v32.protocol').ROOT / 'configs/v32_roi.json'
    out = tmp_path / 'results'
    result = evaluate.run(Namespace(out=out, config=cfg_path, split='dev', count=1, bootstrap=0,
        checkpoint=None, v30_runtime=None, task='od', device='cpu', root=root, annotations=annotation))
    assert result['complete_paired_grid']
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['scope'] == 'fixed_roi_only'
    rows = [json.loads(line) for line in (out / 'rows.jsonl').read_text().splitlines()]
    assert len(rows) == 2 * 5 * 5  # anchor and four independently selected arms
    for row in rows:
        wire = (out / row['packet_path']).read_bytes()
        parsed = unpack_packet(wire)
        assert len(wire) == row['coded_bytes'] == row['elementary_bytes'] + row['metadata_bytes']
        assert list(parsed.source_shape) == row['source_shape']
        assert row['bpp'] == 8 * len(wire) / (320 * 320)
