import importlib
from pathlib import Path

import pytest
import torch

from tests.test_v32_roi import pixels


def baseline():
    return importlib.import_module('adaptive_vcm.v32.baseline')


def test_checkpoint_with_bad_schema_is_rejected_by_unmodified_upstream_loader(tmp_path):
    path = tmp_path / 'invalid.pth'
    torch.save({'schema': 'old', 'task': 'ar'}, path)
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match='schema'):
        baseline().FrozenV30Baseline.load(path, root, 'ar', 'cpu')


def test_runtime_must_be_explicit_and_existing(tmp_path):
    with pytest.raises(ValueError):
        baseline().FrozenV30Baseline.load(tmp_path / 'missing.pth', tmp_path, 'ar', 'cpu')


def test_baseline_source_pixel_or_id_leakage_is_rejected():
    import hashlib
    rgb = pixels()
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    fixture = baseline().FrozenV30Baseline(None, None, {}, {'train_source_sha256': {'seen': digest}})
    with pytest.raises(ValueError, match='TRAIN'):
        fixture.assert_source('unseen-id', rgb)
    with pytest.raises(ValueError, match='TRAIN'):
        fixture.assert_source('seen', pixels(2))


def test_v32_config_rejects_weakened_guard_and_incomplete_qp_grid():
    module = importlib.import_module('adaptive_vcm.v32.protocol')
    cfg = module.load_config()
    cfg['ar_kl_slack'] = .2
    with pytest.raises(ValueError):
        module.validate_config(cfg)
    cfg = module.load_config()
    cfg['qps'] = [30, 35, 40, 45]
    with pytest.raises(ValueError):
        module.validate_config(cfg)
