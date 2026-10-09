import json
from pathlib import Path
import pytest
import shutil


def test_frozen_runtime_rejects_invented_manifest_before_import(tmp_path):
    from adaptive_vcm.v32at.frozen_rf import verify_runtime
    (tmp_path/'origin-manifest.json').write_text(json.dumps({'origin_commit': '0'*40, 'source_files_sha256': {}}))
    with pytest.raises(ValueError, match='immutable'):
        verify_runtime(tmp_path)


def test_real_ar_checkpoint_and_original_runtime_load_without_rehashing():
    from adaptive_vcm.v32at.frozen_rf import load_frozen_ar
    root = Path(__file__).resolve().parents[2]/'output/v32at_runtime'
    cp = Path(__file__).resolve().parents[2]/'output/v32rf_kaggle_packed1/ar/extracted_v1/v32rf-ar-fit48-test32-s303231/training/real_last.pth'
    if not root.exists(): pytest.skip('immutable runtime is prepared by local artifact recovery')
    loaded = load_frozen_ar(cp, root)
    assert loaded.metadata['task'] == 'ar' and loaded.metadata['arm'] == 'real'
    assert loaded.manifest['checkpoint_sha256'] == '12f05e4a2647890f33d27bc0a95b495398f4e1fc2ada1ce1beee017a984cb272'
    assert all(not p.requires_grad for p in loaded.model.parameters())


def test_source_and_checkpoint_tamper_rejected(tmp_path):
    from adaptive_vcm.v32at.frozen_rf import verify_runtime, load_frozen_ar
    root = Path(__file__).resolve().parents[2]/'output/v32at_runtime'
    if not root.exists(): pytest.skip('immutable runtime unavailable')
    clone = tmp_path/'runtime'
    shutil.copytree(root, clone, ignore=shutil.ignore_patterns('.git','__pycache__'))
    name = next(iter(json.loads((clone/'origin-manifest.json').read_text())['source_files_sha256']))
    with (clone/name).open('ab') as f: f.write(b'\n# modified\n')
    with pytest.raises(ValueError, match='runtime file differs'):
        verify_runtime(clone)
    cp = tmp_path/'modified.pth'
    cp.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='checkpoint'):
        load_frozen_ar(cp, root)
