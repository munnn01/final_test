"""An immutable public holdout payload cannot omit models or drift its notebook."""
import importlib
import importlib.util
import json
import pytest


def module():
    assert importlib.util.find_spec('scripts.v34_holdout_kaggle') is not None, 'holdout payload builder missing'
    return importlib.import_module('scripts.v34_holdout_kaggle')


def members():
    values = {n: b'fixture' for n in ['source.zip', 'origin.json', 'job.json', 'protocol.json', 'annotations.csv', 'exclusions.json']}
    values.update({f'controller_{s}.pth': b'fixture' for s in [303401, 303402, 303403]})
    return values


def test_transport_requires_all_three_frozen_checkpoints_and_rejects_extra_members():
    m = module(); raw = members()
    assert m.unpack(m.pack(raw)) == raw
    del raw['controller_303402.pth']
    with pytest.raises(ValueError): m.pack(raw)
    raw = members(); raw['../credentials.json'] = b'not allowed'
    with pytest.raises(ValueError): m.pack(raw)


def test_job_retains_full_protocol_and_changes_only_independent_shard_account():
    m = module()
    protocol = dict(design=dict(count=1024, shards=4), settings={'codecs': ['h264', 'h265']})
    origin = dict(origin_commit='a'*40)
    job = m.job_spec('fixture_account', origin, protocol, 2)
    assert job['shard'] == 2 and job['private'] is False
    assert job['count'] == 1024 and job['training_performed'] is False
    assert job['datasets'] == ['jizeyong/kinetics400-rgb256-f10-val-data']
    assert job['handle'].startswith('fixture_account/')
    with pytest.raises(ValueError): m.job_spec('fixture_account', origin, protocol, 4)


def test_tampered_payload_is_rejected_before_any_submit_call(tmp_path):
    m = module()
    receipt = dict(payload_sha256={'notebook.ipynb': 'a'*64})
    (tmp_path/'local_validation.json').write_text(json.dumps(receipt))
    (tmp_path/'notebook.ipynb').write_text('{}')
    with pytest.raises(ValueError): m.validate_payload(tmp_path)


def test_runtime_transport_preserves_exact_core_bytes_and_committed_provenance():
    import hashlib, io, zipfile
    m = module()
    files = {'adaptive_vcm/v34/holdout.py': b'evaluator\n', 'adaptive_vcm/v34/evaluation.py': b'original core\n',
             'configs/v34_arot_r1.json': b'{"fixed":true}', 'pyproject.toml': b'name="test"\n',
             'docs/old.md': b'non-runtime history', 'tests/test_old.py': b'old test', 'scripts/old.py': b'old script'}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        for name, raw in files.items(): z.writestr(name, raw)
    raw = stream.getvalue()
    origin = dict(origin_commit='a'*40, bundle_sha256=hashlib.sha256(raw).hexdigest(), bundle_bytes=len(raw),
        source_files_sha256={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if not n.endswith('.md')})
    bundle, manifest = m.runtime_source(raw, origin)
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        expected = {'adaptive_vcm/v34/holdout.py', 'adaptive_vcm/v34/evaluation.py', 'configs/v34_arot_r1.json', 'pyproject.toml'}
        assert set(z.namelist()) == expected
        assert all(z.read(n) == files[n] for n in expected)
    assert set(manifest['source_files_sha256']) == expected
    assert manifest['origin_commit'] == 'a'*40 and manifest['original_archive_sha256'] == origin['bundle_sha256']
    assert manifest['bundle_sha256'] == hashlib.sha256(bundle).hexdigest()
