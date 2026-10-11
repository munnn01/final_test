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
