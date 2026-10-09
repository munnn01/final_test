"""The original RF loader runs unmodified under a verified import alias."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from ..v32.protocol import canonical_hash

ORIGIN_COMMIT='50a22e57ad68bba91ad9d1ffc4caec5140ede259'
BUNDLE_HASH='26659e08959250b8dfb297950998f4ca3024006d9e2faefaf2185600d3c19270'
FILES_HASH='09098025eb34379fdc665d4495b0dad5bd0e18e128fb90381d2e08b605bd6ec5'
CHECKPOINT_HASH='12f05e4a2647890f33d27bc0a95b495398f4e1fc2ada1ce1beee017a984cb272'


def verify_runtime(runtime):
    root=Path(runtime).resolve()
    try: origin=json.loads((root/'origin-manifest.json').read_text(encoding='utf-8'))
    except (OSError,ValueError) as error: raise ValueError('immutable RF runtime manifest missing') from error
    files=origin.get('source_files_sha256',{})
    if (origin.get('origin_commit')!=ORIGIN_COMMIT or origin.get('bundle_sha256')!=BUNDLE_HASH
            or canonical_hash(files)!=FILES_HASH): raise ValueError('immutable RF runtime differs')
    for name,digest in files.items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('immutable RF runtime file differs: '+name)
    return root,origin


def load_frozen_ar(checkpoint,runtime,device='cpu'):
    root,origin=verify_runtime(runtime); checkpoint=Path(checkpoint)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_HASH:
        raise ValueError('immutable RF AR checkpoint bytes differ')
    alias='_v32at_rf_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if alias not in sys.modules:
        init=root/'adaptive_vcm/__init__.py'
        spec=importlib.util.spec_from_file_location(alias,init,submodule_search_locations=[str(init.parent)])
        module=importlib.util.module_from_spec(spec); sys.modules[alias]=module; spec.loader.exec_module(module)
    loader=importlib.import_module(alias+'.v32rf.checkpoint')
    loaded=loader.load_checkpoint(checkpoint,'ar',device)
    if loaded.metadata['arm']!='real': raise ValueError('RF real AR model required')
    loaded.model.eval().requires_grad_(False)
    loaded.manifest=dict(loaded.manifest,original_runtime_commit=origin['origin_commit'],runtime_bundle_sha256=BUNDLE_HASH)
    return loaded
