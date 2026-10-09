"""Load the previous plain CNN using its immutable, unmodified runtime."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from .protocol import settings
from ..v32.protocol import canonical_hash
from ..v32vc.checkpoint import state_hash

WARM_FILES_HASH='7fc9d721da25876e3915f14e9dedaf64c67a02737a5c737558761a4107b3e8b5'
WARM_BUNDLE_HASH='4283b114e7dd3dab6621d75661d36b69e910bc22c891275546fd117a381ac6ce'


def verify_runtime(runtime):
    root=Path(runtime).resolve()
    try: origin=json.loads((root/'origin-manifest.json').read_text())
    except (OSError,ValueError) as error: raise ValueError('warm runtime manifest unavailable') from error
    files=origin.get('source_files_sha256',{})
    if (origin.get('origin_commit')!=settings()['warm_origin'] or canonical_hash(files)!=WARM_FILES_HASH
            or origin.get('bundle_sha256')!=WARM_BUNDLE_HASH):
        raise ValueError('warm runtime is not the immutable V32-VC snapshot')
    for name,digest in files.items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('warm runtime source mismatch: '+name)
    return root,origin


def load_warm(path,runtime,task,device='cpu'):
    root,origin=verify_runtime(runtime); path=Path(path)
    alias='_v32rf_warm_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if alias not in sys.modules:
        init=root/'adaptive_vcm/__init__.py'
        spec=importlib.util.spec_from_file_location(alias,init,submodule_search_locations=[str(init.parent)])
        module=importlib.util.module_from_spec(spec); sys.modules[alias]=module; spec.loader.exec_module(module)
    loader=importlib.import_module(alias+'.v32vc.checkpoint')
    warm=loader.load_checkpoint(path,task,device)
    if warm.metadata['arm']!='plain': raise ValueError('warm start must be original plain LAST')
    warm.rf_warm_manifest=dict(origin_commit=origin['origin_commit'],checkpoint_sha256=warm.manifest['checkpoint_sha256'],
        state_sha256=state_hash(warm.model.state_dict()),fit_ids=warm.metadata['fit_ids'],cal_ids=warm.metadata['cal_ids'],
        source_pixel_sha256=warm.metadata['source_pixel_sha256'])
    return warm
