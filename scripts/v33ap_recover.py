"""Recover verified V33-AP LAST checkpoints after a serialization-only fix."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from adaptive_vcm.v32.protocol import canonical_hash,provenance
from adaptive_vcm.v32vc.checkpoint import state_hash
from adaptive_vcm.v33ap.checkpoint import native_metadata,load_checkpoint
from adaptive_vcm.v33ap.protocol import settings


def recover_checkpoint(source,destination,expected_sha256,origin):
    source,destination=Path(source),Path(destination)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=expected_sha256: raise ValueError('original checkpoint hash mismatch')
    if destination.exists(): raise FileExistsError('recovered checkpoint destination must be new')
    # Only the scalar/dtype pair observed in the verified campaign archives.
    with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,np.dtypes.Float64DType]):
        value=torch.load(source,map_location='cpu',weights_only=True)
    if not isinstance(value,dict) or set(value)!={'metadata','metadata_sha256','model'}: raise ValueError('invalid original checkpoint')
    m=value['metadata']; state=state_hash(value['model'])
    if canonical_hash(m)!=value['metadata_sha256'] or state!=m['state_sha256'] or m['settings']!=settings(): raise ValueError('original state/metadata/recipe mismatch')
    previous=m['code']; archived={k:h for k,h in origin['source_files_sha256'].items() if k.startswith('adaptive_vcm/') and k.endswith('.py')}
    if previous['code_files_sha256']!=archived: raise ValueError('original source provenance mismatch')
    current=provenance(settings()); old=previous['code_files_sha256']; new=current['code_files_sha256']
    changes=sorted(k for k in set(old)|set(new) if old.get(k)!=new.get(k))
    if any(k!='adaptive_vcm/v33ap/checkpoint.py' for k in changes): raise ValueError('recovery requires serialization-only runtime changes')
    audit=dict(original_checkpoint_sha256=expected_sha256,original_metadata_sha256=value['metadata_sha256'],
        original_source_commit=origin['origin_commit'],original_code=previous,state_sha256=state,
        changed_runtime_files=changes,parameter_updates=False,reason='NumPy float64 history to native metadata; serialization-only fix')
    restored=native_metadata({**m,'code':current,'checkpoint_recovery':audit})
    output=dict(metadata=restored,metadata_sha256=canonical_hash(restored),model=value['model'])
    destination.parent.mkdir(parents=True,exist_ok=True); torch.save(output,destination)
    loaded=load_checkpoint(destination)
    if state_hash(loaded.model.state_dict())!=state: raise ValueError('recovery changed trained state')
    return dict(arm=restored['arm'],checkpoint_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),metadata_sha256=output['metadata_sha256'],**audit)
