"""A frozen native AR instance in the shared framework, never a retrained AR."""
import hashlib
import json
from pathlib import Path
from ..v34.checkpoint import load_checkpoint
from ..v34.evaluation import choose_packet


def _checked(entry, root):
    path=Path(entry['path'])
    path=(root/path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry.get('sha256'):
        raise ValueError('frozen AR artifact changed')
    return path


def load_reference(manifest, device='cpu'):
    root=Path.cwd()
    if isinstance(manifest,(str,Path)):
        path=Path(manifest).resolve(); root=path.parent
        manifest=json.loads(path.read_text(encoding='utf-8'))
    cfg=json.loads(_checked(manifest['config'],root).read_text(encoding='utf-8'))
    if (cfg.get('revision')!='r1-qp4-single-task' or cfg.get('qps')!=[30,35,40,45]
        or cfg.get('seeds')!=[303401,303402,303403] or not cfg.get('domain_condition')):
        raise ValueError('native four-QP AR recipe required')
    entries=manifest.get('checkpoints',[])
    if len(entries)!=3: raise ValueError('three pinned AR checkpoints required')
    models=[load_checkpoint(_checked(e,root),device) for e in entries]
    for model,seed in zip(models,cfg['seeds']):
        meta=model.metadata
        if (model.conditions!=9 or model.actions!=13 or meta.get('experiment')!='ar'
            or meta.get('settings')!=cfg or meta.get('seed')!=seed
            or meta.get('task_models')!=manifest.get('task_models') or 'calibration' not in meta):
            raise ValueError('frozen AR checkpoint/recipe/model mismatch')
    return cfg,models


def evaluate_reference(groups,heads,models,cfg,meta,out,trackeval_path):
    from ..v34.evaluation import evaluate
    from ..v34.runner import write_json
    from ..v32vc.checkpoint import state_hash
    if heads.experiment!='ar' or any(m.metadata['settings']!=cfg or m.metadata['experiment']!='ar'
        or m.metadata['task_models']!=heads.hashes() for m in models):raise ValueError('matching frozen AR reference required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    evaluate(groups,heads,models,cfg,meta,out,trackeval_path)
    receipt=dict(execution='fresh_frozen_reference_evaluation',task='ar',retrained=False,
        checkpoint_sha256=[m.checkpoint_sha256 for m in models],state_sha256=[state_hash(m.state_dict()) for m in models],
        reports_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*summary.json')})
    write_json(out/'ar_reference_receipt.json',receipt);return receipt
