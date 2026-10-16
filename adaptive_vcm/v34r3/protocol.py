"""R3 recipes and provenance; task identities never stand for metric slots."""
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile
from . import SCHEMA
from ..v34.protocol import ACTIONS, TASKS, digest, codec_factory

ROOT = Path(__file__).resolve().parents[2]
BASELINE = 'c445ef3c690383e6a593c24043fca6d8a3f0cdaa'
STAGES = ('measure-identity','train-warm','measure-rollout','train-final','evaluate','all')
COMPONENTS = {'ar': ('ar','od','ot'), 'od': ('fn','fp','localization'),
              'ot': ('hota','deta','assa')}


def component_names(task):
    if task not in COMPONENTS: raise ValueError('unknown standalone task')
    return COMPONENTS[task]


def task_mask(task):
    component_names(task)
    return [int(task == name) for name in TASKS]


def component_mask(task):
    component_names(task)
    return [1,0,0] if task == 'ar' else [1,1,1]


def settings(path=None):
    cfg = json.loads((Path(path) if path else ROOT/'configs/v34_r3_qp4.json').read_text(encoding='utf-8'))
    if (cfg.get('schema') != SCHEMA or cfg.get('revision') != 'r3-qp4-shared-method'
        or cfg.get('qps') != [30,35,40,45] or cfg.get('codecs') != ['h264','h265']
        or cfg.get('seeds') != [303401,303402,303403] or cfg.get('conditions') != 9
        or cfg.get('frames') != 16 or cfg.get('source_shortlist') != 3
        or set(cfg.get('task_recipes', {})) != set(TASKS)):
        raise ValueError('R3 recipe drift')
    if cfg['task_recipes']['ar'] != {'mode':'frozen_reference','config_path':'configs/v34_r1_qp4.json'}:
        raise ValueError('AR recipe must remain native and frozen')
    for task in ('od','ot'):
        value=cfg['task_recipes'][task]
        if value.get('risk_components') != list(component_names(task)):
            raise ValueError('risk component semantics mismatch')
        for name in ('gt_limits','source_limits'):
            limits=value.get(name, [])
            expected=3 if name=='gt_limits' or task=='ot' else 4
            if len(limits)!=expected or any(not isinstance(x,(int,float)) or not 0<=x<1e6 for x in limits):
                raise ValueError('invalid component limits')
    return cfg


def recipe(cfg, task, stage):
    component_names(task)
    if stage not in STAGES or task=='ar' and stage!='evaluate':
        raise ValueError('AR is evaluation-only; invalid task stage')
    common={k:v for k,v in cfg.items() if k!='task_recipes'}
    return {**common, **cfg['task_recipes'][task], 'task':task,
            'component_mask':component_mask(task)}


def baseline_manifest(root, base_commit=BASELINE):
    if not re.fullmatch('[0-9a-f]{40}',base_commit): raise ValueError('full baseline commit required')
    raw=subprocess.check_output(['git','-C',str(root),'archive','--format=zip',base_commit])
    files={}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in archive.namelist():
            p=Path(name)
            if not name.endswith('/') and (p.suffix=='.py' or p.parts[0]=='configs'
                or p.name in ('pyproject.toml','requirements.txt','requirements.lock','setup.cfg')):
                files[name]=hashlib.sha256(archive.read(name)).hexdigest()
    if not files: raise ValueError('empty baseline source manifest')
    value=dict(base_commit=base_commit,files=files)
    return {**value,'sha256':digest(value)}


def verify_baseline(root, manifest):
    if manifest.get('sha256') != digest({k:v for k,v in manifest.items() if k!='sha256'}):
        raise ValueError('baseline manifest changed')
    root=Path(root).resolve()
    if not manifest.get('files'): raise ValueError('empty baseline manifest')
    for name,expected in manifest['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen baseline source changed: '+name)
    return manifest
