"""Same paid streams with and without the explicitly learned Swin receiver."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .post import load_checkpoint as load_post,receiver_variants
from .checkpoint import load_checkpoint
from .protocol import codec_factory,execution
from .swin import SwinAnalyzer
from ..analyzers import ActionAnalyzer
from ..codec import reference_bpp
from ..data import ar_plan,read_video,fingerprint
from ..evaluate import write_json,_ar_curves
from ..v32.evaluate import empty_output,SourceAudit,prepare_source
from ..v32.protocol import load_config,provenance


def run(args):
    out=empty_output(args.out); cfg=load_config(); device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.count!=32 or args.split not in ('dev','test'): raise ValueError('locked AR post DEV32/TEST32 required')
    torch.set_num_threads(2); torch.manual_seed(303232)
    rf=load_checkpoint(args.real_checkpoint,'ar',device); post=load_post(args.post_checkpoint,device)
    if post.metadata['rf_real']!=rf.manifest: raise ValueError('postprocessor paired RF checkpoint mismatch')
    teachers=[ActionAnalyzer(n,device) for n in cfg['ar_teachers']]
    evaluators={n:next((t for t in teachers if t.name==n),None) or ActionAnalyzer(n,device) for n in cfg['ar_evaluators']}
    evaluators['swin3d_t']=SwinAnalyzer(device)
    plan,meta=ar_plan(args.root,args.split,32); ids=[i['id'] for i in plan]; fingerprint(ids); audit=SourceAudit(); rows=[]
    manifest=dict(**provenance(cfg),scope='v32rf_swin_receiver',status='running',split=args.split,ids=ids,source_pixel_sha256={},rf_real=rf.manifest,post=post.manifest,
        codec_execution=execution('ar'),evaluators=list(evaluators),swin_protocol='same received 16 frames; official spatial transform',
        receiver='standard independent decode, followed optionally by frozen-Swin/trainable-RGB-head restoration',
        rate_scope='identical paid packet; preinstalled post weights excluded per packet',post_checkpoint_bytes=args.post_checkpoint.stat().st_size)
    write_json(out/'manifest.json',manifest)
    for index,item in enumerate(plan):
        rgb=read_video(item['path'],cfg['frames'],cfg['ar_size'],cfg['temporal_stride']); sourcehash=audit.add(item['id'],rgb)
        rf.assert_source(item['id'],rgb); post.assert_source(item['id'],rgb)
        start=time.perf_counter(); _,support,_=prepare_source(rgb,'ar',cfg,teachers); prep=time.perf_counter()-start
        for c in cfg['codecs']:
            for q in cfg['qps']:
                codec=codec_factory('ar')(c,q)
                for stream,pixels in (('anchor',rgb),('rf_real',rf.pixels(rgb,support,codec))):
                    packet=codec.roundtrip(pixels,source_shape=rgb.shape,task='ar'); path=Path('packets')/c/f'{index:05d}-q{q}-{stream}.v32'
                    (out/path).parent.mkdir(parents=True,exist_ok=True); (out/path).write_bytes(packet.data)
                    for variant,(received,seconds) in receiver_variants(packet,post.model).items():
                        arm=stream+('_post' if variant=='restored' else '')
                        row=dict(id=item['id'],codec=c,qp=q,arm=arm,source_sha256=sourcehash,source_shape=list(rgb.shape),
                            packet_sha256=packet.packet_sha256,packet_path=path.as_posix(),decoded_rgb_sha256=hashlib.sha256(packet.decoded.tobytes()).hexdigest(),
                            receiver_rgb_sha256=hashlib.sha256(received.tobytes()).hexdigest(),coded_bytes=packet.coded_bytes,metadata_bytes=packet.metadata_bytes,
                            elementary_bytes=packet.elementary_bytes,bpp=reference_bpp(packet.coded_bytes,rgb.shape),restore_seconds=seconds,codec_seconds=packet.seconds,source_preparation_seconds=prep,
                            correct={n:int(e.probabilities(received).argmax()==item['label']) for n,e in evaluators.items()})
                        rows.append(row)
                        with (out/'rows.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        manifest['source_pixel_sha256']=audit.hashes.copy(); write_json(out/'manifest.json',manifest)
        print(f'Swin receiver {args.split} {index+1}/32 complete',flush=True)
    expected={(i,c,q,a) for i in ids for c in cfg['codecs'] for q in cfg['qps'] for a in ('anchor','anchor_post','rf_real','rf_real_post')}
    keys=[(r['id'],r['codec'],r['qp'],r['arm']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected: raise ValueError('incomplete receiver paired grid')
    results={}
    for c in cfg['codecs']:
        results[c]={}
        for arm in ('anchor_post','rf_real','rf_real_post'):
            paired=[r for r in rows if r['codec']==c and r['arm']=='anchor']+[{**r,'arm':'adaptive'} for r in rows if r['codec']==c and r['arm']==arm]
            results[c][arm]=_ar_curves(paired,sorted(evaluators),cfg['qps'],0,cfg['seed'])
        paired=[{**r,'arm':'anchor'} for r in rows if r['codec']==c and r['arm']=='rf_real']+[{**r,'arm':'adaptive'} for r in rows if r['codec']==c and r['arm']=='rf_real_post']
        results[c]['post_on_same_rf_stream']=_ar_curves(paired,sorted(evaluators),cfg['qps'],0,cfg['seed'])
    write_json(out/'summary.json',dict(scope=manifest['scope'],results=results,complete_paired_grid=True,target_confirmed=False))
    manifest['status']='complete'; write_json(out/'manifest.json',manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--count',type=int,default=32); p.add_argument('--real-checkpoint',type=Path,required=True); p.add_argument('--post-checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--device',choices=['cpu','cuda']); run(p.parse_args())
