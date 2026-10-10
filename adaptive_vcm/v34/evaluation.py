"""Held-out predictions first, GT scoring afterwards; raw paired curves retained."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..analyzers import ActionAnalyzer,DetectionAnalyzer
from ..v33ap.sources import SourceRecord,serial
from ..v33ap.evaluate import inverse_boxes,safe_curve
from ..coco_metrics import coco_map,coco_box
from .protocol import ACTIONS,codec_factory
from .preprocessing import region_maps,filter_clip
from .learning import select_action
from .runner import condition,input_pair,infer,write_json

def choose_packet(model,pair,codec_name,qp,mask,cfg,anchor,encode):
    """Only learned predictions, calibrated bounds and measured bytes enter here."""
    out=infer(model,pair,condition(codec_name,qp,mask,cfg))
    forced=int(np.argmax(out['logits']))
    selected=select_action(out['risk'],out['rate'],model.metadata['calibration'],mask,cfg['damage_tolerance'],cfg['budget_ratio'])
    proposed=selected
    packet=anchor if selected==0 else encode(selected)
    fallback=packet.coded_bytes>anchor.coded_bytes*cfg['budget_ratio']
    if fallback: selected=0; packet=anchor
    return forced,selected,packet,dict(predicted_risk=out['risk'],predicted_rate=out['rate'],proposed_action=proposed,rate_fallback=fallback)

@torch.no_grad()
def received_predictions(packet,heads,transfer_ar,transfer_od,task):
    rgb=packet.decoded; result={}
    if len(rgb)>1:
        result['ar']={h.name:h.probabilities(rgb) for h in (heads.action,transfer_ar)}
    if task=='od': result['od']=transfer_od.predict(rgb)
    elif heads.experiment=='ar_od':
        from ..v33ap.tasks import frame_indices
        result['od_consistency']=heads.detections(rgb[list(frame_indices(len(rgb)))])
    else: result['ot_detections']=heads.detections(rgb)
    return result

def coco_predictions(pred,item,g):
    mapped=inverse_boxes(pred,item,g)
    return [dict(image_id=item['image_id'],category_id=int(label),bbox=coco_box(box),score=float(score))
        for box,label,score in zip(mapped['boxes'],mapped['labels'],mapped['scores'])]

def summaries(rows,groups,meta,cfg,split):
    result={}; all_arms=sorted({r['arm'] for r in rows})
    for task in ('ar','od'):
        records=[r for r in groups[split] if r.task==task]
        if not records: continue
        gt={}
        if task=='od':
            from collections import defaultdict
            gt=defaultdict(list)
            for annotation in meta['od']['annotations']: gt[annotation['image_id']].append(annotation)
        for codec in cfg['codecs']:
            curves={}
            names=('swin3d_t','r3d_18') if task=='ar' else ('resnet50',)
            for name in names:
                curves[name]={}
                for arm in all_arms:
                    curve=dict(bpp=[],quality=[])
                    for qp in cfg['qps']:
                        subset=[r for r in rows if r['task']==task and r['codec']==codec and r['qp']==qp and r['arm']==arm]
                        if len(subset)!=len(records): raise ValueError('incomplete paired held-out grid')
                        quality=float(np.mean([r['correct'][name] for r in subset])) if task=='ar' else coco_map(
                            [p for r in subset for p in r['coco']],gt,[r.item['image_id'] for r in records],meta['od'])[0]
                        curve['bpp'].append(float(np.mean([r['bpp'] for r in subset]))); curve['quality'].append(quality)
                    curves[name][arm]=curve
            result[task+'/'+codec]={'curves':curves,'comparisons':{name:{a:safe_curve(c['anchor'],v) for a,v in c.items() if a!='anchor'} for name,c in curves.items()}}
    # Paired source bootstrap of AR accuracy at each same QP. This CI does not
    # imply equal bitrate or independent evidence across seeds.
    ci={}; rng=np.random.default_rng(303401)
    for key in result:
        if not key.startswith('ar/'): continue
        codec=key.split('/')[1]; ci[codec]={}
        for arm in all_arms:
            if arm=='anchor': continue
            ci[codec][arm]={}
            for qp in cfg['qps']:
                a=sorted([r for r in rows if r['task']=='ar' and r['codec']==codec and r['qp']==qp and r['arm']=='anchor'],key=lambda r:r['id'])
                b=sorted([r for r in rows if r['task']=='ar' and r['codec']==codec and r['qp']==qp and r['arm']==arm],key=lambda r:r['id'])
                delta=np.array([v['correct']['swin3d_t']-u['correct']['swin3d_t'] for u,v in zip(a,b)])
                draws=delta[rng.integers(0,len(delta),(1000,len(delta)))].mean(1)
                ci[codec][arm][qp]=dict(delta_pp=float(delta.mean()*100),lo_pp=float(np.quantile(draws,.025)*100),hi_pp=float(np.quantile(draws,.975)*100))
    return dict(split=split,scope='exploratory historical pilot; no same-video joint GT on K400+COCO/MOT20',results=result,
        secondary_source_consistency={f'{domain}/{codec}/{qp}/{arm}':{
            metric:float(np.mean([r[metric] for r in rows if r['task']==domain and r['codec']==codec and r['qp']==qp and r['arm']==arm and metric in r]))
            for metric in ('od_source_distance','ot_source_cost','ar_source_kl')
            if any(r['task']==domain and r['codec']==codec and r['qp']==qp and r['arm']==arm and metric in r for r in rows)}
            for domain in ('ar','ot') for codec in cfg['codecs'] for qp in cfg['qps'] for arm in all_arms},
        selection_counts={a:dict(total=sum(r['arm']==a for r in rows),identity=sum(r['arm']==a and r['action']==0 for r in rows),
            rate_fallback=sum(r['arm']==a and r.get('rate_fallback',False) for r in rows),
            q8_changed=sum(r['arm']==a and r['q8_edits']>0 for r in rows)) for a in all_arms},
        ar_paired_same_qp_ci=ci,bootstrap_draws=1000,od_ci=None,ot_sequence_ci=None,
        uncertainty_limit='OD curves need larger paired bootstrap; one OT sequence per held-out split prevents a sequence CI',
        rate_scope='paid stream plus common header / canonical pixels; encoder-only maps are not transmitted',target_confirmed=False)

def evaluate(groups,heads,models,cfg,meta,out,trackeval_path):
    transfer_ar=ActionAnalyzer('r3d_18',heads.device)
    transfer_od=DetectionAnalyzer('resnet50',heads.device) if heads.experiment=='ar_od' else None
    from ..v32vc.checkpoint import state_hash
    write_json(out/'evaluators.json',dict(task_models=heads.hashes(),transfer_ar=state_hash(transfer_ar.model.state_dict()),
        transfer_od=state_hash(transfer_od.model.state_dict()) if transfer_od else None,tracker='supervision==0.27.0 ByteTrack' if heads.experiment=='ar_ot' else None))
    for split in ('dev','test'):
        rows=[]; trackers={}; tracks={}
        for index,r in enumerate(groups[split]):
            # Copy only source fields into support/selection. Labels and GT IDs
            # remain in the outer record for scoring after stream selection.
            source=SourceRecord(r.id,r.task,{k:v for k,v in r.item.items() if k!='label'},r.rgb,r.geometry,r.frame_recipe)
            start=time.perf_counter(); heads.source(source)
            map_start=time.perf_counter()
            maps=region_maps(source.rgb,source.source_predictions['od']); pair=input_pair(source.rgb,maps)
            support_seconds=source.source_detector_seconds+time.perf_counter()-map_start
            source_seconds=time.perf_counter()-start
            for codec_name in cfg['codecs']:
                for qp in cfg['qps']:
                    codec=codec_factory(r.task,codec_name,qp,cfg); packets={}; timings={}; predictions={}
                    def encode(action):
                        if action not in packets:
                            start=time.perf_counter(); pixels=filter_clip(r.rgb,maps,ACTIONS[action]); pre=time.perf_counter()-start
                            packets[action]=codec.roundtrip(pixels,source_shape=r.rgb.shape,task=r.task)
                            timings[action]=dict(preprocess_seconds=pre,**codec.last_times,q8_edits=int(np.count_nonzero(pixels!=r.rgb)))
                        return packets[action]
                    anchor=encode(0); arms={'anchor':(0,anchor,{}),'fixed':(cfg['fixed_action'],encode(cfg['fixed_action']),{})}
                    for model in models:
                        start=time.perf_counter(); forced,selected,p,details=choose_packet(model,pair,codec_name,qp,source.mask,cfg,anchor,encode)
                        details['selection_seconds']=time.perf_counter()-start
                        seed=model.metadata['seed']; arms[f'forced_{seed}']=(forced,encode(forced),{**details,'rate_fallback':False})
                        arms[f'selected_{seed}']=(selected,p,details)
                    for arm,(action,packet,details) in arms.items():
                        sha=packet.packet_sha256
                        if sha not in predictions:
                            start=time.perf_counter(); predictions[sha]=received_predictions(packet,heads,transfer_ar,transfer_od,r.task)
                            predictions[sha]['receiver_seconds']=time.perf_counter()-start
                        pred=predictions[sha]
                        row=dict(id=r.id,task=r.task,split=split,codec=codec_name,qp=qp,arm=arm,action=action,
                            packet_sha256=sha,shared_packet=True,decoded_sha256=hashlib.sha256(packet.decoded.tobytes()).hexdigest(),
                            bytes=packet.coded_bytes,elementary_bytes=packet.elementary_bytes,metadata_bytes=packet.metadata_bytes,
                            bpp=packet.coded_bytes*8/np.prod(r.rgb.shape[:3]),source_seconds=source_seconds,
                            support_seconds=support_seconds,diagnostic_teacher_seconds=source.diagnostic_teacher_seconds,
                            receiver_seconds=pred['receiver_seconds'],**timings[action],**details)
                        if arm.startswith('selected_'):
                            proposed=details['proposed_action']
                            row['deployed_encode_seconds']=timings[0]['encode_seconds']+(timings[proposed]['encode_seconds'] if proposed else 0.)
                            row['deployed_encode_count']=1+int(proposed!=0)
                        if r.task=='ar':
                            row['correct']={n:int(np.argmax(p)==r.item['label']) for n,p in pred['ar'].items()}
                            if heads.experiment=='ar_od':
                                from .tasks import detection_distance
                                from ..v33ap.tasks import frame_indices
                                row['od_source_distance']=detection_distance([source.source_predictions['od'][i] for i in frame_indices(len(r.rgb))],pred['od_consistency'])
                            else:
                                from .tracking import track_frames,tracking_cost
                                if 'ot_frames' not in pred: pred['ot_frames']=track_frames(pred['ot_detections'])
                                row['ot_source_cost']=tracking_cost(pred['ot_frames'],source.source_predictions['ot'])
                        elif r.task=='od': row['coco']=coco_predictions(pred['od'],r.item,r.geometry)
                        elif r.task=='ot':
                            from .tracking import ByteTracker,track_frames
                            key=(codec_name,qp,arm,r.item['sequence'])
                            if key not in trackers: trackers[key]=ByteTracker(); tracks[key]=[]
                            tracks[key].extend(track_frames(pred['ot_detections'],trackers[key]))
                            p=pred['ar']['swin3d_t']; ref=source.source_predictions['ar']
                            row['ar_source_kl']=float(np.sum(ref*np.log(np.maximum(ref,1e-10)/np.maximum(p,1e-10))))
                            row['ar_accuracy']=None
                        rows.append(row)
            write_json(out/f'{split}_rows.partial.json',rows)
            print(f'V34 eval {split} {index+1}/{len(groups[split])}',flush=True)
        write_json(out/f'{split}_rows.json',rows)
        summary=summaries(rows,groups,meta,cfg,split)
        if tracks:
            from ..v32.tracking import write_mot,stage_ground_truth,load_trackeval,evaluate_mot
            plan=[p for p in meta['ot'] if p['name'] in {r.item['sequence'] for r in groups[split] if r.task=='ot'}]
            metrics_out=out/(split+'_tracking'); metrics_out.mkdir()
            names=[]
            for (codec,qp,arm,seq),frames in tracks.items():
                name=f'{codec}_{qp}_{arm}'; names.append(name)
                record=next(r for r in groups[split] if r.task=='ot' and r.item['sequence']==seq)
                write_mot(metrics_out/'trackers'/name/'data'/(seq+'.txt'),frames,record.geometry,record.item['width'],record.item['height'])
            write_json(metrics_out/'ground_truth_provenance.json',stage_ground_truth(plan,metrics_out/'ground_truth'))
            metrics=evaluate_mot(load_trackeval(trackeval_path),plan,metrics_out,sorted(set(names)))
            summary['ot_official_metrics']=metrics
            summary['ot_curves']={}
            for codec in cfg['codecs']:
                curves={}
                for arm in sorted({r['arm'] for r in rows}):
                    curves[arm]=dict(bpp=[],quality=[])
                    for qp in cfg['qps']:
                        sub=[r for r in rows if r['task']=='ot' and r['codec']==codec and r['qp']==qp and r['arm']==arm]
                        curves[arm]['bpp'].append(float(np.mean([r['bpp'] for r in sub])))
                        curves[arm]['quality'].append(metrics[f'{codec}_{qp}_{arm}']['combined']['HOTA'])
                summary['ot_curves'][codec]=dict(curves=curves,comparisons={a:safe_curve(curves['anchor'],c) for a,c in curves.items() if a!='anchor'})
        write_json(out/f'{split}_summary.json',summary)
