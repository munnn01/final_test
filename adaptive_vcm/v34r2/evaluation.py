"""One shared decoded packet, measured source constraints, committed prefix."""
import hashlib
import copy
import time
from pathlib import Path
import numpy as np
import torch
from .protocol import ACTIONS,codec_factory
from .tracking import Prefix
from .runner import prepare_source,stream_key,measured_score,condition,infer,input_pair,write_json
from ..v34.preprocessing import filter_clip
from ..v34.evaluation import coco_predictions,tracking_curve
from ..v33ap.evaluate import safe_curve


def choose(candidates,encode,score,active,cfg,prefix=None):
    candidates=[int(a) for a in candidates]
    if len(candidates)!=cfg['source_shortlist'] or len(set(candidates))!=len(candidates) or any(a<=0 or a>=len(ACTIONS) for a in candidates):
        raise ValueError('exact nonidentity probe budget required')
    actions=[0,*candidates]; packets=[]; values=[]; work=[]
    before=prefix.signature if prefix is not None else None
    for action in actions:
        packet=encode(action); start=time.perf_counter(); value=score(packet.decoded)
        work.append(time.perf_counter()-start); packets.append(packet); values.append(value)
    costs=np.array([v['source_cost'] for v in values],float); mask=np.asarray(active,bool)
    if not np.isfinite(costs[0,mask]).all(): raise ValueError('invalid source anchor')
    rates=np.array([p.coded_bytes/packets[0].coded_bytes for p in packets])
    excess=costs-costs[0]; feasible=(rates<=cfg['budget_ratio'])&np.isfinite(excess[:,mask]).all(1)
    feasible &= (excess[:,mask]<=np.array(cfg['source_guard_limits'])[mask]).all(1)
    if not feasible[0]: raise ValueError('identity guard must be feasible')
    selected=int(np.argmin(np.where(feasible,rates,np.inf)))
    if prefix is not None: prefix.commit(values[selected]['probe'])
    clean=lambda a:[[float(v) if np.isfinite(v) else None for v in row] for row in a]
    audit=dict(tried_actions=actions,source_excess=clean(excess),rates=rates.tolist(),feasible=feasible.tolist(),
        guard_seconds=sum(work),guard_work_seconds=work,prefix_before=before,prefix_after=prefix.signature if prefix is not None else None,
        selection_policy='minimum real paid bytes among source-feasible packets',source_guard_scope='source consistency; no task accuracy guarantee')
    return actions[selected],packets[selected],values[selected],audit


def shortlist(model,pair,c,qp,active,cfg,domain,kind,identifier):
    start=time.perf_counter(); out=None
    if kind=='cnn':
        out=infer(model,pair,condition(c,qp,active,cfg,domain))
        # Feasibility is a ranking signal; only real decoded probes enforce constraints.
        rank=out['logits']-np.logaddexp(0,-out['feasible_logits'])
        actions=[int(i) for i in np.argsort(-rank,kind='stable') if i][:cfg['source_shortlist']]
    elif kind=='train_fixed': actions=list(model.metadata['fixed_shortlist'])
    elif kind=='random':
        seed=int(hashlib.sha256(f"{model.metadata['seed']}/{identifier}/{c}/{qp}".encode()).hexdigest()[:16],16)
        actions=np.random.default_rng(seed).choice(np.arange(1,len(ACTIONS)),cfg['source_shortlist'],replace=False).tolist()
    else: raise ValueError('unknown shortlist control')
    return actions,out,time.perf_counter()-start


@torch.no_grad()
def evaluate(groups,meta,cfg,out,heads,models,experiment,package,transfer_ar):
    for split in ('dev','test'):
        rows=[]; clean={}; supports={}; prefixes={}; receivers={}; tracks={}; exhaustive=[]; gt_history={}
        for index,r in enumerate(groups[split]):
            source,maps=prepare_source(r,heads,experiment,cfg,clean,supports)
            pair=input_pair(r.rgb,maps); key=stream_key(r); gt_record=None; gt_context=None
            for c in cfg['codecs']:
                for qp in cfg['qps']:
                    codec=codec_factory(r.task,c,qp,cfg); packets={}; timing={}
                    def encode(action):
                        if action not in packets:
                            begin=time.perf_counter(); pixels=filter_clip(r.rgb,maps,ACTIONS[action]); pre=time.perf_counter()-begin
                            packets[action]=codec.roundtrip(pixels,source_shape=r.rgb.shape,task=r.task)
                            timing[action]=dict(preprocess_seconds=pre,**codec.last_times,q8_edits=int(np.count_nonzero(pixels!=r.rgb)))
                        return packets[action]
                    arms=[('anchor',None,None)]+[(f'{kind}_{m.metadata["seed"]}',m,kind) for m in models for kind in ('cnn','train_fixed','random')]
                    for arm,model,kind in arms:
                        prefix=prefixes.setdefault((key,c,qp,arm),Prefix(context_frames=cfg['context_frames'])) if source.mask[2] and arm!='anchor' else None
                        def score(rgb): return measured_score(heads,rgb,source,cfg,prefix=prefix)
                        diagnostic=None; frozen_prefix=None
                        if arm=='anchor':
                            action=0; packet=encode(0)
                            audit=dict(tried_actions=[0],guard_seconds=0,source_guard_scope='identity baseline')
                        else:
                            actions,predicted,seconds=shortlist(model,pair,c,qp,source.mask,cfg,r.task,kind,r.id)
                            # DEV-only full bank under the exact CNN prefix, before committing.
                            if split=='dev' and cfg['dev_exhaustive'] and kind=='cnn':
                                frozen_prefix=prefix.snapshot() if prefix is not None else None
                                values=[score(encode(a).decoded) for a in range(len(ACTIONS))]
                                costs=np.array([v['source_cost'] for v in values],float); active=np.asarray(source.mask,bool)
                                rates=np.array([encode(a).coded_bytes/encode(0).coded_bytes for a in range(len(ACTIONS))])
                                ok=(rates<=cfg['budget_ratio'])&((costs-costs[0])[:,active]<=np.array(cfg['source_guard_limits'])[active]).all(1)
                                best=int(np.argmin(np.where(ok,rates,np.inf)))
                                diagnostic=dict(id=r.id,codec=c,qp=qp,seed=model.metadata['seed'],action=best,rate=float(rates[best]),
                                    prefix_sha256=prefix.signature if prefix is not None else None,scope='DEV exhaustive diagnostics; never TEST GT selection')
                            action,packet,result,audit=choose(actions,encode,score,source.mask,cfg,prefix)
                            audit.update(shortlist_seconds=seconds,predictions=predicted)
                        # Independent receiver inference consumes the selected decoded packet.
                        begin=time.perf_counter(); pred={}; receiver_tracks=None
                        if source.mask[0]: pred['ar']=heads.action.probabilities(packet.decoded)
                        if source.mask[1] or source.mask[2]: pred['od']=heads.detections(packet.decoded)
                        if source.mask[2]:
                            receiver=receivers.setdefault((key,c,qp,arm),Prefix(context_frames=cfg['context_frames']))
                            received=receiver.probe(pred['od']); receiver.commit(received)
                            receiver_tracks=received.current; pred['ot']=receiver_tracks
                        receiver_seconds=time.perf_counter()-begin
                        row=dict(id=r.id,task=r.task,split=split,codec=c,qp=qp,arm=arm,action=action,active=source.mask,
                            packet_sha256=packet.packet_sha256,shared_packet=True,decoded_sha256=hashlib.sha256(packet.decoded.tobytes()).hexdigest(),
                            receiver_packet_sha256={task:packet.packet_sha256 for task in pred},receiver_seconds=receiver_seconds,
                            bytes=packet.coded_bytes,bpp=packet.coded_bytes*8/np.prod(r.rgb.shape[:3]),**timing[action],**audit)
                        # Labels/held-out annotations are accessed only after selection commits.
                        if r.task=='ar':
                            row['correct']={'swin3d_t':int(np.argmax(pred['ar'])==r.item['label'])}
                            if transfer_ar is not None: row['correct']['r3d_18']=int(np.argmax(transfer_ar.probabilities(packet.decoded))==r.item['label'])
                        elif r.task=='od': row['coco']=coco_predictions(pred['od'][0],r.item,r.geometry)
                        if r.task=='ot': tracks.setdefault((c,qp,arm,key),[]).extend(receiver_tracks)
                        row['receiver_scoring_seconds']=time.perf_counter()-begin-receiver_seconds
                        if diagnostic is not None:
                            # The selected stream has already committed. GT now enters only
                            # an offline DEV upper bound on a frozen pre-selection snapshot.
                            if gt_record is None:
                                gt_record=r
                                if r.task=='ot':
                                    from ..v34.tracking import gt_window
                                    gt_record=copy.copy(r)
                                    gt_record.targets=gt_window(Path(r.item['path'])/'gt/gt.txt',r.item['start'],r.item['start']+len(r.rgb),r.geometry)
                                    gt_context=gt_history.get(key,[])+gt_record.targets
                            gt_values=[measured_score(heads,encode(a).decoded,source,cfg,prefix=frozen_prefix,gt=gt_record,gt_context=gt_context,package=package)['gt_cost'] for a in range(len(ACTIONS))]
                            gt_mask=np.array([r.task==t for t in ('ar','od','ot')]); gt_values=np.array(gt_values,float)
                            gt_ok=(rates<=cfg['budget_ratio'])&np.isfinite(gt_values[:,gt_mask]).all(1)
                            gt_ok &= ((gt_values-gt_values[0])[:,gt_mask]<=np.array(cfg['damage_tolerance'])[gt_mask]).all(1)
                            gt_action=int(np.argmin(np.where(gt_ok,rates,np.inf)))
                            diagnostic.update(gt_action=gt_action,gt_rate=float(rates[gt_action]),gt_mask=gt_mask.tolist(),
                                gt_scope='available labelled task only; OT finite-context HOTA target; source upper bound is separate')
                            exhaustive.append(diagnostic)
                        tried=audit['tried_actions']; row['deployed_encode_count']=len(tried)
                        row['deployed_candidate_work']=[dict(action=a,**timing[a]) for a in tried]
                        row['sender_seconds']=(0 if arm=='anchor' else source.sender_support_seconds)+audit.get('shortlist_seconds',0)+audit['guard_seconds']+sum(
                            sum(timing[a][k] for k in ('preprocess_seconds','encode_seconds','decode_seconds','inverse_seconds')) for a in tried)
                        row['timing_scope']='sum of all deployed probes plus source/support/selector; excludes DEV exhaustive diagnostic'
                        rows.append(row)
            if gt_context is not None: gt_history[key]=gt_context[-cfg['context_frames']:]
            write_json(out/f'{split}_rows.partial.json',rows)
            print(f'V34 R2 evaluate {split} {index+1}/{len(groups[split])}',flush=True)
        write_json(out/f'{split}_rows.json',rows)
        summary=summarize(rows,groups[split],meta,cfg)
        if tracks: summary.update(tracking_summary(tracks,rows,groups[split],meta,cfg,out/split,package))
        write_json(out/f'{split}_summary.json',summary)
        if exhaustive: write_json(out/'dev_exhaustive_source_guard.json',exhaustive)


def coco_stats(results,gt,ids,meta):
    import contextlib,io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    names=['mAP','AP50','AP75','APsmall','APmedium','APlarge']
    if not results: return {k:0. for k in names}
    with contextlib.redirect_stdout(io.StringIO()):
        dataset=COCO(); dataset.dataset=dict(info=meta.get('info',{}),images=[dict(id=i) for i in ids],annotations=[a for i in ids for a in gt[i]],categories=meta['categories'])
        dataset.createIndex(); dt=dataset.loadRes(results); evaluator=COCOeval(dataset,dt,'bbox'); evaluator.params.imgIds=ids
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    return {k:float(evaluator.stats[i]) if evaluator.stats[i]>=0 else None for i,k in enumerate(names)}


def comparison(anchor,other):
    value=safe_curve(anchor,other); monotone=value.get('quality_monotone_by_rate',{})
    if not all(monotone.get(k,False) for k in ('anchor','adaptive')):
        value.update(bd_rate_pct=None,pchip_bd_rate_pct=None,bd_quality_pp=None,invalid_reason='nonmonotone or insufficient curve; inspect raw same-QP points')
    return value


def summarize(rows,records,meta,cfg):
    from collections import defaultdict
    results={}; gt=defaultdict(list)
    for a in meta.get('od',{}).get('annotations',[]): gt[a['image_id']].append(a)
    for task in ('ar','od'):
        subset=[r for r in rows if r['task']==task]
        if not subset: continue
        for c in cfg['codecs']:
            curves={}; points={}
            for arm in sorted({r['arm'] for r in subset}):
                curve=dict(bpp=[],quality=[]); points[arm]=[]
                for qp in cfg['qps']:
                    selected=[r for r in subset if r['codec']==c and r['qp']==qp and r['arm']==arm]
                    ids=[r.item['image_id'] for r in records if r.task=='od']
                    metrics=coco_stats([p for r in selected for p in r['coco']],gt,ids,meta['od']) if task=='od' else {
                        name:float(np.mean([r['correct'][name] for r in selected])) for name in selected[0]['correct']}
                    curve['bpp'].append(float(np.mean([r['bpp'] for r in selected])))
                    curve['quality'].append(metrics['mAP' if task=='od' else 'swin3d_t'])
                    points[arm].append(dict(qp=qp,**metrics,bpp=curve['bpp'][-1]))
                curves[arm]=curve
            result=dict(curves=curves,raw_points=points,comparisons={a:comparison(curves['anchor'],v) for a,v in curves.items() if a!='anchor'})
            if task=='od' and cfg['bootstrap_draws']:
                from ..coco_metrics import paired_bootstrap_detection_bd
                table={arm:{(c,qp):{r.item['image_id']:next((v['bpp'],v['coco']) for v in subset if v['id']==r.id and v['codec']==c and v['qp']==qp and v['arm']==arm)
                    for r in records if r.task=='od'} for qp in cfg['qps']} for arm in curves}
                result['paired_image_bootstrap']=paired_bootstrap_detection_bd(table,codec=c,qps=cfg['qps'],gt_by_id=gt,
                    image_ids=ids,ann_meta=meta['od'],arms=[a for a in curves if a!='anchor'],n_boot=cfg['bootstrap_draws'],seed=303401)
            results[task+'/'+c]=result
    return dict(results=results,scope=cfg['holdout_status'],quality_improvement_confirmed=False,
        joint_gt_scope='one packet per source; only the labelled source task has GT; source constraints on other active tasks',
        rate_scope='actual packet bytes including header; no encoder-only masks transmitted',
        probe_budget='identity + three candidates for CNN/TRAIN-fixed/random',
        selection_counts={a:{str(i):sum(r['arm']==a and r['action']==i for r in rows) for i in range(len(ACTIONS))} for a in {r['arm'] for r in rows}})


def tracking_summary(tracks,rows,records,meta,cfg,out,package):
    from ..v32.tracking import write_mot,stage_ground_truth,evaluate_mot
    plan=[p for p in meta['ot'] if p['name'] in {r.item['sequence'] for r in records if r.task=='ot'}]
    names=set()
    for (c,q,arm,seq),frames in tracks.items():
        name=f'{c}_{q}_{arm}'; names.add(name); r=next(r for r in records if r.task=='ot' and r.item['sequence']==seq)
        write_mot(out/'trackers'/name/'data'/(seq+'.txt'),frames,r.geometry,r.item['width'],r.item['height'])
    write_json(out/'ground_truth_provenance.json',stage_ground_truth(plan,out/'ground_truth'))
    metrics=evaluate_mot(package,plan,out,sorted(names))
    # Preserve DetA/AssA from the official raw report, which the legacy wrapper omits.
    import json
    raw=json.loads((out/'trackeval_raw_results.json').read_text())['results']
    for name,value in metrics.items():
        for seq,entry in [('COMBINED_SEQ',value['combined']),*value['sequences'].items()]:
            hota=raw[name][seq]['pedestrian']['HOTA']
            entry.update(DetA=float(np.mean(hota['DetA'])*100),AssA=float(np.mean(hota['AssA'])*100))
    curves={}
    for c in cfg['codecs']:
        arms=sorted({r['arm'] for r in rows if r['task']=='ot'})
        values={a:dict(bpp=[float(np.mean([r['bpp'] for r in rows if r['task']=='ot' and r['codec']==c and r['qp']==q and r['arm']==a])) for q in cfg['qps']],
            quality=[metrics[f'{c}_{q}_{a}']['combined']['HOTA'] for q in cfg['qps']]) for a in arms}
        curves[c]=dict(curves=values,comparisons={a:tracking_curve(values['anchor'],v) for a,v in values.items() if a!='anchor'})
    return dict(ot_official_metrics=metrics,ot_curves=curves,ot_sequence_ci=None,
        ot_uncertainty='one sequence per held-out split in locked historical recipe; no sequence CI')
