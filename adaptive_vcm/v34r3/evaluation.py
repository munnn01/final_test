"""Independent task evaluation; held-out labels enter after stream decisions."""
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from .protocol import ACTIONS,codec_factory,digest,task_mask
from .measurement import prepare_source
from .selection import choose,shortlist
from .metrics import compare
from .sources import group_id
from ..v34.preprocessing import filter_clip
from ..v34.runner import input_pair,write_json,infer
from ..v34r2.runner import condition
from ..v34r2.tracking import Prefix
from ..v34.evaluation import coco_predictions
from ..v34r2.evaluation import summarize
from ..v32vc.checkpoint import state_hash


def predict(model,pair,c,q,cfg):
    return dict(**infer(model,pair,condition(c,q,task_mask(cfg['task']),cfg,cfg['task'])),model_state_sha256=state_hash(model.state_dict()))


def promotion(report,cohort_manifest,calibration):
    checks=dict(independent_cohort=bool(cohort_manifest.get('confirmation_eligible')),
        enough_cal=bool(calibration.get('promotion_eligible')) and (report['task']!='ot' or calibration.get('source_groups',0)>=2),
        quality=report.get('worst_qp_gap_pp',-np.inf)>=-.5,bytes=bool(report.get('byte_improvement')),
        dev_gate=bool(report.get('dev_gate_supported')),nonidentity=report.get('nonidentity_fraction',0)>0,
        bd_reliable=bool(report.get('bd_reliable')))
    return dict(confirmed=all(checks.values()),checks=checks,scope='confirmation requires every check; historical and single-sequence evidence is exploratory')


def tracking_summary(tracks,rows,records,meta,cfg,out,package):
    from ..v32.tracking import write_mot,stage_ground_truth,TRACKEVAL_COMMIT,json_safe
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plan=[p for p in meta['ot'] if p['name'] in {group_id(r) for r in records}]
    benchmarks={p['name'].split('-')[0] for p in plan}
    if len(benchmarks)!=1 or not benchmarks<={'MOT17','MOT20'}:raise ValueError('one explicit MOT benchmark per evaluation required')
    names=set()
    for (c,q,arm,seq),frames in tracks.items():
        name=f'{c}_{q}_{arm}';names.add(name);r=next(r for r in records if group_id(r)==seq)
        write_mot(out/'trackers'/name/'data'/(seq+'.txt'),frames,r.geometry,r.item['width'],r.item['height'])
    write_json(out/'ground_truth_provenance.json',stage_ground_truth(plan,out/'ground_truth'))
    dataset=package.datasets.MotChallenge2DBox(dict(GT_FOLDER=str(out/'ground_truth'),TRACKERS_FOLDER=str(out/'trackers'),
        OUTPUT_FOLDER=str(out/'official_metrics'),TRACKERS_TO_EVAL=sorted(names),BENCHMARK=next(iter(benchmarks)),
        SPLIT_TO_EVAL='train',SKIP_SPLIT_FOL=True,DO_PREPROC=True,PRINT_CONFIG=False,SEQ_INFO={p['name']:p['frames'] for p in plan}))
    evaluator=package.Evaluator(dict(PRINT_RESULTS=False,PRINT_CONFIG=False,TIME_PROGRESS=False,USE_PARALLEL=False,
        BREAK_ON_ERROR=True,LOG_ON_ERROR=None,OUTPUT_SUMMARY=True,OUTPUT_DETAILED=True,PLOT_CURVES=False))
    metrics=[package.metrics.HOTA(),package.metrics.CLEAR(dict(PRINT_CONFIG=False,THRESHOLD=.5)),package.metrics.Identity(dict(PRINT_CONFIG=False,THRESHOLD=.5))]
    raw,messages=evaluator.evaluate([dataset],metrics);raw=raw[dataset.get_name()]
    write_json(out/'trackeval_raw_results.json',json_safe(dict(results=raw,messages=messages,commit=TRACKEVAL_COMMIT)))
    def extract(entry):
        v=entry['pedestrian'];h=v['HOTA'];c=v['CLEAR']
        return dict(**{k:float(np.mean(h[k])*100) for k in ('HOTA','DetA','AssA')},IDF1=float(v['Identity']['IDF1']*100),
            MOTA=float(c['MOTA']*100),IDs=int(c['IDSW']),FP=int(c['CLR_FP']),FN=int(c['CLR_FN']),TP=int(c['CLR_TP']))
    official={n:dict(combined=extract(raw[n]['COMBINED_SEQ']),sequences={p['name']:extract(raw[n][p['name']]) for p in plan}) for n in names}
    results={}
    for c in cfg['codecs']:
        curves={};points={}
        for arm in sorted({r['arm'] for r in rows}):
            points[arm]=[dict(qp=q,bpp=float(np.mean([r['bpp'] for r in rows if r['codec']==c and r['qp']==q and r['arm']==arm])),
                **official[f'{c}_{q}_{arm}']['combined']) for q in cfg['qps']]
            curves[arm]=dict(bpp=[p['bpp'] for p in points[arm]],quality=[p['HOTA'] for p in points[arm]])
        results['ot/'+c]=dict(curves=curves,raw_points=points)
    return dict(results=results,official_metrics=official,uncertainty='no sequence CI when only one heldout sequence; component targets are finite-context surrogates')


@torch.no_grad()
def evaluate(groups,meta,cfg,out,heads,models,task,package,*,splits=('dev','test'),codecs=None,policies=None):
    if task not in ('od','ot') or any(m.metadata.get('checkpoint_role')!='FINAL' or m.metadata.get('recipe')!=cfg
        or m.metadata.get('task_models')!=heads.hashes() for m in models):raise ValueError('matching FINAL task checkpoints required')
    if not splits or not set(splits)<={'dev','test'}:raise ValueError('heldout evaluation only')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);report_cfg={**cfg,'codecs':list(codecs or cfg['codecs'])}
    if not set(report_cfg['codecs'])<=set(cfg['codecs']):raise ValueError('invalid codec scope')
    policies=policies or [dict(gt_limits=cfg['gt_limits'],source_limits=cfg['source_limits'],risk_gate=True)]
    summaries={}
    for split in splits:
        records=[r for r in groups[split] if r.task==task]
        if not records:continue
        rows=[];clean={};supports={};prefixes={};receivers={};tracks={};gt_history={}
        for index,r in enumerate(records):
            source,maps=prepare_source(r,heads,task,cfg,clean,supports);pair=input_pair(r.rgb,maps);key=group_id(r);offline=[]
            for c in report_cfg['codecs']:
                for q in cfg['qps']:
                    codec=codec_factory(task,c,q,cfg);packets={};times={};sender_predictions={};receiver_predictions={}
                    def encode(action):
                        if action not in packets:
                            begin=time.perf_counter();pixels=filter_clip(r.rgb,maps,ACTIONS[action]);pre=time.perf_counter()-begin
                            packets[action]=codec.roundtrip(pixels,source_shape=r.rgb.shape,task=task)
                            times[action]=dict(preprocess_seconds=pre,**codec.last_times)
                        return packets[action]
                    def frozen_predictions(rgb,cache):
                        sha=hashlib.sha256(rgb.tobytes()).hexdigest()
                        if sha not in cache:
                            start=time.perf_counter();pred=heads.detections(rgb);cache[sha]=(pred,time.perf_counter()-start)
                        return cache[sha]
                    def score(rgb):
                        # Cache only deterministic frozen inference on identical pixels.
                        # Every tracker probe still belongs to its own arm's prefix.
                        detections,seconds=frozen_predictions(rgb,sender_predictions)
                        start=time.perf_counter();value=heads.source_score(rgb,source,prefix=prefix,package=package,detections=detections)
                        value['representative_guard_seconds']=seconds+time.perf_counter()-start
                        return value
                    arms=[('anchor',None,None,None)]+[(f'{kind}_{m.metadata["seed"]}'+(f'_p{i}' if len(policies)>1 else ''),m,kind,p)
                        for i,p in enumerate(policies) for m in models for kind in ('cnn','train_fixed','random')]
                    for arm,model,kind,policy in arms:
                        prefix=prefixes.setdefault((key,c,q,arm),Prefix(context_frames=cfg['context_frames'])) if task=='ot' else None
                        begin=time.perf_counter()
                        if arm=='anchor':
                            action=0;packet=encode(0);value=score(packet.decoded)
                            if prefix is not None:prefix.commit(value['probe'])
                            audit=dict(tried_actions=[0],guard_seconds=0);selector_seconds=0
                        else:
                            prediction=predict(model,pair,c,q,cfg)
                            rng=np.random.default_rng(int(digest([model.metadata['seed'],r.id,c,q])[:16],16))
                            actions=shortlist(prediction,cfg,kind,fixed=model.metadata['fixed_shortlist'],rng=rng)
                            selector_seconds=time.perf_counter()-begin
                            costs=[]
                            def scored(rgb):
                                value=score(rgb);costs.append(value['representative_guard_seconds']);return value
                            action,packet,value,audit=choose(actions,encode,scored,
                                task=task,recipe=cfg,prediction=prediction,calibration=model.metadata.get('calibration'),prefix=prefix,policy=policy)
                            audit['actual_cached_guard_seconds']=audit['guard_seconds'];audit['guard_seconds']=sum(costs)
                        # Independent receiver consumes only the packet selected above.
                        detections,inference_seconds=frozen_predictions(packet.decoded,receiver_predictions);begin=time.perf_counter()
                        receiver_tracks=None
                        if task=='ot':
                            receiver=receivers.setdefault((key,c,q,arm),Prefix(context_frames=cfg['context_frames']))
                            probe=receiver.probe(detections);receiver.commit(probe);receiver_tracks=probe.current
                            tracks.setdefault((c,q,arm,key),[]).extend(receiver_tracks)
                        row=dict(id=r.id,task=task,split=split,codec=c,qp=q,arm=arm,action=action,
                            bytes=packet.coded_bytes,bpp=packet.coded_bytes*8/np.prod(r.rgb.shape[:3]),
                            packet_sha256=packet.packet_sha256,decoded_sha256=hashlib.sha256(packet.decoded.tobytes()).hexdigest(),
                            receiver_seconds=inference_seconds+time.perf_counter()-begin,checkpoint_sha256=getattr(model,'checkpoint_sha256',None),**audit)
                        if task=='od':row['coco']=coco_predictions(detections[0],r.item,r.geometry)
                        if model is not None:row['predicted_gt_damage']=np.asarray(prediction['risk'])[action].tolist()
                        tried=audit['tried_actions'];row.update(deployed_encode_count=len(tried),deployed_candidate_work=[dict(action=a,**times[a]) for a in tried],
                            sender_seconds=(source.sender_support_seconds if arm!='anchor' else 0)+selector_seconds+audit['guard_seconds']+
                                sum(sum(times[a][k] for k in ('preprocess_seconds','encode_seconds','decode_seconds','inverse_seconds')) for a in tried),
                            timing_scope='representative uncached probe/receiver inference work; identical-pixel frozen predictions are memoized across experimental arms; actual cached guard time is separate')
                        rows.append(row)
                        offline.append((row,packet,value))
            # All decisions for this source have committed before any label read.
            if task=='od':
                annotations=[a for a in meta['od']['annotations'] if a['image_id']==r.item['image_id']];context=None
            else:
                from .mot_targets import read_context
                annotations=None;context=gt_history.get(key,[])+read_context(Path(r.item['path'])/'gt/gt.txt',r.item['start'],r.item['start']+len(r.rgb))
                gt_history[key]=context[-cfg['context_frames']:]
            anchors={};cost_cache={}
            for row,packet,value in offline:
                cachekey=(packet.packet_sha256,getattr(value.get('probe'),'signature',None))
                if cachekey not in cost_cache:
                    cost_cache[cachekey]=heads.gt_score(packet.decoded,source,annotations=annotations,context=context,package=package,source_result=value)['gt_cost']
                cost=np.asarray(cost_cache[cachekey],float);condition_key=(row['codec'],row['qp'])
                if row['arm']=='anchor':anchors[condition_key]=cost
                delta=cost-anchors[condition_key];row['offline_gt_cost']=cost.tolist();row['offline_gt_damage']=delta.tolist()
                row['offline_gt_scope']='post-selection component diagnostics; not final cohort accuracy'
                if 'predicted_gt_damage' in row:row['risk_prediction_error']=(delta-row['predicted_gt_damage']).tolist()
            write_json(out/f'{split}_rows.partial.json',rows)
            print(f'V34 R3 {task} evaluate {split} {index+1}/{len(records)}',flush=True)
        write_json(out/f'{split}_rows.json',rows)
        summary=summarize(rows,records,meta,{**report_cfg,'bootstrap_draws':0}) if task=='od' else tracking_summary(tracks,rows,records,meta,report_cfg,out/split,package)
        summary.pop('joint_gt_scope',None);summary.update(task=task,scope=meta.get('cohort_audit',{}).get('status',cfg['holdout_status']),quality_improvement_confirmed=False,
            recipe_sha256=digest(cfg),policies=policies,checkpoint_sha256=[m.checkpoint_sha256 for m in models],
            calibration=[m.metadata.get('calibration') for m in models],rate_scope='actual paid packet including header',
            selection_counts={a:{str(i):sum(r['arm']==a and r['action']==i for r in rows) for i in range(13)} for a in {r['arm'] for r in rows}})
        for result in summary['results'].values():
            result['comparisons']={a:compare(result['curves']['anchor'],curve,task=task) for a,curve in result['curves'].items() if a!='anchor'}
        summary['controller_diagnostics']={}
        for arm in sorted({r['arm'] for r in rows if r['arm']!='anchor'}):
            selected=[r for r in rows if r['arm']==arm];counts=np.bincount([a for r in selected for a in r['tried_actions'][1:]],minlength=13);freq=counts/counts.sum()
            policy=policies[int(arm.rsplit('_p',1)[1])] if len(policies)>1 else policies[0]
            summary['controller_diagnostics'][arm]=dict(shortlist_frequency=freq.tolist(),shortlist_entropy=float(-sum(p*np.log(p) for p in freq if p)),
                identity_fraction=float(np.mean([r['action']==0 for r in selected])),
                measured_component_violations=np.sum(np.asarray([r['offline_gt_damage'] for r in selected])>policy.get('gt_limits',cfg['gt_limits']),axis=0).tolist(),
                risk_mae=np.mean(np.abs([r['risk_prediction_error'] for r in selected]),axis=0).tolist())
        if task=='od' and split=='test' and cfg['bootstrap_draws']:
            _bootstrap(summary,rows,records,meta,report_cfg)
        write_json(out/f'{split}_summary.json',summary);summaries[split]=summary
    return summaries


def select_policy(summary,policies,task):
    """One task policy, selected from DEV across all seeds/codecs/QPs."""
    scored=[]
    for i,p in enumerate(policies):
        gaps=[];rates=[]
        for result in summary['results'].values():
            anchor=result['curves']['anchor']
            for arm,curve in result['curves'].items():
                if arm.startswith('cnn_') and (len(policies)==1 or arm.endswith(f'_p{i}')):
                    gaps.extend((np.asarray(curve['quality'],float)-np.asarray(anchor['quality'],float))*(1 if task=='ot' else 100))
                    rates.extend(np.asarray(curve['bpp'])/anchor['bpp'])
        violations=sum(sum(v['measured_component_violations']) for arm,v in summary.get('controller_diagnostics',{}).items()
            if arm.startswith('cnn_') and (len(policies)==1 or arm.endswith(f'_p{i}')))
        scored.append(dict(index=i,policy=p,worst_gap_pp=float(min(gaps)),rate_ratio=float(np.mean(rates)),component_violations=violations))
    eligible=[s for s in scored if s['policy'].get('risk_gate',True) and s['worst_gap_pp']>=-.5 and s['rate_ratio']<1]
    selected=min(eligible,key=lambda s:(s['rate_ratio'],s['index'])) if eligible else next(s for s in scored if s['policy'].get('risk_gate',True))
    controls=[s for s in scored if not s['policy'].get('risk_gate',True) and s['policy']['gt_limits']==selected['policy']['gt_limits'] and s['policy']['source_limits']==selected['policy']['source_limits']]
    supported=bool(eligible) and bool(controls) and selected['component_violations']<controls[0]['component_violations'] and (controls[0]['worst_gap_pp']<-.5 or selected['rate_ratio']<controls[0]['rate_ratio']-1e-6)
    return dict(policy=selected['policy'],policy_sha256=digest(selected['policy']),fit_split='dev',scores=scored,dev_gate_supported=supported)


def _bootstrap(summary,rows,records,meta,cfg):
    from collections import defaultdict
    from ..coco_metrics import _remap_bootstrap_sample
    from ..v34r2.evaluation import coco_stats
    ids=[r.item['image_id'] for r in records];id_map={r.id:r.item['image_id'] for r in records};gt=defaultdict(list)
    for a in meta['od']['annotations']:gt[a['image_id']].append(a)
    for c in cfg['codecs']:
        result=summary['results']['od/'+c];arms=list(result['comparisons']);values={a:[] for a in arms}
        table={(r['arm'],r['qp'],id_map[r['id']]):r for r in rows if r['codec']==c};rng=np.random.default_rng(303401)
        for _ in range(cfg['bootstrap_draws'] if len(ids)>=2 else 0):
            sample=rng.choice(ids,len(ids),replace=True).tolist();boot_gt,new_ids,mapping=_remap_bootstrap_sample(sample,gt);curves={}
            for arm in ['anchor',*arms]:
                curve=dict(bpp=[],quality=[])
                for q in cfg['qps']:
                    pred=[{**p,'image_id':new} for old,new in mapping for p in table[arm,q,old]['coco']]
                    curve['bpp'].append(float(np.mean([table[arm,q,old]['bpp'] for old,_ in mapping])))
                    curve['quality'].append(coco_stats(pred,boot_gt,new_ids,meta['od'])['mAP'])
                curves[arm]=curve
            for arm in arms:
                value=compare(curves['anchor'],curves[arm],task='od')['bd_rate_pct']
                if value is not None and np.isfinite(value):values[arm].append(value)
        result['paired_image_bootstrap']={a:dict(lo=float(np.quantile(v,.025)) if v else None,hi=float(np.quantile(v,.975)) if v else None,
            valid_draws=len(v),requested_draws=cfg['bootstrap_draws'],valid_fraction=len(v)/cfg['bootstrap_draws'],
            scope='paired image bootstrap, conditional on R3-valid draws; no interval when none valid') for a,v in values.items()}
