"""Explicit guarded-selection attribution; forced arms never count as acceptance."""
from collections import defaultdict
import math


def selector_diagnostics(audits,cfg,task):
    slack=cfg['ar_kl_slack' if task=='ar' else 'od_distance_slack']
    result=defaultdict(lambda:dict(generated=0,selected=0,guard_failed=0,insufficient_savings=0,other_selected=0))
    for audit in audits:
        anchor=audit['candidates'][0]
        selected={v for k,v in audit['selected'].items() if not k.endswith('_forced')}
        for candidate in audit['candidates']:
            if not candidate['name'].startswith('rf_'): continue
            row=result[candidate['name']]; row['generated']+=1
            distances=candidate['relative_distances']; decisions=candidate['preserves_decision']
            if candidate['name'] in selected: row['selected']+=1
            elif (not distances or len(distances)!=len(anchor['relative_distances'])
                    or len(decisions)!=len(distances) or not all(decisions)
                    or any(v is None or not math.isfinite(v) or v>slack for v in distances)):
                row['guard_failed']+=1
            elif candidate['total_bytes']>anchor['total_bytes']*(1-cfg['min_savings']):
                row['insufficient_savings']+=1
            else: row['other_selected']+=1
    return dict(result)
