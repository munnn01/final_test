"""CAL-only fixed scalar proxy fits and unlabeled held-out proxy diagnostics."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ..data import partition
from ..evaluate import write_json
from ..v32vc.train import tensor,support_tensors,fit_rate
from ..v32vc.virtual import VirtualCodec
from ..v32rf.bridge import real_forward
from .protocol import codec_factory
from .tasks import ar_forward,od_forward,distance,frame_indices


def calibrate_domain(records,cal_ids,task,cfg,device,out):
    if len(cal_ids)!=12 or len(set(cal_ids))!=12 or any(i not in records or records[i].task!=task or partition(i[3:])!='train' for i in cal_ids): raise ValueError('twelve distinct TRAIN calibration sources required')
    out.mkdir(parents=True,exist_ok=True); rows=[]
    for identifier in cal_ids:
        r=records[identifier]; rgb,s=r.rgb,r.support
        low=np.stack([cv2.GaussianBlur(f,(0,0),1.5) for f in rgb]).astype(np.float32)
        alpha=((1-s['protection'])*(1-.75*s['motion']))[...,None]
        variants=[('identity',rgb),('bounded_blur',np.clip(np.rint(rgb+.1*alpha*(low-rgb)),0,255).astype(np.uint8))]
        for c in cfg['codecs']:
            for qp in cfg['qps']:
                codec=codec_factory(task,cfg['preset'])(c,qp)
                for name,px in variants:
                    packet=codec.roundtrip(px,source_shape=px.shape,task=task)
                    x=tensor(px,device); cuts=torch.from_numpy(s['cuts'].copy()).to(device)[None]; observations=[]
                    with torch.no_grad():
                        for scale in cfg['step_scales']:
                            step=.015*2**((qp-30)/6)*scale
                            v=VirtualCodec({f'{c}/{qp}':dict(step=step,rate_scale=1.,rate_offset=0.)}).to(device).eval()
                            z,rate=v(x,qp,int(c=='h265'),cuts)
                            observations.append(dict(step=step,bpp=float(rate),mse=float(F.mse_loss(z,tensor(packet.decoded,device)))))
                    rows.append(dict(source_id=identifier,codec=c,qp=qp,variant=name,observations=observations,
                        elementary_bpp=8*packet.elementary_bytes/np.prod(px.shape[:3]),metadata_bytes=packet.metadata_bytes,total_bytes=packet.coded_bytes))
        write_json(out/'calibration_rows.json',rows); print('CAL '+identifier,flush=True)
    entries={}
    for c in cfg['codecs']:
        for qp in cfg['qps']:
            group=[r for r in rows if r['codec']==c and r['qp']==qp]
            idx=min(range(len(cfg['step_scales'])),key=lambda n:sum(r['observations'][n]['mse'] for r in group))
            pred=np.array([r['observations'][idx]['bpp'] for r in group]); true=np.array([r['elementary_bpp'] for r in group])
            scale,offset=fit_rate(pred,true)
            entries[f'{c}/{qp}']=dict(step=group[0]['observations'][idx]['step'],rate_scale=scale,rate_offset=offset,n=len(group),
                pixel_mse=float(np.mean([r['observations'][idx]['mse'] for r in group])),rate_relative_mae=float(np.mean(np.abs(scale*pred+offset-true)/true)))
    write_json(out/'calibration.json',entries); return entries


def feature_error(heads,z,r):
    terms=[]
    if r.task=='ar':
        _,f=ar_forward(heads.action,z); terms.append(distance(f,r.source_features['ar']))
    _,f=od_forward(heads.detector,z,frame_indices(z.shape[2])); terms.append(distance(f,r.source_features['od']))
    return torch.stack(terms).mean()


def diagnose(model,records,val_ids,calibration,heads,cfg,out):
    device=next(model.parameters()).device; before={k:v.clone() for k,v in model.state_dict().items()}; rows=[]
    with torch.no_grad():
        for identifier in val_ids:
            r=records[identifier]; x=tensor(r.rgb,device); p,m,cuts=support_tensors(r.support,x)
            for c in cfg['codecs']:
                for qp in cfg['qps']:
                    codec=codec_factory(r.task,cfg['preset'])(c,qp); v=VirtualCodec(calibration[r.task]).to(device).eval()
                    y=model(x,qp,int(c=='h265'),p,m,cuts); proxy,rate=v(y,qp,int(c=='h265'),cuts)
                    actual,bpp,_=real_forward(y,codec,r.task)
                    # Fixed bounded perturbation: compares the local objective direction in both codecs.
                    low=F.avg_pool2d(y.permute(0,2,1,3,4).reshape(-1,3,*y.shape[-2:]),3,1,1).reshape(1,y.shape[2],3,*y.shape[-2:]).permute(0,2,1,3,4)
                    pert=(y+.01*(1-p)*(low-y)).clamp(0,1)
                    pp,_=v(pert,qp,int(c=='h265'),cuts); aa,_,_=real_forward(pert,codec,r.task)
                    fp,fa=feature_error(heads,proxy,r),feature_error(heads,actual,r)
                    delta_proxy=float(feature_error(heads,pp,r)-fp); delta_real=float(feature_error(heads,aa,r)-fa)
                    mse=float(F.mse_loss(proxy,actual))
                    # Plain packet metadata is determined by the wire schema, never estimated stream bytes.
                    from ..v32.transport import pack_packet
                    from fractions import Fraction
                    header=len(pack_packet(b'x',task=r.task,codec=c,source_shape=r.rgb.shape,coded_shape=r.rgb.shape,fps=Fraction(25)))-1
                    paid=float(rate)+8*header/np.prod(r.rgb.shape[:3])
                    rows.append(dict(source_id=identifier,codec=c,qp=qp,pixel_mse=mse,psnr_db=None if mse==0 else -10*np.log10(mse),
                        proxy_feature_error=float(fp),real_feature_error=float(fa),actual_bpp=bpp,estimated_bpp=paid,
                        rate_relative_error=abs(paid-bpp)/bpp,perturbation_proxy_delta=delta_proxy,perturbation_real_delta=delta_real,
                        descent_sign_agreement=bool(np.sign(delta_proxy)==np.sign(delta_real))))
    if any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()): raise RuntimeError('diagnostics updated model')
    result=dict(rows=rows,updates=False,labels_used=False,perturbation='fixed 1 percent editable local averaging')
    write_json(out/'diagnostics.json',result); return result
