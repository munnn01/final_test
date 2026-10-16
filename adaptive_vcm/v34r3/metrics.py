"""Keep the measurements; interpolation is a diagnostic, never a repair."""
import numpy as np
from ..v33ap.evaluate import safe_curve


def compare(anchor,adaptive,*,task):
    unit=100 if task=='ot' else 1
    a=dict(bpp=anchor['bpp'],quality=(np.asarray(anchor['quality'],float)/unit).tolist())
    b=dict(bpp=adaptive['bpp'],quality=(np.asarray(adaptive['quality'],float)/unit).tolist())
    result=safe_curve(a,b)
    result.update(raw_points=dict(anchor=anchor,adaptive=adaptive),interpolation_sensitive=False,
        raw_cubic_bd_rate_pct=result.get('bd_rate_pct'),raw_pchip_bd_rate_pct=result.get('pchip_bd_rate_pct'))
    reasons=[]
    for curve in (a,b):
        rate=np.asarray(curve['bpp'],float);quality=np.asarray(curve['quality'],float)
        if len(rate)!=4 or not np.isfinite(rate).all() or not np.isfinite(quality).all() or (rate<=0).any():
            reasons.append('four finite positive-rate QP points required');continue
        order=np.argsort(rate);q=quality[order]
        if len(set(rate))<4 or (np.diff(q)<=0).any():reasons.append('flat, duplicate or nonmonotone quality/rate curve');continue
        p=np.polynomial.Polynomial.fit(quality,np.log(rate),3).convert()
        lo=max(min(a['quality']),min(b['quality']));hi=min(max(a['quality']),max(b['quality']))
        if not lo<hi:reasons.append('insufficient quality overlap');continue
        derivative=p.deriv();turns=[lo,hi,*[v.real for v in derivative.deriv().roots() if abs(v.imag)<1e-9 and lo<v.real<hi]]
        extrema=[lo,hi,*[v.real for v in derivative.roots() if abs(v.imag)<1e-9 and lo<v.real<hi]]
        if min(derivative(turns))<-1e-8 or min(p(extrema))<np.log(rate).min()-1e-8 or max(p(extrema))>np.log(rate).max()+1e-8:
            result['interpolation_sensitive']=True
    cubic=result.get('bd_rate_pct');pchip=result.get('pchip_bd_rate_pct')
    if cubic is not None and pchip is not None and abs(cubic-pchip)>5:result['interpolation_sensitive']=True
    if result.get('invalid_reason'):reasons.append(result['invalid_reason'])
    if result['interpolation_sensitive']:reasons.append('cubic overshoot/negative derivative or cubic-PCHIP disagreement >5 percentage points')
    reliable=not reasons and cubic is not None
    result.update(bd_reliable=reliable,diagnostic_policy='predeclared 5-percentage-point engineering sensitivity check')
    if not reliable:
        result.update(bd_rate_pct=None,bd_quality_pp=None,point_passes=False,screen_passes=False,invalid_reason='; '.join(dict.fromkeys(reasons)))
        for name in result.get('guards',{}):
            if name.startswith(('canonical_bd','pchip_bd','bd_quality')):result['guards'][name]=False
    return result
