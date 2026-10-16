"""Offline inference-error components using the official COCO matching rules."""
import contextlib
import copy
import io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ..v34.evaluation import coco_predictions


def _iou(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    low=np.maximum(a[:2],b[:2]); high=np.minimum(a[:2]+a[2:],b[:2]+b[2:])
    overlap=float(np.maximum(0,high-low).prod())
    return overlap/max(float(a[2:].prod()+b[2:].prod()-overlap),1e-12)


def components(predictions, annotations, image, geometry, *, score_cutoff=.10):
    image=dict(image); image_id=int(image.get('image_id',image.get('id')))
    item={**image,'image_id':image_id}
    result=[p for p in coco_predictions(predictions,item,geometry) if p['score']>=score_cutoff]
    anns=copy.deepcopy(annotations)
    if any(a['image_id']!=image_id for a in anns): raise ValueError('foreign COCO annotations')
    categories=sorted({a['category_id'] for a in anns}|{p['category_id'] for p in result})
    if not categories: return np.zeros(3)
    if len({a['id'] for a in anns})!=len(anns): raise ValueError('duplicate COCO annotation IDs')
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(); gt.dataset=dict(info={},images=[{**image,'id':image_id}],annotations=anns,
                                  categories=[dict(id=c) for c in categories]); gt.createIndex()
        if result: dt=gt.loadRes(result)
        else:
            dt=COCO();dt.dataset=dict(images=gt.dataset['images'],categories=gt.dataset['categories'],annotations=[]);dt.createIndex()
        evaluator=COCOeval(gt,dt,'bbox'); evaluator.params.imgIds=[image_id]
        evaluator.params.catIds=categories; evaluator.params.iouThrs=np.linspace(.5,.95,10)
        evaluator.params.areaRng=[[0,1e10]]; evaluator.params.areaRngLbl=['all'];evaluator.params.maxDets=[100]
        evaluator.evaluate()
    entries=[e for e in evaluator.evalImgs if e is not None]
    count=sum(int(np.count_nonzero(~np.asarray(e['gtIgnore'],bool))) for e in entries)
    values=[]
    for threshold in range(10):
        fn=fp=0; errors=[]
        for e in entries:
            ignored=np.asarray(e['gtIgnore'],bool); dt_ignored=np.asarray(e['dtIgnore'][threshold],bool)
            fn+=np.count_nonzero((e['gtMatches'][threshold]==0)&~ignored)
            fp+=np.count_nonzero((e['dtMatches'][threshold]==0)&~dt_ignored)
            for index,gt_id in enumerate(e['dtMatches'][threshold]):
                if gt_id and not dt_ignored[index]:
                    g=gt.anns[int(gt_id)]; d=dt.anns[int(e['dtIds'][index])]
                    if not g.get('iscrowd',0): errors.append(1-_iou(d['bbox'],g['bbox']))
        values.append([float(fn)/max(count,1),float(fp)/max(count,1),float(np.mean(errors)) if errors else 0.])
    return np.mean(values,axis=0)
