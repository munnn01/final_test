import numpy as np
import pytest
from PIL import Image


def test_coco_single_frame_exact_geometry(tmp_path):
    from adaptive_vcm.v33ap.sources import read_source
    from adaptive_vcm.data import read_image
    p=tmp_path/'a.png'; Image.new('RGB',(13,7),(10,40,80)).save(p)
    item=dict(path=str(p),width=13,height=7)
    rgb,g,recipe=read_source('od',item,dict(od_size=16))
    old,og=read_image(item,16)
    assert np.array_equal(rgb,old) and g==og and len(rgb)==1
    assert g[:2]==(16/13,9/7) and recipe['repeated_frames']==0


def test_ar_canonical_short_recipe(tmp_path):
    import cv2
    from adaptive_vcm.v33ap.sources import read_source
    from adaptive_vcm.data import read_video
    p=tmp_path/'a.avi'; writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),25,(16,16))
    for n in range(3): writer.write(np.full((16,16,3),40+n*20,np.uint8))
    writer.release()
    rgb,g,r=read_source('ar',dict(path=str(p)),dict(frames=4,temporal_stride=2,ar_size=16))
    assert np.array_equal(rgb,read_video(str(p),4,16,2)) and g is None
    assert r['decoded_unique_frames']==2 and r['repeated_frames']==2
    with pytest.raises((FileNotFoundError,RuntimeError)): read_source('ar',dict(path='missing.mp4'),dict(frames=4,temporal_stride=2,ar_size=16))


def test_duplicate_pixels_across_namespaces():
    from adaptive_vcm.v33ap.sources import pixel_audit
    from v33_fixtures import record
    a,b=record(),record('od'); b.rgb=a.rgb.copy()
    with pytest.raises(ValueError,match='duplicate'): pixel_audit([a,b])
