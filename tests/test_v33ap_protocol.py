import pytest
from adaptive_vcm.data import partition


def test_arm_masks_and_locked_grid():
    from adaptive_vcm.v33ap.protocol import settings, arm_weights, codec_factory
    c = settings()
    assert c['seed'] == 303251 and c['width'] == 12 and c['epochs'] == 2
    assert c['qps'] == [30, 35, 40, 45, 50]
    assert [arm_weights(a) for a in ('joint', 'ar_only', 'od_only')] == [(1., 1.), (1., 0.), (0., 1.)]
    with pytest.raises(ValueError): arm_weights('ot')
    assert codec_factory('ar')('h264', 40).standard.gop_settings['gop'] == 16
    assert codec_factory('od')('h265', 40).standard.gop_settings['gop'] == 1
    with pytest.raises(ValueError): codec_factory('ot')


def test_train_split_namespace_and_order():
    from adaptive_vcm.v33ap.protocol import source_id, split_train
    ids = [source_id('od', str(i)) for i in range(200) if partition('coco2017/'+str(i)) == 'train'][:64]
    fit, cal, val = split_train(ids, 'od')
    assert [len(fit), len(cal), len(val)] == [48, 12, 4]
    assert not set(fit)&set(cal) and not set(fit)&set(val) and not set(cal)&set(val)
    assert split_train(list(reversed(ids)), 'od') == (fit, cal, val)
    assert source_id('ar', 'jumping/file.mp4') == 'ar/jumping/file.mp4'
    with pytest.raises(ValueError): split_train(ids[:-1], 'od')
    with pytest.raises(ValueError): source_id('od', '../1')
