import numpy as np


def test_separate_codec_timing_and_exact_wire():
    from adaptive_vcm.v33ap.protocol import codec_factory
    from adaptive_vcm.v32rf.protocol import codec_factory as original
    rgb=np.random.default_rng(3).integers(0,256,(1,16,16,3),dtype=np.uint8)
    for c in ('h264','h265'):
        codec=codec_factory('od','ultrafast')(c,35); packet=codec.roundtrip(rgb,source_shape=rgb.shape,task='od')
        other=original('od','ultrafast')(c,35).roundtrip(rgb,source_shape=rgb.shape,task='od')
        assert packet.data==other.data and np.array_equal(packet.decoded,other.decoded)
        assert codec.last_times['encode_seconds']>0 and codec.last_times['decode_seconds']>0
        assert codec.last_times['inverse_seconds']==0
