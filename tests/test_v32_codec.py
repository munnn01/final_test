import importlib
from fractions import Fraction

import numpy as np
import pytest

from adaptive_vcm.codec import locate_ffmpeg
from adaptive_vcm.v32.roi import build_grids, retarget
from tests.test_v32_roi import pixels, support


@pytest.mark.codec
@pytest.mark.parametrize('name', ['h264', 'h265'])
def test_real_codec_packet_reconstructs_from_received_mapping_and_counts_bytes(name):
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    module = importlib.import_module('adaptive_vcm.v32.codec')
    rgb = pixels()
    grids = build_grids(rgb.shape, support(), factor=4, margin=0)
    coded = retarget(rgb, grids)
    codec = module.V32Codec(name, 35, preset='ultrafast', fps=Fraction(25))
    result = codec.roundtrip(coded, source_shape=rgb.shape, task='od', grids=grids)
    assert result.decoded.shape == rgb.shape
    assert result.coded_shape == (1, 28, 28, 3)
    assert result.coded_bytes == len(result.data)
    assert result.coded_bytes == result.elementary_bytes + result.metadata_bytes
    assert result.metadata_bytes > 40
    received = codec.decode_packet(bytes(result.data))
    np.testing.assert_array_equal(received.decoded, result.decoded)
    assert received.packet_sha256 == result.packet_sha256


@pytest.mark.codec
def test_plain_packet_uses_standard_decoder_without_roi_source_side_information():
    if locate_ffmpeg() is None:
        pytest.skip('FFmpeg unavailable')
    module = importlib.import_module('adaptive_vcm.v32.codec')
    rgb = np.full((1, 32, 32, 3), 128, np.uint8)
    codec = module.V32Codec('h264', 35, preset='ultrafast', fps=Fraction(25))
    result = codec.roundtrip(rgb, source_shape=rgb.shape, task='od')
    assert result.decoded.shape == rgb.shape
    assert result.metadata_bytes == 40
    assert np.max(np.abs(result.decoded.astype(int) - 128)) <= 2
