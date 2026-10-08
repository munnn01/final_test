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


@pytest.mark.parametrize('actual_shape', [(4, 32, 32, 3), (1, 32, 128, 3)])
def test_received_equal_byte_count_cannot_substitute_geometry_or_frame_count(actual_shape):
    from adaptive_vcm.codec import StandardCodec
    from adaptive_vcm.v32.codec import V32Codec
    from adaptive_vcm.v32.transport import pack_packet
    actual = np.full(actual_shape, 128, np.uint8)
    stream = StandardCodec('h264', 40, preset='ultrafast').roundtrip(actual).data
    wire = pack_packet(stream, task='od', codec='h264', source_shape=(1,64,64,3),
                       coded_shape=(1,64,64,3), fps=Fraction(25))
    with pytest.raises(ValueError, match='geometry|frames'):
        V32Codec('h264', 40, preset='ultrafast').decode_packet(wire)


def test_excess_receiver_output_is_bounded_before_reading_rgb_into_memory(monkeypatch):
    import subprocess
    from pathlib import Path
    from adaptive_vcm.codec import StandardCodec
    from adaptive_vcm.v32.codec import V32Codec
    from adaptive_vcm.v32.transport import pack_packet
    actual = np.full((40,32,32,3), 128, np.uint8)
    stream = StandardCodec('h264', 40, preset='ultrafast').roundtrip(actual).data
    wire = pack_packet(stream, task='od', codec='h264', source_shape=(1,32,32,3),
                       coded_shape=(1,32,32,3), fps=Fraction(25))
    real_run = subprocess.run
    captured = []
    def bounded_run(command, **kwargs):
        assert kwargs['stdout'] != subprocess.PIPE, 'decoder must not buffer unbounded RGB stdout'
        assert command[command.index('-frames:v')+1] == '2', 'one excess frame must remain detectable'
        assert int(command[command.index('-max_pixels')+1]) <= 16384  # bounded codec alignment allowance
        result = real_run(command, **kwargs)
        captured.append(Path(command[-1]).stat().st_size)
        return result
    monkeypatch.setattr(subprocess, 'run', bounded_run)
    with pytest.raises(ValueError, match='geometry|frames'):
        V32Codec('h264', 40, preset='ultrafast').decode_packet(wire)
    assert captured and max(captured) <= 2 * 32 * 32 * 3


def test_resolution_change_inside_stream_is_rejected_before_reshaping():
    import subprocess
    from adaptive_vcm.codec import StandardCodec
    from adaptive_vcm.v32.codec import V32Codec
    from adaptive_vcm.v32.transport import pack_packet
    encoder = StandardCodec('h264',40,preset='ultrafast')
    a = encoder.roundtrip(np.full((1,32,32,3),128,np.uint8)).data
    b = encoder.roundtrip(np.full((1,16,64,3),128,np.uint8)).data
    wire = pack_packet(a+b,task='ot',codec='h264',source_shape=(2,32,32,3),
                       coded_shape=(2,32,32,3),fps=Fraction(25))
    with pytest.raises((ValueError,subprocess.CalledProcessError)):
        V32Codec('h264',40,preset='ultrafast').decode_packet(wire)
