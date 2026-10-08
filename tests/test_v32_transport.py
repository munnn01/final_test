import importlib
from fractions import Fraction
import struct

import pytest

from adaptive_vcm.v32.roi import build_grids
from tests.test_v32_roi import pixels, support


def transport():
    return importlib.import_module('adaptive_vcm.v32.transport')


def test_wire_carries_all_grid_edges_and_counts_metadata_as_transmitted_bytes():
    rgb = pixels()
    grids = build_grids(rgb.shape, support(), factor=4, margin=0)
    wire = transport().pack_packet(b'elementary', task='od', codec='h264',
        source_shape=rgb.shape, coded_shape=(1, 28, 28, 3), fps=Fraction(25), grids=grids)
    packet = transport().unpack_packet(wire)
    assert packet.stream == b'elementary'
    assert packet.grids == grids
    assert packet.source_shape == rgb.shape
    assert packet.coded_shape == (1, 28, 28, 3)
    assert packet.fps == Fraction(25)
    assert packet.metadata_bytes == len(wire) - 10
    assert packet.metadata_bytes > transport().HEADER_BYTES


def test_identity_also_pays_header_and_does_not_contain_roi_grid():
    shape = (1, 64, 64, 3)
    wire = transport().pack_packet(b'identity', task='od', codec='h265',
        source_shape=shape, coded_shape=shape, fps=Fraction(25), grids=())
    packet = transport().unpack_packet(wire)
    assert packet.metadata_bytes == transport().HEADER_BYTES
    assert packet.grids == ()
    assert len(wire) == 8 + transport().HEADER_BYTES


@pytest.mark.parametrize('change', ['truncate', 'append', 'corrupt', 'huge_length'])
def test_malformed_packet_is_rejected_before_decode(change):
    shape = (1, 64, 64, 3)
    wire = transport().pack_packet(b'identity', task='od', codec='h264',
        source_shape=shape, coded_shape=shape, fps=Fraction(25), grids=())
    if change == 'truncate':
        wire = wire[:-1]
    elif change == 'append':
        wire += b'!'
    elif change == 'corrupt':
        wire = wire[:-1] + bytes([wire[-1] ^ 1])
    else:
        wire = wire[:28] + struct.pack('>I', 0xffffffff) + wire[32:]
    with pytest.raises(ValueError):
        transport().unpack_packet(wire)


def test_incomplete_scene_mapping_and_plain_od_resize_are_rejected():
    rgb = pixels(2)
    grids = build_grids(pixels().shape, support(), factor=4, margin=0)
    with pytest.raises(ValueError):
        transport().pack_packet(b'video', task='ar', codec='h264',
            source_shape=rgb.shape, coded_shape=(2, 28, 28, 3), fps=Fraction(25), grids=grids)
    with pytest.raises(ValueError):
        transport().pack_packet(b'image', task='od', codec='h264',
            source_shape=(1, 64, 64, 3), coded_shape=(1, 32, 32, 3), fps=Fraction(25), grids=())
