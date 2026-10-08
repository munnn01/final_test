"""Standard codec plus a separately parsed, paid ROI reconstruction recipe."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import subprocess
import time

import numpy as np

from ..codec import StandardCodec
from .roi import inverse_retarget
from .transport import pack_packet, unpack_packet


@dataclass(frozen=True)
class Packet:
    data: bytes
    decoded: np.ndarray
    coded_shape: tuple
    elementary_bytes: int
    metadata_bytes: int
    seconds: float

    @property
    def coded_bytes(self):
        return len(self.data)

    @property
    def packet_sha256(self):
        return hashlib.sha256(self.data).hexdigest()


class V32Codec:
    def __init__(self, codec: str, qp: int, preset: str = 'medium', fps: Fraction = Fraction(25)):
        if not isinstance(fps, Fraction) or fps <= 0:
            raise ValueError('V32 requires rational FPS')
        self.codec, self.qp, self.fps = codec, qp, fps
        self.standard = StandardCodec(codec, qp, preset=preset, fps=fps)

    def roundtrip(self, rgb, *, source_shape, task, grids=(), encoded=None):
        start = time.perf_counter()
        # Validate wire recipe before touching the encoder.
        pack_packet(b'validate', task=task, codec=self.codec, source_shape=source_shape,
                    coded_shape=rgb.shape, fps=self.fps, grids=grids)
        result = self.standard.roundtrip(rgb) if encoded is None else encoded
        wire = pack_packet(result.data, task=task, codec=self.codec, source_shape=source_shape,
                           coded_shape=rgb.shape, fps=self.fps, grids=grids)
        received = self.decode_packet(wire)
        return Packet(wire, received.decoded, rgb.shape, len(result.data),
                      received.metadata_bytes, time.perf_counter() - start)

    def decode_packet(self, wire: bytes):
        start = time.perf_counter()
        packet = unpack_packet(wire)
        if packet.codec != self.codec or packet.fps != self.fps:
            raise ValueError('V32 receiver codec/FPS mismatch')
        fmt = 'h264' if packet.codec == 'h264' else 'hevc'
        command = [self.standard.ffmpeg, '-nostdin', '-v', 'error', '-xerror',
                   '-err_detect', 'explode', '-threads', '2', '-f', fmt, '-i', 'pipe:0',
                   '-vsync', '0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
        output = subprocess.run(command, input=packet.stream, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=True, timeout=self.standard.timeout).stdout
        expected = int(np.prod(packet.coded_shape))
        if len(output) != expected:
            raise ValueError('VC32 decoded stream disagrees with transmitted geometry')
        rgb = np.frombuffer(output, np.uint8).reshape(packet.coded_shape).copy()
        if packet.grids:
            rgb = inverse_retarget(rgb, packet.grids, packet.source_shape)
        rgb.setflags(write=False)
        return Packet(wire, rgb, packet.coded_shape, len(packet.stream), packet.metadata_bytes,
                      time.perf_counter() - start)
