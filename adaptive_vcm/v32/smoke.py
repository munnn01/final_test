"""Real-codec synthetic transport check; contains no task quality experiment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..evaluate import write_json
from .codec import V32Codec
from .evaluate import empty_output
from .protocol import load_config, provenance
from .roi import build_grids, retarget, simplify
from .transport import unpack_packet


def run(out):
    out = empty_output(out)
    rng = np.random.default_rng(303201)
    rgb = rng.integers(0, 256, (1, 64, 64, 3), dtype=np.uint8)
    protection = np.zeros((1, 64, 64), np.float32)
    protection[:, 24:40, 24:40] = 1.
    support = dict(protection=protection, motion=np.zeros_like(protection), cuts=np.array([True]))
    grids = build_grids(rgb.shape, support, factor=4, margin=0.)
    simplified = simplify(rgb, support, strength=1., margin=0.)
    records = []
    for codec in ('h264', 'h265'):
        engine = V32Codec(codec, 40, preset='ultrafast')
        for name, pixels, mapping in [('identity', rgb, ()), ('roi_simplify', simplified, ()),
                                      ('roi_retarget', retarget(simplified, grids), grids)]:
            packet = engine.roundtrip(pixels, source_shape=rgb.shape, task='od', grids=mapping)
            path = out / f'{codec}-{name}.v32'
            path.write_bytes(packet.data)
            receiver = engine.decode_packet(path.read_bytes())
            np.savez_compressed(out / f'{codec}-{name}-received.npz', rgb=receiver.decoded)
            records.append(dict(codec=codec, candidate=name, source_shape=list(rgb.shape),
                coded_shape=list(packet.coded_shape), reconstructed_shape=list(receiver.decoded.shape),
                elementary_bytes=packet.elementary_bytes, metadata_bytes=packet.metadata_bytes,
                total_bytes=packet.coded_bytes, packet_sha256=packet.packet_sha256,
                received_bytes_redecode_matches=bool(np.array_equal(packet.decoded, receiver.decoded))))
    result = dict(scope='synthetic_transport_only', quality_measured=False, records=records)
    write_json(out / 'summary.json', result)
    write_json(out / 'manifest.json', provenance(load_config()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--decode-packet', type=Path, help='Receive one .v32 packet into a new output directory')
    args = parser.parse_args()
    if args.decode_packet:
        out = empty_output(args.out)
        wire = args.decode_packet.read_bytes()
        header = unpack_packet(wire)
        packet = V32Codec(header.codec, 40, fps=header.fps).decode_packet(wire)
        np.savez_compressed(out / 'received.npz', rgb=packet.decoded)
        write_json(out / 'receiver.json', dict(codec=header.codec, task=header.task,
            source_shape=header.source_shape, coded_shape=header.coded_shape, total_bytes=len(wire),
            metadata_bytes=header.metadata_bytes, packet_sha256=packet.packet_sha256))
    else:
        result = run(args.out)
        for row in result['records']:
            print(f"{row['codec']} {row['candidate']}: {row['coded_shape']} -> {row['reconstructed_shape']}, {row['total_bytes']} bytes")


if __name__ == '__main__':
    main()
