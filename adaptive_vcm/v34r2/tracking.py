"""Transactional ByteTrack prefixes. No candidate can mutate another probe."""
import copy
from dataclasses import dataclass
import hashlib
import json
from ..v34.tracking import ByteTracker, track_frames


@dataclass
class Probe:
    tracker: object
    frames: list
    current: list
    owner: object
    generation: int
    signature: str


class Prefix:
    def __init__(self, tracker=None, context_frames=64):
        if context_frames < 1: raise ValueError('positive context required')
        self.tracker = tracker if tracker is not None else ByteTracker()
        self.context_frames = context_frames
        self.frames = []
        self.generation = 0
        self.signature = hashlib.sha256(b'v34-r2-empty-prefix').hexdigest()

    def snapshot(self):
        snapshot=copy.copy(self)
        snapshot.tracker=copy.copy(self.tracker)
        snapshot.tracker.tracker=copy.deepcopy(self.tracker.tracker)
        snapshot.frames=copy.deepcopy(self.frames)
        return snapshot

    def probe(self, predictions):
        # The adapter holds the supervision module; clone only its mutable state.
        fork = copy.copy(self.tracker)
        fork.tracker = copy.deepcopy(self.tracker.tracker)
        current = track_frames(predictions, fork)
        native = [{k:v.tolist() for k,v in p.items() if k in ('boxes','scores','labels')} for p in predictions]
        signature = hashlib.sha256((self.signature+json.dumps(native,sort_keys=True,allow_nan=False)).encode()).hexdigest()
        return Probe(fork, self.frames+current, current, self, self.generation, signature)

    def commit(self, probe):
        if probe.owner is not self or probe.generation != self.generation:
            raise ValueError('foreign or stale tracker probe')
        self.tracker = probe.tracker
        self.frames = probe.frames[-self.context_frames:]
        self.signature = probe.signature
        self.generation += 1
