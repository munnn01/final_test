"""Frozen heads consume one decoded packet; mutable track state is per stream."""
import hashlib
import time
import numpy as np
from ..preprocessing import validate_clip
from ..v32.tracking import ReferenceTracker
from .protocol import validate_task


class SharedReceiver:
    def __init__(self, action, detector=None):
        self.action, self.detector = action, detector
        self.trackers = {}
        self._cache = {}
        for head in (action, detector):
            model = getattr(head, 'model', None)
            if model is not None:
                model.eval().requires_grad_(False)

    def predict(self, packet, *, sequence, arm, frame_start, tasks, tracker=None):
        if not tasks or len(set(tasks)) != len(tasks):
            raise ValueError('unique nonempty tasks required')
        for task in tasks: validate_task(task)
        if 'ot' in tasks and self.detector is None:
            raise ValueError('tracking requires detector')
        rgb = packet.decoded
        validate_clip(rgb)
        if type(frame_start) is not int or frame_start < 1:
            raise ValueError('positive frame_start required')
        key = (sequence, arm)
        state = None
        if 'ot' in tasks:
            state = tracker if tracker is not None else self.trackers.get(key, ReferenceTracker())
            if state.frame_number + 1 != frame_start:
                raise ValueError('tracking frames must be ordered and contiguous')
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        cache_key = (rgb.shape, rgb.dtype.str, digest)
        # Keep only the current decoded clip. Cache head predictions, never tracks.
        if self._cache.get('key') != cache_key:
            self._cache = {'key': cache_key}
        result = dict(packet_sha256=packet.packet_sha256, rgb_sha256=digest,
                      coded_bytes=packet.coded_bytes, frame_start=frame_start,
                      frame_end=frame_start+len(rgb)-1, frame_count=len(rgb),
                      latency_seconds={})
        if 'ar' in tasks:
            before = time.perf_counter()
            cached = 'ar' in self._cache
            if not cached:
                p = np.asarray(self.action.probabilities(rgb), float)
                if p.ndim != 1 or len(p)<2 or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-5):
                    raise ValueError('invalid action probabilities')
                self._cache['ar'] = p.copy()
            p = self._cache['ar']
            result['ar'] = dict(probabilities=p.tolist(), top1=int(p.argmax()),
                accuracy=None, bd_rate=None, ground_truth='not supplied to receiver', cached=cached)
            result['latency_seconds']['ar'] = time.perf_counter()-before
        if 'ot' in tasks:
            before = time.perf_counter()
            cached = 'detections' in self._cache
            if not cached:
                self._cache['detections'] = [self.detector.predict(frame[None]) for frame in rgb]
            frames = [state.step(d) for d in self._cache['detections']]
            self.trackers[key] = state
            result['ot'] = dict(frames=frames, detector_frame_calls=0 if cached else len(rgb), cached=cached)
            result['latency_seconds']['ot'] = time.perf_counter()-before
        return result
