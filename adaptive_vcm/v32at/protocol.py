"""Explicit two-task scope and shared video codec conditions."""
from fractions import Fraction
from ..v32.codec import V32Codec
from . import SCHEMA


def validate_task(task):
    if task not in ('ar', 'ot'): raise ValueError('V32-AT supports ar and ot only')
    return task


def settings():
    return dict(schema=SCHEMA, tasks=['ar','ot'], seed=303241,
                codecs=['h264','h265'], qps=[30,35,40,45,50], preset='medium',
                gop_settings=dict(gop=16,bframes=0,scenecut=False,closed_gop=True),
                ar_count=32, tracking_sequences=['MOT20-01'], tracking_frames=128, tracking_size=640)


def codec_factory(task):
    validate_task(task)
    def create(codec,qp,preset='medium',fps=Fraction(25)):
        if preset != 'medium': raise ValueError('V32-AT requires medium preset')
        return V32Codec(codec,qp,preset,fps,gop_settings=settings()['gop_settings'])
    return create
