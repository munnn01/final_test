"""Kinetics benchmark: frozen RF preprocessing, standard decode, frozen Swin."""
import argparse
from pathlib import Path
import time
import torch
from ..v32.selection import Candidate
from ..v32.evaluate import run as evaluate
from ..v32rf.swin import SwinAnalyzer
from .frozen_rf import load_frozen_ar
from .protocol import codec_factory, settings
from .receiver import SharedReceiver
from ..v32.protocol import ROOT


FAMILIES = {'rf_real':{'controls','rf_real_plain'}, 'rf_real_forced':{'controls','rf_real_plain'}}


def prepare_extension(rgb, support, codec, models, cfg):
    before = time.perf_counter()
    candidate = Candidate('rf_real_rgb', models['real'].pixels(rgb,support,codec), 'rf_real_plain')
    return dict(extra_candidates=[candidate], extra_families=FAMILIES,
        forced_names={'rf_real_forced':'rf_real_rgb'}, unconditional_extra=True,
        enable_retarget=False), time.perf_counter()-before


def run(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_frozen_ar(args.real_checkpoint, args.rf_runtime, device)
    swin = SwinAnalyzer(device)
    receiver = SharedReceiver(swin)
    def predict(packet, item, evaluators, arm, codec, qp):
        received = receiver.predict(packet, sequence=item['id'], arm=f'{codec}/q{qp}/{arm}', frame_start=1, tasks=('ar',))
        correct = {n:int(e.probabilities(packet.decoded).argmax()==item['label'])
                   for n,e in evaluators.items() if n != 'swin3d_t'}
        correct['swin3d_t'] = int(received['ar']['top1']==item['label'])
        return dict(correct=correct, shared_receiver=received)
    plugin = dict(models={'real':model}, families=FAMILIES, prepare=prepare_extension,
        evaluators=lambda d:{'swin3d_t':swin}, predict=predict,
        manifest=dict(scope='v32at_ar', active_tasks=['ar','ot'], protocol_v32at=settings(),
            primary_evaluator='swin3d_t', retarget_enabled=False, training=False,
            original_rf=model.manifest, swin_protocol='official K400 spatial transform; received 16-frame clip, exploratory'))
    return evaluate(args, extension=plugin, codec_factory=codec_factory('ar'))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--real-checkpoint',type=Path,required=True)
    p.add_argument('--rf-runtime',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['dev','test'],default='test')
    p.add_argument('--count',type=int,default=32)
    p.add_argument('--bootstrap',type=int,default=0)
    p.add_argument('--device',choices=['cpu','cuda'])
    p.set_defaults(task='ar',config=ROOT/'configs/v32at.json',checkpoint=None,v30_runtime=None,annotations=None)
    return p


if __name__ == '__main__': run(parser().parse_args())
