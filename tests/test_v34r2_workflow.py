import json
import pytest


def test_dispatch_waits_for_verified_parent_and_respects_two_sessions():
    from scripts.v34r2_continue_workflow import ready_jobs
    jobs=[dict(id='a',account='baoancut',depends=[]),dict(id='b',account='baoancut',depends=['a']),
          dict(id='c',account='qktttttttttt',depends=[])]
    states={'a':dict(status='COMPLETE',stage_verified=False)}
    assert [j['id'] for j in ready_jobs(jobs,states,{'qktttttttttt':['x','y']})]==[]
    states['a']['stage_verified']=True
    assert [j['id'] for j in ready_jobs(jobs,states,{'qktttttttttt':['x','y']})]==['b']
    states['b']=dict(status='AMBIGUOUS')
    assert ready_jobs(jobs,states,{'qktttttttttt':['x','y']})==[]


def test_dispatch_reserves_slots_for_jobs_selected_in_same_pass():
    from scripts.v34r2_continue_workflow import ready_jobs
    jobs=[dict(id=str(i),account='baoancut',depends=[]) for i in range(4)]
    assert [j['id'] for j in ready_jobs(jobs,{}, {})]==['0','1']


def test_stage_verification_checks_execution_scope_and_full_checkpoint_set(tmp_path):
    from scripts.v34r2_continue_workflow import verify_stage
    entry=dict(experiment='ot',stage='evaluate',scope=dict(codecs=['h264'],domains=['ot'],splits=['test']))
    (tmp_path/'stage_complete.json').write_text(json.dumps(dict(status='complete',stage='evaluate',experiment='ot',
        scope={**entry['scope'],'settings':{'qps':[30,35,40,45]},'checkpoint_sha256':['a'*64,'b'*64,'c'*64]})))
    assert verify_stage(tmp_path,entry)['status']=='complete'
    value=json.loads((tmp_path/'stage_complete.json').read_text())
    value['scope']['codecs']=['h265']
    (tmp_path/'stage_complete.json').write_text(json.dumps(value))
    with pytest.raises(ValueError): verify_stage(tmp_path,entry)


def test_interrupted_push_intent_is_recoverable_without_another_push(tmp_path,monkeypatch):
    from scripts import v34r2_continue_kaggle as payload
    from scripts.v34r2_continue_workflow import checked_submit
    from tests.test_v34_payload import fixture_source
    monkeypatch.setattr(payload,'source_snapshot',fixture_source)
    parent=dict(kind='kernel',ref='trnhlng/v34-r2-ot-combined-qp4-b37c4937/1',
        files={n:'a'*64 for n in ['measurements.json','origin.json','cohorts.json',
                                 'controller_303401.pth','controller_303402.pth','controller_303403.pth']})
    proof=payload.prepare('ot','qktttttttttt',tmp_path,parent=parent,stage='evaluate',
                          codecs=['h264'],splits=['test'],domains=['ot'])
    receipt=tmp_path/'submission.json'
    receipt.write_text(json.dumps(dict(status='AMBIGUOUS',handle=proof['handle'])))
    class Offline:
        def call(self,*args,**kwargs): raise AssertionError('an interrupted push must never be retried blindly')
    value=checked_submit(Offline(),tmp_path)
    assert value['status']=='AMBIGUOUS' and value['version'] is None
    receipt.write_text(json.dumps(dict(status='AMBIGUOUS',handle='another/notebook')))
    with pytest.raises(ValueError,match='receipt'): checked_submit(Offline(),tmp_path)
