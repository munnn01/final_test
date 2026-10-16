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


def test_rejected_input_warning_is_not_a_successful_submission(tmp_path,monkeypatch):
    from scripts import v34r2_continue_kaggle as payload
    from scripts.v34r2_continue_workflow import checked_submit
    from tests.test_v34_payload import fixture_source
    monkeypatch.setattr(payload,'source_snapshot',fixture_source)
    parent=dict(kind='kernel',ref='trnhlng/v34-r2-ot-combined-qp4-b37c4937/1',
        files={n:'a'*64 for n in ['measurements.json','origin.json','cohorts.json',
                                 'controller_303401.pth','controller_303402.pth','controller_303403.pth']})
    proof=payload.prepare('ot','qktttttttttt',tmp_path,parent=parent,stage='evaluate',
                          codecs=['h264'],splits=['test'],domains=['ot'])
    class RejectedInput:
        secrets=[]
        def call(self,*args,**kwargs):
            return dict(exit_code=0,stderr='',stdout="The following are not valid kernel sources and could not be added to the kernel: ['parent/failed/1']\nKernel version 1 successfully pushed. Please check progress at https://www.kaggle.com/code/"+proof['handle'])
    value=checked_submit(RejectedInput(),tmp_path)
    assert value['status']=='INPUT_ERROR' and value['version']==1


def test_recovered_state_uses_private_dataset_owned_by_each_execution_account(tmp_path):
    from scripts.v34r2_continue_workflow import private_parent,manifest
    files={name:tmp_path/name for name in ['measurements.json','origin.json','cohorts.json',
        'controller_303401.pth','controller_303402.pth','controller_303403.pth']}
    for path in files.values(): path.write_bytes(b'verified recovered bytes')
    hashes=manifest(files)
    plan=dict(source_commit='a'*40,parents={'ot':{'files':hashes}},recovered_files={'ot':{n:str(p) for n,p in files.items()}})
    state=dict(jobs={},datasets={})
    class ReadyDataset:
        secrets=[]
        def __init__(self): self.calls=[]
        def call(self,account,args,**kwargs):
            self.calls.append((account,args))
            return dict(exit_code=0,stderr='',stdout='Your private Dataset is being created.' if args[1]=='create' else 'ready')
    client=ReadyDataset()
    for owner in ('qktttttttttt','huolgggnuyen'):
        parent=private_parent(client,tmp_path/'workflow',plan,state,owner,experiment='ot')
        assert parent['kind']=='dataset' and parent['ref'].startswith(owner+'/') and parent['files']==hashes
    assert len([c for c in client.calls if c[1][1]=='create'])==2
    assert all('--public' not in args and '--keep-tabular' in args for _,args in client.calls if args[1]=='create')
