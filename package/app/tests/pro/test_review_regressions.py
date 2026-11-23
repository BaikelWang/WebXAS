import json
import os
from pathlib import Path
import threading
from uuid import uuid4
import pytest
from xasweb.pro import projects
from xasweb.pro.jobs import JobService,JobError,atomic_json
from xasweb.pro.projects import ProjectStore,ProjectError,canonical
from xasweb.pro.contracts import Dataset,Project
from xasweb.pro.preprocessing import process_dataset,merge_project_scans


@pytest.mark.skipif(os.name!='nt',reason='Windows shared-reader regression')
def test_transient_shared_read_does_not_fail_atomic_heartbeat(tmp_path,monkeypatch):
    target=tmp_path/'heartbeat.json';target.write_text('{}')
    replace=os.replace;attempts=[]
    def transient(source,destination):
        attempts.append(1)
        if len(attempts)<3:
            error=PermissionError('temporary shared reader');error.winerror=5
            raise error
        replace(source,destination)
    monkeypatch.setattr(projects.os,'replace',transient)
    atomic_json(target,{'stage':'fitting'})
    assert json.loads(target.read_bytes())=={'stage':'fitting'} and len(attempts)==3


def test_job_input_snapshot_hash_checked_before_result_commit(tmp_path):
    # A stored completed job suffices; this test does not launch a science worker.
    service=JobService.__new__(JobService);service.root=tmp_path;service._lock=threading.RLock()
    pid,jid=str(uuid4()),str(uuid4())
    folder=service._directory(pid,jid);folder.mkdir(parents=True)
    value={'schema_version':1,'project_id':pid,'operation':'fit','payload':{'project_id':pid,'expected_revision':0}}
    raw=canonical(value)
    import hashlib
    (folder/'input.json').write_bytes(raw)
    service._save({'project_id':pid,'job_id':jid,'operation':'fit','snapshot_sha256':hashlib.sha256(raw).hexdigest()})
    assert service.input_snapshot(pid,jid)==value
    (folder/'input.json').write_bytes(raw+b' ')
    with pytest.raises(JobError,match='哈希'):
        service.input_snapshot(pid,jid)


def test_header_identity_and_imported_processing_state_are_enforced(tmp_path):
    source=Path(__file__).resolve().parents[3]/'validation/sample-spectrum.xdi'
    raw=source.read_text()
    store=ProjectStore(tmp_path);p=store.create()
    ref=store.put_asset(p.id,raw.encode(),'Fe.xdi')
    p.datasets=[Dataset(id='d',filename='Fe.xdi',raw_file=ref,raw_text=raw,element='Cu',edge='K')]
    p=store.save(p,p.revision)
    with pytest.raises(ProjectError,match='文件头'):
        process_dataset(store,p.id,'d')
    p.datasets[0].element='Fe'
    corrected='# Processing.state: normalized\n'+raw
    p.datasets[0].raw_text=corrected;p.datasets[0].raw_file=store.put_asset(p.id,corrected.encode(),'normalized.dat')
    p=store.save(p,p.revision)
    with pytest.raises(ProjectError,match='归一化'):
        process_dataset(store,p.id,'d')


@pytest.mark.parametrize('changed',[{'correction_states':{'self_absorption':'corrected'}},{'mapping':{'mode':'fluorescence'}}])
def test_merge_rejects_different_measurement_or_correction_state(tmp_path,changed):
    source=Path(__file__).resolve().parents[3]/'validation/sample-spectrum.xdi'
    raw=source.read_text();store=ProjectStore(tmp_path);p=store.create()
    ds=[]
    for i in range(2):
        blob=(raw+f'# scan {i}\n').encode();ref=store.put_asset(p.id,blob,'scan.dat')
        value={'id':f'd{i}','filename':'scan.dat','raw_file':ref.model_dump(mode='json'),'raw_text':blob.decode(),
               'element':'Fe','edge':'K','sample_id':'s'}
        if i:value.update(changed)
        ds.append(Dataset.model_validate(value))
    p.datasets=ds;p=store.save(p,p.revision)
    with pytest.raises(ProjectError,match='修正状态'):
        merge_project_scans(store,p.id,['d0','d1'],p.revision)


def test_container_spectra_have_independent_primary_assets_and_preserved_upload(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from xasweb.app import app
    from xasweb.pro import api
    monkeypatch.setenv('XASWEB_PROJECT_ROOT',str(tmp_path))
    # This regression isolates the professional API's container identity logic;
    # the legacy parser has its own archived PRJ import fixtures.
    def parsed(blob,filename):
        return {'datasets':[{'raw_text':'100 1\n101 2\n','filename':'group one','kind':'mu','group_id':'g1'},
                            {'raw_text':'100 2\n101 3\n','filename':'group two','kind':'mu','group_id':'g2'}]}
    monkeypatch.setattr(api,'import_file',parsed)
    original=b'untouched source container'
    with TestClient(app) as client:
        p=client.post('/api/pro/projects',json={}).json()
        response=client.post(f"/api/pro/projects/{p['id']}/datasets?expected_revision=0",files={'file':('two.prj',original)})
        assert response.status_code==200,response.text
        data=response.json()['datasets']
        assert data[0]['raw_file']['sha256']!=data[1]['raw_file']['sha256']
        assert data[0]['metadata']['original_upload']==data[1]['metadata']['original_upload']
        store=ProjectStore(tmp_path)
        assert store.read_asset(p['id'],data[0]['metadata']['original_upload'])==original


def test_merged_import_without_lineage_is_not_an_independent_joint_observation(tmp_path):
    from test_fitting import fixture
    from xasweb.pro.larch_adapter import validate_snapshot,FitError
    from xasweb.pro.contracts import FitSpec
    _,_,spec,_,_=fixture(tmp_path)
    value=spec.model_dump(mode='json');value['datasets'][0]['metadata']['import_role']='merged'
    with pytest.raises(FitError) as found:
        validate_snapshot(FitSpec.model_validate(value))
    assert found.value.code=='unknown_merged_lineage'


def test_invalid_model_resource_reference_returns_located_error(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from xasweb.app import app
    monkeypatch.setenv('XASWEB_PROJECT_ROOT',str(tmp_path))
    with TestClient(app) as client:
        project=client.post('/api/pro/projects',json={}).json()
        response=client.post(f"/api/pro/projects/{project['id']}/models/validate",json={
            'expected_revision':0,'model':{'id':'bad_reference','bindings':[{
                'dataset_id':'missing','processing_revision_id':'missing',
                'paths':[{'id':'binding','path_id':'missing'}]}]}})
        assert response.status_code==422,response.text
        assert response.json()['code']=='contract_violation'
        assert response.json()['details']['errors']
