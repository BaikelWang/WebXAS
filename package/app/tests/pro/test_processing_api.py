import json
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from xasweb.app import app
from xasweb.pro.jobs import close_job_service


def wait_job(client,pid,jid):
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        response=client.get(f'/api/pro/projects/{pid}/jobs/{jid}')
        assert response.status_code==200,response.text
        value=response.json()
        if value['state'] in ('succeeded','failed','cancelled','timed_out','interrupted'):
            return value
        time.sleep(.05)
    pytest.fail('Job did not terminate within test budget')


def test_real_processing_job_commits_once_and_exports_arrays(tmp_path,monkeypatch):
    close_job_service()
    monkeypatch.setenv('XASWEB_PROJECT_ROOT',str(tmp_path/'p'))
    monkeypatch.setenv('XASWEB_JOB_ROOT',str(tmp_path/'j'))
    sample=(Path(__file__).resolve().parents[3]/'validation/sample-spectrum.xdi').read_bytes()
    with TestClient(app) as client:
        project=client.post('/api/pro/projects',json={'title':'processing api'}).json();pid=project['id']
        added=client.post(f'/api/pro/projects/{pid}/datasets?expected_revision=0',files={'file':('sample.xdi',sample)})
        assert added.status_code==200,added.text
        project=added.json();did=project['datasets'][0]['id']
        submitted=client.post(f'/api/pro/projects/{pid}/processing',json={'expected_revision':1,'dataset_id':did,'parameters':{'kmax':10.0}})
        assert submitted.status_code==202,submitted.text
        jid=submitted.json()['job_id']
        job=wait_job(client,pid,jid)
        assert job['state']=='succeeded',job
        response=client.post(f'/api/pro/projects/{pid}/jobs/{jid}/commit')
        assert response.status_code==200,response.text
        result=response.json();assert len(result['processing'])==1 and result['revision']==2
        repeat=client.post(f'/api/pro/projects/{pid}/jobs/{jid}/commit')
        assert repeat.status_code==200 and repeat.json()['revision']==2
        ref=result['processing'][0]['arrays']['chi']
        array=client.get(f'/api/pro/projects/{pid}/arrays',params={'path':ref['path']})
        assert array.status_code==200 and len(array.json()['values'])==ref['shape'][0]
        exported=client.post(f'/api/pro/projects/{pid}/export?expected_revision=2')
        restored=client.post('/api/pro/projects/import',files={'file':('saved.zip',exported.content)})
        assert restored.status_code==200,restored.text
        assert restored.json()['processing']==result['processing']
        assert client.get(f'/api/pro/projects/{pid}/arrays',params={'path':'project.json'}).status_code==422


def test_old_session_cannot_be_misclassified_as_spectrum(tmp_path,monkeypatch):
    monkeypatch.setenv('XASWEB_PROJECT_ROOT',str(tmp_path))
    with TestClient(app) as client:
        project=client.post('/api/pro/projects',json={}).json()
        payload=json.dumps({'kind':'webxas-session','schema_version':3,'spectra':[]}).encode()
        response=client.post(f"/api/pro/projects/{project['id']}/datasets?expected_revision=0",files={'file':('results.json',payload)})
        assert response.status_code==422
        assert response.json()['code']=='project_requires_import'
