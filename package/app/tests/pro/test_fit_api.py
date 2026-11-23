import io
import time
from zipfile import ZipFile
import numpy as np
import pytest
from fastapi.testclient import TestClient
from xasweb.app import app
from xasweb.pro.jobs import close_job_service


def execute(client,project):
    pid=project['id']
    submitted=client.post(f'/api/pro/projects/{pid}/fits',json={'expected_revision':project['revision'],'model_id':'synthetic_joint','max_evaluations':500,'timeout_seconds':120.0})
    assert submitted.status_code==202,submitted.text
    jid=submitted.json()['job_id'];deadline=time.monotonic()+130
    while time.monotonic()<deadline:
        response=client.get(f'/api/pro/projects/{pid}/jobs/{jid}')
        assert response.status_code==200,response.text
        job=response.json()
        if job['state'] in ('succeeded','failed','cancelled','timed_out','interrupted'):break
        time.sleep(.05)
    else:pytest.fail('Fit exceeded registered integration deadline')
    assert job['state']=='succeeded',job
    committed=client.post(f'/api/pro/projects/{pid}/jobs/{jid}/commit')
    assert committed.status_code==200,committed.text
    project=committed.json();run=project['runs'][-1]
    assert run['result']['converged'] and len(run['result']['datasets'])==3
    assert run['result']['statistics']['n_variables']==8
    values={x['id']:x['value'] for x in run['result']['parameters']}
    np.testing.assert_allclose(values['amp'],.86,atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(values['dr'],.012,atol=1e-6,rtol=1e-6)
    return project


def test_joint_job_export_restart_recompute_and_report(tmp_path,monkeypatch):
    close_job_service()
    monkeypatch.setenv('XASWEB_PROJECT_ROOT',str(tmp_path/'projects'))
    monkeypatch.setenv('XASWEB_JOB_ROOT',str(tmp_path/'jobs'))
    with TestClient(app) as client:
        demo=client.post('/api/pro/projects/demo')
        assert demo.status_code==200,demo.text
        project=execute(client,demo.json())
        pid=project['id'];run=project['runs'][0]
        report=client.post(f"/api/pro/projects/{pid}/reports",params={'run_id':run['id']})
        assert report.status_code==200,report.text[:300]
        (tmp_path/'report.zip').write_bytes(report.content)
        with ZipFile(io.BytesIO(report.content)) as z:
            assert z.testzip() is None
            assert any(x.endswith('.svg') for x in z.namelist())
            assert any(x.endswith('.csv') for x in z.namelist())
        archive=client.post(f'/api/pro/projects/{pid}/export?expected_revision={project["revision"]}')
        assert archive.status_code==200,archive.text[:300]
    # TestClient shutdown terminates the old job service. The next context uses new workers.
    with TestClient(app) as client:
        imported=client.post('/api/pro/projects/import',files={'file':('joint.zip',archive.content)})
        assert imported.status_code==200,imported.text
        restored=execute(client,imported.json())
        assert len(restored['runs'])==2
        assert restored['runs'][0]['input_snapshot']==restored['runs'][1]['input_snapshot']
        values1={x['id']:x['value'] for x in restored['runs'][0]['result']['parameters']}
        values2={x['id']:x['value'] for x in restored['runs'][1]['result']['parameters']}
        for name in values1:np.testing.assert_allclose(values1[name],values2[name],rtol=1e-7,atol=1e-9)
        compared=client.post(f"/api/pro/projects/{restored['id']}/compare",json={'run_ids':[r['id'] for r in restored['runs']]})
        assert compared.status_code==200,compared.text
        assert compared.json()['comparability'] and not compared.json()['unique_structure']
