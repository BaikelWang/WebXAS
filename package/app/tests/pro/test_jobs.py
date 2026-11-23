"""Fault tests use a private command factory, never a product sleep endpoint."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import pytest

from xasweb.pro.jobs import JobError, JobService, atomic_json, checked_uuid
from xasweb.pro.worker import JobContext, HANDLERS


FAKE_WORKER = r'''
import json, pathlib, subprocess, sys, time
directory = pathlib.Path(sys.argv[1])
snapshot = json.loads((directory/'input.json').read_bytes())
payload = snapshot['payload']
mode = payload.get('mode', 'echo')
(directory/'heartbeat.json').write_text(json.dumps({'stage':'working','progress':0.5,'updated':'test-heartbeat'}))
if mode == 'fail':
    (directory/'error.json').write_text(json.dumps({'code':'controlled_failure','message':'expected fault'}))
    raise SystemExit(7)
if mode == 'child':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    (directory/'child.pid').write_text(str(child.pid))
if mode in ('hold', 'child'):
    (directory/'ready').write_text('ready')
    while not (directory/'release').exists():
        time.sleep(0.01)
result = {'value': payload.get('value'), 'project_id': snapshot['project_id']}
(directory/'output.json').write_text(json.dumps(result))
'''


def eventually(check, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("Condition was not reached before the bounded test deadline")


@pytest.fixture
def service(tmp_path):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER, encoding="utf-8")
    with JobService(tmp_path / "jobs", command_factory=lambda directory, record: [sys.executable, str(worker), str(directory)]) as value:
        yield value


def directory(service, job):
    return service.root / job['project_id'] / 'jobs' / job['job_id']


def terminal(service, job):
    return eventually(lambda: (r if (r := service.get(job['project_id'], job['job_id']))['state'] in
                               {'succeeded','failed','cancelled','timed_out','interrupted'} else None))


def alive(pid):
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, 0, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    try:
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def test_snapshot_is_immutable_and_success_result_is_checked(service):
    project = str(uuid4())
    payload = {'value': {'n': 12}}
    job = service.submit(project, 'fit', payload)
    payload['value']['n'] = 99
    result = terminal(service, job)
    assert result['state'] == 'succeeded', result
    assert service.result(project, job['job_id'])['value'] == {'n': 12}
    raw = (directory(service, job)/'input.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == job['snapshot_sha256']
    assert result['created'] and result['started'] and result['finished']
    assert result['result_ref']['sha256']
    (directory(service, job)/'output.json').write_text('{}')
    with pytest.raises(JobError, match='哈希'):
        service.result(project, job['job_id'])


def test_scheduler_is_serial_and_queued_cancel_never_starts(service):
    project = str(uuid4())
    first = service.submit(project, 'fit', {'mode':'hold'})
    eventually(lambda: (directory(service, first)/'ready').exists())
    second = service.submit(project, 'fit', {'value':2})
    assert service.get(project, second['job_id'])['state'] == 'queued'
    cancelled = service.cancel(project, second['job_id'])
    assert cancelled['state'] == 'cancelled' and cancelled['started'] is None
    (directory(service, first)/'release').touch()
    assert terminal(service, first)['state'] == 'succeeded'
    assert service.get(project, second['job_id'])['state'] == 'cancelled'
    assert not (directory(service, second)/'output.json').exists()


def test_cancel_kills_own_worker_tree_and_cannot_publish_late_result(service):
    project = str(uuid4())
    job = service.submit(project, 'fit', {'mode':'child'})
    eventually(lambda: (directory(service, job)/'ready').exists())
    child_pid = int((directory(service, job)/'child.pid').read_text())
    worker_pid = service.get(project, job['job_id'])['worker_pid']
    assert alive(child_pid)
    service.cancel(project, job['job_id'])
    eventually(lambda: not alive(worker_pid) and not alive(child_pid))
    # Even an externally injected late output does not turn cancellation into success.
    (directory(service, job)/'output.json').write_text('{"late":true}')
    assert service.get(project, job['job_id'])['state'] == 'cancelled'
    with pytest.raises(JobError, match='成功结果'):
        service.result(project, job['job_id'])


def test_hard_timeout_and_heartbeat(service):
    project = str(uuid4())
    job = service.submit(project, 'feff', {'mode':'hold'}, timeout_seconds=1.5)
    eventually(lambda: service.get(project, job['job_id'])['heartbeat'])
    result = terminal(service, job)
    assert result['state'] == 'timed_out'
    assert result['heartbeat']['stage'] == 'working'
    assert result['result_ref'] is None
    eventually(lambda: not alive(result['worker_pid']))


def test_worker_failure_is_not_success(service):
    project = str(uuid4())
    job = service.submit(project, 'preprocess', {'mode':'fail'})
    result = terminal(service, job)
    assert result['state'] == 'failed'
    assert result['error']['code'] == 'controlled_failure'
    assert result['result_ref'] is None


def test_project_isolation_and_validation(service):
    a, b = str(uuid4()), str(uuid4())
    job = service.submit(a, 'fit', {'value':1})
    terminal(service, job)
    assert service.list(b) == []
    with pytest.raises(JobError) as exc:
        service.get(b, job['job_id'])
    assert exc.value.status_code == 404
    with pytest.raises(JobError):
        service.submit('../outside', 'fit', {})
    with pytest.raises(JobError):
        service.submit(a, 'arbitrary.module.function', {})
    with pytest.raises(JobError):
        service.submit(a, 'fit', {'value':float('nan')})
    with pytest.raises(JobError):
        service.submit(a, 'fit', {}, timeout_seconds=0)
    with pytest.raises(JobError) as scope_error:
        service.submit(a, 'fit', {'project_id':b})
    assert scope_error.value.code=='project_scope_mismatch'


def test_restart_marks_unfinished_records_interrupted(tmp_path):
    project, job_id = str(uuid4()), str(uuid4())
    folder = tmp_path/project/'jobs'/job_id
    folder.mkdir(parents=True)
    atomic_json(folder/'job.json', {'project_id':project,'job_id':job_id,'state':'running',
                                  'created':'before-restart','result_ref':{'path':'output.json'}})
    with JobService(tmp_path) as value:
        record = value.get(project,job_id)
        assert record['state'] == 'interrupted'
        assert record['result_ref'] is None
        assert record['error']['code'] == 'service_restarted'


def test_modified_snapshot_is_rejected_before_execution(service):
    project = str(uuid4())
    first = service.submit(project, 'fit', {'mode':'hold'})
    eventually(lambda: (directory(service, first)/'ready').exists())
    job = service.submit(project, 'fit', {'value':1})
    (directory(service,job)/'input.json').write_text('{}')
    (directory(service,first)/'release').touch()
    assert terminal(service,first)['state']=='succeeded'
    result = terminal(service,job)
    assert result['state']=='failed'
    assert result['error']['code']=='snapshot_hash_mismatch'
    assert result['started'] is None


def test_shutdown_interrupts_active_and_queued_jobs(service):
    project = str(uuid4())
    active = service.submit(project, 'fit', {'mode':'hold'})
    eventually(lambda: (directory(service,active)/'ready').exists())
    queued = service.submit(project, 'fit', {})
    service.close()
    assert service.get(project,active['job_id'])['state']=='interrupted'
    assert service.get(project,queued['job_id'])['state']=='interrupted'
    with pytest.raises(JobError):
        service.submit(project,'fit',{})


def test_worker_context_and_registry(tmp_path):
    context=JobContext(tmp_path,'a'*64)
    context.heartbeat('feff',0.5)
    assert json.loads((tmp_path/'heartbeat.json').read_text())['stage']=='feff'
    with pytest.raises(JobError):
        context.heartbeat('stage',float('nan'))
    assert set(HANDLERS)=={'preprocess','fit','feff','sensitivity','auto','auto_export'}
    assert HANDLERS['auto']==('xasweb.pro.workflows','run_job')
    assert HANDLERS['auto_export']==('xasweb.pro.delivery','run_job')


def test_success_also_cleans_remaining_children(service):
    project = str(uuid4())
    job = service.submit(project, 'feff', {'mode':'child'})
    eventually(lambda: (directory(service,job)/'ready').exists())
    child_pid = int((directory(service,job)/'child.pid').read_text())
    (directory(service,job)/'release').touch()
    assert terminal(service,job)['state']=='succeeded'
    eventually(lambda: not alive(child_pid))


def test_lazy_service_is_reused_and_closed(monkeypatch,tmp_path):
    from xasweb.pro.jobs import close_job_service, get_job_service
    monkeypatch.setenv('XASWEB_JOB_ROOT',str(tmp_path/'singleton'))
    close_job_service()
    try:
        first=get_job_service()
        assert get_job_service() is first
        close_job_service()
        assert not first._thread.is_alive()
        assert get_job_service() is not first
    finally:
        close_job_service()


def test_job_api_uses_service_and_validates_operation(service,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from xasweb.pro import job_api

    class KnownProjects:
        def read(self,project_id):
            checked_uuid(project_id)
            from types import SimpleNamespace
            return SimpleNamespace(revision=0)

    monkeypatch.setattr(job_api,'ProjectStore',KnownProjects)
    monkeypatch.setattr(job_api,'get_job_service',lambda:service)
    application=FastAPI()
    application.include_router(job_api.router)
    project,other=str(uuid4()),str(uuid4())
    with TestClient(application) as client:
        response=client.post(f'/api/pro/projects/{project}/jobs',json={
            'operation':'fit','payload':{'mode':'hold','expected_revision':0},'timeout_seconds':10.0})
        assert response.status_code==202,response.text
        job=response.json()
        assert client.get(f'/api/pro/projects/{other}/jobs/{job["job_id"]}').status_code==404
        assert client.post(f'/api/pro/projects/{project}/jobs',json={
            'operation':'os.system','payload':{}}).status_code==422
        assert client.post(f'/api/pro/projects/{project}/jobs',json={
            'operation':'fit','payload':{'project_id':other,'expected_revision':0}}).status_code==409
        assert client.post(f'/api/pro/projects/{project}/jobs',json={
            'operation':'fit','payload':{'expected_revision':99}}).status_code==409
        assert client.post(f'/api/pro/projects/{project}/jobs',json={
            'operation':'fit','payload':{}}).status_code==400
        assert client.get(f'/api/pro/projects/{project}/jobs/{job["job_id"]}/result').status_code==409
        cancelled=client.post(f'/api/pro/projects/{project}/jobs/{job["job_id"]}/cancel')
        assert cancelled.status_code==200
        assert cancelled.json()['state']=='cancelled'


def test_production_worker_rejects_unknown_operation(tmp_path,monkeypatch):
    from xasweb.pro import worker
    source={'operation':'arbitrary.module','payload':{}}
    atomic_json(tmp_path/'input.json',source)
    digest=hashlib.sha256((tmp_path/'input.json').read_bytes()).hexdigest()
    def forbidden_import(*args):
        raise AssertionError('Unknown request reached dynamic import')
    monkeypatch.setattr(worker.importlib,'import_module',forbidden_import)
    assert worker.main([str(tmp_path),digest])==1
    assert json.loads((tmp_path/'error.json').read_text(encoding='utf-8'))['code']=='unknown_operation'
    assert not (tmp_path/'output.json').exists()


@pytest.mark.skipif(os.name!='nt',reason='Windows extended-length path regression')
def test_deep_windows_job_directory(tmp_path):
    # The project/job UUID components can push an otherwise ordinary workspace
    # path beyond 260 characters; this must work in the service itself.
    root=tmp_path/('nested-workspace-'*6)/('nested-runtime-'*6)
    worker=tmp_path/'deep_worker.py'
    worker.write_text(FAKE_WORKER,encoding='utf-8')
    with JobService(root,command_factory=lambda d,r:[sys.executable,str(worker),str(d)]) as value:
        project=str(uuid4())
        job=value.submit(project,'fit',{'value':'deep'})
        record=terminal(value,job)
        assert record['state']=='succeeded',record
        assert value.result(project,job['job_id'])['value']=='deep'
