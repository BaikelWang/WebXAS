import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import pytest
from fastapi.testclient import TestClient
from xasweb.app import app
from xasweb.pro.contracts import Dataset, Project
from xasweb.pro.projects import ProjectStore, ProjectError, canonical
from xasweb.pro.legacy import import_session, import_old_bundle


def populated(tmp_path):
    store = ProjectStore(tmp_path)
    project = store.create('铜样本 项目')
    raw = b'# energy mu\n100 1\n101 2\n102 3\n'
    ref = store.put_asset(project.id, raw, '原始谱.dat')
    project.datasets.append(Dataset(id='ds1', filename='原始谱.dat', raw_file=ref, raw_text=raw.decode()))
    return store, store.save(project, expected_revision=0), raw


def rewrite_zip(payload, change):
    with ZipFile(io.BytesIO(payload)) as z:
        entries = {name:z.read(name) for name in z.namelist()}
    change(entries)
    output = io.BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as z:
        for name, blob in entries.items():
            z.writestr(name, blob)
    return output.getvalue()


def test_project_roundtrip_in_fresh_process(tmp_path):
    store, project, raw = populated(tmp_path/'one')
    blob = store.export(project.id, project.revision)
    path = tmp_path/'portable.zip'; path.write_bytes(blob)
    code = "from pathlib import Path;from xasweb.pro.projects import ProjectStore;import json; s=ProjectStore(r'%s');p=s.import_archive(Path(r'%s').read_bytes()); print(json.dumps({'datasets':len(p.datasets),'raw':s.read_asset(p.id,p.datasets[0].raw_file).decode(),'title':p.title},ensure_ascii=False))" % (tmp_path/'two', path)
    run = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'},timeout=60)
    assert run.returncode == 0, run.stderr
    result=json.loads(run.stdout)
    assert result=={'datasets':1,'raw':raw.decode(),'title':'铜样本 项目'}


def test_revision_conflict_and_immutable_history(tmp_path):
    store, project, _ = populated(tmp_path)
    old=(store.folder(project.id)/'revisions/00000000.json').read_bytes()
    with pytest.raises(ProjectError,match='已更新'):
        store.save(project, expected_revision=0)
    assert (store.folder(project.id)/'revisions/00000000.json').read_bytes()==old


def test_corrupt_and_missing_archive_rejected(tmp_path):
    store, project, _=populated(tmp_path/'one')
    blob=store.export(project.id)
    asset=project.datasets[0].raw_file.path
    for mutate in [lambda e:e.__setitem__(asset,b'corrupt'),lambda e:e.pop(asset),lambda e:e.__setitem__('../escape',b'x')]:
        target=ProjectStore(tmp_path/'two')
        with pytest.raises(ProjectError):
            target.import_archive(rewrite_zip(blob,mutate))
        assert not list(target.root.glob('*/project.json'))


def test_unknown_schema_and_windows_absolute_path_rejected(tmp_path):
    store, project, _=populated(tmp_path)
    blob=store.export(project.id)
    def newer(entries):
        manifest=json.loads(entries['manifest.json']);manifest['schema_version']=99
        entries['manifest.json']=canonical(manifest)
    with pytest.raises(ProjectError,match='版本'):
        store.import_archive(rewrite_zip(blob,newer))
    with pytest.raises(ProjectError):
        store.read_asset(project.id,{'path':'C:\\escape','sha256':'a'*64,'byte_size':1})


def test_old_schema3_restores_data_not_empty_success(tmp_path):
    raw='100 1\n101 2\n'
    saved={'kind':'webxas-session','schema_version':3,
           'sources':[{'name':'original.dat','encoding':'base64','data':base64.b64encode(raw.encode()).decode()}],
           'spectra':[{'filename':'original.dat','original_raw_text':raw,'context':{'element':'Cu'},'selected_fit_id':'historic'}]}
    store=ProjectStore(tmp_path)
    project=import_session(store,json.dumps(saved).encode())
    assert len(project.datasets)==1
    assert project.datasets[0].raw_text==raw
    assert project.recomputable is False
    assert project.migration_report[0].code=='fit_model_missing'
    assert store.read_asset(project.id,project.datasets[0].raw_file)==raw.encode()
    restored=ProjectStore(tmp_path/'restored').import_archive(store.export(project.id))
    assert restored.datasets[0].metadata['legacy_selected_fit_id']=='historic'


def test_empty_legacy_session_is_error(tmp_path):
    with pytest.raises(ProjectError,match='未包含'):
        import_session(ProjectStore(tmp_path),b'{"kind":"webxas-session","schema_version":3,"spectra":[]}')


def test_legacy_zip_processing_retained(tmp_path):
    from xasweb.models import AnalysisRequest
    request=AnalysisRequest(raw_text='100 1\n101 2\n')
    recipe=request.model_dump(exclude={'raw_text'})
    saved={'schema_version':1,'source':{'sha256':hashlib.sha256(request.raw_text.encode()).hexdigest()},'request':recipe,'run_id':'old1','arrays':{}}
    entries={'recipe.json':canonical(recipe),'input.txt':request.raw_text.encode(),'analysis.json':canonical(saved)}
    manifest={'schema_version':1,'run_id':'old1','files':{k:hashlib.sha256(v).hexdigest() for k,v in entries.items()}}
    output=io.BytesIO()
    with ZipFile(output,'w',ZIP_DEFLATED) as z:
        for name,blob in entries.items():z.writestr(name,blob)
        z.writestr('manifest.json',canonical(manifest))
    project=import_old_bundle(ProjectStore(tmp_path),output.getvalue())
    assert len(project.datasets)==1 and not project.recomputable
    assert project.datasets[0].metadata['legacy_recipe']['parameters']['kweight']==2


def test_project_api_export_import_and_stale_edit(tmp_path,monkeypatch):
    monkeypatch.setenv('XASWEB_PROJECT_ROOT',str(tmp_path))
    with TestClient(app) as client:
        response=client.post('/api/pro/projects',json={'title':'test'})
        assert response.status_code==200,response.text
        project=response.json();pid=project['id']
        response=client.patch(f'/api/pro/projects/{pid}',json={'title':'changed','expected_revision':0})
        assert response.status_code==200,response.text
        stale=client.patch(f'/api/pro/projects/{pid}',json={'title':'stale','expected_revision':0})
        assert stale.status_code==409
        archive=client.post(f'/api/pro/projects/{pid}/export?expected_revision=1')
        assert archive.status_code==200
        imported=client.post('/api/pro/projects/import',files={'file':('工程.zip',archive.content,'application/zip')})
        assert imported.status_code==200,imported.text
        assert imported.json()['title']=='changed' and imported.json()['id']!=pid
