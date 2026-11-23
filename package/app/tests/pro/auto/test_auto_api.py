"""Public automatic entry, snapshot scope and delivery access boundaries."""
import io
import hashlib
import json
from unittest.mock import patch,Mock
from zipfile import ZipFile
import pytest
from fastapi.testclient import TestClient
from xasweb.app import app
from xasweb.pro.projects import ProjectStore
from xasweb.pro.contracts import Project


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('XASWEB_PROJECT_ROOT',str(tmp_path/'projects'))
    with TestClient(app) as client:
        yield client


def test_synthetic_mu_demo_is_explicit_and_portable(client):
    response=client.post('/api/auto/demo')
    assert response.status_code==200,response.text
    p=response.json()
    assert p['metadata']['not_experimental_data'] is True
    assert p['datasets'][0]['kind']=='mu'
    assert p['datasets'][0]['element']=='Cu'
    assert 'Synthetic' in p['datasets'][0]['raw_text']
    archive=client.post(f"/api/pro/projects/{p['id']}/export?expected_revision={p['revision']}")
    assert archive.status_code==200
    restored=client.post('/api/pro/projects/import',files={'file':('project.zip',archive.content)})
    assert restored.status_code==200,restored.text
    assert restored.json()['datasets'][0]['raw_file']['sha256']==p['datasets'][0]['raw_file']['sha256']


def test_workflow_scope_and_revision_rejected_before_scheduling(client):
    p=client.post('/api/auto/demo').json()
    response=client.post(f"/api/auto/projects/{p['id']}/workflows",json={'expected_revision':0,'intent':{'dataset_ids':[p['datasets'][0]['id']]}})
    assert response.status_code==409
    response=client.post(f"/api/auto/projects/{p['id']}/workflows",json={'expected_revision':p['revision'],'intent':{'dataset_ids':['foreign']}})
    assert response.status_code==422
    assert client.get(f"/api/pro/projects/{p['id']}/jobs").json()==[]


def test_delivery_preview_only_allows_recorded_png_and_checks_asset(client):
    target=ProjectStore();p=target.create('preview boundary')
    stream=io.BytesIO()
    with ZipFile(stream,'w') as z:
        z.writestr('figures/F01.png',b'PNG preview boundary fixture')
        z.writestr('methods/private.txt',b'not a preview')
    ref=target.put_asset(p.id,stream.getvalue(),'delivery.zip')
    value=p.model_dump(mode='json')
    value['metadata']['automation_deliveries']=[{'id':'out_1','workflow_id':'wf_1','asset':ref.model_dump(mode='json'),'figures':[{'filename':'figures/F01.png'}]}]
    p=target.save(Project.model_validate(value),p.revision)
    base=f'/api/auto/projects/{p.id}/deliveries/out_1'
    assert client.get(base+'/preview',params={'filename':'figures/F01.png'}).status_code==200
    assert client.get(base+'/preview',params={'filename':'methods/private.txt'}).status_code==422
    assert client.get(base+'/preview',params={'filename':'../../project.json'}).status_code==422
    assert client.get(base+'/download').content==stream.getvalue()
    old=client.get(base+'/figures')
    assert old.status_code==200 and old.json()['figures'][0]['insight']['metrics']==[]
    assert '旧版' in old.json()['figures'][0]['insight']['observations'][0]
    target.folder(p.id).joinpath(*ref.path.split('/')).write_bytes(b'changed')
    corrupted=client.get(base+'/download')
    assert corrupted.status_code==422
    assert corrupted.json()['code']=='asset_integrity'


def _snapshot_delivery(target,p,corrupt=False):
    data={'figures/F04-p01.png':b'PNG frozen preview','data/F04-p01.csv':b'k [A^-1],chi [1]\n1,0.2\n',
          'specs/F04-p01.json':b'{"frozen":true}'}
    figure={'id':'F04-p01','panel_id':'F04-p01','family':'F04','filename':'figures/F04-p01.png',
        'formats':['figures/F04-p01.png'],'data_files':['data/F04-p01.csv'],'spec_file':'specs/F04-p01.json',
        'dataset_ids':['ds_original'],'sample_names':['Frozen Cu.dat'],'sources':[{'dataset_id':'ds_original','snapshot_sha256':'f'*64}],
        'insight':{'observations':['原快照实际覆盖 1–2 Å⁻¹。'],'metrics':[],
            'limitations':['不能直接确定价态。'],'method':{'workflow_id':'wf_frozen'},'sources':[]}}
    manifest={'format':'webxas-scientific-delivery','version':1,'workflow_id':'wf_frozen',
        'presentation':'individual','figures':[figure],'missing':[{'family':'F07','reason':'未执行拟合。'}],
        'files':{n:{'sha256':hashlib.sha256(blob).hexdigest(),'byte_size':len(blob)} for n,blob in data.items()}}
    stream=io.BytesIO()
    with ZipFile(stream,'w') as archive:
        for name,blob in data.items():archive.writestr(name,b'bad CSV' if corrupt and name.endswith('.csv') else blob)
        archive.writestr('manifest.json',json.dumps(manifest))
    ref=target.put_asset(p.id,stream.getvalue(),'delivery.zip')
    value=p.model_dump(mode='json');value['metadata']['automation_deliveries']=[{
        'id':'out_snapshot','workflow_id':'wf_frozen','asset':ref.model_dump(mode='json'),'figures':[figure]}]
    return target.save(Project.model_validate(value),p.revision),data


def test_snapshot_figures_and_whitelisted_files_ignore_current_mutable_data(client):
    target=ProjectStore();p=target.create('current mutable title')
    p,data=_snapshot_delivery(target,p)
    value=p.model_dump(mode='json');value['title']='renamed after delivery'
    target.save(Project.model_validate(value),p.revision)
    base=f'/api/auto/projects/{p.id}/deliveries/out_snapshot'
    result=client.get(base+'/figures')
    assert result.status_code==200,result.text
    body=result.json();fig=body['figures'][0]
    assert body['workflow_id']=='wf_frozen' and body['delivery_id']=='out_snapshot'
    assert fig['sample_names']==['Frozen Cu.dat'] and fig['insight']['observations']==['原快照实际覆盖 1–2 Å⁻¹。']
    assert body['missing'][0]['family']=='F07'
    csvfile=next(f for f in fig['files'] if f['filename'].endswith('.csv'))
    assert 'Frozen-Cu' in csvfile['download_name'] and 'F04-p01' in csvfile['download_name'] and 'wf-' in csvfile['download_name']
    response=client.get(base+'/files',params={'filename':csvfile['filename']})
    assert response.status_code==200 and response.content==data[csvfile['filename']]
    assert 'filename*=' in response.headers['content-disposition']
    assert response.headers['x-content-sha256']==hashlib.sha256(response.content).hexdigest()
    for filename in ('../project.json','/project.json','figures\\F04-p01.png','data//F04-p01.csv','data/F04-p01.csv:stream','not-whitelisted'):
        assert client.get(base+'/files',params={'filename':filename}).status_code==422


def test_manifest_file_hash_and_size_are_verified_in_addition_to_outer_asset(client):
    target=ProjectStore();p,_=_snapshot_delivery(target,target.create('corrupted inner file'),corrupt=True)
    response=client.get(f'/api/auto/projects/{p.id}/deliveries/out_snapshot/files',params={'filename':'data/F04-p01.csv'})
    assert response.status_code==422 and response.json()['code']=='delivery_file_integrity'


def test_delivery_request_preserves_legacy_default_and_explicit_individual_families(client):
    target=ProjectStore();p=target.create('request schema');value=p.model_dump(mode='json')
    value['metadata']['automation_runs']=[{'id':'wf1'}];p=target.save(Project.model_validate(value),p.revision)
    service=Mock();service.submit.return_value={'job_id':'job_fixture'}
    base=f'/api/auto/projects/{p.id}/deliveries'
    with patch('xasweb.pro.auto_api.get_job_service',return_value=service):
        args={'expected_revision':p.revision,'workflow_id':'wf1'}
        assert client.post(base,json=args).status_code==202
        assert service.submit.call_args.args[2]['presentation']=='legacy'
        assert client.post(base,json={**args,'presentation':'individual','figure_families':['F04','F07']}).status_code==202
        assert service.submit.call_args.args[2]['figure_families']==['F04','F07']
        assert client.post(base,json={**args,'presentation':'invalid'}).status_code==422
        assert client.post(base,json={**args,'figure_families':['F99']}).status_code==422
        assert client.post(base,json={**args,'figure_families':['F04','F04']}).status_code==422


def test_task_entry_and_professional_link(client):
    response=client.get('/auto')
    assert response.status_code==200
    assert '开始分析' in response.text
    assert '导出结果' in response.text
    assert 'question-dialog' in response.text
    assert 'auto.js' in response.text
    assert 'id="auto-return" href="/auto"' in client.get('/pro').text
    assert client.get('/').text == response.text
    assert 'simple.js' in client.get('/quick').text
    assert 'workbench-view.js' in response.text


def test_auto_page_versions_all_static_assets(client):
    from xasweb import __version__
    page=client.get('/auto').text
    for name in ('auto.css','auto-controls.js','auto-results.js','auto.js'):
        assert f'/static/{name}?v={__version__}' in page
