from io import BytesIO
from zipfile import ZipFile
from fastapi.testclient import TestClient
from xasweb.app import app


def test_methods_exposes_checked_public_scientific_summary():
    with TestClient(app) as client:
        page = client.get('/methods')
        assert page.status_code == 200 and '/static/validation-closure.json' in page.text
        summary = client.get('/static/validation-closure.json').json()
        assert summary['Fe']['physical_gate_pass'] is False
        assert summary['Cu10K']['gates']['R1'] and summary['Cu10K']['gates']['R2']
        assert summary['Cu10K']['gates']['heldout'] is False
        assert summary['Cu10K']['overall_pass'] is False
        assert summary['KEK']['overall_pass'] and summary['NIMS']['overall_pass']


def test_offline_archive_contains_the_same_scientific_record(tmp_path):
    from test_figures_delivery import make_project
    from xasweb.pro.projects import ProjectStore
    from xasweb.pro.delivery import build_delivery

    store = ProjectStore(tmp_path/'projects')
    project, _ = make_project(store)
    raw = build_delivery(store, project.id, 'workflow1', {'formats':['png']}, 'individual', ['F04'])
    with ZipFile(BytesIO(raw)) as archive, TestClient(app) as client:
        expected = client.get('/static/validation-closure.json').raise_for_status().content
        assert archive.read('methods/validation-closure.json') == expected
        page = archive.read('methods/algorithms.html').decode()
        assert 'href="validation-closure.json"' in page and 'src="/static/' not in page
