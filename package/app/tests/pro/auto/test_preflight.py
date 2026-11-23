"""Read-only plans, explicit scopes and stale-plan rejection at both boundaries."""
from copy import deepcopy
import hashlib
import math

import pytest
from fastapi.testclient import TestClient

from xasweb.app import app
from xasweb.pro.auto_contracts import normalize_intent, validate_calibration_scope
from xasweb.pro.auto_preflight import preflight, verify_plan
from xasweb.pro.contracts import Dataset, DatasetBinding, ModelRevision, PathBinding
from xasweb.pro.projects import ProjectError, ProjectStore
from xasweb.pro.workflows import run_job, run_workflow


def sample(store, project, did='sample', *, kind='mu', metadata=None, raw=None):
    if raw is None:
        raw = '# Column.1: energy eV\n# Column.2: mu\n' if kind == 'mu' else '# Column.1: k\n# Column.2: chi\n'
        raw += '\n'.join(f'{7000 + i * 3 if kind == "mu" else (i + 1) / 5:.12g} {1 + math.tanh((i - 30) / 4) + .0001 * i:.12g}' for i in range(100))
    return Dataset(id=did, filename=did + '.dat', kind=kind, raw_text=raw,
        raw_file=store.put_asset(project.id, raw.encode(), did + '.dat'), element='Fe', edge='K',
        sample_id='iron', metadata=metadata or {})


def project_fixture(tmp_path):
    store = ProjectStore(tmp_path / 'projects')
    project = store.create('preflight contracts')
    project.datasets.append(sample(store, project))
    return store, store.save(project, project.revision)


def files(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def codes(plan):
    return {q['code'] for q in plan['questions']}


def test_preflight_is_readonly_deterministic_and_effective_intent_is_idempotent(tmp_path, monkeypatch):
    store, project = project_fixture(tmp_path)
    before = files(store.folder(project.id))
    def forbidden(*args, **kwargs):
        raise AssertionError('Preflight must not process, fit, or write')
    monkeypatch.setattr(store, 'save', forbidden)
    monkeypatch.setattr(store, 'put_asset', forbidden)
    monkeypatch.setattr('xasweb.pro.auto_processing.process_dataset', forbidden)
    monkeypatch.setattr('xasweb.pro.auto_models.fit_snapshot', forbidden)
    plan = preflight(store, project, {'dataset_ids': ['sample']})
    assert not any(q['blocking'] for q in plan['questions'])
    assert plan == preflight(store, project, plan['effective_intent'])
    assert files(store.folder(project.id)) == before
    assert plan['policy_version'] == 'auto-preflight-v1'
    assert next(a for a in plan['available_actions'] if a['task'] == 'process')['state'] == 'ready'
    assert plan['facts'][0]['key'] == 'mapping'


def test_ambiguous_columns_return_preview_and_fields_without_resolving_by_guess(tmp_path):
    store, project = project_fixture(tmp_path)
    raw = '\n'.join(f'{7000+i*3} {i+1} {i+2}' for i in range(100))
    project.datasets = [sample(store, project, raw=raw)]
    plan = preflight(store, project, {'dataset_ids': ['sample']})
    question = next(q for q in plan['questions'] if q['code'] == 'mapping_ambiguous')
    assert question['blocking'] and question['dataset_id'] == 'sample'
    assert len(question['preview']['rows']) == 5
    assert 'dataset.mapping.mode' in {f['path'] for f in question['fields']}
    assert project.datasets[0].metadata.get('mapping_confirmed') is None


def test_chi_weight_and_external_provenance_questions(tmp_path):
    store, project = project_fixture(tmp_path)
    project.datasets = [sample(store, project, kind='chi', metadata={
        'import_processing_state': 'chi', 'import_analysis_blockers': ['chi_weighting_requires_verification']})]
    plan = preflight(store, project, {'task': 'plot', 'dataset_ids': ['sample']})
    assert {'chi_weighting_requires_verification', 'external_processing_note_required'} <= codes(plan)
    project.datasets[0].metadata.update(input_kweight_confirmed=True, input_kweight=2,
                                       external_processing_note='Measured exported chi times k^2, source log 1')
    plan = preflight(store, project, {'task': 'plot', 'dataset_ids': ['sample']})
    assert not any(q['blocking'] for q in plan['questions'])
    assert 'xanes_requires_mu' in codes(preflight(store, project, {'task': 'xanes', 'dataset_ids': ['sample']}))


def test_normalized_input_only_enters_declared_plot_route(tmp_path):
    store, project = project_fixture(tmp_path)
    project.datasets[0].metadata.update(import_processing_state='normalized', external_processing_note='Athena normalized output, E/eV and norm')
    assert 'derived_input' in codes(preflight(store, project, {'dataset_ids': ['sample']}))
    assert not any(q['blocking'] for q in preflight(store, project, {'task': 'plot', 'dataset_ids': ['sample']})['questions'])


def test_weighted_chi_at_zero_is_not_silently_repaired(tmp_path):
    store, project = project_fixture(tmp_path)
    raw = '# Column.1: k\n# Column.2: chi\n' + '\n'.join(f'{i/5} {math.sin(i)}' for i in range(100))
    project.datasets = [sample(store, project, kind='chi', raw=raw,
                              metadata={'input_kweight': 2, 'input_kweight_confirmed': True})]
    assert 'chi_weight' in codes(preflight(store, project, {'dataset_ids': ['sample']}))


@pytest.mark.parametrize('task', ['process', 'xanes', 'exafs', 'series', 'plot'])
def test_task_scoping_clears_hidden_panel_state_without_changing_input(task):
    data = {'task': task, 'dataset_ids': ['sample'], 'reference_ids': ['reference'], 'model_id': 'model',
        'calibration_ids': {'sample': 'cal', 'reference': 'cal'}, 'processing_overrides': {'rbkg': 1.1},
        'lcf': {'sum_to_one': True}, 'fit': {'max_starts': 3},
        'conditions': {'state_key': 'temperature_K', 'reference_applicability_confirmed': True,
            'reference_applicability_note': 'reference', 'joint_applicability_confirmed': True,
            'joint_applicability_note': 'shared', 'independent_repeats_confirmed': True,
            'independent_repeats_note': 'repeats', 'reference_calibration': {'reference_dataset_id': 'reference'}},
        'series': {'relation': 'independent', 'state_key': 'temperature_K', 'merge_repeats': True}}
    original = deepcopy(data)
    result = normalize_intent(data)
    assert data == original
    assert normalize_intent(result) == result
    if task not in ('xanes', 'series'):
        assert result.reference_ids == [] and result.lcf == {}
        assert 'reference_applicability_confirmed' not in result.conditions
    if task not in ('exafs', 'series'):
        assert result.model_id is None and result.fit == {}
    if task != 'series':
        assert 'state_key' not in result.conditions and 'relation' not in result.series
    if task not in ('process', 'xanes'):
        assert 'independent_repeats_confirmed' not in result.conditions
        if task != 'plot':
            assert result.series['merge_repeats'] is True  # Explicit forbidden action must be rejected, not ignored.
    if task == 'series':
        assert 'joint_applicability_confirmed' not in result.conditions
    if task == 'plot':
        assert result.conditions == result.series == result.calibration_ids == result.processing_overrides == {}


@pytest.mark.parametrize('scope', [[], ['sample', 'sample'], ['foreign'], None, 'sample', [True]])
def test_explicit_calibration_scope_is_nonempty_unique_selected_subset(scope):
    intent = normalize_intent({'dataset_ids': ['sample'], 'conditions': {'reference_calibration': {
        'reference_dataset_id': 'standard', 'apply_to_dataset_ids': scope}}})
    with pytest.raises(ProjectError, match='非空') as caught:
        validate_calibration_scope(intent)
    assert caught.value.code == 'reference_calibration_scope'


def test_new_plan_requires_scope_but_legacy_omission_remains_compatible(tmp_path):
    store, project = project_fixture(tmp_path)
    project.datasets.append(sample(store, project, 'standard'))
    intent = {'dataset_ids': ['sample'], 'conditions': {'reference_calibration': {
        'reference_dataset_id': 'standard', 'target_energy_eV': 7112.,
        'standard_confirmed': True, 'note': 'Known standard derivative feature and scan batch'}}}
    validate_calibration_scope(normalize_intent(intent))
    plan = preflight(store, project, intent)
    assert 'reference_calibration_scope_required' in codes(plan)
    with pytest.raises(ProjectError) as caught:
        verify_plan(store, project, intent, plan['plan_fingerprint'])
    assert caught.value.code == 'preflight_needs_input'
    intent['conditions']['reference_calibration']['apply_to_dataset_ids'] = ['sample']
    assert 'reference_calibration_scope_required' not in codes(preflight(store, project, intent))
    intent['conditions']['reference_calibration']['applicability_conditions'] = {'beamtime': 'A'}
    assert 'calibration_conditions' in codes(preflight(store, project, intent))


def test_repeats_require_explicit_facts_and_duplicate_bytes_never_become_observations(tmp_path):
    store, project = project_fixture(tmp_path)
    second = project.datasets[0].model_copy(deep=True)
    second.id = 'second'
    project.datasets.append(second)
    plan = preflight(store, project, {'dataset_ids': ['sample', 'second'], 'series': {'merge_repeats': True}})
    assert {'repeat_independence_required', 'duplicate_scan'} <= codes(plan)
    second.conditions = {'temperature_K': 500}
    plan = preflight(store, project, {'dataset_ids': ['sample', 'second'], 'series': {'merge_repeats': True}})
    assert 'repeat_no_compatible_group' in codes(plan)
    assert 'duplicate_scan' not in codes(plan)  # Different states were never grouped.


def test_multi_spectrum_exafs_exposes_joint_question_and_series_defaults_independent(tmp_path):
    store, project = project_fixture(tmp_path)
    project.datasets.append(sample(store, project, 'second'))
    project.models.append(ModelRevision(id='joint', bindings=[DatasetBinding(dataset_id=did,
        processing_revision_id='old_' + did, paths=[PathBinding(id='binding_' + did, path_id='unavailable')])
        for did in ['sample', 'second']]))
    value = {'task': 'exafs', 'dataset_ids': ['sample', 'second'], 'model_id': 'joint'}
    plan = preflight(store, project, value)
    assert {'joint_applicability', 'noise_unspecified'} <= codes(plan)
    question = next(q for q in plan['questions'] if q['code'] == 'joint_applicability')
    assert question['task_impacts'] == ['exafs'] and question['blocking']
    value['task'] = 'series'
    plan = preflight(store, project, value)
    assert 'joint_applicability' not in codes(plan)
    assert 'series_state_required' in codes(plan)


def test_fingerprint_binds_full_project_and_rejects_changed_assets_before_work(tmp_path, monkeypatch):
    store, project = project_fixture(tmp_path)
    value = {'dataset_ids': ['sample']}
    plan = preflight(store, project, value)
    changed = project.model_copy(deep=True)
    changed.metadata['reviewed_model_version'] = 'different'
    assert preflight(store, changed, value)['plan_fingerprint'] != plan['plan_fingerprint']
    target = store.folder(project.id).joinpath(*project.datasets[0].raw_file.path.split('/'))
    target.write_bytes(b'corrupted')
    with pytest.raises(ProjectError) as caught:
        run_workflow(store, project.id, value, project.revision, plan_fingerprint=plan['plan_fingerprint'])
    assert caught.value.code == 'plan_stale'
    assert store.read(project.id).metadata.get('automation_runs') is None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('XASWEB_PROJECT_ROOT', str(tmp_path / 'projects'))
    with TestClient(app) as client:
        yield client


def test_api_stale_plans_and_invalid_scopes_never_schedule(client, monkeypatch):
    project = client.post('/api/auto/demo').json()
    base = f"/api/auto/projects/{project['id']}"
    body = {'expected_revision': project['revision'], 'intent': {'dataset_ids': [project['datasets'][0]['id']]}}
    response = client.post(base + '/preflight', json=body)
    assert response.status_code == 200, response.text
    plan = response.json()
    scheduled = []
    class Jobs:
        def submit(self, *args):
            scheduled.append(args)
            return {'job_id': 'captured'}
    monkeypatch.setattr('xasweb.pro.auto_api.get_job_service', lambda: Jobs())
    stale = client.post(base + '/workflows', json={**body, 'plan_fingerprint': '0' * 64})
    assert stale.status_code == 409 and stale.json()['code'] == 'plan_stale'
    invalid = deepcopy(body)
    invalid['intent']['conditions'] = {'reference_calibration': {'apply_to_dataset_ids': []}}
    rejected = client.post(base + '/workflows', json=invalid)
    assert rejected.status_code == 422 and rejected.json()['code'] == 'reference_calibration_scope'
    assert scheduled == []
    accepted = client.post(base + '/workflows', json={**body, 'plan_fingerprint': plan['plan_fingerprint']})
    assert accepted.status_code == 202, accepted.text
    assert scheduled[0][2]['plan_fingerprint'] == plan['plan_fingerprint']
    assert scheduled[0][2]['intent'] == plan['effective_intent']


def test_worker_forwards_fingerprint(monkeypatch):
    received = []
    monkeypatch.setattr('xasweb.pro.workflows.run_workflow', lambda *args: received.append(args))
    run_job({'project_id': 'project', 'intent': {}, 'expected_revision': 3, 'plan_fingerprint': 'digest'}, None)
    assert received[0][-1] == 'digest'
