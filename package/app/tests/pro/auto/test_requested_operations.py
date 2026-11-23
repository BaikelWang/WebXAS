"""Explicit function selection must govern both questions and execution."""
from copy import deepcopy

import numpy as np
import pytest

from xasweb.pro.auto_contracts import AnalysisIntent, effective_operations, normalize_intent
from xasweb.pro.auto_preflight import preflight
from xasweb.pro.projects import ProjectError
from xasweb.pro.workflows import run_workflow
from test_analysis_xanes import mixture_project
from test_analysis_models_series import existing_model_fixture


def lcf_intent(**changes):
    return {'task': 'xanes', 'dataset_ids': ['sample'], 'reference_ids': ['ref0', 'ref1'],
            'use_existing_processing': True,
            'conditions': {'reference_applicability_confirmed': True,
                           'reference_applicability_note': 'Same synthetic input grid and normalization; not experimental validation.'},
            **changes}


def test_none_preserves_legacy_reference_selection_and_numerical_result(tmp_path):
    store, project, *_ = mixture_project(tmp_path)
    implicit = normalize_intent(lcf_intent())
    explicit_null = normalize_intent(lcf_intent(requested_operations=None))
    assert implicit == explicit_null and implicit.requested_operations is None
    assert effective_operations(implicit) == ['lcf']
    plan = preflight(store, project, explicit_null)
    assert next(c for c in plan['capabilities'] if c['id'] == 'lcf')['selected'] is True
    result = run_workflow(store, project.id, explicit_null, project.revision)
    assert result['state'] == 'succeeded', result['issues']
    np.testing.assert_allclose(result['lcf_results'][0]['coefficients'], [.35, .65], atol=2e-7)
    assert result['intent']['requested_operations'] is None


def test_empty_selection_removes_old_inputs_and_never_calls_optional_algorithms(tmp_path, monkeypatch):
    store, project, *_ = mixture_project(tmp_path)
    value = lcf_intent(task='series', requested_operations=[], model_id='stale_model',
                       model_recipe_id='stale_recipe', fit={'noise': {'source': 'unknown'}},
                       series={'relation': 'joint_existing'})
    value['conditions'].update(state_key='potential_V', joint_applicability_confirmed=True,
                               joint_applicability_note='Old sharing declaration')
    original = deepcopy(value)
    def forbidden(*args, **kwargs):
        raise AssertionError('An unselected optional operation was executed')
    monkeypatch.setattr('xasweb.pro.xanes.fit_reference_combination', forbidden)
    monkeypatch.setattr('xasweb.pro.auto_models.run_known_model', forbidden)
    plan = preflight(store, project, value)
    assert not any(q['blocking'] for q in plan['questions'])
    normalized = plan['effective_intent']
    assert normalized['reference_ids'] == []
    assert normalized['model_id'] is normalized['model_recipe_id'] is None
    assert normalized['fit'] == normalized['lcf'] == {}
    assert 'relation' not in normalized['series']
    assert set(normalized['conditions']) == {'state_key'}
    assert value == original
    result = run_workflow(store, project.id, value, project.revision)
    assert result['state'] == 'succeeded', result['issues']
    assert result['processing_ids'] == ['pp_sample']  # References were not processed.
    assert result['features'] and result['series']['points'][0]['state_value'] == .5
    assert result['lcf_results'] == result['fit_run_ids'] == []
    assert not any(step['stage'] in ('lcf', 'exafs') for step in result['steps'])
    saved = store.read(project.id)
    assert saved.runs == project.runs and saved.models == project.models


def test_requested_missing_inputs_are_questions_and_not_a_descriptive_success(tmp_path):
    store, project, *_ = mixture_project(tmp_path)
    value = {'task': 'series', 'requested_operations': ['lcf', 'structural_fit'],
             'dataset_ids': ['sample'], 'use_existing_processing': True,
             'conditions': {'state_key': 'potential_V'}}
    plan = preflight(store, project, value)
    blockers = {q['code'] for q in plan['questions'] if q['blocking']}
    assert {'reference_required', 'model_required'} <= blockers
    assert {c['id'] for c in plan['capabilities'] if c['available'] and c['selected']} == {'lcf','structural_fit'}
    assert not next(c['selected'] for c in plan['capabilities'] if c['id']=='decomposition')
    assert 'LCF' in plan['plan_summary'] and 'EXAFS' in plan['plan_summary']
    result = run_workflow(store, project.id, value, project.revision)
    assert result['state'] == 'needs_input'
    assert {'reference_required', 'model_required'} <= {i['code'] for i in result['issues']}
    assert result['processing_ids'] == ['pp_sample'] and result['features']


def test_exafs_selection_is_mandatory_and_visible_even_when_client_sends_empty(tmp_path):
    store, project, *_ = mixture_project(tmp_path)
    plan = preflight(store, project, {'task': 'exafs', 'dataset_ids': ['sample'],
                                    'requested_operations': [], 'use_existing_processing': True})
    assert plan['effective_intent']['requested_operations'] == ['structural_fit']
    capability = next(c for c in plan['capabilities'] if c['id'] == 'structural_fit')
    assert capability['available'] and capability['selected'] and '必需' in capability['reason']
    assert 'model_required' in {q['code'] for q in plan['questions']}
    assert normalize_intent(plan['effective_intent']).model_dump(mode='json') == plan['effective_intent']


@pytest.mark.parametrize('task,operation', [('process', 'lcf'), ('process', 'structural_fit'),
    ('plot', 'lcf'), ('plot', 'structural_fit'), ('xanes', 'structural_fit'), ('exafs', 'lcf')])
def test_unsupported_operations_are_rejected_before_input_processing(task, operation):
    with pytest.raises(ProjectError) as caught:
        normalize_intent({'task': task, 'dataset_ids': ['sample'], 'requested_operations': [operation]})
    assert caught.value.code == 'operation_task'


def test_operation_contract_rejects_duplicates_and_normalizes_order():
    with pytest.raises(ValueError):
        AnalysisIntent(task='series', dataset_ids=['sample'], requested_operations=['lcf', 'lcf'])
    value = {'task': 'series', 'dataset_ids': ['sample'], 'requested_operations': ['structural_fit', 'lcf']}
    normalized = normalize_intent(value)
    assert normalized.requested_operations == ['lcf', 'structural_fit']
    assert normalize_intent(normalized) == normalized


def test_selection_changes_fingerprint_and_capabilities_without_changing_project(tmp_path):
    store, project, *_ = mixture_project(tmp_path)
    before = project.model_dump(mode='json')
    off = preflight(store, project, lcf_intent(requested_operations=[]))
    on = preflight(store, project, lcf_intent(requested_operations=['lcf']))
    assert off['plan_fingerprint'] != on['plan_fingerprint']
    assert not any(c['selected'] for c in off['capabilities'])
    assert next(c for c in on['capabilities'] if c['id'] == 'lcf')['selected'] is True
    assert next(c for c in on['capabilities'] if c['id'] == 'structural_fit')['available'] is False
    assert not any(q['blocking'] for q in on['questions'])
    assert store.read(project.id).model_dump(mode='json') == before


def test_series_lcf_only_ignores_old_model_but_keeps_reference_result(tmp_path, monkeypatch):
    store, project, *_ = mixture_project(tmp_path)
    value = lcf_intent(task='series', requested_operations=['lcf'], model_id='old_model',
                       series={'relation': 'joint_existing'})
    value['conditions'].update(state_key='potential_V', joint_applicability_confirmed=True,
                               joint_applicability_note='Old model')
    monkeypatch.setattr('xasweb.pro.auto_models.run_known_model', lambda *a, **k: pytest.fail('Unselected structural fit ran'))
    plan = preflight(store, project, value)
    assert not any(q['code'] in ('model_required', 'joint_applicability', 'noise_unspecified') for q in plan['questions'])
    result = run_workflow(store, project.id, value, project.revision)
    assert result['state'] == 'succeeded', result['issues']
    assert result['fit_run_ids'] == [] and len(result['lcf_results']) == 1
    assert result['series']['points'][0]['reference_coefficients']


def test_series_sharing_questions_only_apply_to_requested_structural_fit(tmp_path):
    store, project, *_ = existing_model_fixture(tmp_path, (0, 1))
    for index, dataset in enumerate(project.datasets):
        dataset.conditions['temperature_K'] = 100 + 100 * index
    project = store.save(project, project.revision)
    value = {'task': 'series', 'dataset_ids': ['d0', 'd1'], 'model_id': 'joint_model',
             'requested_operations': ['structural_fit'], 'use_existing_processing': True,
             'series': {'relation': 'joint_existing'}, 'conditions': {'state_key': 'temperature_K'}}
    joint = preflight(store, project, value)
    assert 'joint_applicability' in {q['code'] for q in joint['questions']}
    independent = preflight(store, project, {**value, 'series': {'relation': 'independent'}})
    assert 'joint_applicability' not in {q['code'] for q in independent['questions']}
    disabled = preflight(store, project, {**value, 'requested_operations': []})
    assert not any(q['code'] in ('joint_applicability', 'model_required', 'noise_unspecified') for q in disabled['questions'])
    assert disabled['effective_intent']['model_id'] is None
