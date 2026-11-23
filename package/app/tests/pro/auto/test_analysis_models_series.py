import importlib.util
from pathlib import Path

import numpy as np
import pytest

from xasweb.pro.auto_contracts import AnalysisIntent
from xasweb.pro.auto_models import deterministic_initial_sets, independent_model
from xasweb.pro.auto_processing import estimate_reference_calibration
from xasweb.pro.contracts import Calibration, Dataset
from xasweb.pro.parameters import compile_model
from xasweb.pro.projects import ProjectError, ProjectStore
from xasweb.pro.workflows import run_workflow


def existing_model_fixture(tmp_path, indices=(0, 1)):
    path = Path(__file__).resolve().parents[1] / 'test_fitting.py'
    spec = importlib.util.spec_from_file_location('auto_existing_feff_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture(tmp_path, indices)


def standard_project(tmp_path):
    store = ProjectStore(tmp_path / 'projects')
    p = store.create('Reference calibration vs sample chemical shift')
    energy = np.linspace(6950, 7900, 951)
    for did, edge, state in [('standard', 7115, 0), ('sample0', 7118, 1), ('sample1', 7121, 2)]:
        mu = .2 + .5 + np.arctan((energy - edge) / 3) / np.pi + .00002 * (energy - 7100)
        raw = '\n'.join(f'{x:.17g} {y:.17g}' for x, y in zip(energy, mu)) + '\n'
        p.datasets.append(Dataset(id=did, filename=did + '.dat', raw_text=raw,
            raw_file=store.put_asset(p.id, raw.encode(), did + '.dat'),
            element='Fe', edge='K', sample_id=did, conditions={'time_s': state}))
    return store, store.save(p, p.revision)


def test_reference_standard_estimates_drift_and_keeps_sample_chemical_shifts(tmp_path):
    store, p = standard_project(tmp_path)
    calibration = estimate_reference_calibration(store, p.id, 'standard', 7112.0,
        standard_confirmed=True, note='Known standard, raw-mu derivative maximum defined at 7112 eV; same energy calibration.')
    assert calibration.value == pytest.approx(-3)
    assert store.read(p.id).calibrations == []  # atomic caller owns publication
    intent = {'task': 'series', 'dataset_ids': ['sample0', 'sample1'],
        'conditions': {'check_background_stability': False, 'state_key': 'time_s',
            'reference_calibration': {'reference_dataset_id': 'standard', 'target_energy_eV': 7112.0,
                'standard_confirmed': True, 'note': 'Confirmed known standard and matching measurement energy scale.'}}}
    run = run_workflow(store, p.id, intent, p.revision)
    assert run['state'] == 'succeeded', run['issues']
    saved = store.read(p.id)
    assert len(saved.calibrations) == 1
    assert saved.calibrations[0].value == pytest.approx(-3)
    features = {f['dataset_id']: f for f in run['features']}
    assert features['sample1']['edge_position_eV'] - features['sample0']['edge_position_eV'] == pytest.approx(3)
    assert [p['state_value'] for p in run['series']['points']] == [1, 2]
    assert 'standard' in run['provenance']['source_hashes']
    # Repeating the same intent starts from original energies, not shifted data.
    repeat = run_workflow(store, p.id, intent, saved.revision)
    new = store.read(p.id)
    assert len(new.calibrations) == 2
    assert all(c.value == pytest.approx(-3) for c in new.calibrations)
    assert [f['edge_position_eV'] for f in repeat['features']] == [f['edge_position_eV'] for f in run['features']]


def test_unconfirmed_standard_calibration_does_not_run_uncalibrated_fallback(tmp_path):
    store, p = standard_project(tmp_path)
    with pytest.raises(ProjectError) as exc:
        estimate_reference_calibration(store, p.id, 'standard', 7112.0)
    assert exc.value.code == 'reference_standard_confirmation'
    run = run_workflow(store, p.id, {'dataset_ids': ['sample0'], 'conditions': {
        'reference_calibration': {'reference_dataset_id': 'standard', 'target_energy_eV': 7112.0}}}, p.revision)
    assert run['state'] == 'needs_input'
    assert run['processing_ids'] == []
    assert not store.read(p.id).calibrations


def test_series_missing_actual_state_and_failed_observation_are_not_interpolated(tmp_path):
    store, p = standard_project(tmp_path)
    p.datasets[2].conditions = {}
    raw = '1 2\n' * 45
    p.datasets.append(Dataset(id='failed', filename='failed.dat', kind='chi', raw_text=raw,
        raw_file=store.put_asset(p.id, raw.encode(), 'failed.dat'), conditions={'time_s': 3}))
    p = store.save(p, p.revision)
    run = run_workflow(store, p.id, {'task': 'series', 'dataset_ids': ['sample0', 'sample1', 'failed'],
        'series': {'state_key': 'time_s'}, 'conditions': {'check_background_stability': False}}, p.revision)
    assert run['state'] == 'needs_input'
    points = run['series']['points']
    assert points[1]['state_value'] is None
    assert points[2]['processing_id'] is None
    assert points[2]['state_value'] == 3
    assert run['series']['matrix_dataset_ids'] == ['sample0']


def test_model_starts_are_finite_bounded_and_single_models_release_sharing(tmp_path):
    store, p, spec, *_ = existing_model_fixture(tmp_path)
    initials = deterministic_initial_sets(spec.model, 4)
    assert 1 <= len(initials) <= 3
    assert initials == deterministic_initial_sets(spec.model, 4)
    for start in initials:
        for node in spec.model.parameters.nodes:
            if node.id in start:
                assert node.minimum < start[node.id] < node.maximum
    single = independent_model(spec.model, 'd0', p.paths)
    compiled = compile_model(single, p.paths)
    assert set(compiled.free_ids) == {'amp', 'dr', 'e0_0', 'sig_0'}
    assert all(n.owner_id != 'd1' for n in single.parameters.nodes)
    with pytest.raises(ProjectError):
        deterministic_initial_sets(spec.model, 9)


def test_known_joint_model_runs_independent_benchmarks_and_saves_reusable_recipe(tmp_path):
    store, p, spec, metadata, arrays = existing_model_fixture(tmp_path)
    for i, dataset in enumerate(p.datasets):
        dataset.conditions = {'temperature_K': float(300 + 50 * i)}
    p = store.save(p, p.revision)
    intent = {'task': 'series', 'dataset_ids': ['d0', 'd1'], 'model_id': spec.model.id,
        'use_existing_processing': True, 'series': {'relation': 'joint_existing', 'state_key': 'temperature_K'},
        'fit': {'max_starts': 1, 'max_evaluations': 500, 'timeout_seconds': 120.0},
        'conditions': {'joint_applicability_confirmed': True,
            'joint_applicability_note': 'Frozen synthetic same-engine example shares amp/dr solely for numerical adapter acceptance.'}}
    run = run_workflow(store, p.id, intent, p.revision)
    assert run['state'] == 'succeeded', run['issues']
    assert len(run['fit_run_ids']) == 3
    saved = store.read(p.id)
    assert len(saved.runs) == 3
    assert [len(r.result.datasets) for r in saved.runs] == [1, 1, 2]
    assert all(r.result.converged for r in saved.runs)
    recipe = saved.metadata['automation_model_recipes'][0]
    assert recipe['version'] == 1
    assert recipe['scope'] == 'same_engine_synthetic_engineering_example'
    assert recipe['validation']['real_experimental_standard'].startswith('pending')
    assert run['recipe']['model_recipe']['id'] == recipe['id']
    assert len(run['series']['points'][0]['fits']) == 2
    assert p.models[0].model_dump(mode='json') == saved.models[0].model_dump(mode='json')
    reused = dict(intent, model_recipe_id=recipe['id'])
    reused['series'] = dict(intent['series'], relation='independent')
    reused_run = run_workflow(store, p.id, reused, saved.revision)
    assert len(reused_run['fit_run_ids']) == 2
    assert len(store.read(p.id).metadata['automation_model_recipes']) == 1


def test_joint_relations_require_physical_confirmation(tmp_path):
    store, p, spec, *_ = existing_model_fixture(tmp_path)
    run = run_workflow(store, p.id, {'task': 'exafs', 'dataset_ids': ['d0', 'd1'],
        'model_id': spec.model.id, 'use_existing_processing': True}, p.revision)
    assert run['state'] == 'needs_input'
    assert run['fit_run_ids'] == []
    assert any(i['code'] == 'joint_applicability' for i in run['issues'])


def add_repeat(store, p, *, duplicate=False, different_state=False):
    original = p.datasets[1]
    table = np.array([list(map(float, line.split())) for line in original.raw_text.splitlines()])
    table[:, 1] += .0001 * np.sin(table[:, 0])
    raw = original.raw_text if duplicate else '\n'.join(f'{x:.17g} {y:.17g}' for x, y in table) + '\n'
    second = original.model_copy(deep=True)
    second.id = 'repeat'
    second.filename = 'independent-repeat.dat'
    second.raw_text = raw
    second.raw_file = store.put_asset(p.id, raw.encode(), 'independent-repeat.dat')
    if different_state:
        second.conditions = {'time_s': 200}
    p.datasets.append(second)
    return store.save(p, p.revision)


def test_explicit_independent_repeat_merge_preserves_masks_lineage_and_scatter(tmp_path):
    from xasweb.pro.preprocessing import load_array
    store, p = standard_project(tmp_path)
    p = add_repeat(store, p)
    run = run_workflow(store, p.id, {'task': 'process', 'dataset_ids': ['sample0', 'repeat'],
        'series': {'merge_repeats': True}, 'conditions': {'check_background_stability': False,
            'independent_repeats_confirmed': True, 'independent_repeats_note': 'Synthetic distinct scans; explicit repeat identity for engineering regression.'}}, p.revision)
    assert run['state'] == 'succeeded', run['issues']
    saved = store.read(p.id)
    merged = saved.datasets[-1]
    assert merged.lineage == ['sample0', 'repeat']
    assert merged.metadata['scan_count'] == 2
    assert len(run['processing_ids']) == 1
    assert saved.processing[-1].dataset_id == merged.id
    assert saved.processing[-1].noise.source == 'unknown'
    assert load_array(store, p.id, merged.arrays['scatter']).max() > 0
    mask = load_array(store, p.id, merged.arrays['source_common_mask_0'])
    assert mask.dtype.kind == 'b' and mask.all()
    assert store.read_asset(p.id, p.datasets[1].raw_file) == p.datasets[1].raw_text.encode()
    assert saved.metadata['automation_runs'][0] == run


@pytest.mark.parametrize('variant,expected', [('unconfirmed', 'repeat_independence_required'),
    ('duplicate', 'duplicate_scan'), ('different_state', 'repeat_no_compatible_group'), ('model_task', 'repeat_merge_task')])
def test_repeat_merge_rejects_pseudoreplication_and_wrong_states(tmp_path, variant, expected):
    store, p = standard_project(tmp_path)
    p = add_repeat(store, p, duplicate=variant == 'duplicate', different_state=variant == 'different_state')
    conditions = {'independent_repeats_confirmed': variant != 'unconfirmed',
                  'independent_repeats_note': 'Controlled engineering negative case.'}
    intent = {'task': 'exafs' if variant == 'model_task' else 'process',
        'dataset_ids': ['sample0', 'repeat'], 'series': {'merge_repeats': True}, 'conditions': conditions}
    run = run_workflow(store, p.id, intent, p.revision)
    assert run['state'] == 'needs_input'
    assert any(i['code'] == expected for i in run['issues'])
    assert len(store.read(p.id).datasets) == len(p.datasets)
    assert run['processing_ids'] == []


def test_unrelated_s02_calibration_cannot_hide_unapplied_warning(tmp_path):
    store, p, spec, *_ = existing_model_fixture(tmp_path, (0,))
    p.calibrations.append(Calibration(id='irrelevant_s02', kind='s02', value=.9, unit='1',
        reference_dataset_ids=['d0'], element='Fe', edge='K', applicability_note='Intentionally inapplicable edge/element.',
        conditions={'parameter_id': 'amp', 'temperature_K': 900.0}))
    p = store.save(p, p.revision)
    run = run_workflow(store, p.id, {'task': 'exafs', 'dataset_ids': ['d0'], 'model_id': spec.model.id,
        'use_existing_processing': True, 'fit': {'max_starts': 1, 'timeout_seconds': 90.0}}, p.revision)
    assert run['fit_run_ids']
    assert any(i['code'] == 's02_uncalibrated' for i in run['issues'])
    model_decision = next(d for d in run['decisions'] if d['stage'] == 'model')
    assert model_decision['selected']['uncalibrated_dataset_ids'] == ['d0']
    assert model_decision['selected']['applied_s02_calibrations'] == {}


def test_reference_shift_is_applied_once_before_repeat_merge(tmp_path):
    from xasweb.pro.preprocessing import load_array
    store, p = standard_project(tmp_path)
    p = add_repeat(store, p)
    original_energy = float(p.datasets[1].raw_text.splitlines()[0].split()[0])
    run = run_workflow(store, p.id, {'dataset_ids': ['sample0', 'repeat'],
        'series': {'merge_repeats': True}, 'conditions': {'check_background_stability': False,
            'independent_repeats_confirmed': True, 'independent_repeats_note': 'Synthetic independently declared repeated observations.',
            'reference_calibration': {'reference_dataset_id': 'standard', 'target_energy_eV': 7112.0,
                'standard_confirmed': True, 'note': 'Same-definition known standard; matching energy calibration.'}}}, p.revision)
    assert run['state'] == 'succeeded', run['issues']
    saved = store.read(p.id)
    pp = saved.processing[-1]
    assert load_array(store, p.id, pp.arrays['energy'])[0] == pytest.approx(original_energy - 3)
    assert pp.metadata['effective_energy_shift_eV'] == 0  # already applied before merge
    assert sum(c['operation'] == 'reference_shift_before_repeat_merge' for c in pp.resolved_calls) == 1
    assert not any(i['code'] == 'uncalibrated_energy' for i in run['issues'])


def test_applied_s02_requires_fixed_parameter_link_and_matching_measurement_conditions(tmp_path):
    from types import SimpleNamespace
    from xasweb.pro.auto_models import applied_s02_calibrations, eligible_fit_calibrations
    from xasweb.pro.contracts import FitSpec
    store, p, spec, *_ = existing_model_fixture(tmp_path, (0,))
    value = spec.model_dump(mode='json')
    value['datasets'][0]['conditions'] = {'temperature_K': 300.0}
    amp = next(n for n in value['model']['parameters']['nodes'] if n['id'] == 'amp')
    amp.update(kind='fixed', value=.9, minimum=None, maximum=None)
    cal = Calibration(id='fixed_s02', kind='s02', value=.9, unit='1', reference_dataset_ids=['d0'],
        element='Ce', edge='L3', applicability_note='Explicit fixed S02 transfer; conditional errors.',
        conditions={'parameter_id': 'amp', 'temperature_K': 900.0})
    value['calibrations'] = [cal.model_dump(mode='json')]
    case = FitSpec.model_validate(value)
    run = SimpleNamespace(input_snapshot=case, result=SimpleNamespace(metadata={}), id='fixture_run')
    assert applied_s02_calibrations(p, [run]) == {}
    kept, excluded = eligible_fit_calibrations(case)
    assert kept == [] and excluded[0]['calibration_id'] == 'fixed_s02'
    case.calibrations[0].conditions['temperature_K'] = 300.0
    kept, excluded = eligible_fit_calibrations(case)
    assert [c.id for c in kept] == ['fixed_s02'] and excluded == []
    links = applied_s02_calibrations(p, [run])
    assert links['d0'][0]['calibration_id'] == 'fixed_s02'
    assert links['d0'][0]['uncertainty_basis'] == 'conditional_fixed'
    case.calibrations[0].value = .8
    assert applied_s02_calibrations(p, [run]) == {}


def test_external_plot_only_snapshot_cannot_enter_known_model_quantification(tmp_path):
    store, p, spec, *_ = existing_model_fixture(tmp_path, (0,))
    p.processing[0].metadata['external_plot_only'] = True
    p.processing[0].metadata['quantitative_eligible'] = False
    p = store.save(p, p.revision)
    run = run_workflow(store, p.id, {'task': 'exafs', 'dataset_ids': ['d0'], 'model_id': spec.model.id,
        'use_existing_processing': True}, p.revision)
    assert run['state'] == 'needs_input'
    assert run['fit_run_ids'] == []
    assert any(i['code'] == 'external_plot_only' for i in run['issues'])
