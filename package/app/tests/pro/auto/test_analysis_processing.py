from pathlib import Path

import numpy as np
import pytest

from xasweb.models import Parameters
from xasweb.pro.auto_contracts import AnalysisIntent, WorkflowResult, read_workflows
from xasweb.pro.auto_processing import propose_parameters
from xasweb.pro.contracts import Calibration, Dataset, Project
from xasweb.pro.preprocessing import load_array
from xasweb.pro.projects import ProjectError, ProjectStore
from xasweb.pro.workflows import run_workflow


def sample_project(tmp_path, kind='mu', offset=0.0):
    store = ProjectStore(tmp_path / 'projects')
    project = store.create('automatic scientific boundaries')
    if kind == 'mu':
        source = Path(__file__).resolve().parents[4] / 'validation/sample-spectrum.xdi'
        original = source.read_text(encoding='utf-8')
        rows = []
        for line in original.splitlines():
            if line.startswith('#'):
                rows.append(line)
            elif line.strip():
                x, y = map(float, line.split())
                rows.append(f'{x + offset:.17g} {y:.17g}')
        raw = '\n'.join(rows) + '\n'
    else:
        k = np.linspace(0, 15, 301)
        raw = '\n'.join(f'{x:.17g} {np.sin(4*x)*np.exp(-x/8):.17g}' for x in k) + '\n'
    dataset = Dataset(id='sample', kind=kind, filename='spectrum.dat', raw_text=raw,
                      raw_file=store.put_asset(project.id, raw.encode(), 'spectrum.dat'),
                      element='Fe', edge='K', sample_id='iron')
    project.datasets.append(dataset)
    project = store.save(project, project.revision)
    return store, project


def test_intent_validation_and_old_projects_remain_readable(tmp_path):
    store, p = sample_project(tmp_path, 'chi')
    assert read_workflows(p) == []
    with pytest.raises(ValueError):
        AnalysisIntent(dataset_ids=['sample', 'sample'])
    with pytest.raises(ValueError):
        AnalysisIntent(dataset_ids=['sample'], reference_ids=['sample'])
    intent = AnalysisIntent(dataset_ids=['sample'])
    assert AnalysisIntent.model_validate_json(intent.model_dump_json()) == intent


def test_chi_generates_only_measured_spaces_and_cache_is_traceable(tmp_path):
    store, p = sample_project(tmp_path, 'chi')
    first = run_workflow(store, p.id, {'task': 'process', 'dataset_ids': ['sample']}, p.revision)
    assert first['state'] == 'succeeded'
    saved = store.read(p.id)
    pp = saved.processing[0]
    assert {'k', 'chi', 'r', 'chir_mag'} <= pp.arrays.keys()
    assert not {'energy', 'mu', 'norm'} & pp.arrays.keys()
    assert pp.noise.source == 'unknown'  # no automatic high-R white-noise assumption
    assert first['features'] == []
    second = run_workflow(store, p.id, {'dataset_ids': ['sample']}, saved.revision)
    after = store.read(p.id)
    assert len(after.processing) == 1
    assert first['processing_ids'] == second['processing_ids']
    assert any(d['rule'] == 'content_addressed_reuse' for d in second['decisions'])
    assert after.metadata['automation_runs'][0] == first
    assert second['id'] != first['id']
    assert WorkflowResult.model_validate(second).version == 1
    restored = store.import_archive(store.export(p.id))
    assert len(read_workflows(restored)) == 2


def test_chi_cannot_acquire_xanes_or_energy_calibration(tmp_path):
    store, p = sample_project(tmp_path, 'chi')
    run = run_workflow(store, p.id, {'task': 'xanes', 'dataset_ids': ['sample']}, p.revision)
    assert run['state'] == 'needs_input'
    assert any(i['code'] == 'xanes_requires_mu' for i in run['issues'])
    p = store.read(p.id)
    with pytest.raises(ProjectError) as exc:
        propose_parameters(store, p, p.datasets[0], {'energy_shift': 2})
    assert exc.value.code == 'chi_energy_shift'


def test_real_input_windows_and_calibration_are_idempotent(tmp_path):
    store, p = sample_project(tmp_path, offset=3)
    p.calibrations.append(Calibration(id='reference_shift', kind='energy', value=-3.0,
        unit='eV', reference_dataset_ids=['sample'], element='Fe', edge='K',
        applicability_note='Recorded external reference drift; sample chemical shift is retained.'))
    p = store.save(p, p.revision)
    intent = {'dataset_ids': ['sample'], 'calibration_ids': {'sample': 'reference_shift'},
              'conditions': {'check_background_stability': False}}
    run = run_workflow(store, p.id, intent, p.revision)
    assert run['state'] == 'succeeded', run['issues']
    saved = store.read(p.id)
    pp = saved.processing[0]
    energy = load_array(store, p.id, pp.arrays['energy'])
    raw_energy = np.array([float(line.split()[0]) for line in p.datasets[0].raw_text.splitlines() if line and not line.startswith('#')])
    np.testing.assert_allclose(energy, raw_energy - 3, atol=1e-10, rtol=0)
    assert pp.metadata['effective_energy_shift_eV'] == -3
    assert pp.parameters.e0 + pp.parameters.pre1 >= energy[0]
    assert pp.parameters.e0 + pp.parameters.norm2 <= energy[-1]
    again = run_workflow(store, p.id, intent, saved.revision)
    assert again['processing_ids'] == run['processing_ids']
    assert len(store.read(p.id).processing) == 1


def test_unknown_model_still_delivers_descriptive_products(tmp_path):
    store, p = sample_project(tmp_path, 'chi')
    run = run_workflow(store, p.id, {'task': 'exafs', 'dataset_ids': ['sample']}, p.revision)
    assert run['state'] == 'needs_input'
    assert len(run['processing_ids']) == 1
    assert run['fit_run_ids'] == []
    assert any(i['code'] == 'model_required' for i in run['issues'])


def test_background_sensitivity_retains_cases_without_selecting_by_fit(tmp_path):
    store, p = sample_project(tmp_path)
    run = run_workflow(store, p.id, {'dataset_ids': ['sample']}, p.revision)
    assert run['processing_ids']
    saved = store.read(p.id)
    baseline = next(pp for pp in saved.processing if pp.id == run['processing_ids'][0])
    assert baseline.parameters.rbkg == 1
    assert len(saved.processing) == 3
    assert len(baseline.metadata['background_sensitivity']) == 2
    assert 'no final-fit score' in baseline.metadata['background_selection']
    assert all(p.noise.source == 'unknown' for p in saved.processing)


def test_following_edge_is_preserved_mask_and_windows_stay_below_it(tmp_path):
    store, p = sample_project(tmp_path)
    params, mask, issues, evidence = propose_parameters(store, p, p.datasets[0],
        conditions={'next_edge_eV': 7550.0})
    assert mask is not None and (~mask).any()
    assert params.e0 + params.norm2 < 7520
    assert params.kmax + params.dk / 2 <= evidence['k_coverage']
    assert any(i['code'] == 'following_edge_excluded' for i in issues)


def test_bad_observation_partial_success_and_raw_bytes_preserved(tmp_path):
    store, p = sample_project(tmp_path, 'chi')
    raw = '4 1\n' * 45
    broken = Dataset(id='bad', filename='bad.dat', kind='chi', raw_text=raw,
        raw_file=store.put_asset(p.id, raw.encode(), 'bad.dat'), element='Fe', edge='K')
    p.datasets.append(broken)
    p = store.save(p, p.revision)
    run = run_workflow(store, p.id, {'dataset_ids': ['sample', 'bad']}, p.revision)
    assert run['state'] == 'partial'
    assert len(run['processing_ids']) == 1
    assert any(i['code'] == 'invalid_grid' and i.get('dataset_id') == 'bad' for i in run['issues'])
    assert store.read_asset(p.id, broken.raw_file) == raw.encode()


def test_short_postedge_and_raw_text_conflict_are_not_repaired(tmp_path):
    store, p = sample_project(tmp_path)
    x = np.linspace(7050, 7130, 161)
    mu = 1 / (1 + np.exp(-(x - 7112) / 1.5))
    raw = '\n'.join(f'{a:.17g} {b:.17g}' for a, b in zip(x, mu))
    p.datasets[0].raw_file = store.put_asset(p.id, raw.encode(), 'short.dat')
    p.datasets[0].raw_text = raw
    p = store.save(p, p.revision)
    run = run_workflow(store, p.id, {'dataset_ids': ['sample']}, p.revision)
    assert run['state'] == 'needs_input'
    assert any(i['code'] == 'short_postedge' for i in run['issues'])
    p = store.read(p.id)
    p.datasets[0].raw_text += '\n7131 1'
    p = store.save(p, p.revision)
    run = run_workflow(store, p.id, {'dataset_ids': ['sample']}, p.revision)
    assert any(i['code'] == 'raw_text_mismatch' for i in run['issues'])


def test_revision_conflicts_and_cancel_do_not_publish_half_results(tmp_path):
    store, p = sample_project(tmp_path, 'chi')
    with pytest.raises(ProjectError) as exc:
        run_workflow(store, p.id, {'dataset_ids': ['sample']}, p.revision - 1)
    assert exc.value.code == 'revision_conflict'
    class Cancelled(RuntimeError):
        pass
    class Context:
        def heartbeat(self, stage, progress=None):
            if stage == 'publishing immutable scientific workflow':
                raise Cancelled('worker cancelled before publication')
    with pytest.raises(Cancelled):
        run_workflow(store, p.id, {'dataset_ids': ['sample']}, p.revision, Context())
    assert store.read(p.id).revision == p.revision
    assert not store.read(p.id).processing
    assert read_workflows(store.read(p.id)) == []


def test_external_normalized_plot_does_not_invent_raw_mu_or_chi(tmp_path):
    store, p = sample_project(tmp_path)
    p.datasets[0].metadata.update(import_processing_state='normalized',
        external_processing_note='External normalized absorbance, energy in eV, source and normalization reviewed for descriptive plotting.')
    p = store.save(p, p.revision)
    plot = run_workflow(store, p.id, {'task': 'plot', 'dataset_ids': ['sample']}, p.revision)
    assert plot['state'] == 'succeeded'
    saved = store.read(p.id)
    assert len(saved.processing) == 1
    pp = saved.processing[0]
    assert set(pp.arrays) == {'energy', 'norm'}
    assert pp.metadata['external_plot_only'] is True
    assert pp.resolved_calls[0]['operation'] == 'external_plot_import'
    assert 'autobk' not in [c['operation'] for c in pp.resolved_calls]
    refused = run_workflow(store, p.id, {'task': 'process', 'dataset_ids': ['sample']}, saved.revision)
    assert refused['state'] == 'needs_input'
    assert any(i['code'] == 'derived_input' for i in refused['issues'])


def test_external_plot_requires_semantics_and_processing_source(tmp_path):
    store, p = sample_project(tmp_path, 'chi')
    missing = run_workflow(store, p.id, {'task': 'plot', 'dataset_ids': ['sample']}, p.revision)
    assert missing['state'] == 'needs_input'
    assert any(i['code'] == 'external_processing_note_required' for i in missing['issues'])
    p = store.read(p.id)
    p.datasets[0].metadata.update(import_processing_state='chi', external_processing_note='Original externally extracted unweighted chi(k), k in inverse angstrom.')
    p = store.save(p, p.revision)
    plot = run_workflow(store, p.id, {'task': 'plot', 'dataset_ids': ['sample']}, p.revision)
    assert plot['state'] == 'succeeded'
    pp = store.read(p.id).processing[0]
    assert {'k', 'chi', 'r'} <= pp.arrays.keys()
    assert pp.metadata['external_plot_only']
    saved = store.read(p.id)
    analysis = run_workflow(store, p.id, {'task': 'process', 'dataset_ids': ['sample']}, saved.revision)
    assert analysis['state'] == 'succeeded'
    assert analysis['processing_ids'] != plot['processing_ids']
    newest = store.read(p.id).processing[-1]
    assert newest.metadata.get('external_plot_only') is not True


def test_declared_l3_search_ignores_stronger_l2_and_preserves_chemical_shift(tmp_path):
    from larch.xafs import find_e0
    store = ProjectStore(tmp_path / 'projects')
    p = store.create('L3 identity protection, synthetic two-edge negative case')
    energy = np.arange(5500., 6601.)
    # Deliberately give L3 a measured +8 eV chemical shift and a much stronger
    # following L2 edge. The tabulated prior must not erase that shift.
    mu = .5 + np.arctan((energy - 5731) / 3) / np.pi
    mu += 3 * (.5 + np.arctan((energy - 6164) / 3) / np.pi)
    raw = '\n'.join(f'{x:.17g} {y:.17g}' for x, y in zip(energy, mu)) + '\n'
    d = Dataset(id='ce', filename='two-edge.dat', raw_text=raw, element='Ce', edge='L3',
        raw_file=store.put_asset(p.id, raw.encode(), 'two-edge.dat'))
    p.datasets.append(d)
    p = store.save(p, p.revision)
    assert find_e0(energy, mu) > 6100  # original global heuristic would be wrong
    parameters, mask, issues, evidence = propose_parameters(store, p, d)
    assert parameters.e0 == pytest.approx(5731, abs=1)
    assert parameters.e0 != pytest.approx(5723, abs=1)
    assert mask is not None and energy[mask][-1] < 6164 - 30
    assert parameters.e0 + parameters.norm2 < 6164 - 30
    assert evidence['edge_search_prior']['absolute_calibration'] is False
    assert evidence['edge_search_prior']['energy_shift_applied'] == 0
    assert evidence['next_edge_limit_eV'] == 6164
    assert any(i['code'] == 'following_edge_excluded' for i in issues)
    d.edge = 'K'
    with pytest.raises(ProjectError) as exc:
        propose_parameters(store, p, d)
    assert exc.value.code == 'declared_edge_coverage'


def test_header_identity_conflict_is_not_silently_relabelled(tmp_path):
    store, p = sample_project(tmp_path)
    p.datasets[0].element = 'Cu'  # file explicitly declares Fe K
    with pytest.raises(ProjectError) as exc:
        propose_parameters(store, p, p.datasets[0])
    assert exc.value.code == 'header_identity_conflict'


def test_unknown_element_uses_descriptive_search_with_explicit_limitation(tmp_path):
    store, p = sample_project(tmp_path)
    p.datasets[0].element = None
    p.datasets[0].edge = None
    parameters, mask, issues, evidence = propose_parameters(store, p, p.datasets[0])
    assert parameters.e0 is not None
    assert evidence['edge_search_prior']['identity'] == 'unknown'
    assert any(i['code'] == 'unknown_edge_descriptive' for i in issues)


def detector_project(tmp_path, mode='transmission', unit='eV', both=False, declared_mode=True):
    store, original = sample_project(tmp_path)
    table = np.array([list(map(float, line.split())) for line in original.datasets[0].raw_text.splitlines()
                      if line and not line.startswith('#')])
    energy, mu = table[:, 0], table[:, 1]
    i0 = 1e6 * (1 + .01 * np.sin(energy / 50))
    transmission = i0 * np.exp(-mu)
    fluorescence = i0 * mu
    columns = [energy / (1000 if unit == 'keV' else 1), i0]
    names = ['energy ' + unit, 'i0']
    if mode == 'transmission' or both:
        columns.append(transmission)
        names.append('it')
    if mode == 'fluorescence' or both:
        columns.append(fluorescence)
        names.append('if')
    raw = '# XDI/1.0\n# Element.symbol: Fe\n# Element.edge: K\n'
    raw += ''.join(f'# Column.{i + 1}: {name}\n' for i, name in enumerate(names))
    if declared_mode:
        raw += '# Measurement.mode: ' + mode + '\n'
    raw += '\n'.join(' '.join(f'{v:.17g}' for v in row) for row in np.column_stack(columns)) + '\n'
    project = store.create('Raw named XDI detector channels')
    project.datasets.append(Dataset(id='detector', filename='detectors.xdi', raw_text=raw,
        raw_file=store.put_asset(project.id, raw.encode(), 'detectors.xdi'), element='Fe', edge='K', sample_id='iron'))
    return store, store.save(project, project.revision), energy, mu


@pytest.mark.parametrize('mode,unit', [('transmission', 'eV'), ('transmission', 'keV'),
    ('fluorescence', 'eV'), ('fluorescence', 'keV')])
def test_named_xdi_channels_are_resolved_without_treating_i0_as_mu(tmp_path, mode, unit):
    store, p, energy, mu = detector_project(tmp_path, mode, unit)
    original_declaration = p.datasets[0].model_dump(mode='json')
    intent = {'task': 'process', 'dataset_ids': ['detector'], 'conditions': {'check_background_stability': False}}
    run = run_workflow(store, p.id, intent, p.revision)
    assert run['state'] == 'succeeded', run['issues']
    saved = store.read(p.id)
    assert len(saved.datasets) == 2
    assert saved.datasets[0].model_dump(mode='json') == original_declaration
    resolved = saved.datasets[1]
    assert resolved.raw_file == saved.datasets[0].raw_file
    assert resolved.mapping.mode == mode
    assert resolved.mapping.signal == 2 and resolved.mapping.reference == 1
    assert resolved.mapping.energy_unit == unit
    assert resolved.lineage == ['detector']
    pp = saved.processing[0]
    assert pp.dataset_id == resolved.id
    np.testing.assert_allclose(load_array(store, p.id, pp.arrays['mu']), mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(load_array(store, p.id, pp.arrays['energy']), energy, rtol=1e-12, atol=1e-9)
    decision = next(d for d in run['decisions'] if d['stage'] == 'input_mapping')
    assert decision['selected']['resolved_dataset_id'] == resolved.id
    repeated = run_workflow(store, p.id, intent, saved.revision)
    after = store.read(p.id)
    assert len(after.datasets) == 2
    assert len(after.processing) == 1
    assert repeated['processing_ids'] == run['processing_ids']


def test_unnamed_multicolumn_and_mixed_detector_modes_require_input(tmp_path):
    store, p, energy, mu = detector_project(tmp_path, both=True, declared_mode=False)
    run = run_workflow(store, p.id, {'dataset_ids': ['detector']}, p.revision)
    assert run['state'] == 'needs_input'
    assert run['processing_ids'] == []
    assert any(i['code'] == 'mapping_ambiguous_signal' for i in run['issues'])
    p = store.read(p.id)
    raw = '\n'.join(f'{x:.17g} 1000000 {y:.17g}' for x, y in zip(energy, mu)) + '\n'
    p.datasets[0].raw_text = raw
    p.datasets[0].raw_file = store.put_asset(p.id, raw.encode(), 'unnamed.dat')
    p = store.save(p, p.revision)
    run = run_workflow(store, p.id, {'dataset_ids': ['detector']}, p.revision)
    assert run['state'] == 'needs_input'
    assert run['processing_ids'] == []
    assert any(i['code'] == 'mapping_ambiguous' for i in run['issues'])


def test_user_confirmed_mapping_is_respected_and_two_columns_do_not_clone(tmp_path):
    from xasweb.models import Mapping
    store, p, energy, mu = detector_project(tmp_path, both=True, declared_mode=False)
    p.datasets[0].mapping = Mapping(energy=0, signal=2, reference=1, mode='transmission')
    p.datasets[0].metadata['mapping_confirmed'] = True
    p = store.save(p, p.revision)
    run = run_workflow(store, p.id, {'dataset_ids': ['detector'],
        'conditions': {'check_background_stability': False}}, p.revision)
    assert run['state'] == 'succeeded', run['issues']
    saved = store.read(p.id)
    assert len(saved.datasets) == 1
    assert saved.processing[0].dataset_id == 'detector'
    np.testing.assert_allclose(load_array(store, p.id, saved.processing[0].arrays['mu']), mu, rtol=1e-12, atol=1e-12)
    assert next(d for d in run['decisions'] if d['stage'] == 'input_mapping')['selected']['mapping_source'] == 'user_confirmed'


def test_resolved_mapping_ids_remain_consistent_in_reference_calibration_and_series(tmp_path):
    store, p, energy, mu = detector_project(tmp_path)
    p.datasets[0].conditions = {'temperature_K': 300.0}
    p = store.save(p, p.revision)
    observed = float(energy[np.argmax(np.gradient(mu, energy, edge_order=2))])
    run = run_workflow(store, p.id, {'task': 'series', 'dataset_ids': ['detector'],
        'series': {'state_key': 'temperature_K'}, 'conditions': {'check_background_stability': False,
            'reference_calibration': {'reference_dataset_id': 'detector', 'target_energy_eV': observed - 2,
                'standard_confirmed': True, 'note': 'Declared standard run; target uses the same raw-mu derivative definition.'}}}, p.revision)
    assert run['state'] == 'succeeded', run['issues']
    saved = store.read(p.id)
    resolved = saved.datasets[-1]
    assert resolved.id != 'detector'
    assert saved.calibrations[0].reference_dataset_ids == [resolved.id]
    assert saved.calibrations[0].value == pytest.approx(-2)
    assert saved.processing[0].dataset_id == resolved.id
    assert saved.processing[0].calibration_id == saved.calibrations[0].id
    assert run['series']['points'][0]['dataset_id'] == resolved.id
    assert run['series']['points'][0]['state_value'] == 300.0
