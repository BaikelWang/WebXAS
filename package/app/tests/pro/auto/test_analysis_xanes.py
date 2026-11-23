import numpy as np
import pytest

from xasweb.models import Parameters
from xasweb.pro.contracts import Dataset, ProcessingRevision
from xasweb.pro.preprocessing import load_array, save_array
from xasweb.pro.projects import ProjectError, ProjectStore
from xasweb.pro.workflows import run_workflow
from xasweb.pro.xanes import extract_features, fit_reference_combination, solve_coefficients


def mixture_project(tmp_path, identical=False):
    store = ProjectStore(tmp_path / 'projects')
    p = store.create('Synthetic reference coefficients, not a real standard')
    energy = np.linspace(7000, 7400, 401)
    ref0 = .5 + np.arctan((energy - 7112) / 3) / np.pi + .4 * np.exp(-((energy - 7122) / 4) ** 2)
    ref1 = ref0.copy() if identical else .5 + np.arctan((energy - 7118) / 4) / np.pi + .6 * np.exp(-((energy - 7128) / 5) ** 2)
    signal = .35 * ref0 + .65 * ref1
    for did, data in [('sample', signal), ('ref0', ref0), ('ref1', ref1)]:
        raw = '\n'.join(f'{x:.17g} {y:.17g}' for x, y in zip(energy, data)) + '\n'
        d = Dataset(id=did, filename=did + '.dat', kind='mu', raw_text=raw,
            raw_file=store.put_asset(p.id, raw.encode(), did + '.dat'), element='Fe', edge='K',
            conditions={'potential_V': 0.5 if did == 'sample' else 0.0})
        p.datasets.append(d)
        pp = ProcessingRevision(id='pp_' + did, dataset_id=did, parameters=Parameters(e0=7112.0),
            arrays={'energy': save_array(store, p.id, energy), 'norm': save_array(store, p.id, data)},
            metadata={'source_sha256': d.raw_file.sha256, 'input_mapping': d.mapping.model_dump(mode='json'),
                      'input_kind': 'mu', 'input_kweight': 0})
        p.processing.append(pp)
    return store, store.save(p, p.revision), energy, signal, ref0, ref1


def test_simplex_matches_larch_independent_reference_and_known_mixture():
    from larch import Group
    from larch.math import lincombo_fit
    energy = np.linspace(7080, 7210, 261)
    a = 1 / (1 + np.exp(-(energy - 7112) / 3))
    b = 1 / (1 + np.exp(-(energy - 7118) / 4)) + .3 * np.exp(-((energy - 7128) / 5) ** 2)
    y = .35 * a + .65 * b
    solved = solve_coefficients(np.column_stack([a, b]), y)
    reference = lincombo_fit(Group(energy=energy, norm=y),
        [Group(energy=energy, norm=a, label='a'), Group(energy=energy, norm=b, label='b')],
        arrayname='norm', xmin=float(energy[0]), xmax=float(energy[-1]),
        minvals=[0., 0.], maxvals=[1., 1.], sum_to_one=True, vary_e0=False)
    np.testing.assert_allclose(solved, [.35, .65], atol=2e-7)
    np.testing.assert_allclose(solved, [reference.weights['a'], reference.weights['b']], atol=2e-7)
    np.testing.assert_allclose(np.column_stack([a, b]) @ solved, reference.yfit, atol=2e-7)


def test_nonnegative_boundary_and_sum_option_are_feasible():
    x = np.linspace(0, 2, 101)
    matrix = np.column_stack([1 + x, np.sin(x) + 2])
    bounded = solve_coefficients(matrix, matrix[:, 0])
    np.testing.assert_allclose(bounded, [1, 0], atol=1e-7)
    free_total = solve_coefficients(matrix, 1.8 * matrix[:, 0] + .4 * matrix[:, 1], sum_to_one=False)
    np.testing.assert_allclose(free_total, [1.8, .4], atol=1e-8)


def test_workflow_lcf_reconstruction_references_and_portable_diagnostics(tmp_path):
    store, p, energy, signal, a, b = mixture_project(tmp_path)
    intent = {'task': 'xanes', 'dataset_ids': ['sample'], 'reference_ids': ['ref0', 'ref1'],
        'use_existing_processing': True, 'conditions': {'reference_applicability_confirmed': True,
        'reference_applicability_note': 'Synthetic mathematical mixture; same grid and normalization, not experimental validation.'}}
    run = run_workflow(store, p.id, intent, p.revision)
    assert run['state'] == 'succeeded', run['issues']
    lcf = run['lcf_results'][0]
    np.testing.assert_allclose(lcf['coefficients'], [.35, .65], atol=2e-7)
    from xasweb.pro.contracts import ArrayRef
    read = lambda name: load_array(store, p.id, ArrayRef.model_validate(lcf['arrays'][name]))
    np.testing.assert_allclose(read('model') + read('residual'), read('data'), atol=1e-14)
    np.testing.assert_allclose(read('contribution_0') + read('contribution_1'), read('model'), atol=1e-14)
    assert lcf['diagnostics']['reference_completeness'] == 'assumed_not_proven'
    assert lcf['diagnostics']['uncertainty'] == 'not_estimated_under_constraints'
    assert len(lcf['alternatives']) == 4
    assert len(run['features']) == 3
    assert all(f['absolute_valence'] == 'not_inferred' for f in run['features'])
    assert store.read(p.id).metadata['automation_runs'][0] == run


def test_reference_identity_and_applicability_are_required(tmp_path):
    store, p, *_ = mixture_project(tmp_path)
    for conditions, expected in [({}, 'reference_applicability'),
        ({'reference_applicability_confirmed': True}, 'reference_applicability_note')]:
        with pytest.raises(ProjectError) as exc:
            fit_reference_combination(store, p, p.processing[0], p.processing[1:], conditions=conditions)
        assert exc.value.code == expected
    p.datasets[1].edge = 'L3'
    with pytest.raises(ProjectError) as exc:
        fit_reference_combination(store, p, p.processing[0], p.processing[1:], conditions={})
    assert exc.value.code == 'reference_identity'


def test_collinear_reference_result_is_retained_with_nonunique_warning(tmp_path):
    store, p, *_ = mixture_project(tmp_path, identical=True)
    result = fit_reference_combination(store, p, p.processing[0], p.processing[1:],
        conditions={'reference_applicability_confirmed': True, 'reference_applicability_note': 'Synthetic duplicate references, intentional nonidentifiability.'})
    assert result['diagnostics']['near_collinear']
    assert result['diagnostics']['condition_number'] is None
    assert any('nonunique' in w for w in result['warnings'])
    assert result['diagnostics']['reference_completeness'] != 'verified'


def test_reference_extrapolation_and_free_energy_alignment_are_forbidden(tmp_path):
    store, p, *_ = mixture_project(tmp_path)
    conditions = {'reference_applicability_confirmed': True, 'reference_applicability_note': 'Synthetic test.'}
    with pytest.raises(ProjectError) as exc:
        fit_reference_combination(store, p, p.processing[0], p.processing[1:], {'vary_e0': True}, conditions)
    assert exc.value.code == 'lcf_shift_not_supported'
    with pytest.raises(ProjectError) as exc:
        fit_reference_combination(store, p, p.processing[0], p.processing[1:], {'window_eV': [6990., 7150.]}, conditions)
    assert exc.value.code == 'reference_coverage'


def test_true_chemical_shift_is_reported_without_alignment(tmp_path):
    store, p, *_ = mixture_project(tmp_path)
    first = extract_features(store, p.id, p.processing[1])
    second = extract_features(store, p.id, p.processing[2])
    assert second['edge_position_eV'] - first['edge_position_eV'] >= 5
    assert first['derivative_method']['smoothing'] is False
    assert 'maximum first derivative' in first['edge_definition']
