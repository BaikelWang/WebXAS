"""Finite-step compatibility, including RuntimeWarning-as-error validation."""
from pathlib import Path

import numpy as np
import pytest

from xasweb.ingest import parse_ascii
from xasweb.pro.auto_processing import finite_find_e0, propose_parameters
from xasweb.pro.contracts import Dataset
from xasweb.pro.preprocessing import load_array, process_dataset
from xasweb.pro.projects import ProjectError, ProjectStore


@pytest.mark.parametrize('step', [.25, .5, 2., 4., 5.])
def test_short_local_edge_windows_use_explicit_finite_full_grid_step(step):
    full = np.arange(6900., 7800. + step / 2, step)
    mu = .5 + np.arctan((full - 7112.) / 3) / np.pi
    half_width = 10 if step < 1 else 80
    use = abs(full - 7112.) <= half_width
    assert 8 <= use.sum() < 100
    original = mu.copy()
    e0, record = finite_find_e0(full[use], mu[use], full_energy=full)
    assert abs(e0 - 7112.) <= step
    assert record['raw_energy_step_eV'] == pytest.approx(step)
    expected_scale = .5 * (max(.01, min(1., step)) + max(.01, min(1., record['first_e0_eV'] / 25000.)))
    assert record['smoothing_step_eV'] == pytest.approx(expected_scale)
    assert np.isfinite(e0) and record['smoothing_step_eV'] > 0
    if step < 1:
        assert record['clipped_energy_step_eV'] < 1  # no former NaN-to-one fallback
    np.testing.assert_array_equal(mu, original)


def test_nonuniform_measured_grid_has_finite_step_without_resampling_fit_data():
    full = np.unique(np.concatenate([np.arange(6900., 7080., 3.), np.arange(7080., 7150., .35),
                                    np.arange(7150., 7801., 4.)]))
    mu = .5 + np.arctan((full - 7113.2) / 2) / np.pi
    use = abs(full - 7113.2) <= 14
    assert use.sum() < 100
    e0, record = finite_find_e0(full[use], mu[use], full_energy=full)
    assert abs(e0 - 7113.2) <= .35
    assert record['raw_energy_step_eV'] == pytest.approx(.35)
    assert record['full_grid_points'] == len(full)
    assert record['search_points'] == use.sum()
    assert record['data_smoothing'] is False


@pytest.mark.parametrize('step', [.25, 1., 4.])
def test_normal_long_window_matches_pinned_public_larch(step):
    from larch.xafs import find_e0
    energy = np.arange(6900., 7900. + step / 2, step)
    mu = .5 + np.arctan((energy - 7115.) / 3) / np.pi
    expected = find_e0(energy, mu)
    actual, _ = finite_find_e0(energy, mu, full_energy=energy)
    assert actual == expected


@pytest.mark.parametrize('case', ['short', 'duplicate', 'nonfinite', 'too_fine'])
def test_invalid_edge_grids_raise_explicit_error_before_kernel(case):
    energy = np.arange(7100., 7121.)
    mu = .5 + np.arctan((energy - 7112) / 3) / np.pi
    if case == 'short':
        energy, mu = energy[:5], mu[:5]
    elif case == 'duplicate':
        energy[3] = energy[2]
    elif case == 'nonfinite':
        mu[3] = np.nan
    else:
        energy = 7112 + np.arange(len(energy)) * 1e-5
    with pytest.raises(ProjectError) as exc:
        finite_find_e0(energy, mu)
    assert exc.value.code == 'edge_search_grid'


@pytest.mark.parametrize('name,element,tabulated', [
    ('Cu_Kedge/Cu_Kedge_metal_foil_50K.dat', 'Cu', 8979.),
    ('Fe_Kedge/Fe_Kedge_metal_foil_300K.dat', 'Fe', 7112.),
])
def test_real_foil_edge_and_processing_match_previous_public_local_search(tmp_path, name, element, tabulated):
    from larch.xafs import find_e0
    root = Path(__file__).resolve().parents[5]
    raw = (root / 'benchmarks/inputs/delivered_demo' / name).read_text(encoding='utf-8')
    table, _ = parse_ascii(raw)
    energy, mu = table[:, 0], table[:, 1]
    use = abs(energy - tabulated) <= 80
    assert use.sum() >= 100  # public reference is numerically defined here
    previous_e0 = find_e0(energy[use], mu[use])
    store = ProjectStore(tmp_path / 'projects')
    project = store.create('Real foil finite-step regression; no structural validation claim')
    dataset = Dataset(id='foil', filename=Path(name).name, raw_text=raw,
        raw_file=store.put_asset(project.id, raw.encode(), 'foil.dat'), element=element, edge='K')
    project.datasets.append(dataset)
    project = store.save(project, project.revision)
    parameters, mask, _, evidence = propose_parameters(store, project, dataset)
    assert parameters.e0 == previous_e0
    previous_parameters, previous_mask, _, _ = propose_parameters(store, project, dataset, {'e0': previous_e0})
    assert previous_parameters == parameters
    actual = process_dataset(store, project.id, 'foil', parameters, mask)
    baseline = process_dataset(store, project.id, 'foil', previous_parameters, previous_mask)
    for field in ('energy', 'mu', 'norm', 'k', 'chi', 'chir_re', 'chir_im', 'chir_mag'):
        np.testing.assert_array_equal(load_array(store, project.id, actual.arrays[field]),
                                      load_array(store, project.id, baseline.arrays[field]))
    assert evidence['edge_search_prior']['e0_estimator']['algorithm'].endswith('finite-step-v1')
