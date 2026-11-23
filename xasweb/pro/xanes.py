"""Defined XANES features and bounded nonnegative reference coefficients.

Fits use the same measured intersection and normalization convention. Coefficients
are never relabelled as concentration or oxidation state. Constraint boundaries,
reference collinearity and window/reference sensitivity are retained.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, nnls

from .preprocessing import load_array, save_array
from .projects import ProjectError


def _spectrum(store, project_id, processing):
    if not {'energy', 'norm'} <= processing.arrays.keys():
        raise ProjectError('xanes_unavailable', 'XANES requires measured energy and normalized mu(E); chi(k) cannot supply them.', processing.dataset_id)
    energy = load_array(store, project_id, processing.arrays['energy'])
    norm = load_array(store, project_id, processing.arrays['norm'])
    if energy.ndim != 1 or norm.shape != energy.shape or len(energy) < 8 or np.any(np.diff(energy) <= 0):
        raise ProjectError('xanes_grid', 'XANES requires matching, strictly increasing measured energy and normalized absorption.')
    return energy, norm


def extract_features(store, project_id, processing):
    energy, norm = _spectrum(store, project_id, processing)
    e0 = processing.parameters.e0 or processing.metadata.get('metrics', {}).get('e0_eV')
    if e0 is None:
        raise ProjectError('feature_e0', 'XANES feature extraction requires a recorded normalization E0.', processing.dataset_id)
    use = (energy >= e0 - 20) & (energy <= e0 + 30)
    if use.sum() < 8 or np.max(np.diff(energy[use]), initial=0) > 5:
        raise ProjectError('feature_sampling', 'Near-edge features require at least eight points with gaps no larger than 5 eV.', processing.dataset_id)
    # This derivative does not replace raw or fitted data and includes its
    # exact estimator and boundary convention in the portable record.
    derivative = np.gradient(norm, energy, edge_order=2)
    indices = np.flatnonzero(use)
    edge_index = int(indices[np.argmax(derivative[use])])
    white = (energy >= e0) & (energy <= e0 + 40)
    widx = np.flatnonzero(white)
    white_index = int(widx[np.argmax(norm[white])])
    return {'dataset_id': processing.dataset_id, 'processing_id': processing.id,
        'edge_position_eV': float(energy[edge_index]),
        'edge_definition': 'maximum first derivative of normalized mu on recorded E0 [-20,+30] eV window',
        'white_line_energy_eV': float(energy[white_index]), 'white_line_height': float(norm[white_index]),
        'white_line_definition': 'maximum normalized mu in recorded E0 [0,+40] eV window; not a peak-model decomposition',
        'white_line_at_window_boundary': white_index in (int(widx[0]), int(widx[-1])),
        'derivative_method': {'algorithm': 'numpy.gradient', 'edge_order': 2, 'smoothing': False,
                              'sampling': 'original measured energy points'},
        'absolute_valence': 'not_inferred',
        'arrays': {'energy': processing.arrays['energy'].model_dump(mode='json'),
                   'derivative': save_array(store, project_id, derivative).model_dump(mode='json')}}


def solve_coefficients(matrix, data, sum_to_one=True):
    """Nonnegative least squares, with an optional exact simplex constraint.

    No unconstrained covariance is passed off as uncertainty at the simplex
    boundary. Acceptance tests check reconstruction and constraint feasibility.
    """
    matrix, data = np.asarray(matrix, float), np.asarray(data, float)
    if matrix.ndim != 2 or data.ndim != 1 or matrix.shape[0] != len(data) or matrix.shape[1] < 1:
        raise ProjectError('lcf_shape', 'Reference matrix and measured data have incompatible shapes.')
    if not np.isfinite(matrix).all() or not np.isfinite(data).all():
        raise ProjectError('lcf_nonfinite', 'LCF inputs must be finite.')
    if len(data) <= matrix.shape[1] + 2:
        raise ProjectError('lcf_points', 'Insufficient measured points for selected reference coefficients.')
    initial, _ = nnls(matrix, data)
    if not sum_to_one:
        return initial
    if matrix.shape[1] == 1:
        return np.ones(1)
    initial = initial / initial.sum() if initial.sum() > 1e-15 else np.ones(matrix.shape[1]) / matrix.shape[1]
    scale = max(float(np.dot(data, data)), 1.0)
    result = minimize(lambda c: float(np.sum((matrix @ c - data) ** 2) / scale), initial,
        jac=lambda c: 2 * (matrix.T @ (matrix @ c - data)) / scale,
        method='SLSQP', bounds=[(0, 1)] * matrix.shape[1],
        constraints=[{'type': 'eq', 'fun': lambda c: float(c.sum() - 1),
                      'jac': lambda c: np.ones(len(c))}], options={'ftol': 1e-14, 'maxiter': 1000})
    if not result.success or not np.isfinite(result.x).all() or np.min(result.x) < -1e-9 or abs(result.x.sum() - 1) > 1e-8:
        raise ProjectError('lcf_optimization', 'Constrained reference-coefficient optimization did not return a feasible solution.')
    return np.maximum(result.x, 0.0)


def fit_reference_combination(store, project, sample, references, options=None, conditions=None):
    options, conditions = dict(options or {}), dict(conditions or {})
    if not 1 <= len(references) <= 8:
        raise ProjectError('reference_count', 'LCF accepts one to eight explicitly selected references.', sample.dataset_id)
    if any(p.metadata.get('external_plot_only') is True for p in [sample, *references]):
        raise ProjectError('external_plot_only', '外部直绘快照没有经过本软件的统一处理验收，不能直接进入定量 LCF；请建立可核查的分析版本。', sample.dataset_id)
    datasets = {d.id: d for d in project.datasets}
    target = datasets[sample.dataset_id]
    selected = [datasets[p.dataset_id] for p in references]
    if not target.element or not target.edge or any((d.element, d.edge) != (target.element, target.edge) for d in selected):
        raise ProjectError('reference_identity', 'Quantitative reference comparison needs confirmed matching element and absorption edge.', target.id)
    if conditions.get('reference_applicability_confirmed') is not True:
        raise ProjectError('reference_applicability', 'Confirm reference identities, consistent normalization and measurement applicability before quantitative coefficients.', target.id)
    if not str(conditions.get('reference_applicability_note', '')).strip():
        raise ProjectError('reference_applicability_note', 'Record the physical basis for reference applicability; a residual alone cannot establish it.', target.id)
    if options.get('vary_e0') or options.get('energy_shifts'):
        raise ProjectError('lcf_shift_not_supported', 'Reference energy shifts are disabled in this recipe; retain chemical shifts and use explicit calibrated revisions.', target.id)
    if options.get('nonnegative', True) is not True:
        raise ProjectError('lcf_constraints', 'The validated automation scope requires nonnegative reference coefficients.', target.id)
    sum_to_one = options.get('sum_to_one', True)
    if not isinstance(sum_to_one, bool):
        raise ProjectError('lcf_constraints', 'sum_to_one must be a boolean.', target.id)
    e, y = _spectrum(store, project.id, sample)
    ref_arrays = [_spectrum(store, project.id, p) for p in references]
    e0 = sample.parameters.e0 or sample.metadata.get('metrics', {}).get('e0_eV')
    if e0 is None:
        raise ProjectError('lcf_e0', 'LCF requires a recorded near-edge reference E0.', target.id)
    window = options.get('window_eV', [float(e0 - 20), float(e0 + 80)])
    if not isinstance(window, list) or len(window) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in window) or window[0] >= window[1]:
        raise ProjectError('lcf_window', 'LCF window must contain two ordered absolute energies in eV.', target.id)
    lo, hi = map(float, window)
    if lo < e[0] or hi > e[-1] or any(lo < re[0] or hi > re[-1] for re, _ in ref_arrays):
        raise ProjectError('reference_coverage', 'Every reference and sample must cover the complete declared LCF window; extrapolation is forbidden.', target.id)
    mask = (e >= lo) & (e <= hi)
    energy, data = e[mask], y[mask]
    if len(energy) < max(15, len(references) + 3):
        raise ProjectError('lcf_points', 'LCF window lacks sufficient measured sample points.', target.id)
    matrix = np.column_stack([np.interp(energy, re, ry) for re, ry in ref_arrays])
    coeff = solve_coefficients(matrix, data, sum_to_one)
    model = matrix @ coeff
    residual = data - model
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-15 * singular[0] else None
    collinear = rank < len(references) or condition is None or condition > 1e6
    alternatives = []
    for trim in (5.0, 10.0):
        use = (energy >= lo + trim) & (energy <= hi - trim)
        record = {'kind': 'window', 'window_eV': [lo + trim, hi - trim], 'status': 'not_applicable'}
        if use.sum() >= max(15, len(references) + 3):
            alt = solve_coefficients(matrix[use], data[use], sum_to_one)
            record.update(status='succeeded', coefficients=alt.tolist(),
                          rms=float(np.sqrt(np.mean((data[use] - matrix[use] @ alt) ** 2))))
        alternatives.append(record)
    # Leave-one-out alternatives preserve plausible incomplete-reference
    # explanations. They are sensitivity checks, not an automatic winner list.
    if len(references) > 1:
        for index in range(min(len(references), 7)):
            keep = [i for i in range(len(references)) if i != index]
            alt = solve_coefficients(matrix[:, keep], data, sum_to_one)
            alternatives.append({'kind': 'reference_omission', 'omitted_reference_id': references[index].dataset_id,
                'reference_ids': [references[i].dataset_id for i in keep], 'coefficients': alt.tolist(),
                'rms': float(np.sqrt(np.mean((data - matrix[:, keep] @ alt) ** 2))), 'status': 'succeeded'})
    boundaries = [bool(c <= 1e-6 or (sum_to_one and c >= 1 - 1e-6)) for c in coeff]
    window_coeffs = [a['coefficients'] for a in alternatives if a['kind'] == 'window' and a['status'] == 'succeeded']
    spread = np.max(np.abs(np.array(window_coeffs) - coeff), axis=0).tolist() if window_coeffs else None
    warnings = ['Reference coverage/completeness and measurement applicability remain assumptions; a small residual cannot prove them.',
                'Coefficients are conditional reference-component coefficients, not automatically concentration or average valence.',
                'No Gaussian standard errors are assigned under nonnegativity/sum constraints; sensitivities are not confidence intervals.']
    if collinear:
        warnings.append('References are nearly collinear under the declared matrix diagnostic; coefficients may be nonunique.')
    if any(boundaries):
        warnings.append('At least one coefficient is on a constraint boundary.')
    if spread and max(spread) > .1:
        warnings.append('Reference coefficients change by more than the engineering attention threshold (0.1) under retained window changes.')
    arrays = {'energy': energy, 'data': data, 'model': model, 'residual': residual}
    arrays.update({'contribution_' + str(i): matrix[:, i] * coeff[i] for i in range(len(coeff))})
    return {'dataset_id': target.id, 'reference_ids': [p.dataset_id for p in references],
        'processing_ids': [sample.id, *[p.id for p in references]], 'coefficients': coeff.tolist(),
        'coefficient_label': 'conditional reference-component coefficient',
        'constraints': {'nonnegative': True, 'sum_to_one': sum_to_one, 'energy_shifts': False},
        'window_eV': [lo, hi], 'applicability_note': conditions['reference_applicability_note'],
        'engine': {'solver': 'scipy.optimize.SLSQP' if sum_to_one else 'scipy.optimize.nnls',
                   'objective': 'unweighted least squares on measured sample grid; linear reference interpolation, no extrapolation',
                   'validation_level': 'engineering and synthetic-mixture regression; real standard acceptance pending'},
        'diagnostics': {'rms': float(np.sqrt(np.mean(residual ** 2))), 'matrix_rank': rank,
                        'condition_number': condition, 'near_collinear': collinear,
                        'coefficient_at_boundary': boundaries, 'window_coefficient_max_abs_change': spread,
                        'uncertainty': 'not_estimated_under_constraints', 'reference_completeness': 'assumed_not_proven'},
        'alternatives': alternatives, 'warnings': warnings,
        'arrays': {name: save_array(store, project.id, value).model_dump(mode='json') for name, value in arrays.items()}}
