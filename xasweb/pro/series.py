"""Series products use actual state metadata and retain failed observations."""
from __future__ import annotations

import numpy as np

from .preprocessing import load_array, save_array
from .projects import ProjectError


def build_series(store, project, intent, processing, features, lcf_results, fit_runs):
    state_key = intent.series.get('state_key') or intent.conditions.get('state_key')
    if not isinstance(state_key, str) or not state_key.strip():
        raise ProjectError('series_state_required', 'Select an actual state metadata key (time, potential, temperature, etc.); file order is not a measured state.')
    by_id = {d.id: d for d in project.datasets}
    feature_map = {f['dataset_id']: f for f in features}
    lcf_map = {f['dataset_id']: f for f in lcf_results}
    points = []
    for did in intent.dataset_ids:
        dataset = by_id[did]
        state = dataset.conditions.get(state_key)
        if isinstance(state, bool) or not isinstance(state, (int, float, str)) or (isinstance(state, str) and not state.strip()):
            state = None
        pp = processing.get(did)
        point = {'dataset_id': did, 'state_value': state, 'processing_id': pp.id if pp else None,
                 'state': 'succeeded' if pp and state is not None else 'needs_input',
                 'reason': None if pp and state is not None else ('processing unavailable' if pp is None else 'actual state metadata missing'),
                 'edge_position_eV': feature_map.get(did, {}).get('edge_position_eV'),
                 'white_line_height': feature_map.get(did, {}).get('white_line_height'),
                 'reference_coefficients': lcf_map.get(did, {}).get('coefficients'),
                 'fits': []}
        for run in fit_runs:
            if run.result is not None and any(r.dataset_id == did for r in run.result.datasets):
                point['fits'].append({'fit_run_id': run.id, 'converged': run.result.converged,
                    'parameters': [p.model_dump(mode='json') for p in run.result.parameters],
                    'uncertainty_basis': run.result.uncertainty_basis})
        points.append(point)
    # No smoothing, interpolation over failed states, or fabricated integer
    # sequence coordinate is used. Stable input order remains explicit.
    result = {'state_key': state_key, 'points': points, 'order': 'selected observation order',
              'trend_inference': 'descriptive association only; not causal evidence',
              'failed_state_policy': 'retain point with null products and reason; never interpolate as success',
              'arrays': {}, 'display_intersection': None}
    usable = [p for p in points if p['processing_id'] and p['state_value'] is not None]
    pp_list = [processing[p['dataset_id']] for p in usable]
    if not pp_list:
        return result
    axis, signal = ('energy', 'norm') if all({'energy', 'norm'} <= p.arrays.keys() for p in pp_list) else ('k', 'chi')
    compatible = [(point, pp) for point, pp in zip(usable, pp_list) if {axis, signal} <= pp.arrays.keys()]
    if len(compatible) != len(pp_list):
        result['display_reason'] = 'Mixed mu(E)/chi(k) products have no common display space; individual curves remain available.'
        return result
    xs = [load_array(store, project.id, pp.arrays[axis]) for _, pp in compatible]
    lo, hi = max(x[0] for x in xs), min(x[-1] for x in xs)
    grid = xs[0][(xs[0] >= lo) & (xs[0] <= hi)]
    if len(grid) < 8:
        result['display_reason'] = 'Insufficient common measured range; original processing ranges retained.'
        return result
    matrix = np.vstack([np.interp(grid, x, load_array(store, project.id, pp.arrays[signal]))
                        for x, (_, pp) in zip(xs, compatible)])
    arrays = {axis: grid, 'matrix': matrix}
    if all(isinstance(p['state_value'], (int, float)) and not isinstance(p['state_value'], bool) for p, _ in compatible):
        arrays['state_values'] = np.asarray([p['state_value'] for p, _ in compatible], float)
    result.update(arrays={k: save_array(store, project.id, v).model_dump(mode='json') for k, v in arrays.items()},
                  matrix_dataset_ids=[p['dataset_id'] for p, _ in compatible],
                  display_intersection={'axis': axis, 'signal': signal, 'minimum': float(lo), 'maximum': float(hi),
                                        'scope': 'display only; fit windows and original processing arrays remain unchanged'})
    return result
