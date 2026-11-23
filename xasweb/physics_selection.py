"""Reference-free, bounded coarse-to-fine recipe search with conservative fallback.

The thresholds are development guardrails, not calibrated scientific acceptance.
The initial recipe defines the comparison domain so candidates cannot win by
shortening the FT interval. Reference spectra are intentionally not accepted.
"""
from __future__ import annotations

import numpy as np

from .models import Parameters


VERSION = 'physics-coordinate-v2-fixed-diagnostic-domains'


def preservation(candidate, initial):
    p0 = initial['request']['parameters']
    p = candidate['request']['parameters']
    domain = np.arange(max(2.,p0['kmin']), p0['kmax'] + .00001, .05)
    k0, k = np.asarray(initial['arrays']['k']), np.asarray(candidate['arrays']['k'])
    flags = []
    if k[-1] < domain[-1] or k[0] > domain[0] or p['kmax'] < p0['kmax']:
        return {'eligible':False,'flags':['shortened_comparison_domain'],'comparison_k_range':[float(domain[0]),float(domain[-1])]}
    a = np.interp(domain,k0,initial['arrays']['chi']) * domain**2
    b = np.interp(domain,k,candidate['arrays']['chi']) * domain**2
    scale = max(float(np.linalg.norm(a)),1e-12)
    ratio = float(np.linalg.norm(b)/scale)
    change = float(np.linalg.norm(b-a)/scale)
    correlation = float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
    if not .8 <= ratio <= 1.25: flags.append('amplitude_change_requires_review')
    if change > .3: flags.append('processing_instability_requires_review')
    if correlation < .97: flags.append('spectral_shape_change_requires_review')
    if not all(np.isfinite([ratio,change,correlation])): flags.append('nonfinite_comparison')
    return {'eligible':not flags,'flags':flags,'weighted_amplitude_ratio':ratio,
            'weighted_relative_change':change,'weighted_correlation':correlation,
            'comparison_k_range':[float(domain[0]),float(domain[-1])]}


def common_diagnostics(result, initial):
    """Diagnostic masks belong to the initial recipe, never the candidate."""
    arrays=result['arrays'];energy=np.asarray(arrays['energy'])
    p=initial['request']['parameters'];edge=initial['metrics']['e0_eV']
    pre=(energy>=edge+p['pre1'])&(energy<=edge+p['pre2'])
    post=(energy>edge+p['norm1'])&(energy<edge+p['norm2'])
    if not pre.any() or not post.any():raise ValueError('Initial diagnostic intervals lack support.')
    residual=(np.asarray(arrays['mu'])-np.asarray(arrays['pre_edge']))/result['metrics']['edge_step']
    return {'pre_edge_noise':float(np.sqrt(np.mean(residual[pre]**2))),
            'post_edge_offset':float(abs(np.median(np.asarray(arrays['norm'])[post])-1)),
            'fixed_pre_edge_eV':[edge+p['pre1'],edge+p['pre2']],
            'fixed_post_edge_eV':[edge+p['norm1'],edge+p['norm2']]}


def merit(diagnostics, comparison):
    # Roughness is reported by the engine but intentionally excluded here.
    # Penalize changes that consume signal-preservation and stability margins.
    return (diagnostics['pre_edge_noise'] + .1*diagnostics['post_edge_offset']
            + .03*comparison.get('weighted_relative_change',0)
            + .03*abs(np.log(max(comparison.get('weighted_amplitude_ratio',1),1e-12))))


def search(session, candidates):
    """Initial engine call already belongs to session; no extra numerical calls."""
    if not session.results:
        return {'selector_version':VERSION,'fallback_reason':'initial_recipe_failed',
                'selection':'no_initial_recipe','selected_attempt':None}
    initial = session.results[0][1]
    initial_diag = common_diagnostics(initial,initial)
    initial_comparison = preservation(initial,initial)
    best, best_score, best_attempt = initial, merit(initial_diag,initial_comparison), 1
    initial_score = best_score
    session.trace[0]['selection_diagnostics'] = {**initial_comparison,'merit':best_score,'stage':'initial','common_diagnostics':initial_diag}
    base = Parameters.model_validate(initial['request']['parameters'])
    seen = {base.model_dump_json()}
    # Coarse scan spans all key parameter families within the input-supported recipe.
    coarse = [dict(e0=base.e0-1), dict(e0=base.e0+1), dict(rbkg=.8), dict(rbkg=1.2),
              dict(nnorm=1), dict(norm1=base.norm1*.8,norm2=base.norm2*.9),
              dict(bkg_kweight=2)]
    queue = [('coarse',base.model_copy(update=change)) for change in coarse]
    while len(session.trace) < session.budget and not session.hard_stopped:
        if not queue:
            center = Parameters.model_validate(best['request']['parameters'])
            # Refinement preserves FT coverage; taper and lower bound vary explicitly.
            deltas = [('e0',.5),('rbkg',.1),('pre2',5.),('norm1',20.),('norm2',30.),
                      ('dk',.2),('kmin',.2)]
            for name,delta in deltas:
                for direction in (-1,1):
                    values = center.model_dump()
                    values[name] += direction*delta
                    if name=='norm2': values[name]=min(values[name],base.norm2)
                    try: parameter=Parameters.model_validate(values)
                    except ValueError: continue
                    if parameter.model_dump_json() not in seen:
                        queue.append(('fine',parameter))
            if not queue: break
        stage, parameter = queue.pop(0)
        identity = parameter.model_dump_json()
        if identity in seen: continue
        seen.add(identity)
        result = session.evaluate(parameter)
        if result is None: continue
        comparison = preservation(result,initial)
        common = common_diagnostics(result,initial)
        score = merit(common,comparison)
        session.trace[-1]['selection_diagnostics'] = {**comparison,'merit':score,'stage':stage,'common_diagnostics':common}
        # Require an actual margin to prevent floating-point tie-driven recipe churn.
        if comparison['eligible'] and score < best_score - max(1e-5,.01*initial_score):
            best,best_score,best_attempt=result,score,len(session.trace)
    session.selected_result = best
    fallback = best_attempt == 1
    return {'selector_version':VERSION,'selection':'physics_coarse_to_fine',
            'selected_attempt':best_attempt,'fallback_used':fallback,
            'fallback_reason':'no_supported_candidate_improved_joint_diagnostics' if fallback else None,
            'initial_merit':initial_score,'selected_merit':best_score,
            'threshold_status':'uncalibrated development guardrails; Requires review',
            'reference_inputs_used':False,
            'limitations':['Signal preservation is relative to the initial recipe, not an independent accuracy reference.',
                           'Agreement with the initial recipe does not rule out shared processing bias.',
                           'No structural model is supplied by this reduction selector.']}
