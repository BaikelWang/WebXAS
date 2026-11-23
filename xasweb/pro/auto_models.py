"""Bounded orchestration of explicitly supplied, existing FEFF models.

There is deliberately no automatic invention of an absorber geometry. Existing
paths, physical expressions and noise declarations pass through the frozen
FEFFIT adapter, including its information and amplitude-degeneracy gates.
"""
from __future__ import annotations

import time
import math
from uuid import uuid4

from .contracts import ModelRevision, NoiseSpec
from .larch_adapter import build_fit_spec, fit_snapshot
from .parameters import compile_model
from .projects import ProjectError


def model_recipe(project, model):
    selected = {b.dataset_id for b in model.bindings}
    datasets = [d for d in project.datasets if d.id in selected]
    synthetic = all(d.metadata.get('data_kind') == 'same_engine_synthetic' and d.metadata.get('not_experimental_data') is True for d in datasets)
    path_ids = {p.path_id for b in model.bindings for p in b.paths}
    return {'id': 'mr_' + uuid4().hex, 'version': 1, 'source_model_id': model.id,
        'scope': 'same_engine_synthetic_engineering_example' if synthetic else 'user_supplied_model_not_independently_validated',
        'model': model.model_dump(mode='json'),
        'paths': [p.model_dump(mode='json') for p in project.paths if p.id in path_ids],
        'supported_identities': sorted({f'{d.element}-{d.edge}' for d in datasets}),
        'structure_basis': 'Existing explicit FEFF path geometry and supplied parameter graph; no unknown-structure recognition.',
        'initialization': {'policy': 'baseline plus deterministic interior bound fractions', 'maximum_starts': 8, 'random_seed': 0},
        'validation': {'engineering': 'Existing FEFFIT adapter, numerical regression and physical/information gates',
                       'real_experimental_standard': 'pending independent acceptance', 'expert_structure_review': 'not asserted by automation'}}


def independent_model(model, dataset_id, paths):
    """Release sharing by creating a separate immutable single-spectrum model."""
    compiled = compile_model(model, paths)
    binding = next(b for b in model.bindings if b.dataset_id == dataset_id)
    path_ids = {p.id for p in binding.paths}
    used = set()
    for pid in path_ids:
        for expression in compiled.path_expressions.get(pid, {}).values():
            used.update(compiled._closure(expression.names))
    value = model.model_dump(mode='json')
    value.update(id='am_' + uuid4().hex, revision=model.revision + 1,
        bindings=[binding.model_dump(mode='json')],
        parameters={'nodes': [p.model_dump(mode='json') for p in model.parameters.nodes if p.id in used]},
        coordination_groups=[g.model_dump(mode='json') for g in model.coordination_groups if set(g.path_binding_ids) <= path_ids],
        notes=model.notes + '\nIndependent per-spectrum baseline; no cross-state sharing.',
        title=model.title + ' · independent ' + dataset_id)
    return ModelRevision.model_validate(value)


def deterministic_initial_sets(model, limit=3):
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
        raise ProjectError('fit_budget', 'Total starting-point count must be an integer from one to eight.')
    output = []
    # The baseline is always run by fit_snapshot; additions remain inside the
    # declared physical bounds and are never a Cartesian product search.
    fractions = (.25, .75, .4, .6, .15, .85, .5)
    for fraction in fractions[:limit - 1]:
        initial = {p.id: p.minimum + fraction * (p.maximum - p.minimum)
                   for p in model.parameters.nodes if p.kind == 'free' and p.minimum is not None and p.maximum is not None}
        if initial and initial not in output:
            output.append(initial)
    return output


def _rebind(project, original, processing, intent):
    value = original.model_dump(mode='json')
    value.update(id='am_' + uuid4().hex, revision=original.revision + 1,
                 title=original.title + ' · automatic run',
                 notes=original.notes + '\nDerived automatically from ' + original.id + '; model identity and physical relations supplied by the user.')
    old_processing = {p.id: p for p in project.processing}
    for b in value['bindings']:
        pp = processing.get(b['dataset_id'])
        if pp is None:
            raise ProjectError('model_dataset_scope', 'Every model observation must belong to the selected, successfully processed inputs.', b['dataset_id'])
        previous = old_processing.get(b['processing_revision_id'])
        b['processing_revision_id'] = pp.id
        declared = intent.fit.get('noise')
        if declared is not None:
            if not isinstance(declared, dict):
                raise ProjectError('noise_contract', 'fit.noise must be a portable NoiseSpec object.')
            noise = dict(declared)
        elif b['noise']['source'] != 'unknown':
            noise = dict(b['noise'])
        elif previous is not None:
            noise = previous.noise.model_dump(mode='json')
        else:
            noise = pp.noise.model_dump(mode='json')
        noise['processing_revision_id'] = pp.id
        b['noise'] = NoiseSpec.model_validate(noise).model_dump(mode='json')
        # Preserve the model's fit windows; the existing adapter rejects any
        # new processing that fails to cover them rather than silently cropping.
    return ModelRevision.model_validate(value)


def applied_s02_calibrations(project, runs):
    """Recognize applied links, not merely a calibration somewhere in Project."""
    applied = {}
    for run in runs:
        if run.result is None:
            continue
        spec = run.input_snapshot
        compiled = compile_model(spec.model, spec.paths)
        data = {d.id: d for d in spec.datasets}
        joint_links = {link['calibration_id']: link for link in run.result.metadata.get('calibration_links', [])}
        for calibration in spec.calibrations:
            if calibration.kind != 's02' or not calibration.applicability_note.strip():
                continue
            parameter_id = calibration.conditions.get('parameter_id')
            node = compiled.nodes.get(parameter_id) if isinstance(parameter_id, str) else None
            fixed_link = node is not None and node.kind == 'fixed' and math.isclose(node.value, calibration.value, rel_tol=1e-9, abs_tol=1e-12)
            joint_link = joint_links.get(calibration.id)
            if not fixed_link and not joint_link:
                continue
            # These three fields identify the calibration fit, rather than
            # measurement conditions. All other declared conditions must match.
            measurement_conditions = {k: v for k, v in calibration.conditions.items()
                                      if k not in ('source_run_id', 'parameter_id', 'basis')}
            for binding in spec.model.bindings:
                dataset = data[binding.dataset_id]
                if (dataset.element, dataset.edge) != (calibration.element, calibration.edge):
                    continue
                if any(dataset.conditions.get(k) != v for k, v in measurement_conditions.items()):
                    continue
                used = set()
                for path in binding.paths:
                    if path.enabled:
                        used.update(compiled._closure(compiled.path_expressions[path.id]['s02'].names))
                if parameter_id not in used:
                    continue
                applied.setdefault(dataset.id, []).append({'calibration_id': calibration.id,
                    'fit_run_id': run.id, 'parameter_id': parameter_id,
                    'uncertainty_basis': 'joint_reference' if joint_link else 'conditional_fixed'})
    return applied


def eligible_fit_calibrations(spec):
    """Do not let inapplicable metadata trigger a joint-uncertainty label.

    The general adapter preserves all calibration provenance. The automatic
    recipe additionally checks transfer conditions for every model observation
    affected by the calibration's amplitude parameter before passing that
    calibration as a usable fit assumption.
    """
    compiled = compile_model(spec.model, spec.paths)
    data = {d.id: d for d in spec.datasets}
    kept, excluded = [], []
    for calibration in spec.calibrations:
        if calibration.kind != 's02':
            kept.append(calibration)
            continue
        pid = calibration.conditions.get('parameter_id')
        affected = []
        if isinstance(pid, str):
            for binding in spec.model.bindings:
                used = set()
                for path in binding.paths:
                    if path.enabled:
                        used.update(compiled._closure(compiled.path_expressions[path.id]['s02'].names))
                if pid in used:
                    affected.append(data[binding.dataset_id])
        conditions = {k: v for k, v in calibration.conditions.items() if k not in ('source_run_id', 'parameter_id', 'basis')}
        eligible = bool(affected) and bool(calibration.applicability_note.strip()) and all(
            (d.element, d.edge) == (calibration.element, calibration.edge) and
            all(d.conditions.get(k) == v for k, v in conditions.items()) for d in affected)
        if eligible:
            kept.append(calibration)
        else:
            excluded.append({'calibration_id': calibration.id, 'reason': 'No explicit amplitude link or at least one affected element/edge/measurement condition is inapplicable.'})
    return kept, excluded


def run_known_model(store, project, processing, intent, context=None):
    if any(pp.metadata.get('external_plot_only') is True for pp in processing.values()):
        raise ProjectError('external_plot_only', '外部直绘快照只用于描述展示，不能直接进入自动定量 EXAFS；请先建立明确处理/噪声和来源的分析版本。')
    recipe = None
    if intent.model_recipe_id:
        recipe = next((r for r in project.metadata.get('automation_model_recipes', []) if r.get('id') == intent.model_recipe_id), None)
        if recipe is None or recipe.get('version') != 1:
            raise ProjectError('model_recipe_required', '找不到可复用的版本化模型配方，或配方版本不受支持。')
        original = ModelRevision.model_validate(recipe['model'])
        if intent.model_id is not None and intent.model_id != original.id:
            raise ProjectError('model_recipe_identity', '模型 ID 与所选配方来源不一致。')
        available = {p.id: p.model_dump(mode='json') for p in project.paths}
        if any(available.get(p['id']) != p for p in recipe['paths']):
            raise ProjectError('model_recipe_paths', '配方所依据的 FEFF 路径已变化；请建立新配方，不能静默替换来源。')
    else:
        original = next((m for m in project.models if m.id == intent.model_id), None)
    if original is None:
        raise ProjectError('model_required', 'Supply an existing path model or a separately reviewed structure template; descriptive E/k/R/WT results remain available.')
    recipe = recipe or model_recipe(project, original)
    selected = set(intent.dataset_ids)
    if {b.dataset_id for b in original.bindings} != selected:
        raise ProjectError('model_dataset_scope', 'The selected model must bind exactly the selected samples. Create a new explicit model for another observation set.', original.id)
    model = _rebind(project, original, processing, intent)
    compiled = compile_model(model, project.paths)
    if any(d['severity'] == 'error' for d in compiled.diagnostics):
        raise ProjectError('AMPLITUDE_DEGENERACY', compiled.diagnostics[0]['message'], model.id)
    relation = intent.series.get('relation', 'independent') if intent.task == 'series' else 'provided_model'
    if relation not in ('independent', 'provided_model', 'joint_existing'):
        raise ProjectError('relation_template', 'This release supports independent observations or an explicitly supplied existing joint model; new shared relations need expert modeling.')
    if len(model.bindings) > 1 and relation in ('provided_model', 'joint_existing'):
        if intent.conditions.get('joint_applicability_confirmed') is not True or not str(intent.conditions.get('joint_applicability_note', '')).strip():
            raise ProjectError('joint_applicability', 'Confirm and explain which model relations may be shared across the actual measurement states.')
        identities = {(d.element, d.edge) for d in project.datasets if d.id in selected}
        if len(identities) != 1:
            raise ProjectError('joint_cross_edge', 'Automatic shared fitting across absorption edges is outside the approved relation recipe; keep separate models.')
    baseline_models = [independent_model(model, b.dataset_id, project.paths) for b in model.bindings] if len(model.bindings) > 1 else []
    planned = baseline_models + ([model] if relation != 'independent' or len(model.bindings) == 1 else [])
    max_starts = intent.fit.get('max_starts', 3)
    max_evaluations = intent.fit.get('max_evaluations', 500)
    total_seconds = intent.fit.get('timeout_seconds', 180.0)
    if isinstance(total_seconds, bool) or not isinstance(total_seconds, (int, float)) or not 0 < total_seconds <= 3600:
        raise ProjectError('fit_budget', 'Automatic fitting total time budget must be positive and at most 3600 seconds.')
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int) or not 1 <= max_evaluations <= 20000:
        raise ProjectError('fit_budget', 'Automatic fit evaluations must be a bounded integer from one to 20000.')
    started = time.monotonic()
    runs, models, attempts, issues = [], [], [], []
    working = project.model_copy(deep=True)
    for pp in processing.values():
        if pp.id not in {p.id for p in working.processing}:
            working.processing.append(pp)
    for index, candidate in enumerate(planned):
        record = {'model_id': candidate.id, 'dataset_ids': [b.dataset_id for b in candidate.bindings],
                  'kind': 'independent_baseline' if candidate in baseline_models else 'provided_model', 'state': 'failed'}
        remaining = total_seconds - (time.monotonic() - started)
        if remaining <= 0:
            record.update(state='not_run', code='fit_budget_exhausted', message='Total fit budget exhausted; unattempted cases remain unexplored.')
            attempts.append(record)
            issues.append({'code': 'fit_budget_exhausted', 'message': record['message'], 'severity': 'warning'})
            continue
        working.models.append(candidate)
        models.append(candidate)
        try:
            initials = intent.fit.get('initial_sets')
            if initials is None:
                initials = deterministic_initial_sets(candidate, max_starts)
            if not isinstance(initials, list) or len(initials) > 7:
                raise ProjectError('fit_budget', 'At most seven additional initial sets may follow the baseline.')
            spec = build_fit_spec(working, candidate.id, initial_sets=initials, random_seed=0,
                max_evaluations=max_evaluations, timeout_seconds=float(remaining / max(1, len(planned) - index)))
            spec.calibrations, excluded = eligible_fit_calibrations(spec)
            if excluded:
                record['excluded_calibrations'] = excluded
            run = fit_snapshot(store, project.id, spec, context=context)
            runs.append(run)
            record.update(state=run.task_state, fit_run_id=run.id, converged=run.result.converged,
                          rfactor=run.result.statistics.rfactor, attempts=run.result.metadata.get('attempts', []))
            successful = [a for a in record['attempts'] if a.get('success') and a.get('chi_square') is not None]
            if successful:
                best = min(a['chi_square'] for a in successful)
                record['objective_near_equivalent_attempt_indices'] = [a['index'] for a in successful
                    if abs(a['chi_square'] - best) <= max(1e-10, .01 * abs(best))]
                record['near_equivalence_definition'] = 'objective within 1% of best on identical data/settings (engineering diagnostic, not statistical equivalence or structural uniqueness)'
            if not run.result.converged:
                issues.append({'code': 'fit_not_converged', 'message': 'A retained model attempt did not converge; inspect its residuals and termination record.', 'severity': 'warning'})
        except (ValueError, RuntimeError) as exc:
            record.update(code=getattr(exc, 'code', 'model_fit_failed'), message=str(exc))
            issues.append({'code': record['code'], 'message': record['message'], 'severity': 'warning'})
        attempts.append(record)
    applied_calibrations = applied_s02_calibrations(project, runs)
    uncalibrated = sorted(selected - applied_calibrations.keys())
    if uncalibrated:
        issues.append({'code': 's02_uncalibrated', 'message': 'No applicable and actually applied S0² calibration link for datasets: ' + ', '.join(uncalibrated) + '. Parameter errors remain conditional.', 'severity': 'warning'})
    issues.append({'code': 'model_scientific_validation_pending', 'message': 'Running a supplied model and synthetic regression does not independently validate its structure, reference applicability, or production scientific readiness.', 'severity': 'info'})
    decision = {'stage': 'model', 'rule': 'provided_model_bounded_initialization_v1',
        'reason': 'Reuse original FEFF paths and parameter relations, information gates, bounded deterministic starts and independent baselines; no automatic structural uniqueness claim.',
        'selected': {'source_model_id': original.id, 'relation': relation, 'random_seed': 0,
                     'total_timeout_seconds': total_seconds, 'max_starts': max_starts,
                     'comparison_basis': 'same processing, per-spectrum fit windows, weights and declared noise; no ranking of backgrounds',
                     'applied_s02_calibrations': applied_calibrations, 'uncalibrated_dataset_ids': uncalibrated,
                     'model_recipe': recipe},
        'alternatives': attempts}
    return models, runs, issues, decision
