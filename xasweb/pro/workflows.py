"""One disposable worker performs a bounded scientific workflow.

The captured project is the only scientific input. Intermediate assets are
content-addressed; scientific records and the immutable workflow are committed
together under an optimistic revision check. A cancelled/killed worker cannot
publish a half-rebound model or a false succeeded run. No nested job is queued.
"""
from __future__ import annotations

from uuid import uuid4

from .auto_contracts import AnalysisIntent, WorkflowResult, read_workflows, effective_operations, normalize_intent, validate_calibration_scope
from .auto_processing import FrozenProjectStore, estimate_reference_calibration, import_external_plot, prepare_input_mappings, prepare_repeat_merges, process_automatically
from .contracts import Project
from .jobs import utc_now
from .preprocessing import load_array
from .projects import ProjectError, ProjectStore


_MESSAGES = {
    'model_required': '尚未提供可用的结构/路径模型。已保留描述性处理结果；请在专家模式建立模型或选择已有模型后再拟合。',
    'model_dataset_scope': '模型包含的谱与本次选定样品不一致，或部分谱处理失败。请选择匹配的模型，不能静默删掉模型中的观测。',
    'raw_text_mismatch': '工程中的可编辑文本与保留的原始字节不一致。请明确导入为新数据，避免来源哈希失真。',
    'invalid_grid': '原始坐标存在重复点或倒序；本次未自动排序、丢点或修复。请检查原文件。',
    'short_postedge': '边后能区不足，无法完成当前支持的归一化和背景流程；请补充测量范围。',
    'preedge_coverage': '边前窗口内有效测量点不足；请补充能区或检查吸收边位置。',
    'norm_coverage': '归一化窗口没有足够的测量覆盖；请检查边位或补充边后能区。',
    'uncalibrated_energy': '能量按原文件标尺比较；没有参考标准时不生成绝对校准，也不把样品边移强行对齐。',
    'background_sensitive': 'rbkg 改变 0.1 Å 时 χ(k) 的 RMS 变化超过工程关注阈值 25%；已保留敏感性分支，请检查背景选择。该阈值不是置信度。',
    'background_sensitivity_incomplete': '附近背景参数候选失败；基准结果保留，但不能声称已通过稳定性验证。',
    'following_edge_excluded': '已用保留的掩码排除所声明下一吸收边前 30 eV 及其以上的数据；原始点没有删除。',
    'fluorescence_correction_unverified': '没有编造死时间或自吸收修正；荧光谱的定量解释仍需核查外部校正依据。',
    'sparse_near_edge': '近边采样偏稀；导数特征和边位需要人工检查。',
    'large_sampling_gap': '谱中存在较大采样间隙；插值不会增加独立信息，相关结果需检查。',
    'suspected_edge_spike': '近边存在疑似尖峰的突跳回落；本次未平滑或掩掉原始点，请先核查测量。',
    'reference_identity': '参考谱与样品必须明确具有相同吸收元素和吸收边，才能执行条件组分分析。',
    'reference_applicability': '请确认参考样身份、统一归一化和测量条件适用性，再计算条件参考组分系数。',
    'reference_applicability_note': '请记录参考样适用于目标样品的物理依据；小残差不能证明参考库完整。',
    'reference_coverage': '样品与全部参考谱必须完整覆盖同一 LCF 窗口；本次不做参考外推。',
    'reference_processing_failed': '部分选定参考谱处理失败；本次没有悄悄缩减参考库后继续定量拟合。',
    'lcf_near_collinear': '参考谱近共线，系数可能不唯一；已保留窗口/参考组合敏感性与近等价解释。',
    'lcf_shift_not_supported': '本配方关闭参考谱自由能移；保留真实化学位移，请使用明确的校准版本。',
    'joint_applicability': '联合拟合需要确认并说明哪些物理关系可以跨状态共享。各状态的真实结构参数不会被默认共享。',
    'joint_cross_edge': '当前自动共享配方不跨吸收边使用；不同吸收边请建立独立模型。',
    'noise_unspecified': '定量 EXAFS 拟合缺少明确噪声模型；请提供实验噪声，或在专家模型中明确选择带假设的估计。',
    's02_uncalibrated': '未记录适用的 S₀² 标定。振幅或配位数及其误差仅在所给模型和标定假设下成立，未包含标定总不确定度。',
    'model_scientific_validation_pending': '已有模型可计算及合成回归通过，不等于结构、标准样适用性或真实科研生产验收已完成。',
    'fit_not_converged': '保留的模型候选未收敛；请检查残差、参数边界和停止原因。',
    'fit_budget_exhausted': '本次有限拟合预算已耗尽；尚未尝试的候选保留为未探索范围，不能声称得到全局唯一解。',
    'series_state_required': '请指定实际状态元数据字段，例如时间、电位或温度；文件顺序不能替代测量状态。',
    'feature_sampling': '近边特征窗口需要至少 8 个测量点且相邻间隔不大于 5 eV；当前特征不予输出。',
    'processing_required': '缺少可直接绘图的处理快照；请先处理原始谱，或导入具有明确含义和来源的处理数据。',
}

_DECISIONS = {
    'coverage_first_v1': '按测量覆盖和 Larch 正边跃检查选择基准窗口；保留附近背景敏感性候选，不按最终拟合残差倒选背景。',
    'content_addressed_reuse': '原始字节、映射、软件、校准、掩码及处理参数完全一致，复用经过校验的不可变处理快照。',
    'explicit_existing_snapshot': '读取已有处理快照，直接复用数值结果。',
    'nonnegative_reference_coefficients_v1': '在已确认适用性和完整公共能窗下计算非负参考组分系数；不自由移动参考谱，保留边界及敏感性诊断。',
    'provided_model_bounded_initialization_v1': '复用给定 FEFF 路径和物理参数关系，运行有限确定性初值；联合分析另有逐谱独立基准，不声称未知结构识别或唯一性。',
    'actual_state_metadata_v1': '只使用真实状态元数据，失败点留空；共同显示区间与各谱的原拟合窗口分别记录。',
}


def _issue(exc, dataset_id=None, severity='warning'):
    item = {'code': getattr(exc, 'code', 'analysis_branch_failed'), 'message': str(exc), 'severity': severity}
    if dataset_id is not None:
        item['dataset_id'] = dataset_id
    return item


def _existing_processing(store, project, dataset):
    from .larch_adapter import _processing_identity
    for pp in reversed(project.processing):
        if pp.dataset_id == dataset.id:
            _processing_identity(dataset, pp)
            for ref in pp.arrays.values():
                load_array(store, project.id, ref)
            return pp
    raise ProjectError('processing_required', 'This input has no existing processing snapshot; run automatic processing or import explicit supported processed data.', dataset.id)


def run_workflow(store, project_id, intent, expected_revision, context=None, plan_fingerprint=None):
    from .auto_models import resolve_model, run_known_model
    from .larch_adapter import _processing_identity, current_engine_versions, current_source_sha256
    from .recipes import descriptive_recipe
    from .series import build_series
    from .xanes import extract_features, fit_reference_combination

    intent = normalize_intent(intent)
    operations = effective_operations(intent)
    project = store.read(project_id)
    if project.revision != expected_revision:
        raise ProjectError('revision_conflict', 'Project changed after automatic-workflow submission; submit against the current revision.', project_id)
    validate_calibration_scope(intent)
    if plan_fingerprint is not None:
        from .auto_preflight import verify_plan
        verify_plan(store, project, intent, plan_fingerprint)
    # Validate existing versioned metadata before appending. Legacy projects
    # have an empty list and remain fully readable through the v1 project API.
    read_workflows(project)
    datasets = {d.id: d for d in project.datasets}
    all_ids = list(dict.fromkeys(intent.dataset_ids + intent.reference_ids))
    if any(did not in datasets for did in all_ids):
        raise ProjectError('dataset_not_found', 'A selected sample/reference is absent from the captured project.')
    result = WorkflowResult(id='wf_' + uuid4().hex, task=intent.task, input_revision=expected_revision,
        intent=intent, created_at=utc_now(), recipe=descriptive_recipe(),
        provenance={'engine_versions': current_engine_versions(), 'source_sha256': current_source_sha256(),
                    'intent_scope_policy': 'auto-preflight-v1', 'plan_fingerprint': plan_fingerprint,
                    'source_hashes': {did: datasets[did].raw_file.sha256 for did in all_ids},
                    'commit_policy': 'atomic processing/models/fit runs/workflow; expected revision checked at publication',
                    'retry_policy': 'new immutable run, reuse exact successful processing content when available',
                    'cancellation': 'JobService terminates this worker; no nested scientific job or unpublished result is reported as complete'})
    values = result.model_dump(mode='json')
    processing, additions, models, runs, calibrations = {}, [], [], [], []
    failed = False
    needs_input = False
    project, intent, mapping_blocked, mapping_decisions, mapping_issues = prepare_input_mappings(store, project, intent)
    datasets = {d.id: d for d in project.datasets}
    all_ids = [did for did in dict.fromkeys(intent.dataset_ids + intent.reference_ids) if did not in mapping_blocked]
    values['decisions'].extend(mapping_decisions)
    values['issues'].extend(mapping_issues)
    values['provenance']['source_hashes'].update({did: datasets[did].raw_file.sha256 for did in all_ids})
    if mapping_blocked:
        needs_input = True
        for did in sorted(mapping_blocked):
            values['steps'].append({'stage': 'input_mapping', 'dataset_id': did,
                'state': 'needs_input', 'finished_at': utc_now()})
    reference_request = intent.conditions.get('reference_calibration')
    if reference_request is not None:
        try:
            if not isinstance(reference_request, dict):
                raise ProjectError('reference_calibration_contract', '参考校准条件必须是一张明确的标准样条件卡。')
            required = {'reference_dataset_id', 'target_energy_eV'}
            if not required <= reference_request.keys():
                raise ProjectError('reference_calibration_contract', '参考校准需要标准样 ID 和同一定义的已知目标能量。')
            calibration = estimate_reference_calibration(FrozenProjectStore(store, project), project_id,
                reference_request['reference_dataset_id'], reference_request['target_energy_eV'],
                standard_confirmed=reference_request.get('standard_confirmed', False), note=reference_request.get('note', ''),
                search_half_width_eV=reference_request.get('search_half_width_eV', 30.0),
                max_shift_eV=reference_request.get('max_shift_eV', 20.0),
                applicability_conditions=reference_request.get('applicability_conditions'))
            apply_to = reference_request.get('apply_to_dataset_ids', all_ids)
            if not isinstance(apply_to, list) or not apply_to or not set(apply_to) <= set(all_ids):
                raise ProjectError('reference_calibration_scope', '参考校准应用范围必须属于本次选定谱。')
            if any((datasets[did].element, datasets[did].edge) != (calibration.element, calibration.edge) for did in apply_to):
                raise ProjectError('calibration_edge', '标准样校准只能应用于已确认相同元素和吸收边的谱。')
            if any(datasets[did].kind != 'mu' for did in apply_to):
                raise ProjectError('chi_energy_shift', 'χ(k) 输入不能应用能量漂移；请提供原始 μ(E)。')
            if set(apply_to) & set(intent.calibration_ids):
                raise ProjectError('calibration_duplicate', '同一谱只能选择一次参考校准，避免重复加偏移。')
            intent = intent.model_copy(deep=True)
            for did in apply_to:
                intent.calibration_ids[did] = calibration.id
            project.calibrations.append(calibration)
            calibrations.append(calibration)
            values['decisions'].append({'stage': 'calibration', 'rule': 'confirmed_reference_derivative_v1',
                'reason': '仅由明确标准样估计仪器漂移；原始数据不变，样品之间的真实化学位移保留。',
                'selected': {'calibration': calibration.model_dump(mode='json'), 'apply_to_dataset_ids': apply_to},
                'alternatives': []})
            reference = next(d for d in project.datasets if d.id == reference_request['reference_dataset_id'])
            values['provenance']['source_hashes'][reference.id] = reference.raw_file.sha256
        except (ValueError, RuntimeError) as exc:
            # Do not continue uncalibrated after an explicitly requested
            # calibration failed: that would change the submitted intent.
            needs_input = True
            values['issues'].append(_issue(exc))
            values['steps'].append({'stage': 'calibration', 'state': 'needs_input', 'code': getattr(exc, 'code', 'calibration_failed'), 'finished_at': utc_now()})
            all_ids = []
    if intent.series.get('merge_repeats') is True and all_ids:
        try:
            project, intent, decision = prepare_repeat_merges(store, project, intent)
            datasets = {d.id: d for d in project.datasets}
            all_ids = list(dict.fromkeys(intent.dataset_ids + intent.reference_ids))
            values['decisions'].append(decision)
            values['provenance']['source_hashes'].update({did: datasets[did].raw_file.sha256 for did in all_ids})
            values['steps'].append({'stage': 'repeat_scans', 'state': 'succeeded', 'finished_at': utc_now()})
        except (ValueError, RuntimeError) as exc:
            needs_input = True
            values['issues'].append(_issue(exc))
            values['steps'].append({'stage': 'repeat_scans', 'state': 'needs_input', 'code': getattr(exc, 'code', 'repeat_merge_failed'), 'finished_at': utc_now()})
            all_ids = []
    bound_processing = {}
    if 'structural_fit' in operations and (intent.model_id or intent.model_recipe_id):
        try:
            reviewed, _ = resolve_model(project, intent)
        except ProjectError:
            pass  # The fit branch records the model error alongside descriptive results.
        else:
            bound_processing = {b.dataset_id: b.processing_revision_id for b in reviewed.bindings}
    snapshots = {pp.id: pp for pp in project.processing}
    for index, did in enumerate(all_ids):
        stage = {'stage': 'processing', 'dataset_id': did, 'state': 'running', 'started_at': utc_now()}
        if context:
            context.heartbeat(f'processing {index + 1}/{len(all_ids)}: {did}', .05 + .5 * index / len(all_ids))
        try:
            if did in bound_processing:
                pp = snapshots.get(bound_processing[did])
                if pp is None:
                    raise ProjectError('processing_required', '所选模型关联的处理版本缺失。', did)
                _processing_identity(datasets[did], pp)
                if intent.processing_overrides or intent.calibration_ids.get(did, pp.calibration_id) != pp.calibration_id:
                    raise ProjectError('model_processing_override', '请先在专业工作台应用新的预处理设置或校准，并保存新模型。', did)
                created, issues = [], []
                decision = {'stage': 'processing', 'rule': 'reviewed_model_snapshot', 'dataset_id': did,
                            'reason': 'Reused the processing revision bound to the reviewed model.', 'selected': pp.id}
            elif intent.task == 'plot' or intent.use_existing_processing:
                try:
                    pp = _existing_processing(store, project, datasets[did])
                    created, issues = [], []
                    decision = {'stage': 'processing', 'rule': 'explicit_existing_snapshot',
                                'reason': 'Read the selected project processing snapshot without scientific recomputation.',
                                'dataset_id': did, 'selected': pp.id}
                except ProjectError as exc:
                    if intent.task != 'plot' or exc.code != 'processing_required':
                        raise
                    pp, created, issues, decision = import_external_plot(store, project, datasets[did], intent, context)
            else:
                pp, created, issues, decision = process_automatically(store, project, datasets[did], intent, context)
            processing[did] = pp
            additions.extend(created)
            values['processing_ids'].append(pp.id)
            values['decisions'].append(decision)
            values['issues'].extend(issues)
            stage.update(state='succeeded', processing_id=pp.id)
        except (ValueError, RuntimeError) as exc:
            failed = True
            values['issues'].append(_issue(exc, did, 'error'))
            stage.update(state='needs_input', code=getattr(exc, 'code', 'processing_failed'), message=str(exc))
        stage['finished_at'] = utc_now()
        values['steps'].append(stage)
    if not processing or not any(did in processing for did in intent.dataset_ids):
        needs_input = True
    # Descriptive processing may provide derivatives even without references.
    # A chi-only project never acquires fabricated energy or mu arrays.
    for did in intent.dataset_ids + intent.reference_ids:
        pp = processing.get(did)
        if pp is None:
            continue
        if not {'energy', 'norm'} <= pp.arrays.keys():
            if intent.task == 'xanes':
                needs_input = True
                values['issues'].append({'code': 'xanes_requires_mu', 'message': 'chi(k) has no recoverable XANES or energy-space derivative; supply mu(E) for this branch.', 'dataset_id': did, 'severity': 'warning'})
            continue
        try:
            values['features'].append(extract_features(store, project_id, pp))
        except (ValueError, RuntimeError) as exc:
            values['issues'].append(_issue(exc, did))
            if intent.task == 'xanes':
                failed = True
    if 'lcf' in operations and not intent.reference_ids:
        needs_input = True
        values['issues'].append({'code': 'reference_required', 'message': '已选择参考组分拟合，但没有指定适用参考谱；描述性结果仍保留。', 'severity': 'warning'})
        values['steps'].append({'stage': 'lcf', 'state': 'needs_input', 'code': 'reference_required', 'finished_at': utc_now()})
    elif 'lcf' in operations:
        references = [processing[did] for did in intent.reference_ids if did in processing]
        for did in intent.dataset_ids:
            if did not in processing:
                continue
            try:
                if len(references) != len(intent.reference_ids):
                    raise ProjectError('reference_processing_failed', 'Some selected references could not be processed; LCF was not silently run with a different library.')
                lcf = fit_reference_combination(store, project, processing[did], references, intent.lcf, intent.conditions)
                values['lcf_results'].append(lcf)
                values['steps'].append({'stage': 'lcf', 'dataset_id': did, 'state': 'succeeded', 'finished_at': utc_now()})
                values['decisions'].append({'stage': 'lcf', 'rule': 'nonnegative_reference_coefficients_v1',
                    'reason': 'Confirmed reference applicability, complete window coverage, no reference shifts and retained boundary/sensitivity diagnostics.',
                    'dataset_id': did, 'selected': {'window_eV': lcf['window_eV'], 'constraints': lcf['constraints']},
                    'alternatives': lcf['alternatives']})
                if lcf['diagnostics']['near_collinear']:
                    values['issues'].append({'code': 'lcf_near_collinear', 'message': 'Reference coefficients are not separately identifiable under the retained matrix diagnostic; near-equivalent alternatives remain visible.', 'dataset_id': did, 'severity': 'warning'})
            except (ValueError, RuntimeError) as exc:
                needs_input = True
                values['issues'].append(_issue(exc, did))
                values['steps'].append({'stage': 'lcf', 'dataset_id': did, 'state': 'needs_input', 'code': getattr(exc, 'code', 'lcf_failed'), 'finished_at': utc_now()})
    elif intent.task == 'xanes':
        values['issues'].append({'code': 'descriptive_xanes_only', 'message': 'XANES features are descriptive. Upload applicable references and confirm their conditions to obtain conditional reference-component coefficients.', 'severity': 'info'})
    if 'structural_fit' in operations:
        if context:
            context.heartbeat('validating supplied structural model', .65)
        try:
            models, runs, issues, decision = run_known_model(store, project, processing, intent, context)
            values['fit_run_ids'] = [run.id for run in runs]
            values['issues'].extend(issues)
            values['decisions'].append(decision)
            values['recipe']['model_recipe'] = decision['selected']['model_recipe']
            if not runs:
                needs_input = True
            elif any(run.result is None or not run.result.converged for run in runs) or any(a['state'] in ('failed', 'not_run') for a in decision['alternatives']):
                failed = True
            values['steps'].append({'stage': 'exafs', 'state': 'succeeded' if runs else 'needs_input', 'fit_run_ids': values['fit_run_ids'], 'finished_at': utc_now()})
        except (ValueError, RuntimeError) as exc:
            needs_input = True
            values['issues'].append(_issue(exc))
            values['steps'].append({'stage': 'exafs', 'state': 'needs_input', 'code': getattr(exc, 'code', 'fit_failed'), 'finished_at': utc_now()})
    if intent.task == 'series':
        try:
            values['series'] = build_series(store, project, intent, processing, values['features'], values['lcf_results'], runs)
            if any(p['state_value'] is None for p in values['series']['points']):
                needs_input = True
                values['issues'].append({'code': 'series_state_missing', 'message': 'Some actual state metadata are missing; those observations remain blank in trends.', 'severity': 'warning'})
            values['decisions'].append({'stage': 'series', 'rule': 'actual_state_metadata_v1',
                'reason': 'Use original measured-state metadata, retain failed states, and keep display intersection separate from fit windows.',
                'selected': {'state_key': values['series']['state_key'], 'display_intersection': values['series'].get('display_intersection')}})
        except (ValueError, RuntimeError) as exc:
            needs_input = True
            values['issues'].append(_issue(exc))
    if 'preedge' in operations:
        from .xanes import preedge_integral
        from .projects import canonical
        values['preedge_results']=[]
        for did in intent.dataset_ids:
            try:
                bounds=intent.preedge_windows or {}
                if did not in processing or set(bounds)!={'left_eV','peak_eV','right_eV'}:
                    raise ProjectError('preedge_windows','需要有效处理版本及三个已确认能量窗口。', did)
                integral=preedge_integral(store,project_id,processing[did],**bounds)
                ref=store.put_asset(project_id,canonical(integral),'preedge.json')
                values['preedge_results'].append(dict(dataset_id=did,artifact=ref.model_dump(mode='json')))
            except (ValueError,RuntimeError) as exc:
                needs_input=True
                values['issues'].append(_issue(exc,did))
    if 'decomposition' in operations:
        from .decomposition import decompose_series
        from .projects import canonical
        try:
            if any(did not in processing for did in intent.dataset_ids):
                raise ProjectError('decomposition_processing', '部分输入缺少有效处理版本。')
            result = decompose_series(store, project, [processing[did] for did in intent.dataset_ids], window_eV=intent.decomposition_window_eV, n_components=intent.decomposition_components)
            values['decomposition'] = store.put_asset(project_id, canonical(result), 'decomposition.json').model_dump(mode='json')
            values['steps'].append({'stage':'decomposition','state':'succeeded','finished_at':utc_now()})
        except ProjectError as exc:
            needs_input = True
            values['issues'].append(_issue(exc))
            values['steps'].append({'stage':'decomposition','state':'needs_input','code':exc.code,'finished_at':utc_now()})
    values['state'] = 'needs_input' if needs_input else 'partial' if failed else 'succeeded'
    values['finished_at'] = utc_now()
    for issue in values['issues']:
        issue['message'] = _MESSAGES.get(issue['code'], issue['message'])
    for decision in values['decisions']:
        decision['reason'] = _DECISIONS.get(decision['rule'], decision['reason'])
    result = WorkflowResult.model_validate(values)
    value = project.model_dump(mode='json')
    known = {p['id'] for p in value['processing']}
    value['processing'].extend(pp.model_dump(mode='json') for pp in additions if pp.id not in known)
    value['models'].extend(m.model_dump(mode='json') for m in models)
    value['runs'].extend(r.model_dump(mode='json') for r in runs)
    value['metadata'].setdefault('automation_runs', []).append(result.model_dump(mode='json'))
    model_recipe = values['recipe'].get('model_recipe')
    if model_recipe:
        recipes = value['metadata'].setdefault('automation_model_recipes', [])
        if not any(r.get('id') == model_recipe['id'] for r in recipes):
            recipes.append(model_recipe)
    if context:
        context.heartbeat('publishing immutable scientific workflow', .95)
    # A concurrent editor cannot overwrite this snapshot. Revision conflicts
    # leave only unreferenced immutable assets, never changed prior records.
    store.save(Project.model_validate(value), expected_revision)
    return result.model_dump(mode='json')


def run_job(payload, context):
    return run_workflow(ProjectStore(), payload['project_id'], payload['intent'],
                        payload['expected_revision'], context, payload.get('plan_fingerprint'))
