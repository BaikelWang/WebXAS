"""Read-only declaration checks for the automatic UI, never a scientific run.

Readiness means the declared input contract is satisfied. It does not predict
normalization/fit success, estimate a parameter, or validate a physical model.
"""
from __future__ import annotations

import hashlib
import math

from ..import_formats import column_labels, describe
from ..ingest import parse_ascii
from .auto_contracts import SUPPORTED_OPERATIONS, effective_operations, normalize_intent, validate_calibration_scope
from .auto_processing import _raw, inspect_input, resolve_input_mapping
from .contracts import ModelRevision, NoiseSpec
from .projects import ProjectError, canonical

POLICY_VERSION = 'auto-preflight-v1'
TASKS = ('process', 'xanes', 'exafs', 'series', 'plot')


def _field(path, label, value=None, kind='text', options=None):
    return {'path': path, 'label': label, 'type': kind, 'value': value,
            'options': [{'value': v, 'label': str(v)} for v in (options or [])]}


def _mapping_fields(dataset):
    return [_field('dataset.kind', '数据类型', dataset.kind, 'select', ['mu', 'chi']),
            _field('dataset.mapping.energy', '横轴列（从 0 起）', dataset.mapping.energy, 'integer'),
            _field('dataset.mapping.signal', '信号列（从 0 起）', dataset.mapping.signal, 'integer'),
            _field('dataset.mapping.reference', '参考通道列（从 0 起）', dataset.mapping.reference, 'integer'),
            _field('dataset.mapping.energy_unit', '能量单位', dataset.mapping.energy_unit, 'select', ['eV', 'keV']),
            _field('dataset.mapping.mode', '信号定义', dataset.mapping.mode, 'select', ['mu', 'transmission', 'fluorescence']),
            _field('dataset.metadata.mapping_confirmed', '确认以上列含义', False, 'boolean')]


def _inspect(store, project, dataset):
    result = {'dataset': dataset, 'state': dataset.metadata.get('import_processing_state', 'unknown'),
              'facts': [], 'error': None, 'preview': None, 'range': None}
    try:
        raw = _raw(store, project.id, dataset)
        table, header = parse_ascii(raw)
        result['preview'] = {'columns': column_labels(header, table.shape[1]), 'rows': table[:5].tolist()}
        declaration = describe(raw, dataset.filename, dataset.kind)
        result['state'] = dataset.metadata.get('import_processing_state') or declaration.get('processing_state', 'unknown')
        mapping, source, _ = resolve_input_mapping(raw, dataset)
        resolved = dataset.model_copy(deep=True)
        resolved.mapping = mapping
        result['dataset'] = resolved
        x, _ = inspect_input(store, project.id, resolved)
        result['range'] = [float(x[0]), float(x[-1])]
        result['facts'].append({'dataset_id': dataset.id, 'key': 'mapping', 'value': mapping.model_dump(mode='json'),
            'unit': 'angstrom-1' if dataset.kind == 'chi' else mapping.energy_unit,
            'source': source, 'status': 'confirmed' if source == 'user_confirmed' else 'resolved'})
    except (ValueError, RuntimeError, UnicodeError) as exc:
        result['error'] = {'code': getattr(exc, 'code', 'input_invalid'), 'message': str(exc)}
    for key, value in [('kind', dataset.kind), ('element', dataset.element), ('edge', dataset.edge),
                       ('processing_state', result['state']), ('conditions', dataset.conditions)]:
        result['facts'].append({'dataset_id': dataset.id, 'key': key, 'value': value, 'unit': None,
            'source': 'project_declaration', 'status': 'declared' if value is not None else 'unknown'})
    return result


def _questions(store, project, intent, inspected, *, explicit_scope=True):
    questions = []
    task = intent.task
    operations = effective_operations(intent)
    selected = intent.dataset_ids + intent.reference_ids
    datasets = {d.id: d for d in project.datasets}

    def add(code, message, did=None, fields=(), *, blocking=True, ids=None, preview=None):
        questions.append({'code': code, 'dataset_id': did, 'dataset_ids': ids or ([did] if did else list(selected)),
            'blocking': blocking, 'task_impacts': [task], 'message': message, 'fields': list(fields), 'preview': preview})

    def identity_fields(d):
        return [_field('dataset.element', '吸收元素', d.element),
                _field('dataset.edge', '吸收边', d.edge, 'select', ['K', 'L1', 'L2', 'L3'])]

    for did in selected:
        d = datasets[did]
        record = inspected[did]
        if record['error']:
            error = record['error']
            mapping_error = error['code'].startswith('mapping_') or error['code'] in ('invalid_mapping', 'chi_mapping')
            fields = identity_fields(d) if error['code'] == 'header_identity_conflict' else _mapping_fields(d) if mapping_error else []
            add(error['code'], error['message'], did, fields, preview=record['preview'])
            continue
        state = record['state']
        if d.kind == 'support':
            add('not_spectrum', '支持材料不能作为光谱分析。', did)
            continue
        if task == 'plot' or intent.use_existing_processing:
            try:
                from .workflows import _existing_processing
                _existing_processing(store, project, record['dataset'])
                continue
            except (ValueError, RuntimeError) as exc:
                if getattr(exc, 'code', '') != 'processing_required':
                    add(getattr(exc, 'code', 'processing_invalid'), str(exc), did)
                    continue
                if task != 'plot':
                    add('processing_required', '所选数据没有可复用的处理快照；请先进行基础处理。', did)
                    continue
        if task == 'plot':
            if state not in ('normalized', 'chi', 'corrected_chi'):
                add('external_plot_state', '没有已有快照；仅明确归一化谱或 χ(k) 可直接绘图。原始谱请先进行基础处理。', did)
            if not str(d.metadata.get('external_processing_note', '')).strip():
                add('external_processing_note_required', '请说明外部处理方法、列含义与来源。', did,
                    [_field('dataset.metadata.external_processing_note', '处理来源与列含义')])
        elif state in ('normalized', 'difference', 'background', 'chi_fit_table', 'r_space', 'q_space'):
            add('derived_input', '此输入已是处理结果，不能作为原始 μ(E) 再做背景处理；请选择适用的直接绘图入口。', did)
        if state in ('corrected_mu', 'corrected_chi') and not str(d.metadata.get('external_processing_note', '')).strip():
            add('processing_provenance', '外部已校正谱需要校正方法与适用条件。', did,
                [_field('dataset.metadata.external_processing_note', '外部校正依据')])
        if d.kind == 'chi' and 'chi_weighting_requires_verification' in d.metadata.get('import_analysis_blockers', []) and d.metadata.get('input_kweight_confirmed') is not True:
            add('chi_weighting_requires_verification', '请确认输入 χ(k) 已乘的 k 权重。', did,
                [_field('dataset.metadata.input_kweight', '输入 k 权重', d.metadata.get('input_kweight', 0), 'select', [0, 1, 2, 3]),
                 _field('dataset.metadata.input_kweight_confirmed', '确认输入权重', False, 'boolean')])
        if d.kind == 'chi':
            weight = d.metadata.get('input_kweight', 0)
            if isinstance(weight, bool) or weight not in (0, 1, 2, 3):
                add('chi_weight', '输入 k 权重必须为 0、1、2 或 3。', did,
                    [_field('dataset.metadata.input_kweight', '输入 k 权重', weight, 'select', [0, 1, 2, 3])])
            elif weight and record['range'][0] <= 0:
                add('chi_weight', '带权 χ(k) 包含 k=0，不能无损去权；请提供原始未加权谱或核对来源，不能自动丢点。', did)
        if d.kind == 'chi' and task == 'xanes':
            add('xanes_requires_mu', 'χ(k) 不含可恢复的 XANES 信息，请提供 μ(E)。', did)
        if (not d.element or not d.edge) and task != 'plot':
            add('edge_identity_missing', '元素或吸收边未声明；仅可进行描述性处理，不能将自动边位当作已核实身份。', did, identity_fields(d), blocking=False)

    if 'lcf' in operations and not intent.reference_ids:
        add('reference_required', '已选择参考组分拟合；请导入并指定至少一条适用参考谱。', fields=[
            _field('intent.reference_ids', '组分参考谱', [], 'multiselect', [d.id for d in project.datasets if d.id not in intent.dataset_ids])])
    if 'lcf' in operations and intent.reference_ids:
        conditions = intent.conditions
        if conditions.get('reference_applicability_confirmed') is not True or not str(conditions.get('reference_applicability_note', '')).strip():
            add('reference_applicability', '请核对参考谱身份、归一化与测量适用条件，并记录依据。', fields=[
                _field('intent.conditions.reference_applicability_confirmed', '已确认参考适用性', False, 'boolean'),
                _field('intent.conditions.reference_applicability_note', '参考适用依据')])
        for did in intent.dataset_ids:
            d = datasets[did]
            if not d.element or not d.edge or any((datasets[r].element, datasets[r].edge) != (d.element, d.edge) for r in intent.reference_ids):
                add('reference_identity', '定量参考比较要求样品与全部参考明确具有相同元素和吸收边。', did, identity_fields(d))
        for did in selected:
            if datasets[did].kind != 'mu' and not any(q['code'] == 'xanes_requires_mu' and q['dataset_id'] == did for q in questions):
                add('xanes_requires_mu', '参考组分拟合需要样品与参考的 μ(E)，不能由 χ(k) 恢复。', did)

    model = None
    if 'structural_fit' in operations:
        if intent.model_recipe_id:
            recipe = next((r for r in project.metadata.get('automation_model_recipes', []) if r.get('id') == intent.model_recipe_id), None)
            if recipe and recipe.get('version') == 1:
                model = ModelRevision.model_validate(recipe['model'])
                available = {p.id: p.model_dump(mode='json') for p in project.paths}
                if any(available.get(p['id']) != p for p in recipe['paths']):
                    add('model_recipe_paths', '模型配方依据的 FEFF 路径已变化，请建立新配方。')
                if intent.model_id is not None and intent.model_id != model.id:
                    add('model_recipe_identity', '所选模型与配方来源不一致。')
            else:
                add('model_recipe_required', '找不到所选版本化模型配方。')
        else:
            model = next((m for m in project.models if m.id == intent.model_id), None)
        if model is None:
            add('model_required', '请选择已有结构/路径模型；也可先完成基础处理与谱图。', fields=[
                _field('intent.model_id', '已有模型', intent.model_id, 'select', [m.id for m in project.models])])
        else:
            if {b.dataset_id for b in model.bindings} != set(intent.dataset_ids):
                add('model_dataset_scope', '模型必须恰好绑定本次选定样品；不能自动删改观测。')
            relation = intent.series.get('relation', 'independent') if task == 'series' else 'provided_model'
            if relation not in ('independent', 'provided_model', 'joint_existing'):
                add('relation_template', '请选择独立分析或已有联合模型。')
            if len(model.bindings) > 1 and relation in ('provided_model', 'joint_existing'):
                if intent.conditions.get('joint_applicability_confirmed') is not True or not str(intent.conditions.get('joint_applicability_note', '')).strip():
                    add('joint_applicability', '此多谱模型包含跨谱关系，请确认哪些关系可共享及其物理依据。', fields=[
                        _field('intent.conditions.joint_applicability_confirmed', '已审查共享关系', False, 'boolean'),
                        _field('intent.conditions.joint_applicability_note', '共享关系适用依据')])
                if len({(datasets[did].element, datasets[did].edge) for did in intent.dataset_ids}) != 1:
                    add('joint_cross_edge', '当前自动联合配方不支持跨吸收边共享。')
            previous = {p.id: p for p in project.processing}
            for binding in model.bindings:
                pp = previous.get(binding.processing_revision_id)
                noise = intent.fit.get('noise') or (binding.noise.model_dump(mode='json') if binding.noise.source != 'unknown' else pp.noise.model_dump(mode='json') if pp else {'source': 'unknown'})
                try:
                    if NoiseSpec.model_validate(noise).source == 'unknown':
                        add('noise_unspecified', '定量拟合需要明确实验噪声或带有假设的估计；请在专业工作台补充。', binding.dataset_id)
                except ValueError:
                    add('noise_contract', '噪声声明不符合 NoiseSpec 合同，请在专业工作台修正。', binding.dataset_id)
            try:
                from .parameters import compile_model
                compiled = compile_model(model, project.paths)
                if not compiled.free_ids:
                    add('no_free_parameters', '所选模型全部参数固定，仅适用于模型预览。')
                for issue in compiled.diagnostics:
                    if issue['severity'] == 'error':
                        add(issue.get('code', 'model_invalid'), issue['message'])
                path_ids = {b.path_id for binding in model.bindings for b in binding.paths}
                paths = {p.id: p for p in project.paths}
                for binding in model.bindings:
                    d = datasets.get(binding.dataset_id)
                    for pb in binding.paths:
                        path = paths.get(pb.path_id)
                        if d is not None and path is not None and (d.element, d.edge) != (path.absorber_element, path.edge):
                            add('path_edge', '模型路径的吸收元素/边与数据声明不一致，请核对模型。', d.id, identity_fields(d))
                for path in project.paths:
                    if path.id in path_ids:
                        store.read_asset(project.id, path.source_file)
            except (ValueError, RuntimeError) as exc:
                add(getattr(exc, 'code', 'model_invalid'), str(exc))
            add('model_scientific_validation_pending', '已有模型可计算不等于结构已通过真实标准样验收；S₀² 是否实际应用及误差依据由运行结果记录。', blocking=False)

    if task == 'series':
        key = intent.series.get('state_key') or intent.conditions.get('state_key')
        if not isinstance(key, str) or not key.strip():
            keys = sorted({k for did in intent.dataset_ids for k in datasets[did].conditions})
            add('series_state_required', '请选择真实测量状态字段；文件顺序不能替代状态。', fields=[
                _field('intent.conditions.state_key', '状态字段', None, 'text', keys)])
        else:
            for did in intent.dataset_ids:
                state = datasets[did].conditions.get(key)
                if isinstance(state, bool) or not isinstance(state, (int, float, str)) or (isinstance(state, str) and not state.strip()):
                    add('series_state_missing', '请补充此数据的真实状态：' + key, did,
                        [_field('dataset.conditions.' + key, key)])

    if intent.series.get('merge_repeats') is True and task not in ('process', 'xanes'):
        add('repeat_merge_task', '重复扫描合并只适用于基础处理或 XANES；请关闭此操作，或切换到支持的任务。', fields=[
            _field('intent.series.merge_repeats', '合并重复扫描', True, 'boolean')])
    if intent.series.get('merge_repeats') is True and task in ('process', 'xanes'):
        if intent.conditions.get('independent_repeats_confirmed') is not True or not str(intent.conditions.get('independent_repeats_note', '')).strip():
            add('repeat_independence_required', '合并前需确认独立重复、同样品与测量条件；复制文件不算重复观测。', fields=[
                _field('intent.conditions.independent_repeats_confirmed', '确认独立同状态重复', False, 'boolean'),
                _field('intent.conditions.independent_repeats_note', '重复测量依据')])
        groups = {}
        for did in intent.dataset_ids:
            d = inspected[did]['dataset']
            if d.kind != 'mu' or not d.sample_id or not d.element or not d.edge:
                add('repeat_metadata', '重复扫描需要原始 μ(E)、样品 ID、元素与吸收边。', did,
                    [_field('dataset.sample_id', '样品 ID', d.sample_id), *identity_fields(d)])
            if inspected[did]['state'] not in ('mu', 'unknown') or (d.lineage and not d.metadata.get('mapping_source_dataset_id')) or d.metadata.get('import_role') == 'merged':
                add('repeat_original_required', '已合并或其他衍生谱不能再充当独立原始重复。', did)
            key = canonical({'sample_id': d.sample_id, 'element': d.element, 'edge': d.edge,
                'mapping': d.mapping.model_dump(mode='json'), 'conditions': d.conditions, 'correction_states': d.correction_states})
            groups.setdefault(key, []).append(d)
        if not any(len(g) > 1 for g in groups.values()):
            add('repeat_no_compatible_group', '没有完全同样品、状态、映射和修正状态的重复组；不会按文件名合并。')
        for group in groups.values():
            if len({d.raw_file.sha256 for d in group}) != len(group):
                add('duplicate_scan', '相同原始字节的副本不能作为独立重复改善误差估计。', ids=[d.id for d in group])

    request = intent.conditions.get('reference_calibration')
    if request is not None:
        scope_fields = [_field('intent.conditions.reference_calibration.apply_to_dataset_ids', '校准适用扫描',
                              request.get('apply_to_dataset_ids') if isinstance(request, dict) else None, 'multiselect', selected)]
        try:
            validate_calibration_scope(intent, require_explicit=explicit_scope)
        except ProjectError as exc:
            add(exc.code, exc.message, fields=scope_fields)
        if isinstance(request, dict):
            rid = request.get('reference_dataset_id')
            ref = datasets.get(rid) if isinstance(rid, str) else None
            if ref is None or ref.kind != 'mu' or not ref.element or not ref.edge:
                add('reference_standard_identity', '请选择有明确元素/边的原始 μ(E) 标准样。', rid if ref else None,
                    [_field('intent.conditions.reference_calibration.reference_dataset_id', '标准样', rid, 'select', list(datasets))])
            elif inspected[rid]['state'] not in ('mu', 'unknown', 'corrected_mu'):
                add('reference_standard_identity', '标准校准需要原始 μ(E)，不能将归一化或其他衍生谱当作原始标准样。', rid)
            elif rid not in selected and inspected[rid]['error']:
                error = inspected[rid]['error']
                add(error['code'], error['message'], rid, _mapping_fields(ref), preview=inspected[rid]['preview'])
            if request.get('standard_confirmed') is not True or not str(request.get('note', '')).strip():
                add('reference_standard_confirmation', '请确认标准样、目标特征定义和测量适用性；不能用样品对齐代替校准。', fields=[
                    _field('intent.conditions.reference_calibration.standard_confirmed', '确认标准样及目标特征', False, 'boolean'),
                    _field('intent.conditions.reference_calibration.note', '标准样和适用依据')])
            energy = request.get('target_energy_eV')
            if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(energy) or not 2000 <= energy < 200000:
                add('reference_calibration_range', '请填写同定义标准样特征的已知能量（eV），范围为 2000 至 200000 eV。', fields=[
                    _field('intent.conditions.reference_calibration.target_energy_eV', '标准特征能量（eV）', energy, 'number')])
            for name, default, low, high in [('search_half_width_eV', 30.0, 5.0, 100.0), ('max_shift_eV', 20.0, 0.0, 100.0)]:
                number = request.get(name, default)
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not low <= number <= high or (name == 'max_shift_eV' and number == 0):
                    add('reference_calibration_range', '标准校准搜索或漂移范围超出当前方法允许值。', fields=[
                        _field('intent.conditions.reference_calibration.' + name, name, number, 'number')])
            if request.get('applicability_conditions') is not None and not isinstance(request['applicability_conditions'], dict):
                add('reference_calibration_contract', '校准适用条件必须是字段与实际测量值组成的对象。')
            scope = request.get('apply_to_dataset_ids', selected)
            if isinstance(scope, list):
                for did in scope:
                    if not isinstance(did, str) or did not in datasets:
                        continue
                    d = datasets[did]
                    if ref and (d.element, d.edge) != (ref.element, ref.edge):
                        add('calibration_edge', '所选标准校准与此谱元素/边不一致。', did, identity_fields(d))
                    if d.kind != 'mu':
                        add('chi_energy_shift', 'χ(k) 不能施加能量偏移，请提供原始 μ(E)。', did)
                    applicability = request.get('applicability_conditions', {})
                    if isinstance(applicability, dict) and any(d.conditions.get(k) != v for k, v in applicability.items()):
                        add('calibration_conditions', '此扫描不满足校准声明的测量适用条件。', did)
                    if did in intent.calibration_ids:
                        add('calibration_duplicate', '此扫描已指定另一校准，不能重复加偏移。', did)
    for did, cid in intent.calibration_ids.items():
        d = datasets[did]
        cal = next((c for c in project.calibrations if c.id == cid), None)
        if cal is None or cal.kind != 'energy' or (d.element, d.edge) != (cal.element, cal.edge):
            add('calibration_edge', '所选能量校准缺失或与扫描元素/边不一致。', did)
        elif any(d.conditions.get(k) != v for k, v in cal.conditions.items()):
            add('calibration_conditions', '此扫描不满足所选校准的测量适用条件。', did)
        if d.kind != 'mu':
            add('chi_energy_shift', 'χ(k) 不能施加能量偏移。', did)
    return questions


def preflight(store, project, value):
    intent = normalize_intent(value)
    by_id = {d.id: d for d in project.datasets}
    selected = list(dict.fromkeys(intent.dataset_ids + intent.reference_ids))
    if not set(selected) <= by_id.keys():
        raise ProjectError('dataset_scope', '所选样品或参考谱不属于当前工程。')
    request = intent.conditions.get('reference_calibration')
    if isinstance(request, dict) and isinstance(request.get('reference_dataset_id'), str) and request['reference_dataset_id'] in by_id:
        selected = list(dict.fromkeys([*selected, request['reference_dataset_id']]))
    inspected = {did: _inspect(store, project, by_id[did]) for did in selected}
    questions = _questions(store, project, intent, inspected)
    actions = []
    incompatible = {'not_spectrum', 'xanes_requires_mu', 'derived_input', 'external_plot_state', 'joint_cross_edge'}
    # Alternatives retain only this plan's scoped facts/intent; changing target
    # requires a fresh preflight, never reuse a hidden confirmation from the UI.
    for task in TASKS:
        alternative_value = {**intent.model_dump(mode='json'), 'task': task}
        if intent.requested_operations is not None:
            alternative_value['requested_operations'] = [op for op in intent.requested_operations if op in SUPPORTED_OPERATIONS[task]]
        alternative = normalize_intent(alternative_value)
        checks = questions if task == intent.task else _questions(store, project, alternative, inspected)
        blockers = [q for q in checks if q['blocking']]
        state = 'not_applicable' if any(q['code'] in incompatible for q in blockers) else 'needs_input' if blockers else 'ready'
        actions.append({'task': task, 'state': state, 'reason': blockers[0]['message'] if blockers else '输入声明可进入此流程；数值覆盖和科学诊断将在运行中继续检查。'})
    effective = intent.model_dump(mode='json')
    selected_operations = effective_operations(intent)
    operation_labels = {'lcf': '参考组分拟合（LCF）', 'structural_fit': '结构拟合（EXAFS）'}
    capabilities = []
    for operation, label in operation_labels.items():
        available = operation in SUPPORTED_OPERATIONS[intent.task]
        selected_operation = operation in selected_operations
        if not available:
            reason = '当前模式不支持此功能。'
        elif intent.task == 'exafs':
            reason = 'EXAFS 模式必需结构拟合；请提供适用模型与噪声声明。'
        elif operation == 'lcf':
            reason = '需指定适用参考谱并确认测量条件。' if selected_operation else '选择后将检查参考谱与适用依据。'
        else:
            reason = '需指定已有模型；联合模型另需共享关系依据。' if selected_operation else '选择后将检查模型、噪声与共享关系。'
        capabilities.append({'id': operation, 'label': label, 'available': available,
                             'selected': selected_operation, 'reason': reason})
    facts = [fact for record in inspected.values() for fact in record['facts']]
    evidence = {'policy_version': POLICY_VERSION, 'project': project.model_dump(mode='json'),
                'effective_intent': effective, 'facts': facts, 'questions': questions}
    fingerprint = hashlib.sha256(canonical(evidence)).hexdigest()
    labels = {'process': '基础处理与谱图', 'xanes': 'XANES 比较', 'exafs': '已有模型 EXAFS', 'series': '真实状态序列', 'plot': '复用处理结果绘图'}
    count = sum(q['blocking'] for q in questions)
    functions = '、'.join(operation_labels[op] for op in selected_operations) or '基础／描述性分析'
    summary = f'{len(intent.dataset_ids)} 条样品谱 · {labels[intent.task]} · {functions} · ' + (f'需补充 {count} 项条件' if count else '输入声明已就绪')
    return {'policy_version': POLICY_VERSION, 'project_id': project.id, 'revision': project.revision,
        'facts': facts, 'questions': questions, 'available_actions': actions, 'capabilities': capabilities, 'effective_intent': effective,
        'plan_summary': summary, 'plan_fingerprint': fingerprint}


def verify_plan(store, project, intent, fingerprint):
    plan = preflight(store, project, intent)
    if plan['plan_fingerprint'] != fingerprint:
        raise ProjectError('plan_stale', '输入、模型、声明或策略已变化，请刷新预检后再执行。', project.id)
    if any(q['blocking'] for q in plan['questions']):
        raise ProjectError('preflight_needs_input', '请补充预检指出的必要条件。', project.id,
                           {'questions': plan['questions']})
    return plan
