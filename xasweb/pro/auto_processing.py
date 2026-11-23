"""Deterministic coverage-first processing on original input bytes.

This layer chooses supported Larch inputs. It never aligns sample edges,
smooths fit data, estimates experimental corrections, or selects backgrounds by
EXAFS fit quality. Every trial is retained as a separate processing revision.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from ..ingest import construct_mu, parse_ascii
from ..models import Mapping, Parameters
from .contracts import Calibration, Dataset, NoiseSpec, ProcessingRevision
from .preprocessing import load_array, merge_scan_arrays, process_dataset, save_array
from .projects import ProjectError, canonical
from uuid import uuid4


class FrozenProjectStore:
    def __init__(self, store, project):
        self.store, self.project = store, project.model_copy(deep=True)

    def read(self, project_id):
        if self.project.id != project_id:
            raise ProjectError('project_scope', 'Automation snapshot belongs to another project.')
        return self.project.model_copy(deep=True)

    def __getattr__(self, name):
        return getattr(self.store, name)


def _issue(code, message, dataset_id, severity='warning'):
    return {'code': code, 'message': message, 'dataset_id': dataset_id, 'severity': severity}


def _raw(store, project_id, dataset):
    original = store.read_asset(project_id, dataset.raw_file)
    # Prefer the verified bytes. raw_text is an editable import preview, not a
    # second authoritative observation that may silently diverge from its hash.
    text = original.decode('utf-8-sig')
    if dataset.raw_text is not None and dataset.raw_text.lstrip('\ufeff') != text:
        raise ProjectError('raw_text_mismatch', 'Stored text differs from original bytes; import an explicit derived dataset.', dataset.id)
    return text


def _mapping_meaning(mapping):
    return (mapping.energy, mapping.signal, mapping.mode, mapping.energy_unit,
            mapping.reference if mapping.mode != 'mu' else None)


def resolve_input_mapping(raw_text, dataset):
    """Reuse the existing declared-column resolver, refusing screening guesses."""
    from ..automation import declared_columns, infer_mapping
    from ..import_formats import column_labels
    table, header = parse_ascii(raw_text)
    if dataset.metadata.get('mapping_confirmed') is True:
        return dataset.mapping, 'user_confirmed', []
    columns, _, declaration_error = declared_columns(header, table.shape[1])
    if declaration_error:
        raise ProjectError('mapping_ambiguous', '原文件的列声明存在冲突或单位不明，请确认列映射和 eV/keV 后再分析。', dataset.id)
    if dataset.kind == 'chi':
        kcol = columns.get('k')
        chicol = next((columns[n] for n in ('chi', 'chi(k)') if n in columns), None)
        if kcol is not None and chicol is not None:
            return Mapping(energy=kcol, signal=chicol), 'declared_k_chi', []
        mapping, reason = None, 'confirm_chi_columns'
    else:
        mapping, reason = infer_mapping(raw_text)
    if reason == 'fluorescence_preferred_over_transmission':
        raise ProjectError('mapping_ambiguous_signal', '文件同时包含透射与荧光样品通道，且未声明测量模式；请明确选择，不能沿用初筛的通道偏好。', dataset.id)
    if mapping is not None:
        return mapping, reason, []
    if table.shape[1] == 2:
        labels = [str(s).strip().lower() for s in column_labels(header, 2)]
        forbidden = {'i0', 'i_0', 'io', 'it', 'i1', 'itrans', 'i_trans', 'if', 'ifluor',
                     'fluorescence', 'ifl', 'ir', 'i2', 'mur', 'mut', 'muf'}
        if any(label in forbidden for label in labels) or reason in ('ambiguous_signal_mode', 'reference_signal_only', 'declared_signal_mode_unavailable'):
            raise ProjectError('mapping_requires_confirmation', '二维表含原始探测通道或测量模式声明不完整，不能把 I₀/It 等强度直接当成 μ。请确认映射。', dataset.id)
        # The two-column import contract explicitly declares mu(E) or chi(k).
        # Preserve its ordinary direct-signal defaults, without guessing units
        # from magnitudes or inventing a detector conversion.
        if dataset.mapping.mode != 'mu' or dataset.mapping.energy != 0 or dataset.mapping.signal != 1:
            raise ProjectError('mapping_requires_confirmation', '非默认二维映射尚未标记为用户确认，请先明确声明列和单位。', dataset.id)
        return dataset.mapping, 'two_column_direct_declaration', []
    raise ProjectError('mapping_ambiguous', '多列表未提供可唯一识别的能量单位和样品信号，不能默认取前两列；请确认能量/信号/参考通道及测量模式。', dataset.id)


def prepare_input_mappings(store, project, intent):
    """Create immutable resolved declarations before any calibration/merge."""
    working, local = project.model_copy(deep=True), intent.model_copy(deep=True)
    by_id = {d.id: d for d in project.datasets}
    ids = list(dict.fromkeys(intent.dataset_ids + intent.reference_ids))
    reference_request = intent.conditions.get('reference_calibration')
    if isinstance(reference_request, dict) and isinstance(reference_request.get('reference_dataset_id'), str):
        ids = list(dict.fromkeys([*ids, reference_request['reference_dataset_id']]))
    replacements, blocked, decisions, issues = {}, set(), [], []
    for did in ids:
        if did not in by_id:
            continue  # The dedicated reference validator gives its precise error.
        dataset = by_id[did]
        try:
            mapping, source, assumptions = resolve_input_mapping(_raw(store, project.id, dataset), dataset)
            resolved_id = dataset.id
            if _mapping_meaning(mapping) != _mapping_meaning(dataset.mapping):
                # Reuse an identical immutable declaration on retry; do not
                # proliferate clones or invalidate exact processing caches.
                previous = next((d for d in reversed(working.datasets)
                    if d.metadata.get('mapping_source_dataset_id') == did and d.raw_file == dataset.raw_file
                    and _mapping_meaning(d.mapping) == _mapping_meaning(mapping)
                    and (d.sample_id, d.element, d.edge, d.conditions, d.correction_states) ==
                        (dataset.sample_id, dataset.element, dataset.edge, dataset.conditions, dataset.correction_states)), None)
                if previous is None:
                    value = dataset.model_dump(mode='json')
                    resolved_id = 'ds_map_' + uuid4().hex
                    value.update(id=resolved_id, mapping=mapping.model_dump(mode='json'),
                        lineage=list(dict.fromkeys([dataset.id, *dataset.lineage])))
                    value['metadata'].update(mapping_resolution={'version': 1, 'source': source,
                        'original_mapping': dataset.mapping.model_dump(mode='json'),
                        'resolved_mapping': mapping.model_dump(mode='json')},
                        mapping_source_dataset_id=dataset.id, mapping_confirmed=False)
                    previous = Dataset.model_validate(value)
                    working.datasets.append(previous)
                resolved_id = previous.id
                replacements[did] = resolved_id
            decisions.append({'stage': 'input_mapping', 'rule': 'declared_signal_mapping_v1', 'dataset_id': did,
                'reason': '复用可信表头识别能量单位和样品 μ/透射/荧光通道；多列歧义先确认，原始声明和字节保留。',
                'selected': {'source_dataset_id': did, 'resolved_dataset_id': resolved_id,
                             'mapping': mapping.model_dump(mode='json'), 'mapping_source': source,
                             'user_confirmed': dataset.metadata.get('mapping_confirmed') is True,
                             'assumptions': assumptions}, 'alternatives': []})
        except (ValueError, RuntimeError) as exc:
            blocked.add(did)
            issues.append({'code': getattr(exc, 'code', 'mapping_failed'), 'message': str(exc),
                           'dataset_id': did, 'severity': 'error'})
    local.dataset_ids = [replacements.get(did, did) for did in local.dataset_ids]
    local.reference_ids = [replacements.get(did, did) for did in local.reference_ids]
    local.calibration_ids = {replacements.get(did, did): cid for did, cid in local.calibration_ids.items()}
    if isinstance(reference_request, dict):
        request = dict(reference_request)
        did = request.get('reference_dataset_id')
        if did in replacements:
            request['reference_dataset_id'] = replacements[did]
        if isinstance(request.get('apply_to_dataset_ids'), list):
            request['apply_to_dataset_ids'] = [replacements.get(did, did) for did in request['apply_to_dataset_ids']]
        local.conditions['reference_calibration'] = request
    return working, local, blocked, decisions, issues


def finite_find_e0(energy, mu, *, full_energy=None):
    """Pinned-Larch two-pass edge search with an explicit finite energy step.

    Larch 2026.3.1's public find_e0 has no estep argument. Its private kernel
    accepts one, whereas find_energy_step on fewer than 100 points accidentally
    takes [0:-0]. This compatibility adapter preserves the public two-pass,
    index and smoothing-scale rules without relying on that empty-slice NaN.

    Define the input step from the *complete valid measured grid*: sort its
    positive differences, discard the lowest floor(1%) and average the next at
    most ten. At least one difference must remain. This robust finite estimator
    is recorded, not interpreted as additional measured resolution.
    """
    from importlib.metadata import version
    from larch.xafs.pre_edge import _finde0
    from larch.xafs.xafsutils import TINY_ENERGY
    if version('xraylarch') != '2026.3.1':
        raise ProjectError('engine_version', '有限步长边搜索适配固定于 xraylarch 2026.3.1，升级引擎前需重新验证。')
    x, y = np.asarray(energy, dtype=float), np.asarray(mu, dtype=float)
    full = x if full_energy is None else np.asarray(full_energy, dtype=float)
    if (x.ndim != 1 or y.shape != x.shape or len(x) < 8 or full.ndim != 1 or len(full) < 8
            or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(full).all()
            or np.any(x <= 0) or np.any(full <= 0)
            or np.any(np.diff(x) < TINY_ENERGY) or np.any(np.diff(full) < TINY_ENERGY)):
        raise ProjectError('edge_search_grid', '边位搜索需要有限、正值、严格递增的能量网格和至少 8 个对应测量点。')
    steps = np.diff(full)
    ordered = np.sort(steps)
    skipped = int(.01 * len(ordered))
    selected = ordered[skipped:min(len(ordered), skipped + 10)]
    if len(selected) == 0:
        raise ProjectError('edge_search_step', '完整有效测量网格没有可用于估计步长的正差分。')
    step = float(np.mean(selected))
    if not np.isfinite(step) or step <= 0:
        raise ProjectError('edge_search_step', '无法从完整有效测量网格得到有限正能量步长。')
    first_e0, first_index, _ = _finde0(x, y, estep=step, use_smooth=False)
    if not np.isfinite(first_e0) or not 0 <= first_index < len(x):
        raise ProjectError('edge_search_nonfinite', 'Larch 初步边位搜索返回无效结果。')
    # Preserve exactly the public 2026.3.1 index selection and low-index branch.
    start, stop = max(3, first_index - 75), min(first_index + 75, len(x) - 3)
    smoothing_origin = float(first_e0)
    if first_index < .05 * len(x):
        smoothing_origin = float(x.mean())
        start, stop = max(3, first_index - 20), len(x) - 3
    clipped_step = max(.01, min(1., step))
    smoothing_step = .5 * (clipped_step + max(.01, min(1., smoothing_origin / 25000.)))
    if stop - start < 8 or not np.isfinite(smoothing_step) or smoothing_step <= 0:
        raise ProjectError('edge_search_sampling', '局部边位搜索的二次估计缺少足够测量上下文；请补充近边采样或明确给出 E₀。')
    e0, refined_index, _ = _finde0(x[start:stop], y[start:stop], estep=smoothing_step, use_smooth=True)
    if refined_index < 1:
        e0 = x[start + 2]
    if not np.isfinite(e0) or not x[0] <= e0 <= x[-1]:
        raise ProjectError('edge_search_nonfinite', 'Larch 二次边位搜索未返回有限且在测量范围内的 E₀。')
    return float(e0), {'algorithm': 'larch-2026.3.1-two-pass-finite-step-v1',
        'kernel': 'larch.xafs.pre_edge._finde0; explicit estep on both passes',
        'step_basis': 'complete valid measured grid; sorted positive differences, discard floor(1%), mean next <=10',
        'full_grid_points': len(full), 'search_points': len(x), 'discarded_smallest_steps': skipped,
        'step_average_count': len(selected), 'raw_energy_step_eV': step,
        'clipped_energy_step_eV': clipped_step, 'smoothing_step_eV': smoothing_step,
        'first_e0_eV': float(first_e0), 'refinement_slice': [int(start), int(stop)],
        'data_smoothing': False, 'derivative_smoothing': 'Larch search-only smoothing; fit/feature arrays unchanged'}


def _edge_prior(dataset, energy, mu, overrides, conditions):
    """xraydb values constrain search identity, never shift sample energies."""
    if not dataset.element or not dataset.edge:
        if overrides.get('e0') is not None:
            e0, estimator = float(overrides['e0']), {'algorithm': 'user_supplied'}
        else:
            e0, estimator = finite_find_e0(energy, mu, full_energy=energy)
        return e0, None, {'identity': 'unknown', 'search': 'global finite-step Larch derivative search; descriptive only',
                          'e0_estimator': estimator,
                          'absolute_calibration': False}
    import xraydb
    try:
        edges = xraydb.xray_edges(dataset.element)
    except (ValueError, KeyError, TypeError) as exc:
        raise ProjectError('edge_identity_unknown', '无法在本机 xraydb 中核对所声明的元素/吸收边，请检查样品事实。', dataset.id) from exc
    declared = edges.get(dataset.edge) if edges else None
    if declared is None:
        raise ProjectError('edge_identity_unknown', '本机 xraydb 不支持所声明的吸收边；请明确核对身份后使用专家流程。', dataset.id)
    tabulated = float(declared.energy)
    following = sorted(float(item.energy) for item in edges.values() if item.energy > tabulated)
    previous = sorted(float(item.energy) for item in edges.values() if item.energy < tabulated)
    next_edge = following[0] if following else None
    center = conditions.get('edge_search_center_eV', dataset.metadata.get('edge_search_center_eV'))
    if isinstance(center, dict):
        center = center.get(dataset.id)
    if center is not None:
        note = conditions.get('edge_search_note', dataset.metadata.get('edge_search_note'))
        if isinstance(center, bool) or not isinstance(center, (int, float)) or not np.isfinite(center) or not isinstance(note, str) or not note.strip():
            raise ProjectError('edge_search_basis', '覆盖表值的边位搜索中心必须是有限 eV 数值，并记录化学位移或仪器标尺依据。', dataset.id)
        center = float(center)
    else:
        center, note = tabulated, 'xraydb tabulated edge is a search prior, not an absolute energy calibration.'
    lo, hi = center - 80.0, center + 80.0
    if previous:
        lo = max(lo, previous[-1] + 30.0)
    if next_edge is not None:
        hi = min(hi, next_edge - 30.0)
    use = (energy >= lo) & (energy <= hi)
    if use.sum() < 8 or energy[0] > center + 20 or energy[-1] < center - 20:
        raise ProjectError('declared_edge_coverage', '测量能区不支持所声明元素/吸收边的局部搜索；不会改用全谱中更强的另一吸收边。', dataset.id)
    if overrides.get('e0') is not None:
        e0, estimator = float(overrides['e0']), {'algorithm': 'user_supplied'}
    else:
        e0, estimator = finite_find_e0(energy[use], mu[use], full_energy=energy)
    if not lo <= e0 <= hi:
        raise ProjectError('declared_edge_conflict', '指定 E₀ 与声明边的搜索先验冲突；若有实际化学位移，请记录 edge_search_center_eV 和 edge_search_note。', dataset.id)
    positions = np.flatnonzero(use)
    if e0 <= energy[positions[0]] or e0 >= energy[positions[-1]]:
        raise ProjectError('edge_search_boundary', '目标边位落在局部搜索边界；需要补充测量能区或明确搜索依据。', dataset.id)
    return e0, next_edge, {'identity': f'{dataset.element}-{dataset.edge}', 'library': 'xraydb',
        'library_version': xraydb.__version__,
        'e0_estimator': estimator,
        'tabulated_edge_eV': tabulated, 'search_center_eV': center, 'search_range_eV': [lo, hi],
        'next_tabulated_edge_eV': next_edge, 'basis_note': note,
        'role': 'identity/search/next-edge prior only; sample chemical shifts remain measured',
        'absolute_calibration': False, 'energy_shift_applied': 0.0}


def inspect_input(store, project_id, dataset):
    text = _raw(store, project_id, dataset)
    table, header = parse_ascii(text)
    if dataset.metadata.get('mapping_confirmed') is not True:
        declared, _, _ = resolve_input_mapping(text, dataset)
        if _mapping_meaning(declared) != _mapping_meaning(dataset.mapping):
            raise ProjectError('mapping_resolution_required', '可信表头与当前默认映射不同，请通过自动工作流建立新的映射声明，或明确确认映射；不能直接把通道强度当成 μ。', dataset.id)
    for field, pattern in [('element', r'Element\.symbol:\s*([A-Z][a-z]?)\b'),
                           ('edge', r'Element\.edge:\s*(K|L[123]|M[1-5])\b')]:
        found = re.search(pattern, '\n'.join(header))
        if found and getattr(dataset, field) not in (None, found[1]) and not dataset.metadata.get('header_identity_override'):
            raise ProjectError('header_identity_conflict', '元素/吸收边与原文件头不一致，请明确修正或记录覆盖依据。', dataset.id)
    mapping = dataset.mapping
    columns = [mapping.energy, mapping.signal] + ([mapping.reference] if mapping.mode != 'mu' else [])
    if len(set(columns)) != len(columns) or max(columns) >= table.shape[1]:
        raise ProjectError('invalid_mapping', 'Selected columns do not identify a valid spectrum.', dataset.id)
    if np.any(np.diff(table[:, mapping.energy]) <= 0):
        raise ProjectError('invalid_grid', 'Input coordinates must be strictly increasing; no rows were reordered or discarded.', dataset.id)
    if dataset.kind == 'chi':
        x, y = table[:, mapping.energy], table[:, mapping.signal]
        if mapping.mode != 'mu' or mapping.energy_unit != 'eV':
            raise ProjectError('chi_mapping', 'chi(k) requires direct k/chi columns, without energy-unit conversion or detector ratios.', dataset.id)
        if x[0] < 0:
            raise ProjectError('invalid_k', 'k must be nonnegative.', dataset.id)
    elif dataset.kind == 'mu':
        x, y, _ = construct_mu(text, mapping)
    else:
        raise ProjectError('not_spectrum', 'Support materials cannot enter automatic spectrum processing.', dataset.id)
    return x, y


def estimate_reference_calibration(store, project_id, reference_dataset_id, target_energy_eV,
        *, standard_confirmed=False, note='', search_half_width_eV=30.0, max_shift_eV=20.0,
        applicability_conditions=None):
    """Estimate reference drift without touching any sample chemical shift.

    The target must use the same *maximum raw-mu first derivative* definition.
    A standard identity/target supplied by a human is an explicit prerequisite;
    a tabulated absorption threshold is not assumed to equal this feature.
    Return an uncommitted Calibration so callers can publish it atomically.
    """
    project = store.read(project_id)
    reference = next((d for d in project.datasets if d.id == reference_dataset_id), None)
    if reference is None or reference.kind != 'mu' or not reference.element or not reference.edge:
        raise ProjectError('reference_standard_identity', '参考校准需要已确认元素/吸收边的原始 μ(E) 标准样。', reference_dataset_id)
    if standard_confirmed is not True or not str(note).strip():
        raise ProjectError('reference_standard_confirmation', '请确认标准样身份、目标特征定义及测量适用性，并记录依据；不能通过对齐样品边位代替仪器校准。', reference_dataset_id)
    for value in (target_energy_eV, search_half_width_eV, max_shift_eV):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ProjectError('reference_calibration_range', '标准样目标能量和搜索/漂移范围必须是有限的 eV 数值。')
    if not 2000 <= target_energy_eV < 200000 or not 5 <= search_half_width_eV <= 100 or not 0 < max_shift_eV <= 100:
        raise ProjectError('reference_calibration_range', '参考校准参数超出当前硬 X 射线方法的显式范围。')
    energy, mu = inspect_input(store, project_id, reference)
    lo, hi = target_energy_eV - search_half_width_eV, target_energy_eV + search_half_width_eV
    use = (energy >= lo) & (energy <= hi)
    if lo < energy[0] or hi > energy[-1] or use.sum() < 8 or np.max(np.diff(energy[use]), initial=0) > 5:
        raise ProjectError('reference_calibration_coverage', '参考谱未完整覆盖校准搜索窗，或采样不足；不外推标准样边位。', reference_dataset_id)
    derivative = np.gradient(mu, energy, edge_order=2)
    positions = np.flatnonzero(use)
    index = int(positions[np.argmax(derivative[use])])
    if index in (int(positions[0]), int(positions[-1])) or derivative[index] <= 0:
        raise ProjectError('reference_calibration_feature', '参考校准特征位于搜索窗边界或没有正导数边跃，需检查标准样。', reference_dataset_id)
    shift = float(target_energy_eV - energy[index])
    if abs(shift) > max_shift_eV:
        raise ProjectError('reference_drift_limit', '参考漂移超出明确允许范围，请检查能量单位、吸收边和标准样身份。', reference_dataset_id)
    return Calibration(id='cal_auto_' + uuid4().hex, kind='energy', value=shift, unit='eV',
        reference_dataset_ids=[reference_dataset_id], element=reference.element, edge=reference.edge,
        uncertainty_basis='not_estimated', conditions=dict(applicability_conditions or {}),
        applicability_note=(str(note).strip() + f'\nCalibration algorithm v1: maximum numpy.gradient(raw mu, measured energy, edge_order=2), no smoothing; '
            f'observed feature {energy[index]:.12g} eV, same-definition standard target {target_energy_eV:.12g} eV, '
            f'search [{lo:.12g},{hi:.12g}] eV, applied target-minus-observed {shift:.12g} eV. '
            f'Original reference sha256={reference.raw_file.sha256}. Statistical calibration uncertainty not estimated; sample edges never aligned.'))


def prepare_repeat_merges(store, project, intent):
    """Merge only human-confirmed independent, explicitly identical states.

    This pure project preparation writes immutable assets, never Project.save.
    Original measurements, observed-range masks and repeat scatter remain in the
    final project. Interpolation covariance is not represented as white noise.
    """
    from ..import_formats import describe
    if intent.task not in ('process', 'xanes'):
        raise ProjectError('repeat_merge_task', '自动重复扫描合并首期仅支持处理/XANES描述任务；不能改变结构模型或序列任务的既有观测绑定。')
    if intent.conditions.get('independent_repeats_confirmed') is not True or not str(intent.conditions.get('independent_repeats_note', '')).strip():
        raise ProjectError('repeat_independence_required', '合并前请确认这些是独立重复测量，并记录依据；复制文件和相似文件名不算独立重复。')
    by_id = {d.id: d for d in project.datasets}
    groups = {}
    for did in intent.dataset_ids:
        dataset = by_id[did]
        if dataset.kind != 'mu' or not dataset.sample_id or not dataset.element or not dataset.edge:
            raise ProjectError('repeat_metadata', '重复扫描合并需明确原始 μ(E)、同一样品 ID、元素和吸收边。', did)
        raw = _raw(store, project.id, dataset)
        declared = dataset.metadata.get('import_processing_state') or describe(raw, dataset.filename).get('processing_state')
        if declared not in ('mu', 'unknown') or (dataset.lineage and not dataset.metadata.get('mapping_source_dataset_id')) or dataset.metadata.get('import_role') == 'merged':
            raise ProjectError('repeat_original_required', '本配方只合并明确的原始重复扫描；已归并、归一化或其他衍生谱不能再次当成独立观测。', did)
        key = canonical({'sample_id': dataset.sample_id, 'element': dataset.element, 'edge': dataset.edge,
                         'mapping': dataset.mapping.model_dump(mode='json'), 'conditions': dataset.conditions,
                         'correction_states': dataset.correction_states})
        groups.setdefault(key, []).append(dataset)
    if not any(len(g) > 1 for g in groups.values()):
        raise ProjectError('repeat_no_compatible_group', '没有找到完全同样品、元素/边、列映射、测量状态和修正状态的重复组；未按文件名或谱形强行合并。')
    working = project.model_copy(deep=True)
    local = intent.model_copy(deep=True)
    replacements, records = {}, []
    for scans in groups.values():
        if len(scans) == 1:
            records.append({'dataset_ids': [scans[0].id], 'state': 'kept_separate',
                            'reason': 'No second compatible observation with identical explicit state.'})
            continue
        if len({d.raw_file.sha256 for d in scans}) != len(scans):
            raise ProjectError('duplicate_scan', '相同原始字节的副本不能作为独立重复扫描改善误差估计。')
        energies, signals, shifts, calibration_ids = [], [], {}, {}
        for dataset in scans:
            x, y = inspect_input(store, project.id, dataset)
            cal_id = intent.calibration_ids.get(dataset.id)
            if cal_id:
                calibration = next((c for c in project.calibrations if c.id == cal_id), None)
                if calibration is None or calibration.kind != 'energy' or (calibration.element, calibration.edge) != (dataset.element, dataset.edge):
                    raise ProjectError('calibration_edge', '重复扫描的参考校准与元素/吸收边不一致。', dataset.id)
                if calibration.conditions and any(dataset.conditions.get(k) != v for k, v in calibration.conditions.items()):
                    raise ProjectError('calibration_conditions', '重复扫描不满足所选参考校准的测量条件。', dataset.id)
                x = x + calibration.value
                shifts[dataset.id] = calibration.value
                calibration_ids[dataset.id] = calibration.id
            energies.append(x)
            signals.append(y)
        merged = merge_scan_arrays(energies, signals)
        lo, hi = float(merged['energy'][0]), float(merged['energy'][-1])
        arrays = {name: save_array(store, project.id, merged[name]) for name in ('energy', 'mu', 'scatter', 'weights')}
        for index, x in enumerate(energies):
            arrays['source_common_mask_' + str(index)] = save_array(store, project.id, (x >= lo) & (x <= hi))
        raw = '# Processing.state: mu\n# Repeat merge: confirmed independent original observations; arithmetic mean on measured intersection\n'
        raw += '\n'.join(f'{x:.17g} {y:.17g}' for x, y in zip(merged['energy'], merged['mu'])) + '\n'
        first = scans[0]
        merged_dataset = Dataset(id='ds_merge_' + uuid4().hex, filename=first.sample_id + '-merged-repeats.dat',
            raw_file=store.put_asset(project.id, raw.encode(), 'merged-repeats.dat'), raw_text=raw,
            sample_id=first.sample_id, element=first.element, edge=first.edge, conditions=first.conditions,
            correction_states=first.correction_states, lineage=[d.id for d in scans], arrays=arrays,
            metadata={'merge_method': merged['method'], 'independent_repeats_confirmed': True,
                'independent_repeats_note': intent.conditions['independent_repeats_note'],
                'source_dataset_ids': [d.id for d in scans], 'source_hashes': [d.raw_file.sha256 for d in scans],
                'source_common_masks': ['source_common_mask_' + str(i) for i in range(len(scans))],
                'measured_common_range_eV': [lo, hi], 'scan_count': len(scans),
                'scatter_definition': 'pointwise sample standard deviation of linearly interpolated original scans, ddof=1',
                'noise_propagation': 'unknown; interpolation covariance and normalization/background propagation not estimated',
                'reference_energy_shifts_eV': shifts, 'reference_calibration_ids': calibration_ids,
                'calibration_policy': 'reference shifts applied once to original scan energies before merge; no sample edge alignment',
                'input_mapping': 'merged direct mu(E), energy in eV; original channel mappings remain on source datasets'})
        working.datasets.append(merged_dataset)
        for dataset in scans:
            replacements[dataset.id] = merged_dataset.id
            local.calibration_ids.pop(dataset.id, None)
        records.append({'dataset_ids': [d.id for d in scans], 'merged_dataset_id': merged_dataset.id,
                        'state': 'merged', 'method': merged['method'], 'common_energy_range_eV': [lo, hi],
                        'source_common_mask_arrays': {d.id: arrays['source_common_mask_' + str(i)].model_dump(mode='json') for i, d in enumerate(scans)},
                        'reference_energy_shifts_eV': shifts, 'noise': 'unknown'})
    local.dataset_ids = list(dict.fromkeys(replacements.get(did, did) for did in intent.dataset_ids))
    return working, local, {'stage': 'repeat_scans', 'rule': 'confirmed_independent_same_state_merge_v1',
        'reason': '只合并明确独立、完全相同测量状态的重复扫描；保留原始谱、公共范围掩码和散布，插值不增加信息量。',
        'selected': {'dataset_ids_after_merge': local.dataset_ids, 'independence_note': intent.conditions['independent_repeats_note']},
        'alternatives': records}


def propose_parameters(store, project, dataset, overrides=None, conditions=None, calibration_id=None):
    """Select only measured windows. Return Parameters, mask, issues, evidence."""
    overrides, conditions = dict(overrides or {}), dict(conditions or {})
    x, y = inspect_input(store, project.id, dataset)
    issues = []
    calibration = None
    if calibration_id:
        calibration = next((c for c in project.calibrations if c.id == calibration_id), None)
        if calibration is None or calibration.kind != 'energy' or (calibration.element, calibration.edge) != (dataset.element, dataset.edge):
            raise ProjectError('calibration_edge', 'Selected energy calibration is absent or inapplicable to this element/edge.', dataset.id)
        if calibration.conditions and any(dataset.conditions.get(k) != v for k, v in calibration.conditions.items()):
            raise ProjectError('calibration_conditions', 'Energy-calibration applicability conditions do not match this scan.', dataset.id)
        if not calibration.applicability_note:
            issues.append(_issue('calibration_applicability', 'Energy calibration has no applicability note; inspect its reference conditions.', dataset.id))
    if dataset.kind == 'chi':
        if calibration_id or overrides.get('energy_shift', 0) != 0:
            raise ProjectError('chi_energy_shift', 'chi(k) cannot be recalibrated by an energy offset.', dataset.id)
        span = float(x[-1])
        dk = float(overrides.get('dk', 1.0))
        kmin = max(2.0, float(x[0]) + dk / 2)
        kmax = min(12.0, np.floor((span - dk / 2) * 10) / 10)
        selected = {'kmin': kmin, 'kmax': kmax, 'dk': dk}
        selected.update(overrides)
        parameters = Parameters.model_validate(selected)
        if parameters.kmin - parameters.dk / 2 < x[0] - 1e-10 or parameters.kmax + parameters.dk / 2 > x[-1] + 1e-10:
            raise ProjectError('chi_coverage', 'Complete FT taper exceeds measured k coverage.', dataset.id)
        return parameters, None, issues, {'input_kind': 'chi', 'measured_k_range': [float(x[0]), float(x[-1])],
                                         'energy_figures': 'not_available', 'e0': 'not_applicable'}
    from larch.xafs.xafsutils import ETOK
    shift = float(overrides.get('energy_shift', 0)) + (calibration.value if calibration else 0)
    energy = x + shift
    e0, known_next_edge, edge_prior = _edge_prior(dataset, energy, y, overrides, conditions)
    if edge_prior['identity'] == 'unknown':
        issues.append(_issue('unknown_edge_descriptive', '元素或吸收边未确定：当前仅执行描述性处理；全谱 E₀ 不代表已核实边身份，也不构成绝对校准。', dataset.id))
    if not energy[0] < e0 < energy[-1]:
        raise ProjectError('e0_coverage', 'E0 does not lie inside measured energy coverage.', dataset.id)
    if energy[0] < 2000:
        raise ProjectError('unsupported_soft_xray', 'Automatic quantitative processing below 2 keV is outside the current hard-X-ray recipe; use an independently validated specialist method.', dataset.id)
    mask = np.ones(len(energy), dtype=bool)
    cutoff = conditions.get('next_edge_eV', dataset.metadata.get('next_edge_eV'))
    if isinstance(cutoff, dict):
        cutoff = cutoff.get(dataset.id)
    if cutoff is None:
        cutoff = known_next_edge
    elif known_next_edge is not None:
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)):
            raise ProjectError('next_edge', '下一吸收边能量必须是有限的 eV 数值。', dataset.id)
        cutoff = min(float(cutoff), known_next_edge)
    if cutoff is not None:
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not np.isfinite(cutoff):
            raise ProjectError('next_edge', 'Following-edge energy must be a finite eV value.', dataset.id)
        mask = energy < float(cutoff) - 30.0
        if not mask.all():
            issues.append(_issue('following_edge_excluded', 'Points within 30 eV of the known/declared following edge and above are excluded by a preserved mask.', dataset.id, 'info'))
    supported = energy[mask]
    if len(supported) < 40 or supported[-1] <= e0 + 40:
        raise ProjectError('short_postedge', 'Insufficient post-edge coverage for the supported normalization/background workflow.', dataset.id)
    spacing = float(np.median(np.diff(supported)))
    margin = max(spacing, 0.01)
    pre1 = max(-150.0, float(supported[0] - e0 + margin))
    pre2 = -30.0
    if pre1 >= pre2 - 10:
        pre2 = min(-10.0, pre1 / 3)
    norm2 = min(700.0, float(supported[-1] - e0 - margin))
    norm1 = min(150.0, max(30.0, norm2 / 3))
    near = energy[(energy >= e0 - 15) & (energy <= e0 + 30)]
    if len(near) < 8 or np.max(np.diff(near), initial=0) > 5:
        issues.append(_issue('sparse_near_edge', 'Near-edge sampling is sparse; derivative features and E0 require review.', dataset.id))
    if np.max(np.diff(supported), initial=0) > max(20.0, 8 * spacing):
        issues.append(_issue('large_sampling_gap', 'The measured spectrum has a large sampling gap; no interpolation creates independent observations.', dataset.id))
    # Reject isolated jump-and-return spikes near the edge rather than moving
    # them, masking them, or accepting their derivative maximum as an edge.
    slopes = np.diff(y) / np.diff(energy)
    slope_scale = max(float(np.percentile(np.abs(slopes), 75)), np.finfo(float).eps)
    near_pairs = (energy[1:-1] >= e0 - 30) & (energy[1:-1] <= e0 + 50)
    reversals = (slopes[:-1] * slopes[1:] < 0) & (np.minimum(np.abs(slopes[:-1]), np.abs(slopes[1:])) > 20 * slope_scale)
    if np.any(near_pairs & reversals):
        raise ProjectError('suspected_edge_spike', 'An isolated jump-and-return spike intersects the edge region; inspect original points before processing.', dataset.id)
    kgrid = np.floor(np.sqrt((supported[-1] - e0) * ETOK) / .05) * .05
    dk = float(overrides.get('dk', 1.0))
    kmax = min(12.0, np.floor((kgrid - dk / 2 - .05) * 10) / 10)
    kmin = 2.0 if kmax > 3.1 else .5
    settings = {'e0': e0, 'pre1': pre1, 'pre2': pre2, 'norm1': norm1, 'norm2': norm2,
                'nnorm': 2 if norm2 - norm1 >= 150 else 1, 'kmin': kmin, 'kmax': kmax,
                'dk': dk, 'bkg_kmax': float(kgrid), 'rbkg': 1.0}
    settings.update(overrides)
    parameters = Parameters.model_validate(settings)
    if parameters.kmax + parameters.dk / 2 > kgrid + 1e-10:
        raise ProjectError('ft_coverage', 'Requested FT taper exceeds the measured/background-supported k range.', dataset.id)
    for lo, hi, count, code in [(parameters.pre1, parameters.pre2, 5, 'preedge_coverage'),
                                 (parameters.norm1, parameters.norm2, max(5, parameters.nnorm + 2), 'norm_coverage')]:
        selected = supported[(supported >= parameters.e0 + lo) & (supported < parameters.e0 + hi)]
        if parameters.e0 + lo < supported[0] or parameters.e0 + hi > supported[-1] or len(selected) < count:
            raise ProjectError(code, 'Selected normalization windows lack sufficient measured points; the input was not repaired.', dataset.id)
    if calibration is None and not dataset.metadata.get('reference_calibration_ids'):
        issues.append(_issue('uncalibrated_energy', 'Energy is compared on the supplied scale; sample edges were not shifted into alignment.', dataset.id, 'info'))
    if dataset.mapping.mode == 'fluorescence':
        issues.append(_issue('fluorescence_correction_unverified', 'No dead-time or self-absorption correction was invented. Quantitative interpretation needs verified external correction provenance.', dataset.id))
    evidence = {'input_kind': 'mu', 'measured_energy_range_eV': [float(energy[0]), float(energy[-1])],
                'background_energy_limit_eV': float(supported[-1]), 'excluded_points': int((~mask).sum()),
                'k_coverage': float(kgrid), 'e0_role': 'chi extraction; distinct from reference calibration and fitted delta E0',
                'calibration_shift_eV': calibration.value if calibration else 0.0,
                'edge_search_prior': edge_prior, 'next_edge_limit_eV': cutoff,
                'sample_edge_alignment': False, 'data_smoothing': False,
                'normalization_selection': 'measured coverage and positive signed edge jump; no FEFFIT objective'}
    return parameters, mask if not mask.all() else None, issues, evidence


def processing_fingerprint(dataset, parameters, calibration_id, calibration, mask):
    from .larch_adapter import current_source_sha256, current_engine_versions
    import xraydb
    value = {'version': 1, 'source': dataset.raw_file.sha256, 'mapping': dataset.mapping.model_dump(mode='json'),
             'kind': dataset.kind, 'metadata': dataset.metadata, 'correction_states': dataset.correction_states,
             'parameters': parameters.model_dump(mode='json'), 'calibration_id': calibration_id,
             'calibration': calibration.model_dump(mode='json') if calibration else None,
             'mask': mask.tolist() if mask is not None else None,
             'source_sha256': current_source_sha256(), 'engine_versions': current_engine_versions(),
             'edge_prior_library': {'xraydb': xraydb.__version__}}
    return hashlib.sha256(canonical(value)).hexdigest()


def process_automatically(store, project, dataset, intent, context=None):
    if dataset.metadata.get('import_processing_state') in ('normalized', 'difference', 'background', 'chi_fit_table', 'r_space', 'q_space'):
        raise ProjectError('derived_input', '导入元数据已声明此谱为归一化/差谱/背景或其他衍生结果，不能作为原始 μ(E) 再做 AUTOBK。', dataset.id)
    parameters, mask, issues, evidence = propose_parameters(store, project, dataset,
        intent.processing_overrides, intent.conditions, intent.calibration_ids.get(dataset.id))
    calibration_id = intent.calibration_ids.get(dataset.id)
    calibration = next((c for c in project.calibrations if c.id == calibration_id), None)
    fingerprint = processing_fingerprint(dataset, parameters, calibration_id, calibration, mask)
    fingerprint = hashlib.sha256(canonical({'base': fingerprint,
        'check_background_stability': intent.conditions.get('check_background_stability', True)})).hexdigest()
    # Cache only identical, fully specified automatic processing from this same
    # source revision. Updated conditions and noise can never silently reuse it.
    for previous in reversed(project.processing):
        if previous.metadata.get('external_plot_only') is True and intent.task != 'plot':
            continue
        if previous.dataset_id == dataset.id and previous.metadata.get('automation_fingerprint') == fingerprint:
            for ref in previous.arrays.values():
                load_array(store, project.id, ref)
            return previous, [], issues, {'stage': 'processing', 'rule': 'content_addressed_reuse',
                'reason': 'Original bytes, mapping, calibration, mask and resolved settings are identical.',
                'dataset_id': dataset.id, 'selected': previous.id, 'alternatives': []}
    frozen = FrozenProjectStore(store, project)
    # Explicitly retain unknown noise. The scientific fit branch requires the
    # user to select an experimental noise model; high-R is not assumed empty.
    noise = NoiseSpec(source='unknown', assumptions=['No independent repeated-scan noise estimate supplied; no high-R region presumed signal-free.'])
    if context:
        context.heartbeat('automatic preprocessing: ' + dataset.id, .15)
    selected = process_dataset(frozen, project.id, dataset.id, parameters, mask, calibration_id, noise)
    selected.metadata.update(automation_fingerprint=fingerprint, auto_recipe='hard-xas-descriptive-v1',
                             auto_evidence=evidence, background_selection='first coverage-valid baseline; no final-fit score')
    if dataset.kind == 'mu' and evidence.get('edge_search_prior', {}).get('library_version'):
        selected.engine_versions['xraydb'] = evidence['edge_search_prior']['library_version']
    if dataset.metadata.get('reference_calibration_ids'):
        selected.resolved_calls.append({'operation': 'reference_shift_before_repeat_merge', 'arguments': {
            'calibration_ids': dataset.metadata['reference_calibration_ids'],
            'energy_shifts_eV': dataset.metadata['reference_energy_shifts_eV'],
            'applied_once_before_merge': True, 'sample_edge_alignment': False}})
    trials, alternatives = [], []
    if dataset.kind == 'mu' and intent.conditions.get('check_background_stability', True):
        for delta in (-.1, .1):
            settings = parameters.model_dump(mode='json')
            settings['rbkg'] = round(parameters.rbkg + delta, 6)
            trial_record = {'rbkg': settings['rbkg'], 'status': 'failed'}
            try:
                if context:
                    context.heartbeat('background sensitivity: ' + dataset.id, .2)
                candidate = process_dataset(frozen, project.id, dataset.id, settings, mask, calibration_id, noise)
                candidate.metadata['automation_sensitivity'] = {'source_processing_id': selected.id,
                    'parameter': 'rbkg', 'delta_angstrom': delta, 'not_statistical_uncertainty': True}
                trials.append(candidate)
                k = load_array(store, project.id, selected.arrays['k'])
                chi = load_array(store, project.id, selected.arrays['chi'])
                ck = load_array(store, project.id, candidate.arrays['k'])
                cy = load_array(store, project.id, candidate.arrays['chi'])
                use = (k >= parameters.kmin) & (k <= parameters.kmax) & (k >= ck[0]) & (k <= ck[-1])
                rms = float(np.sqrt(np.mean((chi[use] - np.interp(k[use], ck, cy)) ** 2)))
                base_rms = float(np.sqrt(np.mean(chi[use] ** 2)))
                trial_record.update(status='succeeded', processing_id=candidate.id, chi_rms_difference=rms,
                                    relative_rms_difference=rms / max(base_rms, 1e-12))
                # An engineering attention threshold, explicitly not a validated
                # confidence score or a reason to choose a different background.
                if rms / max(base_rms, 1e-12) > .25:
                    issues.append(_issue('background_sensitive', 'chi(k) changes by more than the engineering attention threshold (25% RMS) when rbkg changes by 0.1 Å; inspect the retained sensitivity cases.', dataset.id))
            except (ValueError, RuntimeError) as exc:
                trial_record.update(code=getattr(exc, 'code', 'sensitivity_failed'), message=str(exc))
                issues.append(_issue('background_sensitivity_incomplete', 'A nearby rbkg candidate failed; baseline retained, without a stability claim.', dataset.id))
            alternatives.append(trial_record)
    selected.metadata['background_sensitivity'] = alternatives
    return selected, [selected, *trials], issues, {'stage': 'processing', 'rule': 'coverage_first_v1',
        'reason': 'Measured-window checks and the existing Larch signed-edge-step gate select the baseline; all sensitivity cases retained.',
        'dataset_id': dataset.id, 'selected': {'processing_id': selected.id, 'parameters': parameters.model_dump(mode='json'),
                                            'evidence': evidence}, 'alternatives': alternatives}


def import_external_plot(store, project, dataset, intent, context=None):
    """Explicit external processed data may be plotted without inventing mu.

    Only the already identified normalized mu(E) or chi(k) states are accepted;
    difference/background/fit tables and unknown coordinates are not guessed.
    """
    from ..import_formats import describe
    text = _raw(store, project.id, dataset)
    declaration = describe(text, dataset.filename, dataset.kind)
    state = dataset.metadata.get('import_processing_state') or declaration.get('processing_state')
    if state == 'unknown':
        state = declaration.get('processing_state')
    note = dataset.metadata.get('external_processing_note')
    if not isinstance(note, str) or not note.strip():
        raise ProjectError('external_processing_note_required', '外部处理谱直接绘图需记录处理方法、列含义和来源；请补充 external_processing_note。', dataset.id)
    if state not in ('normalized', 'chi', 'corrected_chi'):
        raise ProjectError('external_plot_state', '当前外部直绘支持明确归一化 E 谱或 χ(k)；未知语义、差谱、背景和历史拟合表不能冒充这些量。', dataset.id)
    if dataset.kind == 'chi' and state in ('chi', 'corrected_chi'):
        pp, additions, issues, decision = process_automatically(store, project, dataset, intent, context)
        # A reused scientific revision stays immutable; an external plot-only
        # tag cannot be appended to a previously published cached record.
        if not additions:
            value = pp.model_dump(mode='json')
            value['id'] = 'pp_' + uuid4().hex
            if value['noise']['processing_revision_id']:
                value['noise']['processing_revision_id'] = value['id']
            pp = ProcessingRevision.model_validate(value)
            additions = [pp]
        pp.metadata.update(external_plot_only=True, external_processing_note=note,
                           quantitative_eligible=False, external_validation='user_declared_not_independently_verified')
        return pp, additions, issues, decision
    if dataset.kind != 'mu' or dataset.mapping.mode != 'mu':
        raise ProjectError('external_plot_mapping', '归一化 E 谱直绘需要明确的能量/归一化信号列，不能再做通道比值。', dataset.id)
    energy, norm = inspect_input(store, project.id, dataset)
    if intent.calibration_ids.get(dataset.id) or intent.processing_overrides.get('energy_shift', 0):
        raise ProjectError('external_plot_recalibration', '外部直绘保留既有数值；如需重新校准，请建立有来源的分析版本。', dataset.id)
    pp = ProcessingRevision(id='pp_' + uuid4().hex, dataset_id=dataset.id, input_revision=project.revision,
        arrays={'energy': save_array(store, project.id, energy), 'norm': save_array(store, project.id, norm)},
        noise=NoiseSpec(source='unknown'),
        resolved_calls=[{'operation': 'external_plot_import', 'arguments': {'declared_state': state,
            'source_note': note, 'normalization': 'external; not recomputed', 'data_smoothing': False}}],
        warnings=['外部归一化谱按用户声明直接绘图，未验证原始处理方法；未补造 μ(E)、背景或 χ(k)。'],
        metadata={'external_plot_only': True, 'external_processing_note': note, 'quantitative_eligible': False,
                  'external_validation': 'user_declared_not_independently_verified',
                  'source_sha256': dataset.raw_file.sha256, 'input_mapping': dataset.mapping.model_dump(mode='json'),
                  'input_kind': dataset.kind, 'input_kweight': dataset.metadata.get('input_kweight', 0)})
    return pp, [pp], [], {'stage': 'processing', 'rule': 'external_plot_import_v1',
        'reason': '按已声明语义导入外部处理谱，仅做科研绘图，不补造原始吸收谱或背景结果。',
        'dataset_id': dataset.id, 'selected': {'processing_id': pp.id, 'declared_state': state, 'source_note': note}, 'alternatives': []}
