"""Versioned, portable automation records, stored in Project.metadata.

The project v1 reader already preserves JSON metadata and verifies nested asset
references. Keeping automation here therefore preserves the v1 archive reader;
no scientific record is silently rewritten during an upgrade.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .contracts import Contract, EntityId

AnalysisOperation = Literal['lcf', 'structural_fit', 'decomposition','preedge']
SUPPORTED_OPERATIONS = {
    'process': (), 'xanes': ('lcf','decomposition','preedge'), 'exafs': ('structural_fit',),
    'series': ('lcf', 'structural_fit', 'decomposition','preedge'), 'plot': (),
}


class AnalysisIntent(Contract):
    task: Literal['process', 'xanes', 'exafs', 'series', 'plot'] = 'process'
    requested_operations: list[AnalysisOperation] | None = Field(default=None, max_length=2)
    dataset_ids: list[EntityId] = Field(min_length=1, max_length=64)
    reference_ids: list[EntityId] = Field(default_factory=list, max_length=8)
    model_id: EntityId | None = None
    model_recipe_id: EntityId | None = None
    conditions: dict[str, JsonValue] = Field(default_factory=dict)
    processing_overrides: dict[str, JsonValue] = Field(default_factory=dict)
    calibration_ids: dict[EntityId, EntityId] = Field(default_factory=dict)
    decomposition_window_eV: list[float] | None = Field(default=None,min_length=2,max_length=2,exclude_if=lambda v:v is None)
    decomposition_components: int = Field(default=2,ge=1,le=8,exclude_if=lambda v:v==2)
    preedge_windows: dict[str, list[float]] | None = Field(default=None, exclude_if=lambda v:v is None)
    lcf: dict[str, JsonValue] = Field(default_factory=dict)
    fit: dict[str, JsonValue] = Field(default_factory=dict)
    series: dict[str, JsonValue] = Field(default_factory=dict)
    use_existing_processing: bool = False

    @model_validator(mode='after')
    def identities(self):
        if self.requested_operations is not None:
            if len(set(self.requested_operations)) != len(self.requested_operations):
                raise ValueError('Requested operations must be unique.')
            if set(self.requested_operations) - set(SUPPORTED_OPERATIONS[self.task]):
                raise ValueError('The selected task does not support these operations.')
        if len(set(self.dataset_ids)) != len(self.dataset_ids):
            raise ValueError('Selected sample datasets must be unique.')
        if len(set(self.reference_ids)) != len(self.reference_ids):
            raise ValueError('Selected reference datasets must be unique.')
        if set(self.dataset_ids) & set(self.reference_ids):
            raise ValueError('A sample cannot also be its own LCF reference.')
        if set(self.calibration_ids) - set(self.dataset_ids + self.reference_ids):
            raise ValueError('Energy calibrations must belong to selected inputs.')
        return self


def effective_operations(intent):
    """Resolve legacy opt-in semantics without altering the saved declaration."""
    if intent.requested_operations is not None:
        return ['structural_fit'] if intent.task == 'exafs' else list(intent.requested_operations)
    operations = []
    if intent.task in ('xanes', 'series') and intent.reference_ids:
        operations.append('lcf')
    if intent.task == 'exafs' or (intent.task == 'series' and (intent.model_id or intent.model_recipe_id)):
        operations.append('structural_fit')
    return operations


def normalize_intent(value):
    """Remove task-owned UI state before validating the scientific request.

    This never manufactures confirmations or chooses a model. Historical saved
    intents remain readable; only newly submitted execution intent is scoped.
    """
    from copy import deepcopy
    data = deepcopy(value.model_dump(mode='json') if hasattr(value, 'model_dump') else value)
    task = data.get('task', 'process')
    requested = data.get('requested_operations')
    if requested is not None:
        if not isinstance(requested, list) or any(op not in ('lcf', 'structural_fit', 'decomposition','preedge') for op in requested) or len(set(requested)) != len(requested):
            return AnalysisIntent.model_validate(data)
        if task in SUPPORTED_OPERATIONS and set(requested) - set(SUPPORTED_OPERATIONS[task]):
            from .projects import ProjectError
            raise ProjectError('operation_task', '当前分析模式不支持所选功能，请重新选择模式与功能。')
        # EXAFS is an explicit structural-fit mode, never a disguised plotting
        # mode when its checkbox is omitted. The effective plan shows it fixed.
        requested = ['structural_fit'] if task == 'exafs' else [op for op in ('lcf', 'structural_fit', 'decomposition','preedge') if op in requested]
        data['requested_operations'] = requested
    if requested is None or 'decomposition' not in requested:
        data.pop('decomposition_window_eV',None)
        data.pop('decomposition_components',None)
    if requested is None or 'preedge' not in requested:
        data.pop('preedge_windows',None)
    lcf_enabled = task in ('xanes', 'series') and (requested is None or 'lcf' in requested)
    fit_enabled = task in ('exafs', 'series') and (requested is None or 'structural_fit' in requested)
    conditions = data.get('conditions', {})
    series = data.get('series', {})
    if not isinstance(conditions, dict) or not isinstance(series, dict):
        return AnalysisIntent.model_validate(data)
    if not lcf_enabled:
        removed = set(data.get('reference_ids', [])) - set(data.get('dataset_ids', []))
        data['reference_ids'], data['lcf'] = [], {}
        if isinstance(data.get('calibration_ids'), dict):
            data['calibration_ids'] = {k: v for k, v in data['calibration_ids'].items() if k not in removed}
        for key in ('reference_applicability_confirmed', 'reference_applicability_note'):
            conditions.pop(key, None)
    if not fit_enabled:
        data['model_id'], data['model_recipe_id'], data['fit'] = None, None, {}
        series.pop('relation', None)
    if task != 'series':
        conditions.pop('state_key', None)
        series = {k: v for k, v in series.items() if k == 'merge_repeats' and (task in ('process', 'xanes') or v is True)}
    if task not in ('process', 'xanes') or series.get('merge_repeats') is not True:
        # An explicitly enabled, unsupported operation is rejected by the
        # workflow, not silently converted into a different analysis. False UI
        # residue is irrelevant and can be discarded safely.
        if series.get('merge_repeats') is not True:
            series.pop('merge_repeats', None)
        for key in ('independent_repeats_confirmed', 'independent_repeats_note'):
            conditions.pop(key, None)
    if task not in ('exafs', 'series') or not (data.get('model_id') or data.get('model_recipe_id')) or (task == 'series' and series.get('relation', 'independent') == 'independent'):
        for key in ('joint_applicability_confirmed', 'joint_applicability_note'):
            conditions.pop(key, None)
    if task == 'plot':
        conditions, series = {}, {}
        data['calibration_ids'], data['processing_overrides'] = {}, {}
    data['conditions'], data['series'] = conditions, series
    return AnalysisIntent.model_validate(data)


def validate_calibration_scope(intent, *, require_explicit=False):
    """Legacy calls may omit scope; an explicit scope is never empty or foreign."""
    from .projects import ProjectError
    request = intent.conditions.get('reference_calibration')
    if request is None:
        return
    if not isinstance(request, dict):
        raise ProjectError('reference_calibration_contract', '参考校准条件必须是一张明确的标准样条件卡。')
    scope = request.get('apply_to_dataset_ids')
    if scope is None and 'apply_to_dataset_ids' not in request and not require_explicit:
        return
    if scope is None and 'apply_to_dataset_ids' not in request:
        raise ProjectError('reference_calibration_scope_required', '请选择本次标准样校准适用的扫描；相同元素和吸收边不能证明属于同一次漂移。')
    selected = set(intent.dataset_ids + intent.reference_ids)
    if not isinstance(scope, list) or not scope or any(not isinstance(did, str) for did in scope) or len(set(scope)) != len(scope) or not set(scope) <= selected:
        raise ProjectError('reference_calibration_scope', '参考校准范围必须是本次选定谱的非空、不重复子集。')


class AnalysisIssue(Contract):
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    dataset_id: EntityId | None = None
    severity: Literal['info', 'warning', 'error'] = 'warning'


class AnalysisDecision(Contract):
    stage: str
    rule: str
    reason: str
    dataset_id: EntityId | None = None
    selected: JsonValue = None
    alternatives: list[JsonValue] = Field(default_factory=list)


class WorkflowResult(Contract):
    id: EntityId
    version: Literal[1] = 1
    task: Literal['process', 'xanes', 'exafs', 'series', 'plot']
    state: Literal['succeeded', 'partial', 'needs_input'] = 'succeeded'
    input_revision: int = Field(ge=0)
    intent: AnalysisIntent
    created_at: str
    finished_at: str | None = None
    recipe: dict[str, JsonValue] = Field(default_factory=dict)
    processing_ids: list[EntityId] = Field(default_factory=list)
    decisions: list[AnalysisDecision] = Field(default_factory=list)
    issues: list[AnalysisIssue] = Field(default_factory=list)
    features: list[dict[str, JsonValue]] = Field(default_factory=list)
    lcf_results: list[dict[str, JsonValue]] = Field(default_factory=list)
    preedge_results: list[dict[str, JsonValue]] = Field(default_factory=list, exclude_if=lambda value:not value)
    decomposition: dict[str, JsonValue] | None = Field(default=None, exclude_if=lambda value: value is None)
    fit_run_ids: list[EntityId] = Field(default_factory=list)
    series: dict[str, JsonValue] = Field(default_factory=dict)
    steps: list[dict[str, JsonValue]] = Field(default_factory=list)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)


def read_workflows(project):
    """Validate supported records; legacy projects simply contain no workflows."""
    values = project.metadata.get('automation_runs', [])
    if not isinstance(values, list):
        raise ValueError('automation_runs must be a list of versioned workflows.')
    return [WorkflowResult.model_validate(item) for item in values]
