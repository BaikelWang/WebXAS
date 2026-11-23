"""Portable scientific records. Larch objects never cross this boundary.

The records validate shape, identity and references. Evaluation of parameter
expressions and verification of files belong to their dedicated services.
"""
from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, field_validator, model_validator

from ..models import Mapping, Parameters

EntityId = Annotated[str, StringConstraints(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]
ParameterId = Annotated[str, StringConstraints(min_length=1, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]
Expression = Annotated[str, StringConstraints(min_length=1, max_length=2000)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Unit = Literal["1", "eV", "angstrom", "angstrom2", "angstrom-1", "K"]
JobState = Literal["queued", "running", "cancelling", "succeeded", "failed", "cancelled", "timed_out", "interrupted"]
Assessment = Literal["conditional", "needs_review", "not_identifiable", "validated"]
CorrectionState = Literal["not_applicable", "unknown", "uncorrected", "corrected"]


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite numbers are not portable scientific values.")
    if isinstance(value, BaseModel):
        value = value.__dict__
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True, strict=True)

    @model_validator(mode="after")
    def finite_tree(self):
        _finite_tree(self.__dict__)
        return self


class AssetRef(Contract):
    path: str = Field(min_length=1, max_length=500)
    sha256: Sha256
    byte_size: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def relative_path(cls, value):
        parts = value.split("/")
        if "\\" in value or ":" in value or value.startswith("/") or any(p in ("", ".", "..") for p in parts):
            raise ValueError("Asset path must be a project-relative POSIX path without empty, dot or parent segments.")
        if PurePosixPath(value).is_absolute():
            raise ValueError("Asset path must be relative.")
        return value


class ArrayRef(AssetRef):
    dtype: Literal["bool", "int32", "int64", "float64", "complex128"]
    shape: list[int] = Field(min_length=1, max_length=3)

    @field_validator("shape")
    @classmethod
    def positive_shape(cls, value):
        if any(n <= 0 for n in value):
            raise ValueError("Array dimensions must be positive.")
        return value


class MigrationIssue(Contract):
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
    object_id: EntityId | None = None
    severity: Literal["info", "warning", "error"] = "warning"


class Dataset(Contract):
    id: EntityId
    raw_file: AssetRef
    filename: str = Field(min_length=1, max_length=250)
    kind: Literal["mu", "chi", "support"] = "mu"
    raw_text: str | None = Field(default=None, max_length=2_000_000)
    mapping: Mapping = Field(default_factory=Mapping)
    element: str | None = Field(default=None, max_length=3)
    edge: str | None = Field(default=None, max_length=10)
    sample_id: EntityId | None = None
    conditions: dict[str, JsonValue] = Field(default_factory=dict)
    arrays: dict[str, ArrayRef] = Field(default_factory=dict)
    mask: ArrayRef | None = None
    lineage: list[EntityId] = Field(default_factory=list)
    reference_dataset_id: EntityId | None = None
    correction_states: dict[Literal["self_absorption", "dead_time", "saturation", "thickness"], CorrectionState] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def mask_is_boolean(self):
        if self.mask is not None and self.mask.dtype != "bool":
            raise ValueError("Dataset mask must have bool dtype.")
        if len(self.lineage) != len(set(self.lineage)):
            raise ValueError("Dataset lineage IDs must not repeat.")
        return self


class NoiseSpec(Contract):
    source: Literal["unknown", "explicit_scalar", "explicit_array", "engine_estimate"] = "unknown"
    value: float | None = Field(default=None, gt=0)
    array: ArrayRef | None = None
    unit: Literal["chi", "mu"] = "chi"
    processing_revision_id: EntityId | None = None
    estimation_range: list[float] | None = None
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def source_contract(self):
        if self.source == "explicit_scalar" and (self.value is None or self.array is not None):
            raise ValueError("Explicit scalar noise requires only a positive value.")
        if self.source == "explicit_array" and (self.array is None or self.value is not None):
            raise ValueError("Explicit array noise requires only an array reference.")
        if self.source in ("unknown", "engine_estimate") and (self.value is not None or self.array is not None):
            raise ValueError("Unknown/engine-estimated noise cannot contain an unlabelled explicit value.")
        if self.array is not None and (self.array.dtype != "float64" or len(self.array.shape) != 1):
            raise ValueError("Noise array must be one-dimensional float64.")
        if self.estimation_range is not None and (len(self.estimation_range) != 2 or self.estimation_range[0] >= self.estimation_range[1]):
            raise ValueError("Noise estimation range requires ordered lower and upper values.")
        return self


class Calibration(Contract):
    id: EntityId
    kind: Literal["energy", "s02"]
    value: float
    unit: Literal["eV", "1"]
    reference_dataset_ids: list[EntityId] = Field(min_length=1)
    element: str
    edge: str
    reference_model_id: EntityId | None = None
    stderr: float | None = Field(default=None, ge=0)
    uncertainty_basis: Literal["not_estimated", "conditional_fixed", "joint_reference"] = "not_estimated"
    conditions: dict[str, JsonValue] = Field(default_factory=dict)
    applicability_note: str = ""

    @model_validator(mode="after")
    def units_for_kind(self):
        if self.unit != ("eV" if self.kind == "energy" else "1"):
            raise ValueError("Calibration unit does not match its kind.")
        if self.kind == "s02" and self.value <= 0:
            raise ValueError("S02 calibration must be positive.")
        return self


class ProcessingRevision(Contract):
    id: EntityId
    dataset_id: EntityId
    input_revision: int = Field(default=0, ge=0)
    parameters: Parameters = Field(default_factory=Parameters)
    calibration_id: EntityId | None = None
    arrays: dict[str, ArrayRef] = Field(default_factory=dict)
    mask: ArrayRef | None = None
    noise: NoiseSpec = Field(default_factory=NoiseSpec)
    engine_versions: dict[str, str] = Field(default_factory=dict)
    resolved_calls: list[dict[str, JsonValue]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class GeometryAtom(Contract):
    element: str = Field(min_length=1, max_length=3)
    xyz: list[float] = Field(min_length=3, max_length=3)
    occupancy: float = Field(default=1, gt=0, le=1)


class Structure(Contract):
    id: EntityId
    source_file: AssetRef
    source_format: Literal["cif", "xyz", "feff"]
    absorber_element: str
    edge: str
    absorber_site: str = Field(min_length=1, max_length=100)
    atoms: list[GeometryAtom] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PathAsset(Contract):
    id: EntityId
    source_file: AssetRef
    feff_input: AssetRef | None = None
    feff_program_sha256: Sha256 | None = None
    structure_id: EntityId | None = None
    absorber_element: str
    edge: str
    absorber_site: str = Field(min_length=1)
    nleg: int = Field(ge=2)
    reff: float = Field(gt=0)
    degen: float = Field(gt=0)
    geometry: list[GeometryAtom] = Field(default_factory=list)
    k_min: float = Field(default=0, ge=0)
    k_max: float = Field(gt=0)
    engine_versions: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def k_range(self):
        if self.k_min >= self.k_max:
            raise ValueError("Path k_min must be below k_max.")
        return self


class Parameter(Contract):
    id: ParameterId
    label: str = ""
    unit: Unit = "1"
    scope: Literal["global", "dataset", "path"] = "global"
    owner_id: EntityId | None = None
    kind: Literal["free", "fixed", "expr"] = "free"
    value: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    expression: str | None = Field(default=None, min_length=1, max_length=2000)
    physical_basis: str = ""

    @model_validator(mode="after")
    def parameter_contract(self):
        if (self.scope == "global") != (self.owner_id is None):
            raise ValueError("Global parameters have no owner; local parameters require an owner.")
        if self.kind == "expr":
            if self.expression is None or self.value is not None or self.minimum is not None or self.maximum is not None:
                raise ValueError("Expression parameters require only an expression, without value or bounds.")
        elif self.value is None or self.expression is not None:
            raise ValueError("Free/fixed parameters require a value and no expression.")
        if self.kind == "fixed" and (self.minimum is not None or self.maximum is not None):
            raise ValueError("Fixed parameters cannot have optimizer bounds.")
        if self.minimum is not None and self.maximum is not None and self.minimum >= self.maximum:
            raise ValueError("Parameter minimum must be below maximum.")
        if self.value is not None and ((self.minimum is not None and self.value < self.minimum) or (self.maximum is not None and self.value > self.maximum)):
            raise ValueError("Initial parameter value must lie inside its bounds.")
        return self


class ParameterGraph(Contract):
    nodes: list[Parameter] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_nodes(self):
        _unique(self.nodes, "parameter")
        return self


class TransformSpec(Contract):
    fitspace: Literal["r"] = "r"
    kmin: float = Field(default=2, ge=0)
    kmax: float = Field(default=12, gt=0)
    rmin: float = Field(default=1, ge=0)
    rmax: float = Field(default=3, gt=0)
    kweight: int = Field(default=2, ge=0, le=3)
    dk: float = Field(default=1, gt=0)
    window: Literal["hanning", "kaiser", "parzen", "welch", "sine"] = "hanning"
    kstep: float = Field(default=0.05, gt=0)
    nfft: int = Field(default=2048, ge=256, le=65536)

    @model_validator(mode="after")
    def windows(self):
        if self.kmin >= self.kmax or self.rmin >= self.rmax or self.kmax-self.kmin <= self.dk:
            raise ValueError("Transform windows must be ordered and contain the taper.")
        if self.nfft & (self.nfft - 1):
            raise ValueError("FFT size must be a power of two.")
        return self


class PathBinding(Contract):
    id: EntityId
    path_id: EntityId
    enabled: bool = True
    expressions: dict[Literal["s02", "e0", "deltar", "sigma2", "third", "fourth"], Expression] = Field(default_factory=lambda: {"s02":"1", "e0":"0", "deltar":"0", "sigma2":"0"})
    amplitude_semantics: Literal["scale", "single_path_cn", "group_cn"] = "scale"
    cn_group_id: EntityId | None = None
    geometry_basis: str = ""

    @model_validator(mode="after")
    def amplitude_group(self):
        if (self.amplitude_semantics == "group_cn") != (self.cn_group_id is not None):
            raise ValueError("group_cn amplitude requires a coordination-group ID; other amplitude modes do not.")
        return self


class DatasetBinding(Contract):
    dataset_id: EntityId
    processing_revision_id: EntityId
    paths: list[PathBinding] = Field(min_length=1)
    transform: TransformSpec = Field(default_factory=TransformSpec)
    noise: NoiseSpec = Field(default_factory=NoiseSpec)

    @model_validator(mode="after")
    def unique_paths(self):
        _unique(self.paths, "path binding")
        return self


class CoordinationGroup(Contract):
    id: EntityId
    path_binding_ids: list[EntityId] = Field(min_length=1)
    coordination_parameter_id: ParameterId
    reference_degeneracy: float = Field(gt=0)
    counting_basis: str = Field(min_length=1)


class ModelRevision(Contract):
    id: EntityId
    revision: int = Field(default=0, ge=0)
    title: str = "Untitled model"
    bindings: list[DatasetBinding] = Field(min_length=1)
    parameters: ParameterGraph = Field(default_factory=ParameterGraph)
    coordination_groups: list[CoordinationGroup] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def model_references(self):
        dataset_ids=[x.dataset_id for x in self.bindings]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("A model can bind each dataset only once.")
        paths=[p for d in self.bindings for p in d.paths]
        _unique(paths,"model path binding")
        path_ids={p.id for p in paths}
        parameter_ids={p.id for p in self.parameters.nodes}
        _unique(self.coordination_groups,"coordination group")
        groups={g.id for g in self.coordination_groups}
        counted_paths=set()
        for group in self.coordination_groups:
            if len(group.path_binding_ids)!=len(set(group.path_binding_ids)) or not set(group.path_binding_ids)<=path_ids:
                raise ValueError(f"Unknown or repeated path in coordination group {group.id}.")
            if group.coordination_parameter_id not in parameter_ids:
                raise ValueError(f"Unknown parameter in coordination group {group.id}.")
            if counted_paths.intersection(group.path_binding_ids):
                raise ValueError("A path binding cannot be counted in multiple coordination groups.")
            counted_paths.update(group.path_binding_ids)
        for path in paths:
            if path.cn_group_id is not None and path.cn_group_id not in groups:
                raise ValueError(f"Unknown coordination group for path {path.id}.")
        for p in self.parameters.nodes:
            if p.scope == "dataset" and p.owner_id not in dataset_ids:
                raise ValueError(f"Unknown dataset owner for parameter {p.id}.")
            if p.scope == "path" and p.owner_id not in path_ids:
                raise ValueError(f"Unknown path owner for parameter {p.id}.")
        fft={(b.transform.kstep,b.transform.nfft) for b in self.bindings}
        if len(fft)>1:
            raise ValueError("A joint run requires one kstep/nfft convention.")
        return self


class FitSpec(Contract):
    model: ModelRevision
    datasets: list[Dataset] = Field(default_factory=list)
    processing: list[ProcessingRevision] = Field(default_factory=list)
    paths: list[PathAsset] = Field(default_factory=list)
    calibrations: list[Calibration] = Field(default_factory=list)
    engine_versions: dict[str, str] = Field(default_factory=dict)
    source_sha256: Sha256 | None = None
    max_evaluations: int = Field(default=1000, gt=0, le=1_000_000)
    timeout_seconds: float = Field(default=300, gt=0, le=86400)
    random_seed: int = Field(default=0, ge=0)
    initial_sets: list[dict[ParameterId, float]] = Field(default_factory=list)


class ParameterEstimate(Contract):
    id: ParameterId
    value: float
    unit: Unit
    stderr: float | None = Field(default=None, ge=0)
    uncertainty_reason: str | None = "not_estimated"
    at_bound: bool = False

    @model_validator(mode="after")
    def missing_error_reason(self):
        if self.stderr is None and not self.uncertainty_reason:
            raise ValueError("Unavailable stderr requires a reason.")
        return self


class CovarianceRecord(Contract):
    var_names: list[ParameterId]
    matrix: list[list[float]] | None = None
    scale: Literal["raw", "scaled_stderr"] = "scaled_stderr"
    scaling_description: str = ""
    missing_reason: str | None = None

    @model_validator(mode="after")
    def square_matrix(self):
        n=len(self.var_names)
        if len(set(self.var_names)) != n:
            raise ValueError("Covariance variable names must be unique.")
        if self.matrix is None and not self.missing_reason:
            raise ValueError("Unavailable covariance requires a reason.")
        if self.matrix is not None and (len(self.matrix)!=n or any(len(row)!=n for row in self.matrix)):
            raise ValueError("Covariance dimensions must match var_names.")
        return self


class FitStatistics(Contract):
    residual_count: int = Field(ge=0)
    n_independent: float = Field(ge=0)
    n_variables: int = Field(ge=0)
    degrees_of_freedom: float
    nind_convention: Literal["larch_2026_3_1_plus1"] = "larch_2026_3_1_plus1"
    raw_chisquare: float | None = Field(default=None, ge=0)
    chisquare: float | None = Field(default=None, ge=0)
    reduced_chisquare: float | None = Field(default=None, ge=0)
    rfactor: float | None = Field(default=None, ge=0)
    aic: float | None = None
    bic: float | None = None
    definition: str = "larch 2026.3.1 feffit"


class DatasetFitResult(Contract):
    dataset_id: EntityId
    processing_revision_id: EntityId
    arrays: dict[str, ArrayRef] = Field(default_factory=dict)
    path_contributions: dict[EntityId, dict[str, ArrayRef]] = Field(default_factory=dict)
    statistics: FitStatistics | None = None


class FitResult(Contract):
    run_id: EntityId
    assessment: Assessment = "conditional"
    converged: bool
    termination_reason: str
    statistics: FitStatistics
    parameters: list[ParameterEstimate] = Field(default_factory=list)
    covariance: CovarianceRecord | None = None
    datasets: list[DatasetFitResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    uncertainty_basis: Literal["not_estimated", "conditional_fixed", "joint_reference"] = "not_estimated"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class FitRun(Contract):
    id: EntityId
    model_id: EntityId
    input_snapshot: FitSpec
    snapshot_sha256: Sha256
    task_state: JobState = "queued"
    result: FitResult | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def run_result_identity(self):
        if self.input_snapshot.model.id != self.model_id:
            raise ValueError("FitRun model_id must match its input snapshot.")
        if self.result is not None and self.result.run_id != self.id:
            raise ValueError("Fit result must belong to this run.")
        if self.task_state == "succeeded" and self.result is None:
            raise ValueError("A succeeded run requires a result.")
        return self


def _unique(items, label):
    ids=[x.id for x in items]
    if len(ids)!=len(set(ids)):
        raise ValueError(f"Duplicate {label} ID.")


class Project(Contract):
    format: Literal["webxas-project"] = "webxas-project"
    schema_version: Literal[1] = 1
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(default="Untitled", min_length=1, max_length=250)
    revision: int = Field(default=0, ge=0)
    datasets: list[Dataset] = Field(default_factory=list)
    processing: list[ProcessingRevision] = Field(default_factory=list)
    calibrations: list[Calibration] = Field(default_factory=list)
    structures: list[Structure] = Field(default_factory=list)
    paths: list[PathAsset] = Field(default_factory=list)
    models: list[ModelRevision] = Field(default_factory=list)
    runs: list[FitRun] = Field(default_factory=list)
    migration_report: list[MigrationIssue] = Field(default_factory=list)
    recomputable: bool = True
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def uuid_identity(cls,value):
        if str(UUID(value)) != value:
            raise ValueError("Project ID must be a canonical UUID string.")
        return value

    @model_validator(mode="after")
    def project_references(self):
        for name in ("datasets","processing","calibrations","structures","paths","models","runs"):
            _unique(getattr(self,name),name)
        datasets={d.id for d in self.datasets}
        processing={p.id:p for p in self.processing}
        calibrations={c.id for c in self.calibrations}
        structures={s.id for s in self.structures}
        paths={p.id:p for p in self.paths}
        models={m.id for m in self.models}
        for d in self.datasets:
            if d.reference_dataset_id is not None and d.reference_dataset_id not in datasets:
                raise ValueError(f"Unknown reference dataset for {d.id}.")
        for p in self.processing:
            if p.dataset_id not in datasets or (p.calibration_id is not None and p.calibration_id not in calibrations):
                raise ValueError(f"Unknown processing input for {p.id}.")
        for c in self.calibrations:
            if not set(c.reference_dataset_ids)<=datasets or (c.reference_model_id is not None and c.reference_model_id not in models):
                raise ValueError(f"Unknown calibration reference for {c.id}.")
        for p in self.paths:
            if p.structure_id is not None and p.structure_id not in structures:
                raise ValueError(f"Unknown structure for path {p.id}.")
        for m in self.models:
            for b in m.bindings:
                if b.dataset_id not in datasets or b.processing_revision_id not in processing:
                    raise ValueError(f"Unknown dataset/processing in model {m.id}.")
                if processing[b.processing_revision_id].dataset_id != b.dataset_id:
                    raise ValueError(f"Processing belongs to another dataset in model {m.id}.")
                if any(p.path_id not in paths for p in b.paths):
                    raise ValueError(f"Unknown path in model {m.id}.")
                if any(p.amplitude_semantics != "scale" and paths[p.path_id].nleg != 2 for p in b.paths):
                    raise ValueError("Multiple-scattering paths cannot use a single-scattering CN amplitude mode.")
        for r in self.runs:
            if r.model_id not in models:
                raise ValueError(f"Unknown model for run {r.id}.")
        return self
