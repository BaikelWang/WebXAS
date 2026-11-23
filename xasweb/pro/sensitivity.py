"""Three bounded, auditable sensitivity cases; never publish Project changes."""
from __future__ import annotations

import hashlib
import time
from uuid import uuid4

from pydantic import ValidationError

from ..models import Parameters
from .contracts import FitSpec,ModelRevision,NoiseSpec,Project
from .larch_adapter import fit_snapshot,validate_snapshot
from .preprocessing import load_array,process_dataset
from .projects import ProjectError,ProjectStore,canonical

PRESETS=("baseline","window","background")


class _FrozenReadStore:
    """Processing sees the captured project while writing only immutable assets."""
    def __init__(self,store,project):self.store,self.project=store,project.model_copy(deep=True)
    def read(self,project_id):
        if project_id!=self.project.id:raise ProjectError("project_scope","Sensitivity project scope mismatch.")
        return self.project.model_copy(deep=True)
    def __getattr__(self,name):return getattr(self.store,name)


class _CaseContext:
    def __init__(self,parent,case_id,index):self.parent,self.case_id,self.index=parent,case_id,index
    def heartbeat(self,stage,progress=None):
        if self.parent:self.parent.heartbeat(f"{self.case_id}: {stage}",(self.index+(progress or 0))/len(PRESETS))


def _new_model(model,case_id):
    value=model.model_dump(mode="json")
    value.update(id="sens_"+case_id+"_"+uuid4().hex,revision=model.revision+1,
                 title=model.title+" · "+case_id,
                 notes=model.notes+f"\nSensitivity preset {case_id}; derived from {model.id}. Window/background changes are not directly statistically comparable.")
    return value


def run_sensitivity(store,project_id,spec,expected_revision,context=None):
    spec=FitSpec.model_validate(spec.model_dump(mode="json") if hasattr(spec,"model_dump") else spec)
    project=store.read(project_id)
    if project.revision!=expected_revision:
        raise ProjectError("revision_conflict","Project changed after sensitivity submission.",project_id)
    original=next((m for m in project.models if m.id==spec.model.id),None)
    if original is None or original.model_dump(mode="json")!=spec.model.model_dump(mode="json"):
        raise ProjectError("sensitivity_snapshot","Submitted model differs from its saved immutable revision.",spec.model.id)
    validate_snapshot(spec)
    # The project revision check freezes the source lookup used by processing.
    # Validate the captured computational resources against that same revision.
    for field in ("datasets","processing","paths","calibrations"):
        stored={item.id:item.model_dump(mode="json") for item in getattr(project,field)}
        if any(stored.get(item.id)!=item.model_dump(mode="json") for item in getattr(spec,field)):
            raise ProjectError("sensitivity_snapshot","Snapshot resources differ from the selected project revision.",field)
    frozen=_FrozenReadStore(store,project)
    datasets={d.id:d for d in spec.datasets};processing={p.id:p for p in spec.processing}
    output={"project_id":project_id,"expected_revision":expected_revision,"model_id":spec.model.id,
            "runs":[],"models":[],"processing_revisions":[],"cases":[]}
    start=time.monotonic();deadline=start+spec.timeout_seconds
    origin_hash=hashlib.sha256(canonical(spec.model_dump(mode="json"))).hexdigest()
    for index,case_id in enumerate(PRESETS):
        case={"case_id":case_id,"status":"failed"}
        ctx=_CaseContext(context,case_id,index)
        ctx.heartbeat("preparing preset",0)
        if time.monotonic()>=deadline:
            case["reason"]="Total sensitivity budget exhausted before this preset."
            output["cases"].append(case);continue
        model_data=_new_model(spec.model,case_id)
        selected_processing=list(spec.processing)
        try:
            if case_id=="window":
                for binding in model_data["bindings"]:
                    binding["transform"]["kmax"]-=.5
                try:ModelRevision.model_validate(model_data)
                except ValidationError:
                    case.update(status="not_applicable",reason="Reducing kmax by 0.5 would violate the declared transform/taper contract.")
                    output["cases"].append(case);continue
                case["reason"]="Each spectrum uses kmax minus 0.5 inverse angstrom; changed window is not a direct statistical model-comparison basis."
            elif case_id=="background":
                applicable=[b for b in spec.model.bindings if datasets[b.dataset_id].kind=="mu"]
                details=[{"dataset_id":b.dataset_id,"status":"not_applicable","reason":"chi(k) has no recoverable mu(E) background to reprocess."} for b in spec.model.bindings if datasets[b.dataset_id].kind!="mu"]
                case["details"]=details
                if not applicable:
                    case.update(status="not_applicable",reason="All selected spectra are chi(k); the background preset requires original mu(E).")
                    output["cases"].append(case);continue
                requested={}
                for b in applicable:
                    old=processing[b.processing_revision_id]
                    parameters=old.parameters.model_dump(mode="json");parameters["rbkg"]+=.1
                    try:requested[b.dataset_id]=Parameters.model_validate(parameters)
                    except ValidationError:
                        case.update(status="not_applicable",reason=f"rbkg plus 0.1 exceeds the processing contract for {b.dataset_id}.")
                        break
                if case["status"]=="not_applicable":output["cases"].append(case);continue
                replacement={}
                for b in applicable:
                    ctx.heartbeat("reprocessing original mu(E)",.05)
                    old=processing[b.processing_revision_id]
                    original_mask=load_array(store,project_id,old.mask) if old.mask is not None else None
                    noise=b.noise if b.noise.source!="unknown" else old.noise
                    noise_data=noise.model_dump(mode="json");noise_data["processing_revision_id"]=None
                    new=process_dataset(frozen,project_id,b.dataset_id,requested[b.dataset_id],original_mask,old.calibration_id,NoiseSpec.model_validate(noise_data))
                    new.metadata["sensitivity"]={"preset":"background","source_processing_id":old.id,
                        "delta_rbkg_angstrom":.1,"noise_basis":"Explicit chi-noise declarations held fixed for a controlled sensitivity check; engine-estimated noise is recalculated. This is not uncertainty propagation."}
                    replacement[old.id]=new
                    output["processing_revisions"].append(new.model_dump(mode="json"))
                    details.append({"dataset_id":b.dataset_id,"status":"reprocessed","processing_revision_id":new.id})
                selected_processing=[replacement.get(p.id,p) for p in spec.processing]
                for b in model_data["bindings"]:
                    if b["processing_revision_id"] in replacement:
                        new=replacement[b["processing_revision_id"]]
                        b["processing_revision_id"]=new.id;b["noise"]=new.noise.model_dump(mode="json")
                case["reason"]="Original mu(E) reprocessed with rbkg plus 0.1 angstrom; chi inputs retained. Changed processing is not directly statistically comparable."
            else:
                case["reason"]="Original processing and fit windows retained as the baseline control."
            model=ModelRevision.model_validate(model_data)
            output["models"].append(model.model_dump(mode="json"))
            variant=spec.model_dump(mode="json")
            variant.update(model=model.model_dump(mode="json"),processing=[p.model_dump(mode="json") for p in selected_processing],
                           timeout_seconds=max(.001,deadline-time.monotonic()))
            run=fit_snapshot(store,project_id,FitSpec.model_validate(variant),context=ctx)
            run.result.metadata["sensitivity"]={"case_id":case_id,"source_model_id":spec.model.id,
                "source_fit_spec_sha256":origin_hash,"direct_statistical_comparison":case_id=="baseline"}
            output["runs"].append(run.model_dump(mode="json"))
            case.update(status="succeeded" if run.result.converged else "nonconverged",run_id=run.id,model_id=model.id)
        except Exception as exc:
            case.update(status="failed",reason=str(exc),code=getattr(exc,"code","sensitivity_case_failed"),model_id=model_data["id"])
        output["cases"].append(case)
    if context:context.heartbeat("sensitivity presets completed",1)
    return output


def run_job(payload,context):
    return run_sensitivity(ProjectStore(),payload["project_id"],payload["fit_spec"],payload["expected_revision"],context)
