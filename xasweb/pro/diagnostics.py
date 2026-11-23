"""Extract Larch statistics with explicit covariance and calibration meaning."""
from __future__ import annotations

import math
from uuid import uuid4
import numpy as np

from .contracts import Calibration,CovarianceRecord,ParameterEstimate


def finite(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (TypeError,ValueError,OverflowError):
        return None


def parameter_diagnostics(result,compiled):
    """Return estimates, matching scaled covariance, and review observations."""
    estimates=[];warnings=[];details={"covariance_scale":"scaled_stderr","correlation":None}
    inverse={v:k for k,v in compiled.name_map.items()}
    actual_names=list(getattr(result.fit_details,"var_names",[]) or [])
    if set(actual_names)!={compiled.name_map[x] for x in compiled.free_ids}:
        raise ValueError("The optimizer changed the declared free parameter set.")
    for pid,node in compiled.nodes.items():
        parameter=result.params[compiled.name_map[pid]]
        error=finite(getattr(parameter,"stderr",None))
        if node.kind=="fixed":error=None;reason="fixed_parameter"
        elif error is None:reason="optimizer_did_not_estimate_uncertainty"
        else:reason=None
        value=finite(parameter.value)
        if value is None:raise ValueError(f"Non-finite optimizer parameter {pid}.")
        limits=[x for x in (node.minimum,node.maximum) if x is not None]
        scale=(node.maximum-node.minimum) if node.minimum is not None and node.maximum is not None else max(1.,abs(value))
        at_bound=any(abs(value-limit)<=max(1e-10,abs(scale)*1e-6) for limit in limits)
        if at_bound:warnings.append(f"Parameter {pid} is near a declared boundary.")
        estimates.append(ParameterEstimate(id=pid,value=value,unit=node.unit,stderr=error,uncertainty_reason=reason,at_bound=at_bound))
    raw=getattr(result.fit_details,"covar",None)
    scaled=getattr(result.fit_details,"scaled_covar",None)
    names=[inverse[x] for x in actual_names]
    reason=None
    if scaled is None:reason="optimizer_covariance_unavailable"
    else:
        scaled=np.asarray(scaled,float)
        if scaled.shape!=(len(names),len(names)) or not np.isfinite(scaled).all():
            reason="nonfinite_or_invalid_covariance"
    if reason:
        warnings.append("Parameter covariance is unavailable; missing errors remain null.")
        cov=CovarianceRecord(var_names=names,missing_reason=reason,scaling_description="Larch fit_details.scaled_covar unavailable")
    else:
        by_id={p.id:p for p in estimates}
        errors=np.array([by_id[name].stderr if by_id[name].stderr is not None else np.nan for name in names])
        consistent=np.isfinite(errors).all() and np.allclose(np.diag(scaled),errors**2,rtol=1e-6,atol=np.finfo(float).tiny)
        if not consistent:raise ValueError("Scaled covariance does not match the final parameter stderr values.")
        cov=CovarianceRecord(var_names=names,matrix=scaled.tolist(),scaling_description="Larch 2026.3.1 fit_details.scaled_covar; its diagonal is the final free-parameter stderr squared")
        variance=np.sqrt(np.maximum(np.diag(scaled),0))
        denominator=np.outer(variance,variance)
        correlation=np.divide(scaled,denominator,out=np.zeros_like(scaled),where=denominator>0)
        # Zero variance has undefined correlation, not an invented zero.
        details["correlation"]=[[float(correlation[i,j]) if denominator[i,j]>0 else None for j in range(len(names))] for i in range(len(names))]
        details["correlation_var_names"]=names
        details["high_correlation_threshold"]=.95
        for i,a in enumerate(names):
            for j in range(i):
                if denominator[i,j]>0 and abs(correlation[i,j])>.95:
                    warnings.append(f"Parameters {a} and {names[j]} have |correlation| > 0.95 (review threshold).")
        if raw is not None:
            raw=np.asarray(raw,float)
            if raw.shape==scaled.shape and np.isfinite(raw).all():details["raw_covariance"]=raw.tolist()
    return estimates,cov,warnings,details


def make_s02_calibration(run,parameter_id,reference_dataset_ids,element,edge,applicability_note):
    """Record a reference fit; selecting the known-N standard remains explicit."""
    if run.result is None or not run.result.converged:
        raise ValueError("S02 calibration requires a converged reference fit.")
    if not applicability_note.strip():raise ValueError("S02 calibration requires an applicability statement.")
    reference_ids=set(reference_dataset_ids)
    if not reference_ids or not reference_ids<={b.dataset_id for b in run.input_snapshot.model.bindings}:
        raise ValueError("Calibration reference datasets must be in the fitted model.")
    groups=[g for g in run.input_snapshot.model.coordination_groups if any(p.id in g.path_binding_ids for b in run.input_snapshot.model.bindings if b.dataset_id in reference_ids for p in b.paths)]
    nodes={p.id:p for p in run.input_snapshot.model.parameters.nodes}
    if not groups or any(nodes[g.coordination_parameter_id].kind!="fixed" for g in groups):
        raise ValueError("Reference S02 calibration requires declared fixed known-N counting groups.")
    parameter=next((p for p in run.result.parameters if p.id==parameter_id),None)
    if parameter is None or parameter.unit!="1" or parameter.value<=0:
        raise ValueError("Selected calibration parameter must be a positive dimensionless value.")
    from .parameters import compile_model
    compiled=compile_model(run.input_snapshot.model,run.input_snapshot.paths)
    reference_paths=[p.id for b in run.input_snapshot.model.bindings if b.dataset_id in reference_ids for p in b.paths if p.enabled]
    used=set().union(*(compiled._closure(compiled.path_expressions[bid]["s02"].names) for bid in reference_paths))
    if nodes[parameter_id].kind!="free" or parameter_id not in used:
        raise ValueError("Selected calibration parameter must be an estimated amplitude parameter of the reference model.")
    data={d.id:d for d in run.input_snapshot.datasets}
    if any(data[did].element!=element or data[did].edge!=edge for did in reference_ids):
        raise ValueError("Calibration applicability element/edge does not match the reference data.")
    return Calibration(id="cal_"+uuid4().hex,kind="s02",value=parameter.value,unit="1",stderr=parameter.stderr,
        reference_dataset_ids=list(reference_dataset_ids),element=element,edge=edge,reference_model_id=run.model_id,
        uncertainty_basis="conditional_fixed",applicability_note=applicability_note,
        conditions={"source_run_id":run.id,"parameter_id":parameter_id,"basis":"reference fit conditional on fixed known-N model"})


def joint_reference_basis(spec,compiled):
    """Recognize inspectable standard+sample coupling, never a metadata flag alone.

    This marks covariance propagation inside the stated model. It is not a
    claim that the known-N model, calibration transfer or systematic errors have
    received independent experimental validation.
    """
    nodes=compiled.nodes;bindings={b.dataset_id:b for b in spec.model.bindings}
    datasets={d.id:d for d in spec.datasets}
    groups=spec.model.coordination_groups
    linked=[]
    for calibration in spec.calibrations:
        if calibration.kind!="s02":continue
        refs=set(calibration.reference_dataset_ids)
        targets=set(bindings)-refs
        if not refs or not refs<=bindings.keys() or not targets:continue
        if not calibration.applicability_note.strip():continue
        if any(datasets[did].element!=calibration.element or datasets[did].edge!=calibration.edge for did in refs):continue
        def counted(did):
            bids={p.id for p in bindings[did].paths if p.enabled and p.amplitude_semantics!="scale"}
            return [g for g in groups if bids.intersection(g.path_binding_ids)]
        ref_groups=[g for did in refs for g in counted(did)]
        if not ref_groups or any(not counted(did) for did in refs):continue
        if any(nodes[g.coordination_parameter_id].kind!="fixed" or nodes[g.coordination_parameter_id].value<=0 for g in ref_groups):continue
        free_target_groups=[(did,g) for did in targets for g in counted(did) if nodes[g.coordination_parameter_id].kind=="free"]
        if not free_target_groups:continue
        def amplitude_ids(did):
            return set().union(*(compiled._closure(compiled.path_expressions[p.id]["s02"].names) for p in bindings[did].paths if p.enabled))
        shared=set(compiled.free_ids)
        for did in refs:shared.intersection_update(amplitude_ids(did))
        shared={pid for pid in shared if nodes[pid].scope=="global" and nodes[pid].unit=="1"}
        selected=calibration.conditions.get("parameter_id")
        if not isinstance(selected,str) or selected not in shared:continue
        usable=[did for did,g in free_target_groups if selected in amplitude_ids(did)
                and datasets[did].element==calibration.element and datasets[did].edge==calibration.edge]
        # A calibrated amplitude may not silently serve a different edge.
        affected={did for did in bindings if selected in amplitude_ids(did)}
        if any(datasets[did].element!=calibration.element or datasets[did].edge!=calibration.edge for did in affected):continue
        if usable:linked.append({"calibration_id":calibration.id,"reference_dataset_ids":sorted(refs),"target_dataset_ids":sorted(set(usable)),"shared_parameter_id":selected,"fixed_reference_cn_ids":sorted({g.coordination_parameter_id for g in ref_groups})})
    return ("joint_reference" if linked else "conditional_fixed"),linked
