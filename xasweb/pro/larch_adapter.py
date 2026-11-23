"""One frozen Larch FEFFIT adapter for single and multiple spectra."""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
from pathlib import Path
import platform
import time
from uuid import uuid4

import numpy as np

from .contracts import DatasetFitResult,FitResult,FitRun,FitSpec,FitStatistics
from .diagnostics import finite,parameter_diagnostics,joint_reference_basis
from .parameters import compile_model
from .preprocessing import load_array,save_array,validate_noise
from .projects import ProjectError,ProjectStore,canonical,safe_path


class FitError(ProjectError):
    pass


def current_source_sha256():
    source=Path(__file__).resolve().parents[1]
    return hashlib.sha256(canonical({p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*.py")})).hexdigest()


def current_engine_versions():
    return {"xraylarch":importlib.metadata.version("xraylarch"),"python":platform.python_version(),
            "numpy":np.__version__,"scipy":importlib.metadata.version("scipy"),"lmfit":importlib.metadata.version("lmfit")}


def build_fit_spec(project,model_id,**options):
    model=next((m for m in project.models if m.id==model_id),None)
    if model is None:raise FitError("model_not_found","Requested model does not exist.",model_id)
    selected={b.dataset_id for b in model.bindings}
    processed={b.processing_revision_id for b in model.bindings}
    path_ids={p.path_id for b in model.bindings for p in b.paths}
    calibration_ids={p.calibration_id for p in project.processing if p.id in processed and p.calibration_id}
    calibrations=[c for c in project.calibrations if c.id in calibration_ids or c.kind=="s02"]
    selected.update(did for c in calibrations for did in c.reference_dataset_ids)
    # Retain provenance sources so lineage remains inspectable after recovery.
    while True:
        before=set(selected)
        for d in project.datasets:
            if d.id in selected:selected.update(d.lineage)
        if before==selected:break
    spec=FitSpec(model=model,datasets=[d for d in project.datasets if d.id in selected],
        processing=[p for p in project.processing if p.id in processed],paths=[p for p in project.paths if p.id in path_ids],
        calibrations=calibrations,engine_versions=current_engine_versions(),source_sha256=current_source_sha256(),**options)
    validate_snapshot(spec)
    return spec


def _processing_identity(dataset,processing):
    expected={"input_mapping":dataset.mapping.model_dump(mode="json"),"input_kind":dataset.kind,
              "input_kweight":dataset.metadata.get("input_kweight",0),"source_sha256":dataset.raw_file.sha256}
    for field,value in expected.items():
        if field in processing.metadata and processing.metadata[field]!=value:
            raise FitError("stale_processing",f"Dataset {field} changed; reprocess the raw spectrum before fitting.",dataset.id)
    missing=set(expected)-processing.metadata.keys()
    synthetic=(dataset.metadata.get("data_kind")=="same_engine_synthetic" and dataset.metadata.get("not_experimental_data") is True)
    if missing and not synthetic:
        raise FitError("missing_processing_provenance","Processing identity is incomplete; reprocess real data before fitting.",processing.id)


def _lineage(dataset,datasets,visiting=None):
    visiting=set(visiting or ())
    if dataset.id in visiting:raise FitError("lineage_cycle","Dataset lineage contains a cycle.",dataset.id)
    visiting.add(dataset.id)
    hashes={dataset.raw_file.sha256}
    for parent in dataset.lineage:
        if parent not in datasets:raise FitError("missing_lineage","A provenance dataset is missing from the fit snapshot.",parent)
        hashes.update(_lineage(datasets[parent],datasets,visiting))
    return hashes


def validate_snapshot(spec):
    actual_versions=current_engine_versions()
    if spec.engine_versions.get("xraylarch","2026.3.1")!="2026.3.1" or actual_versions["xraylarch"]!="2026.3.1":
        raise FitError("engine_version","Professional adapter is frozen to xraylarch 2026.3.1.")
    if any(spec.engine_versions.get(name,value)!=value for name,value in actual_versions.items()):
        raise FitError("engine_version","Scientific runtime differs from the immutable submitted snapshot; submit a new run.")
    if spec.source_sha256 is not None and spec.source_sha256!=current_source_sha256():
        raise FitError("engine_source_changed","Application source changed after snapshot creation; rebuild and submit the run again.",spec.model.id)
    datasets={d.id:d for d in spec.datasets};processing={p.id:p for p in spec.processing};paths={p.id:p for p in spec.paths}
    if len(datasets)!=len(spec.datasets) or len(processing)!=len(spec.processing) or len(paths)!=len(spec.paths):
        raise FitError("duplicate_snapshot_id","Snapshot resource IDs must be unique.")
    observed=[]
    for binding in spec.model.bindings:
        if binding.dataset_id not in datasets or binding.processing_revision_id not in processing:
            raise FitError("missing_snapshot_resource","Fitting dataset or processing revision is missing.",binding.dataset_id)
        d=datasets[binding.dataset_id];p=processing[binding.processing_revision_id]
        if p.dataset_id!=d.id:raise FitError("processing_dataset","Processing revision belongs to a different dataset.",p.id)
        _processing_identity(d,p)
        if d.kind=="support":raise FitError("not_spectrum","Support materials cannot be fitted as spectra.",d.id)
        if len(spec.model.bindings)>1 and d.metadata.get('import_role')=='merged' and not d.lineage:
            raise FitError('unknown_merged_lineage','Imported merged data need explicit provenance before quantitative joint fitting with other observations.',d.id)
        lineage=_lineage(d,datasets)
        for other_id,other in observed:
            if lineage&other:raise FitError("duplicate_observation",f"Datasets {d.id} and {other_id} share original observations; reject or explicitly merge before FEFFIT.",d.id)
        observed.append((d.id,lineage))
        for pb in binding.paths:
            if pb.path_id not in paths:raise FitError("missing_snapshot_path","A model path is missing from the immutable snapshot.",pb.path_id)
            asset=paths[pb.path_id]
            if d.element!=asset.absorber_element or d.edge!=asset.edge:
                raise FitError("path_edge","Path absorber and edge do not match the dataset.",pb.id)
    compiled=compile_model(spec.model,spec.paths)
    if not compiled.free_ids:raise FitError("no_free_parameters","Use model preview for an entirely fixed model.",spec.model.id)
    nind=sum(1+2*(b.transform.kmax-b.transform.kmin)*(b.transform.rmax-b.transform.rmin)/np.pi for b in spec.model.bindings)
    if nind<=len(compiled.free_ids):raise FitError("insufficient_information","Non-positive information-based degrees of freedom: model preview remains available.",spec.model.id)
    return compiled


def validate_fit_input(store, project_id, dataset, processing, binding, noise_override=None):
    """Use the same measured-support and noise gate in preflight and the worker."""
    _processing_identity(dataset, processing)
    if not {'k', 'chi'} <= processing.arrays.keys():
        raise FitError('missing_chi', '处理版本没有 k/χ 数组。', processing.id)
    k = load_array(store, project_id, processing.arrays['k'])
    chi = load_array(store, project_id, processing.arrays['chi'])
    if k.ndim != 1 or chi.shape != k.shape or len(k) < 8 or k[0] < 0 or np.any(np.diff(k) <= 0):
        raise FitError('invalid_fit_grid', '结构拟合需要递增的非负 k 网格和对应 χ(k)。', dataset.id)
    t = binding.transform
    # Require the full declared taper to lie on the measured, processed domain.
    lo, hi = t.kmin - t.dk / 2, t.kmax + t.dk / 2
    if lo < k[0] - 1e-10 or hi > k[-1] + 1e-10:
        raise FitError('fit_coverage', f'拟合含渐变窗需覆盖 {lo:.3f}–{hi:.3f} Å⁻¹，处理版本仅支持 '
                       f'{k[0]:.3f}–{k[-1]:.3f} Å⁻¹。请显式缩小窗口或重选处理版本。', dataset.id)
    if k[-1] > t.kstep * (t.nfft // 2 - 1):
        raise FitError('fft_coverage', 'FFT 网格不足以覆盖已处理 k 范围。', dataset.id)
    noise = noise_override if noise_override is not None else (
        binding.noise if binding.noise.source != 'unknown' else processing.noise)
    noise = validate_noise(store, project_id, noise, processing.id, len(k))
    if noise.source == 'unknown':
        raise FitError('noise_unspecified', '请为定量拟合声明实验噪声或选择带假设的估计。', dataset.id)
    if noise.source == 'engine_estimate':
        lo_r, hi_r = noise.estimation_range or (15., 30.)
        if lo_r < 0 or hi_r > np.pi / (2 * t.kstep):
            raise FitError('noise_estimation_range', '噪声估计区间必须在 FFT R 域内。', dataset.id)
    return k, chi, noise


def _prepare(store,project_id,spec,compiled,overrides=None):
    from larch import Group
    from larch.fitting import group2params
    from larch.xafs import feffpath,feffit_dataset,feffit_transform
    parameters,path_expressions=compiled.to_larch()
    initial=compiled.values(overrides)
    for pid in compiled.free_ids:getattr(parameters,compiled.name_map[pid]).value=initial[pid]
    datasets={d.id:d for d in spec.datasets};processing={p.id:p for p in spec.processing};assets={p.id:p for p in spec.paths}
    output=[];resolved_noise=[]
    for binding in spec.model.bindings:
        d=datasets[binding.dataset_id];p=processing[binding.processing_revision_id]
        store.read_asset(project_id,d.raw_file)
        k,chi,noise=validate_fit_input(store,project_id,d,p,binding)
        t=binding.transform
        epsilon=None
        if noise.source=="explicit_scalar":epsilon=noise.value
        elif noise.source=="explicit_array":
            epsilon=load_array(store,project_id,noise.array)
            if np.any(epsilon<1e-12):raise FitError("noise_engine_floor","Array noise is below Larch's 1e-12 floor; choose a documented model rather than silent clipping.",d.id)
        fps={}
        for pb in binding.paths:
            if not pb.enabled:continue
            asset=assets[pb.path_id]
            store.read_asset(project_id,asset.source_file)
            pathfile=store.folder(project_id).joinpath(*safe_path(asset.source_file.path).parts)
            fp=feffpath(str(pathfile),label=pb.id,**path_expressions[pb.id])
            if int(fp.nleg)!=asset.nleg or not math.isclose(float(fp.degen),asset.degen,rel_tol=1e-10,abs_tol=1e-10) or not math.isclose(float(fp.reff),asset.reff,rel_tol=1e-10,abs_tol=1e-10):
                raise FitError("path_metadata","FEFF path metadata does not match its saved asset record.",asset.id)
            if t.kmax+t.dk/2>float(np.max(fp._feffdat.k))+1e-10:
                raise FitError("path_k_coverage","FEFF path does not cover the fit k range.",asset.id)
            fps[pb.id]=fp
        data = Group(k=k, chi=chi, groupname=d.id)
        includes_background = noise.source == 'engine_estimate' and 'delta_chi' in p.arrays
        if includes_background:
            data.delta_chi = load_array(store, project_id, p.arrays['delta_chi'])
            if data.delta_chi.shape != chi.shape or np.any(data.delta_chi < 0):
                raise FitError('background_uncertainty', 'AUTOBK 条件不确定度与 χ(k) 不匹配。', p.id)
        ds=feffit_dataset(data=data,paths=fps,transform=feffit_transform(**t.model_dump(mode="json")),epsilon_k=epsilon)
        if noise.source=="engine_estimate":
            limits=noise.estimation_range or [15.,30.]
            native_estimate=ds.estimate_noise
            def declared_estimate(chi=None,rmin=15.,rmax=30.,all_kweights=True,
                                  _native=native_estimate,_limits=tuple(limits),_dataset_id=d.id,_dataset=ds):
                # Preserve both native epsilon_k and epsilon_r. Feeding the
                # estimated epsilon_k back through set_epsilon_k changes the
                # pinned library's R noise by sqrt(2), hence chi-square by 2.
                output=_native(chi=chi,rmin=_limits[0],rmax=_limits[1],all_kweights=all_kweights)
                for value in (_dataset.epsilon_k,_dataset.epsilon_r):
                    estimated=np.asarray(value)
                    if not np.isfinite(estimated).all() or np.any(estimated<=0):
                        raise FitError("noise_estimate","Noise estimation did not produce a positive finite value.",_dataset_id)
                return output
            # prepare_fit invokes this once. Explicit sigma follows the native
            # set_epsilon_k route; estimated noise follows native Parseval scaling.
            ds.estimate_noise=declared_estimate
        resolved_noise.append({"dataset_id":d.id,"declaration":noise.model_dump(mode="json"),
                               'autobk_conditional_spline_included': includes_background})
        output.append(ds)
    return parameters,output,resolved_noise


def fit_snapshot(store,project_id,spec,*,run_id=None,context=None):
    """Fit immutable input; write only content-addressed result assets, not Project."""
    from larch import Group
    from larch.xafs import feffit
    spec=FitSpec.model_validate(spec.model_dump(mode="json") if hasattr(spec,"model_dump") else spec)
    compiled=validate_snapshot(spec)
    from .jobs import utc_now
    run_id=run_id or "run_"+uuid4().hex
    started_at=utc_now()
    digest=hashlib.sha256(canonical(spec.model_dump(mode="json"))).hexdigest()
    attempts=[];candidates=[];started=time.monotonic();deadline=started+spec.timeout_seconds
    for index,initial in enumerate([{},*spec.initial_sets]):
        if time.monotonic()>=deadline:break
        if context:context.heartbeat(f"fitting attempt {index+1}",.1)
        attempt_start=time.monotonic()
        try:
            parameters,datasets,resolved_noise=_prepare(store,project_id,spec,compiled,initial)
            last_heartbeat=[0.]
            def callback(params,iteration,residual,**kws):
                now=time.monotonic()
                if context and now-last_heartbeat[0]>.5:
                    context.heartbeat(f"optimizing attempt {index+1}; evaluation {iteration}",.3)
                    last_heartbeat[0]=now
                return now>=deadline
            result=feffit(parameters,datasets,max_nfev=spec.max_evaluations,path_outputs=True,fix_unused_variables=False,iter_cb=callback)
            if finite(result.chi_square) is None:raise FitError("nonfinite_fit","Optimizer returned a non-finite objective.",spec.model.id)
            final={pid:float(result.params[name].value) for pid,name in compiled.name_map.items()}
            compiled.path_values(overrides={pid:final[pid] for pid in compiled.free_ids})
            success=bool(result.success) and time.monotonic()<deadline
            record={"index":index,"initial_overrides":initial,"success":success,"parameters":final,"rfactor":finite(result.rfactor),"chi_square":finite(result.chi_square),"message":str(result.message),"duration_seconds":time.monotonic()-attempt_start}
            attempts.append(record);candidates.append((not success,float(result.chi_square),result,datasets,resolved_noise))
        except Exception as exc:
            attempts.append({"index":index,"initial_overrides":initial,"success":False,"code":getattr(exc,"code","fit_exception"),"message":str(exc),"duration_seconds":time.monotonic()-attempt_start})
            if index==0 and not spec.initial_sets:raise
    if not candidates:raise FitError("fit_failed","No fitting attempt returned an evaluable result.",spec.model.id)
    _,_,result,datasets,resolved_noise=min(candidates,key=lambda x:x[:2])
    estimates,cov,warnings,diagnostic_metadata=parameter_diagnostics(result,compiled)
    actual_free={name for name,p in result.params.items() if p.vary}
    if actual_free!={compiled.name_map[x] for x in compiled.free_ids}:raise FitError("free_set_changed","FEFFIT changed the declared free parameter set.",spec.model.id)
    residuals=[np.asarray(ds._residual(result.params),float).copy() for ds in datasets]
    total_count=sum(len(x) for x in residuals)
    nind=float(result.n_independent);nvarys=int(result.nvarys)
    scale=nind/total_count
    raw_total=sum(float(np.dot(x,x)) for x in residuals)
    if not np.isclose(raw_total*scale,result.chi_square,rtol=1e-6,atol=1e-25):
        raise FitError("statistic_mismatch","Residual blocks do not reproduce Larch's global chi-square.")
    statistics=FitStatistics(residual_count=total_count,n_independent=nind,n_variables=nvarys,degrees_of_freedom=nind-nvarys,
        raw_chisquare=raw_total,chisquare=float(result.chi_square),reduced_chisquare=float(result.chi2_reduced),rfactor=float(result.rfactor),aic=finite(result.aic),bic=finite(result.bic))
    output=[];layout=[];offset=0
    for binding,ds,residual,noise_record in zip(spec.model.bindings,datasets,residuals,resolved_noise):
        noise_record['epsilon_r']=finite(ds.epsilon_r)
        noise_record['conversion']='native prepare_fit estimate_noise' if noise_record['declaration']['source']=='engine_estimate' else 'native set_epsilon_k for explicit noise'
        data_fft=Group();ds.transform._xafsft(ds._chi,group=data_fft,rmax_out=10,with_chiq=False)
        arrays={"k":ds.model.k,"data_chi":ds._chi,"model_chi":ds.model.chi,"r":ds.model.r,
                "data_re":data_fft.chir_re,"data_im":data_fft.chir_im,"data_mag":data_fft.chir_mag,
                "model_re":ds.model.chir_re,"model_im":ds.model.chir_im,"model_mag":ds.model.chir_mag,
                "residual_chi":ds._chi-ds.model.chi,"weighted_residual":residual}
        eps=np.asarray(ds.epsilon_k)
        arrays["epsilon_k"]=np.full(len(ds.model.k),float(eps)) if eps.ndim==0 else eps
        contributions={bid:{"k":save_array(store,project_id,path.k),"chi":save_array(store,project_id,path.chi)} for bid,path in ds.paths.items()}
        summed=sum((np.asarray(path.chi) for path in ds.paths.values()),np.zeros_like(ds.model.chi))
        if not np.allclose(summed,ds.model.chi,rtol=1e-10,atol=1e-12):raise FitError("path_sum","Path contributions do not sum to the total chi(k).",binding.dataset_id)
        raw=float(np.dot(residual,residual));data_only=np.asarray(ds._residual(result.params,data_only=True),float)
        data_norm=float(np.dot(data_only,data_only))
        ids=set()
        for pb in binding.paths:
            for expression in compiled.path_expressions.get(pb.id,{}).values():ids.update(compiled._closure(expression.names))
        affected=len(ids&set(compiled.free_ids))
        per=FitStatistics(residual_count=len(residual),n_independent=float(ds.n_idp),n_variables=affected,degrees_of_freedom=float(ds.n_idp)-affected,
            raw_chisquare=raw,chisquare=raw*scale,rfactor=raw/data_norm if data_norm>0 else None,
            definition="Per-spectrum global chi-square contribution uses global Nind/residual_count scale. Shared affecting variables are not additive; no per-spectrum joint reduced chi-square is defined.")
        output.append(DatasetFitResult(dataset_id=binding.dataset_id,processing_revision_id=binding.processing_revision_id,
            arrays={key:save_array(store,project_id,value) for key,value in arrays.items()},path_contributions=contributions,statistics=per))
        layout.append({"dataset_id":binding.dataset_id,"slice":[offset,offset+len(residual)],"raw_sum_squares":raw,"global_scaled_chisquare_contribution":raw*scale})
        offset+=len(residual)
    success=bool(result.success) and time.monotonic()<deadline
    if not success:warnings.append("Optimizer did not converge within the requested budget; retain this conditional attempt for review.")
    if statistics.reduced_chisquare and statistics.reduced_chisquare>1:
        warnings.append("Reduced chi-square exceeds one; inspect the stated noise/model assumptions rather than interpreting R-factor as structural accuracy.")
    correction={d.id:d.correction_states for d in spec.datasets if d.id in {b.dataset_id for b in spec.model.bindings}}
    if any(x in ("unknown","uncorrected") for states in correction.values() for x in states.values()):warnings.append("One or more experimental correction states remain unverified.")
    for d in spec.datasets:
        if d.id in correction and d.mapping.mode=="fluorescence":
            warnings.append(f"Dataset {d.id} uses fluorescence. This release has no validated fluorescence correction pipeline; inspect external correction provenance before quantitative N interpretation, including when correction_states is empty.")
    uncertainty_basis,calibration_links=joint_reference_basis(spec,compiled)
    if uncertainty_basis!="joint_reference" and any(c.uncertainty_basis=="joint_reference" for c in spec.calibrations):
        warnings.append("Calibration metadata requests joint-reference propagation, but the model does not contain a verified fixed-N reference/shared-amplitude/free-target-N linkage; errors remain conditional.")
    fit_result=FitResult(run_id=run_id,assessment="needs_review" if warnings or not success else "conditional",converged=success,
        termination_reason=str(result.message),statistics=statistics,parameters=estimates,covariance=cov,datasets=output,warnings=warnings,uncertainty_basis=uncertainty_basis)
    if "metadata" in FitResult.model_fields:
        fit_result.metadata={**diagnostic_metadata,"attempts":attempts,"resolved_noise":resolved_noise,"residual_layout":layout,
            "data_only_global_scale":scale,"elapsed_seconds":time.monotonic()-started,"experimental_correction_states":correction,
            "calibration_links":calibration_links,
            "scope":"conditional on selected paths, processing, noise and calibration assumptions; not independent scientific validation"}
    return FitRun(id=run_id,model_id=spec.model.id,input_snapshot=spec,snapshot_sha256=digest,
        task_state="timed_out" if time.monotonic()>=deadline else "succeeded",result=fit_result,started_at=started_at,finished_at=utc_now())


def fit_project(store,project_id,model_id,expected_revision=None,context=None,**options):
    project=store.read(project_id)
    if expected_revision is not None and project.revision!=expected_revision:
        raise FitError("revision_conflict","Project changed before the fit was submitted.",project_id)
    return fit_snapshot(store,project_id,build_fit_spec(project,model_id,**options),context=context)


def run_job(payload,context):
    context.heartbeat("validating immutable fit snapshot",.05)
    store=ProjectStore();project_id=payload["project_id"]
    run=fit_snapshot(store,project_id,payload["fit_spec"],run_id=payload.get("run_id"),context=context)
    return {"run":run.model_dump(mode="json"),"project_id":project_id,"expected_revision":payload["expected_revision"]}
