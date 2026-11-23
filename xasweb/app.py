import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .export import build_bundle, replay_bundle
from .ingest import InputError, parse_ascii
from .import_formats import import_file, column_labels, MAX_UPLOAD
from .models import AnalysisRequest, CheckedModel
from pydantic import Field, field_validator
from .edges import normalize_edge
from .automation import AutoRequest, automatic_analysis, infer_mapping
from .science import ROOT, analyze, software_identity
from . import __version__
from .assessment import (analyze_assessed, assess_request, attach_import_assessment, token_context)


@asynccontextmanager
async def application_lifespan(_):
    yield
    from .pro.jobs import close_job_service
    close_job_service()


app = FastAPI(title="WebXAS professional preview", version=__version__, lifespan=application_lifespan)

from .pro.api import router as pro_router, project_exception_handler
from .pro.projects import ProjectError
from .pro.job_api import router as job_router
from .pro.jobs import JobError
from .pro.parameters import ParameterGraphError
from .pro.auto_api import router as auto_router
from .knowledge_api import router as knowledge_router
app.include_router(pro_router)
app.include_router(job_router)
app.include_router(auto_router)
app.include_router(knowledge_router)
app.add_exception_handler(ProjectError, project_exception_handler)


@app.exception_handler(JobError)
async def job_error(_, exc):
    return JSONResponse(status_code=exc.status_code,content={'code':exc.code,'message':exc.message,'details':{},'object_id':None})


@app.exception_handler(ParameterGraphError)
async def parameter_error(_, exc):
    return JSONResponse(status_code=422,content={'code':exc.code,'message':str(exc),'details':{},'object_id':exc.object_id})

lock = threading.RLock()  # Nested workflow stages share Larch's process-global state.
static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static), name="static")
validation_root = os.environ.get("WEBXAS_VALIDATION_ROOT", static.parent / "gallery")
if validation_root and Path(validation_root).is_dir():
    app.mount("/validation", StaticFiles(directory=validation_root), name="validation")


@app.exception_handler(InputError)
async def input_error(_, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/")
def index():
    return FileResponse(static / "auto.html")


@app.get("/quick")
def quick_screen():
    return FileResponse(static / "simple.html")


@app.get('/pro')
def professional_index():
    return FileResponse(static / 'pro.html')


@app.get('/auto')
def automatic_index():
    return FileResponse(static / 'auto.html')


@app.get('/methods')
def method_index():
    return FileResponse(static / 'methods.html')


@app.get('/knowledge')
def knowledge_index():
    return FileResponse(static / 'knowledge.html')


@app.get("/api/health")
def health():
    import importlib.metadata
    return {
        "status": "ready",
        "engine": "xraylarch",
        "version": importlib.metadata.version("xraylarch"),
        "build_commit": os.environ.get("XASWEB_BUILD_COMMIT"),
        "application_version": __version__,
        "application_root": str(ROOT.resolve()),
    }


@app.post("/api/import")
def import_spectra(file: UploadFile = File(...)):
    blob = file.file.read(MAX_UPLOAD + 1)
    if len(blob) > MAX_UPLOAD:
        raise InputError('Upload exceeds 50 MB.')
    from .local_structure import classify
    text = blob.decode('utf-8-sig', errors='replace')
    role = classify(file.filename or '', text)
    if role:
        if len(text.strip())<12:
            raise InputError('支持材料内容不完整，未发现可核对信息。')
        return {'datasets': [], 'warnings': [], 'support': {
            'name': file.filename or 'support', 'role':role, 'text':text,
            'status':'uploaded_not_yet_evaluated'}}
    imported = import_file(blob, file.filename or "spectrum")
    for dataset in imported['datasets']:
        if not dataset['raw_text']:
            dataset['simple_input'] = dict(mapping=None, mapping_reason='supporting_material', chi_mapping=None, table=[], header=[])
            attach_import_assessment(dataset)
            continue
        table, header = parse_ascii(dataset['raw_text'], preview=True)
        import numpy as np
        if not np.isfinite(table).all():
            raise InputError('Non-finite numeric input cannot be previewed safely; original upload has not been changed.')
        try:
            mapping, reason = infer_mapping(dataset['raw_text'])
        except InputError:
            mapping, reason = None, 'preview_only_short_or_invalid_analysis_input'
        # Legacy declared mappings remain available; the screening workflow
        # independently records bounded inference and all resulting assumptions.
        import re
        k_columns = [i for i, label in enumerate(dataset['labels']) if label.lower() == 'k']
        chi_columns = [i for i, label in enumerate(dataset['labels']) if label.lower() == 'chi']
        chi_mapping = None
        if len(k_columns) == len(chi_columns) == 1:
            column = k_columns[0] + 1
            units = [m[1].lower() for line in header
                     if (m := re.search(rf'Column\.{column}:\s*k\s+(\S+)', line, re.I))]
            if units and all(unit in ('inverse_angstrom', '1/angstrom', 'angstrom^-1', 'a^-1', 'å⁻¹') for unit in units):
                chi_mapping = dict(energy=k_columns[0], signal=chi_columns[0], unit='inverse_angstrom')
        dataset['simple_input'] = dict(mapping=mapping.model_dump() if mapping else None,
                                       mapping_reason=reason, chi_mapping=chi_mapping,
                                       table=table.tolist(), header=header)
        from .screening_input import identify_input
        from .local_structure import sample_context
        dataset['input_identification']=identify_input(dataset['raw_text'],sample_context(dataset['raw_text'],{},[]))
        attach_import_assessment(dataset)
    return imported


class ChiDeclaration(CheckedModel):
    k_column: int = Field(ge=0, le=255)
    chi_column: int = Field(ge=0, le=255)
    unit: str = Field(pattern='^inverse_angstrom$')
    input_weight: int = Field(ge=0, le=3)


class ChiRequest(CheckedModel):
    object_id: str | None = Field(default=None, max_length=80)
    assessment_token: str | None = Field(default=None, max_length=16000)
    raw_text: str = Field(max_length=2_000_000)
    kmin: float = Field(default=2, ge=0, le=100)
    kmax: float = Field(default=12, gt=0, le=100)
    kweight: int = Field(default=2, ge=0, le=3)
    dk: float = Field(default=1, ge=0, le=10)
    declaration: ChiDeclaration | None = None


@app.post("/api/chi-transform")
def chi_transform(request: ChiRequest):
    from .import_formats import transform_chi
    from .assessment import assess_chi
    with lock:
        original = request
        if request.declaration is not None:
            context = token_context(request)
            blockers=set(context.get('analysis_blockers') or [])-{'chi_weighting_requires_verification'}
            if request.object_id or blockers or context.get('role') in ('fit', 'residual', 'derived_fit'):
                raise InputError('This input requires model or source review; a column declaration cannot override it.')
            table, header = parse_ascii(request.raw_text)
            import re
            weights = re.findall(r'Chi\.weighting:\s*(\S+)', '\n'.join(header), re.I)
            decl = request.declaration
            if any(value.lower() not in (str(decl.input_weight), 'unweighted' if decl.input_weight == 0 else str(decl.input_weight)) for value in weights):
                raise InputError('The confirmed weighting conflicts with the source weighting.')
            if decl.k_column == decl.chi_column or max(decl.k_column, decl.chi_column) >= table.shape[1]:
                raise InputError('Select distinct, existing k and chi columns.')
            if decl.input_weight and (table[:,decl.k_column]<0).any():
                raise InputError('Weighted chi requires nonnegative k.')
            raw = '# Column.1: k inverse_angstrom\n# Column.2: chi\n' + '\n'.join(
                f'{row[decl.k_column]:.17g} {row[decl.chi_column]/row[decl.k_column]**decl.input_weight:.17g}'
                for row in table if not decl.input_weight or row[decl.k_column]>0)
            request = request.model_copy(update={'raw_text':raw, 'assessment_token':None, 'declaration':None})
        assessment = assess_chi(request)
        if original.declaration:
            assessment['provenance']['user_confirmed_mapping'] = original.declaration.model_dump()
            import hashlib
            assessment['provenance']['original_source_sha256'] = hashlib.sha256(original.raw_text.encode()).hexdigest()
            if original.declaration.input_weight:
                assessment['adjustments'].append('Removed declared k weighting; k=0, if present, excluded because unweighting is undefined there. Original input is preserved.')
        result = transform_chi(**request.model_dump(exclude={'object_id','assessment_token','declaration'}))
        assessment.update(stage='S4', completed_stages=['S4'], completed_operations=['fourier_transform'])
        result['assessment'] = assessment
        return result


class SupportInput(CheckedModel):
    name: str = Field(max_length=200)
    role: str = Field(max_length=40)
    text: str = Field(max_length=2_000_000)


class WorkflowRequest(CheckedModel):
    raw_text: str = Field(max_length=2_000_000)
    filename: str = Field(default='spectrum.dat', max_length=200)
    kind: str | None = Field(default=None, pattern='^(mu|chi)$')
    mapping: dict | None = None
    declaration: ChiDeclaration | None = None
    assessment_token: str | None = Field(default=None, max_length=16000)
    element: str | None = Field(default=None, max_length=3)
    edge: str | None = Field(default=None, max_length=3)
    sample: str = Field(default='', max_length=200)
    supports: list[SupportInput] = Field(default_factory=list, max_length=30)

    @field_validator('edge', mode='before')
    @classmethod
    def canonical_edge(cls, value):
        return normalize_edge(value) if isinstance(value, str) else value


@app.post('/api/workflow')
def workflow(request: WorkflowRequest):
    """Process actual uploaded spectra, then evaluate a fresh structural model."""
    from .local_structure import classify, sample_context, evaluate_structure
    from .screening_input import identify_input
    from .spectral_views import spectral_views
    import numpy as np
    import hashlib
    if sum(len(s.text) for s in request.supports)>8_000_000:
        raise InputError('Supporting text exceeds 8 MB.')
    supports=[]
    for s in request.supports:
        # Roles are inspected again; client labels are not model evidence.
        supports.append(dict(name=s.name,text=s.text,role=classify(s.name,s.text) or 'reference_spectrum'))
    context=sample_context(request.raw_text,{'element':request.element,'edge':request.edge,'sample':request.sample},supports)
    response={'schema_version':3,'kind':'webxas-workflow','filename':request.filename,
              'source_sha256':hashlib.sha256(request.raw_text.encode()).hexdigest(),
              'processing':None,'structure':None,'stop_reasons':[], 'context':context}
    arrays,fft={},{}
    inference_energy,inference_mu=None,None
    with lock:
        try:
            identification=identify_input(request.raw_text,context)
            response['input_identification']=identification
            selection=identification.get('selection')
            kind=request.kind or (selection or {}).get('kind')
            if not selection and not (request.mapping or request.declaration):
                raise InputError(identification.get('reason','无法可靠识别输入类型。'))
            if kind=='mu':
                mapping=request.mapping or {key:value for key,value in selection.items() if key!='kind'}
                if not context.get('element') or not context.get('edge'):
                    from .ingest import construct_mu
                    from .models import Mapping
                    inference_energy,inference_mu,_=construct_mu(request.raw_text,Mapping(**mapping))
                    # Complete the absorber context before processing so the
                    # same automatic element/edge is used by every later stage.
                    from .edge_inference import infer_missing_context
                    infer_missing_context(context,inference_energy,inference_mu)
                processed=automatic_analysis(AutoRequest(raw_text=request.raw_text,filename=request.filename,
                    assessment_token=request.assessment_token,mapping=mapping,
                    element=context.get('element'),edge=context.get('edge'),sample=context.get('sample') or '',
                    method='rules',budget=1,experimental_search=False))
                response['processing']=dict(kind='mu',data=processed)
                if processed.get('result'):
                    arrays=processed['result']['arrays']
                    fft=processed['result'].get('resolved_calls',{}).get('xftf',processed.get('request',{}).get('parameters',{}))
                else:
                    response['stop_reasons'].append(processed.get('reason') or next((t.get('reason') for t in processed.get('automation',{}).get('trace',[]) if t.get('reason')), '光谱处理未完成。'))
            else:
                table,_=parse_ascii(request.raw_text)
                declaration=request.declaration or ChiDeclaration(k_column=selection['energy'],chi_column=selection['signal'],
                    unit=selection['unit'],input_weight=selection['input_weight'])
                col=declaration.k_column
                if col>=table.shape[1]:raise InputError('k 列不存在。')
                k=table[:,col]
                chi_request=ChiRequest(raw_text=request.raw_text,assessment_token=request.assessment_token,
                    declaration=declaration,kmin=max(2.,float(k[0])+.5),kmax=min(12.,float(k[-1])-.5))
                processed=chi_transform(chi_request)
                if request.declaration is None:
                    provenance=processed['assessment']['provenance']
                    provenance['automatic_mapping']=provenance.pop('user_confirmed_mapping',None)
                response['processing']=dict(kind='chi',data=processed,request=chi_request.model_dump())
                arrays=processed;fft=processed['parameters']
        except Exception as exc:
            response['stop_reasons'].append(str(exc))
        # This is a no-op for energy spectra completed above. It retains the
        # recoverable manual fallback for chi(k) and parsing failures.
        from .edge_inference import infer_missing_context
        infer_missing_context(context,inference_energy,inference_mu)
        try:
            response['spectral_views']=spectral_views(arrays,fft)
        except Exception as exc:
            response['spectral_views']={'status':'unavailable','reason':str(exc)}
            response['stop_reasons'].append('衍生图谱停止；已保留光谱处理阶段：'+str(exc))
        try:
            response['structure']=evaluate_structure(arrays.get('k') if arrays.get('chir_mag') else None,arrays.get('chi'),context,supports,fft)
        except Exception as exc:
            response['structure']={'status':'unresolved','fit':None,'stop_reasons':['模型评估停止：'+str(exc)],
                                   'unresolved_parameters':['N','R','sigma_squared','energy_shift','uncertainties']}
        assumptions=response.get('input_identification',{}).get('assumptions',[])
        response['structure'].setdefault('assumptions',[]).extend(assumptions)
        for fit in response['structure'].get('fits',[]):
            fit['assumptions']=list(dict.fromkeys(fit['assumptions']+assumptions))
    response['stop_reasons']+=response['structure'].get('stop_reasons',[])
    response['reference_checks']=[]
    for support in supports:
        if support['role']!='reference_spectrum':
            continue
        check={'filename':support['name'], 'status':'unresolved',
               'note':'参考谱用于吸收边一致性检查；缺少已知配位数及独立振幅标定时不用于标定 S₀²。'}
        try:
            from .metadata import inspect_metadata
            reference_mapping,_=infer_mapping(support['text'])
            reference_context=sample_context(support['text'],{},[])
            if reference_mapping is None or not reference_context.get('element') or not reference_context.get('edge'):
                raise InputError('参考谱缺少明确列、单位或吸收边。')
            if (reference_context['element'],reference_context['edge'])!=(context.get('element'),context.get('edge')):
                raise InputError('参考谱吸收元素或吸收边与当前光谱不匹配。')
            with lock:
                metadata=inspect_metadata(support['text'],reference_mapping,context['element'],context['edge'])
            check.update(status=metadata['status'],metadata=metadata)
            observed=((response.get('processing') or {}).get('data',{}).get('result') or {}).get('metrics',{}).get('e0_eV')
            if observed is not None and metadata.get('detected_edge_eV') is not None:
                check['observed_edge_difference_eV']=observed-metadata['detected_edge_eV']
                check['note']+=' 观测边位差仅供检查，不自动校正或解释为化学位移。'
        except Exception as exc:
            check['reason']=str(exc)
        response['reference_checks'].append(check)
    response['status']='completed' if response['structure'].get('fit') else 'partial' if arrays else 'needs_input'
    if not context.get('element') or not context.get('edge'):
        response['required']=[key for key in ('element','edge') if not context.get(key)]
    return response


@app.get('/api/path-templates')
def path_templates():
    from .local_structure import TEMPLATES
    return {'templates':TEMPLATES, 'qualified_by':['absorber','edge','scatterer','reference_environment'],
            'database_search':False, 'note':'模板是模型分类，不能替换元素标签作为真实 FEFF 路径。'}


@app.post("/api/inspect")
def inspect_table(request: AnalysisRequest):
    table, header = parse_ascii(request.raw_text, preview=True)
    import numpy as np
    preview = [[float(v) if np.isfinite(v) else None for v in row] for row in table[:6]]
    assessment = assess_request(request)
    return {"rows": len(table), "columns": table.shape[1], "header": header,
            "labels": column_labels(header, table.shape[1]), "preview": preview,
            "assessment":assessment,
            "issues": ['nonfinite_input'] if not np.isfinite(table).all() else []}


@app.post("/api/analyze")
def run(request: AnalysisRequest):
    with lock:
        try:
            return analyze_assessed(request)
        except InputError:
            raise
        except Exception as exc:
            logging.exception("Scientific processing failed")
            raise HTTPException(422, "Larch could not process this spectrum. Inspect the input and fitting windows; no result was saved.") from exc


@app.post("/api/auto-analyze")
def auto_run(request: AutoRequest):
    with lock:
        return automatic_analysis(request)


@app.post("/api/export")
def export(request: AnalysisRequest, x_expected_run_id: str | None = Header(default=None)):
    with lock:
        result = analyze_assessed(request)
        if x_expected_run_id and result["run_id"] != x_expected_run_id:
            raise HTTPException(409, "Analysis or software changed. Run the analysis again before exporting.")
        blob = build_bundle(request, result)
    return Response(blob, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="xas-{result["run_id"][:12]}.zip"'})


@app.post("/api/replay")
def replay(file: UploadFile = File(...)):
    blob = file.file.read(50_000_001)
    if len(blob) > 50_000_000:
        raise InputError("Bundle exceeds 50 MB.")
    with lock:
        request, result, report = replay_bundle(blob)
    return {"request": request.model_dump(), "result": result, "verification": report,
            "session": None, "original_raw_text": request.raw_text}


@app.post("/api/export-file")
def export_file(request_json: str = Form(...), expected_run_id: str = Form(...), automation_json: str | None = Form(default=None)):
    """Native form/attachment flow: keeps browser download handling out of JS."""
    try:
        request = AnalysisRequest.model_validate_json(request_json)
    except ValueError as exc:
        raise HTTPException(422, "Invalid export recipe. Return to the main page and rerun analysis.") from exc
    if automation_json is None:
        return export(request, expected_run_id)
    if len(automation_json) > 200_000:
        raise HTTPException(422, "Automation record exceeds 200 KB.")
    try:
        record = json.loads(automation_json)
        if not isinstance(record, dict) or record.get("selected_run_id") != expected_run_id:
            raise ValueError("Automation record does not match this run.")
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    with lock:
        result = analyze_assessed(request)
        if result["run_id"] != expected_run_id:
            raise HTTPException(409, "The displayed result is stale. Run analysis again.")
        blob = build_bundle(request, result, automation=record)
    return Response(blob, media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="xas-auto-analysis.zip"'})


@app.get("/api/evidence")
def evidence():
    report = ROOT / "evidence/validation.json"
    if not report.exists():
        return {"status": "not_run", "checks": []}
    saved = json.loads(report.read_text(encoding="utf-8"))
    saved["matches_current_software"] = saved.get("software") == software_identity()
    return saved
