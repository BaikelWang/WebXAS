"""Project-scoped professional API. Numerical operations are added incrementally."""
import io
import json
import hashlib
import re
from typing import Literal
from uuid import uuid4
from zipfile import ZipFile, BadZipFile
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..import_formats import import_file, MAX_UPLOAD
from .contracts import Contract, Dataset, Project, ProcessingRevision, Parameters, ModelRevision, Calibration
from .projects import MAX_ARCHIVE_BYTES, ProjectError, ProjectStore, canonical
from .legacy import import_session, import_old_bundle

class ProfessionalRoute(APIRoute):
    def get_route_handler(self):
        original=super().get_route_handler()
        async def checked(request):
            try:
                return await original(request)
            except ValidationError as exc:
                raise ProjectError('contract_violation','数据、模型或资源引用未通过校验。',
                    details={'errors':exc.errors(include_url=False,include_context=False,include_input=False)}) from exc
        return checked


router = APIRouter(prefix='/api/pro', tags=['professional'],route_class=ProfessionalRoute)


class DatasetMetadata(Contract):
    expected_revision: int = Field(ge=0)
    dataset: Dataset


class ProcessingRequest(Contract):
    expected_revision: int = Field(ge=0)
    dataset_id: str
    parameters: Parameters = Field(default_factory=Parameters)
    mask: list[bool] | None = None
    calibration_id: str | None = None
    noise: dict | None = None
    normalization_only: bool = False


class ModelRequest(Contract):
    expected_revision: int = Field(ge=0)
    model: ModelRevision


class FitRequest(Contract):
    expected_revision: int = Field(ge=0)
    model_id: str
    max_evaluations: int = Field(default=1000,gt=0,le=1000000)
    timeout_seconds: float = Field(default=300.0,gt=0,le=3600)


class MergeRequest(Contract):
    expected_revision: int = Field(ge=0)
    dataset_ids: list[str]
    energy_shifts: dict[str,float] = Field(default_factory=dict)
    variances: list[float | list[float]] | None = None


class GeneratePathsRequest(Contract):
    expected_revision: int = Field(ge=0)
    structure_id: str
    profile: dict = Field(default_factory=dict)


class CalibrationRequest(Contract):
    expected_revision: int = Field(ge=0)
    calibration: Calibration


class CompareRequest(Contract):
    run_ids: list[str] = Field(min_length=2,max_length=30)


class CreateProject(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str = Field(default='Untitled', min_length=1, max_length=200)


class UpdateProject(CreateProject):
    expected_revision: int = Field(ge=0)


def store():
    return ProjectStore()


@router.post('/projects')
def create_project(request: CreateProject):
    return store().create(request.title).model_dump(mode='json')


@router.post('/projects/import')
def import_project(file: UploadFile = File(...)):
    blob = file.file.read(MAX_ARCHIVE_BYTES + 1)
    if len(blob) > MAX_ARCHIVE_BYTES:
        raise ProjectError('archive_too_large', '工程文件超过 50 MB。')
    target = store()
    if blob.startswith(b'PK'):
        try:
            with ZipFile(io.BytesIO(blob)) as archive:
                is_project = 'project.json' in archive.namelist()
        except BadZipFile as exc:
            raise ProjectError('invalid_archive', '无法读取工程压缩包。') from exc
        project = target.import_archive(blob) if is_project else import_old_bundle(target, blob)
    else:
        project = import_session(target, blob)
    return project.model_dump(mode='json')


@router.get('/projects/{project_id}')
def get_project(project_id: str):
    return store().read(project_id).model_dump(mode='json')


@router.patch('/projects/{project_id}')
def update_project(project_id: str, request: UpdateProject):
    target = store()
    project = target.read(project_id)
    value = project.model_dump(mode='json')
    value['title'] = request.title
    return target.save(Project.model_validate(value), request.expected_revision).model_dump(mode='json')


@router.post('/projects/{project_id}/datasets')
def add_datasets(project_id: str, expected_revision: int, file: UploadFile = File(...)):
    blob = file.file.read(MAX_UPLOAD+1)
    if len(blob) > MAX_UPLOAD:
        raise ProjectError('upload_too_large', '上传超过 50 MB。')
    # Session files belong to the project import route, never spectrum inference.
    try:
        parsed = json.loads(blob)
    except (ValueError, UnicodeDecodeError):
        parsed = None
    if isinstance(parsed, dict) and parsed.get('kind') == 'webxas-session':
        raise ProjectError('project_requires_import', '请使用打开工程导入旧会话。')
    target = store()
    project = target.read(project_id)
    if project.revision != expected_revision:
        raise ProjectError('revision_conflict', '项目版本已更新。', project_id)
    filename = file.filename or 'spectrum.dat'
    imported = import_file(blob, filename)
    if not imported.get('datasets'):
        raise ProjectError('no_datasets', '未找到可导入的光谱。')
    raw_ref = target.put_asset(project_id, blob, filename)
    datasets = []
    for item in imported['datasets']:
        text = item.get('raw_text') or ''
        if not text.strip():
            raise ProjectError('support_only', '该文件只含支持材料，请使用对应路径/结构入口。')
        # A container can hold independent spectra. Their primary numeric assets
        # need separate identities; the untouched container remains an asset too.
        extracted=len(imported['datasets'])>1 or text!=blob.decode('utf-8-sig',errors='replace')
        spectrum_ref=target.put_asset(project_id,text.encode('utf-8'),'extracted.dat') if extracted else raw_ref
        datasets.append(Dataset(id='ds_' + uuid4().hex, filename=item.get('filename',filename),
            raw_file=spectrum_ref, raw_text=text,kind=item.get('kind') if item.get('kind') in ('mu','chi') else 'mu',
            element=(re.search(r'Element\.symbol:\s*([A-Z][a-z]?)\b',text) or [None,None])[1],
            edge=(re.search(r'Element\.edge:\s*(K|L[123]|M[1-5])\b',text) or [None,None])[1],
            metadata={'import_warnings':imported.get('warnings',[]),'import_processing_state':item.get('processing_state','unknown'),
                      'import_analysis_blockers':item.get('analysis_blockers',[]),'import_labels':item.get('labels',[]),
                      'original_upload':raw_ref.model_dump(mode='json'),'extracted_from_container':extracted,
                      'import_group_id':item.get('group_id'),'import_role':item.get('role','spectrum')}))
    value = project.model_dump(mode='json')
    value['datasets'].extend(d.model_dump(mode='json') for d in datasets)
    return target.save(Project.model_validate(value), expected_revision).model_dump(mode='json')


@router.post('/projects/{project_id}/export')
def export_project(project_id: str, expected_revision: int | None = None):
    return Response(store().export(project_id, expected_revision), media_type='application/zip',
                    headers={'Content-Disposition':'attachment; filename="webxas-project.zip"'})


@router.patch('/projects/{project_id}/datasets/{dataset_id}')
def update_dataset(project_id: str,dataset_id: str,request: DatasetMetadata):
    target=store();project=target.read(project_id)
    previous=next((d for d in project.datasets if d.id==dataset_id),None)
    if previous is None or request.dataset.id!=dataset_id:
        raise ProjectError('dataset_not_found','数据集身份不一致。',dataset_id)
    if request.dataset.raw_file!=previous.raw_file or request.dataset.raw_text!=previous.raw_text or request.dataset.lineage!=previous.lineage:
        raise ProjectError('immutable_raw','原始数据与来源不可被元数据编辑覆盖。',dataset_id)
    value=project.model_dump(mode='json')
    value['datasets']=[request.dataset.model_dump(mode='json') if d.id==dataset_id else d.model_dump(mode='json') for d in project.datasets]
    return target.save(Project.model_validate(value),request.expected_revision).model_dump(mode='json')


@router.post('/projects/{project_id}/processing',status_code=202)
def submit_processing(project_id: str,request: ProcessingRequest):
    from .job_api import get_service
    project=store().read(project_id)
    if project.revision!=request.expected_revision:
        raise ProjectError('revision_conflict','项目版本已更新。',project_id)
    if request.dataset_id not in {d.id for d in project.datasets}:
        raise ProjectError('dataset_not_found','数据集不存在。',request.dataset_id)
    return get_service().submit(project_id,'preprocess',{'project_id':project_id,**request.model_dump(mode='json')},300)


@router.post('/projects/{project_id}/merge')
def merge_scans(project_id: str,request: MergeRequest):
    from .preprocessing import merge_project_scans
    return merge_project_scans(store(),project_id,request.dataset_ids,request.expected_revision,request.energy_shifts,request.variances).model_dump(mode='json')


@router.get('/projects/{project_id}/arrays')
def get_array(project_id: str,path: str):
    from .projects import asset_refs
    from .contracts import ArrayRef
    from .preprocessing import load_array
    target=store();project=target.read(project_id)
    ref=next((r for r in asset_refs(project.model_dump(mode='json')) if r['path']==path and 'dtype' in r),None)
    if ref is None:
        raise ProjectError('array_not_found','该数组不是当前项目的资源。')
    array=load_array(target,project_id,ArrayRef.model_validate(ref))
    return {'dtype':ref['dtype'],'shape':ref['shape'],'values':array.tolist()} if array.dtype.kind!='c' else {'dtype':ref['dtype'],'shape':ref['shape'],'real':array.real.tolist(),'imag':array.imag.tolist()}


def validate_model_in_project(project,model):
    from .parameters import compile_model
    value=project.model_dump(mode='json')
    value['models']=[m for m in value['models'] if m['id']!=model.id]+[model.model_dump(mode='json')]
    Project.model_validate(value)
    return compile_model(model,project.paths)


@router.post('/projects/{project_id}/models/validate')
def validate_model(project_id: str,request: ModelRequest):
    from .larch_adapter import validate_fit_input
    project=store().read(project_id)
    if project.revision!=request.expected_revision:
        raise ProjectError('revision_conflict','项目版本已更新。')
    compiled=validate_model_in_project(project,request.model)
    diagnostics = list(compiled.diagnostics)
    datasets = {d.id: d for d in project.datasets}
    snapshots = {p.id: p for p in project.processing}
    for binding in request.model.bindings:
        try:
            validate_fit_input(store(), project_id, datasets[binding.dataset_id], snapshots[binding.processing_revision_id], binding)
        except ProjectError as exc:
            diagnostics.append({'severity': 'error', 'code': exc.code, 'message': str(exc), 'dataset_id': binding.dataset_id})
    if not compiled.free_ids:
        diagnostics.append({'severity': 'error', 'code': 'no_free_parameters', 'message': '模型没有自由参数。'})
    return {'valid':True,'fit_allowed':not any(x.get('severity')=='error' for x in diagnostics),'free_ids':compiled.free_ids,'dependency_order':compiled.dependency_order,'diagnostics':diagnostics}


@router.post('/projects/{project_id}/models')
def save_model(project_id: str,request: ModelRequest):
    target=store();project=target.read(project_id)
    validate_model_in_project(project,request.model)
    if any(m.id==request.model.id for m in project.models):
        raise ProjectError('immutable_model','模型版本已存在，请以新 ID 保存新版本。',request.model.id)
    value=project.model_dump(mode='json');value['models'].append(request.model.model_dump(mode='json'))
    return target.save(Project.model_validate(value),request.expected_revision).model_dump(mode='json')


@router.post('/projects/{project_id}/fits',status_code=202)
def submit_fit(project_id: str,request: FitRequest):
    from .larch_adapter import build_fit_spec
    from .job_api import get_service
    project=store().read(project_id)
    if project.revision!=request.expected_revision:
        raise ProjectError('revision_conflict','项目版本已更新。')
    spec=build_fit_spec(project,request.model_id)
    compiled=validate_model_in_project(project,spec.model)
    if any(x.get('severity')=='error' for x in compiled.diagnostics):
        raise ProjectError('model_not_identifiable','参数图包含不可辨识的自由振幅，请修正约束后拟合。',request.model_id)
    value=spec.model_dump(mode='json');value.update(max_evaluations=request.max_evaluations,timeout_seconds=request.timeout_seconds)
    return get_service().submit(project_id,'fit',{'project_id':project_id,'expected_revision':project.revision,'fit_spec':value},request.timeout_seconds)


@router.post('/projects/{project_id}/sensitivity',status_code=202)
def submit_sensitivity(project_id: str,request: FitRequest):
    from .larch_adapter import build_fit_spec
    from .job_api import get_service
    project=store().read(project_id)
    if project.revision!=request.expected_revision:
        raise ProjectError('revision_conflict','项目版本已更新。')
    spec=build_fit_spec(project,request.model_id,max_evaluations=request.max_evaluations,timeout_seconds=request.timeout_seconds)
    compiled=validate_model_in_project(project,spec.model)
    if any(x.get('severity')=='error' for x in compiled.diagnostics):
        raise ProjectError('model_not_identifiable','请先修正模型的不可辨识参数。')
    return get_service().submit(project_id,'sensitivity',{'project_id':project_id,'expected_revision':project.revision,
        'fit_spec':spec.model_dump(mode='json')},min(3600,request.timeout_seconds*3))


@router.post('/projects/{project_id}/jobs/{job_id}/commit')
def commit_job(project_id: str,job_id: str):
    from .job_api import get_service
    from .contracts import FitRun,PathAsset,Structure
    output=get_service().result(project_id,job_id)
    snapshot=get_service().input_snapshot(project_id,job_id)
    submitted=snapshot['payload']
    if output.get('project_id')!=project_id:
        raise ProjectError('job_project_mismatch','任务输出项目身份不一致。')
    if output.get('expected_revision')!=submitted.get('expected_revision'):
        raise ProjectError('job_revision_mismatch','任务输出与提交版本不一致。')
    target=store();project=target.read(project_id);value=project.model_dump(mode='json')
    if 'runs' in output:
        from .contracts import FitSpec
        if snapshot['operation']!='sensitivity':raise ProjectError('job_operation_mismatch','结果类型与任务不一致。')
        expected=FitSpec.model_validate(submitted['fit_spec'])
        models=[ModelRevision.model_validate(m) for m in output.get('models',[])]
        runs=[FitRun.model_validate(r) for r in output['runs']]
        revisions=[ProcessingRevision.model_validate(r) for r in output.get('processing_revisions',[])]
        models_by_id={m.id:m for m in models}
        for run in runs:
            digest=hashlib.sha256(canonical(run.input_snapshot.model_dump(mode='json'))).hexdigest()
            if run.snapshot_sha256!=digest or run.model_id not in models_by_id or run.input_snapshot.model!=models_by_id[run.model_id] or run.input_snapshot.source_sha256!=expected.source_sha256 or run.input_snapshot.engine_versions!=expected.engine_versions:
                raise ProjectError('run_snapshot_mismatch','敏感性结果未匹配冻结输入与模型。',run.id)
        for key,items in [('models',models),('processing',revisions),('runs',runs)]:
            old={item['id']:item for item in value[key]}
            for item in items:
                serialized=item.model_dump(mode='json')
                if item.id in old and old[item.id]!=serialized:
                    raise ProjectError('immutable_result','同一身份的敏感性结果不同。',item.id)
                if item.id not in old:value[key].append(serialized)
        scans=value['metadata'].setdefault('sensitivity_scans',[])
        if any(scan['id']==job_id for scan in scans):return project.model_dump(mode='json')
        cases=output.get('cases',[])
        if {case.get('case_id') for case in cases}!={'baseline','window','background'}:
            raise ProjectError('sensitivity_cases','敏感性检查必须记录固定三种配置的结果。')
        scans.append({'id':job_id,'model_id':expected.model.id,'cases':cases})
        return target.save(Project.model_validate(value),output['expected_revision']).model_dump(mode='json')
    added=[]
    if 'processing' in output:
        item=ProcessingRevision.model_validate(output['processing']);key='processing';added=[item]
        if snapshot['operation']!='preprocess' or item.dataset_id!=submitted.get('dataset_id') or item.input_revision!=submitted.get('expected_revision'):
            raise ProjectError('processing_snapshot_mismatch','处理结果与提交数据/版本不一致。',item.id)
    elif 'run' in output:
        from .contracts import FitSpec
        item=FitRun.model_validate(output['run'])
        if snapshot['operation']!='fit':raise ProjectError('job_operation_mismatch','结果类型与任务操作不一致。')
        expected_spec=FitSpec.model_validate(submitted['fit_spec'])
        actual_digest=hashlib.sha256(canonical(item.input_snapshot.model_dump(mode='json'))).hexdigest()
        expected_digest=hashlib.sha256(canonical(expected_spec.model_dump(mode='json'))).hexdigest()
        if item.snapshot_sha256!=actual_digest or actual_digest!=expected_digest or item.model_id!=expected_spec.model.id or (submitted.get('run_id') and item.id!=submitted['run_id']):
            raise ProjectError('run_snapshot_mismatch','拟合结果不匹配已提交的输入快照。',item.id)
        key='runs';added=[item]
    elif 'paths' in output:
        key='paths';added=[PathAsset.model_validate(item) for item in output['paths']]
        if snapshot['operation']!='feff' or any(p.structure_id!=submitted.get('structure_id') for p in added):
            raise ProjectError('path_snapshot_mismatch','路径结果与提交结构不一致。')
        if 'structure' in output:
            structure=Structure.model_validate(output['structure'])
            if structure.id not in {s.id for s in project.structures}:value['structures'].append(structure.model_dump(mode='json'))
    else:
        raise ProjectError('job_output','任务未返回可保存的分析对象。')
    old={item['id']:item for item in value[key]}
    for item in added:
        serialized=item.model_dump(mode='json')
        if item.id in old and old[item.id]!=serialized:
            raise ProjectError('immutable_result','同一身份的已保存结果不同。',item.id)
    if added and all(item.id in old for item in added):
        return project.model_dump(mode='json')
    value[key].extend(item.model_dump(mode='json') for item in added if item.id not in old)
    return target.save(Project.model_validate(value),output['expected_revision']).model_dump(mode='json')


@router.get('/projects/{project_id}/runs/{run_id}')
def get_run(project_id: str,run_id: str):
    project=store().read(project_id)
    for run in project.runs:
        if run.id==run_id:return run.model_dump(mode='json')
    raise ProjectError('run_not_found','未找到拟合结果。',run_id)


@router.post('/projects/{project_id}/paths')
def upload_paths(project_id: str,expected_revision: int,element: str,edge: str,absorber_site: str,
                 files: list[UploadFile] = File(...)):
    from .paths import import_paths
    target=store();project=target.read(project_id)
    if project.revision!=expected_revision:
        raise ProjectError('revision_conflict','项目版本已更新。')
    contents=[];total=0
    for file in files:
        blob=file.file.read(MAX_UPLOAD+1);total+=len(blob)
        if total>MAX_UPLOAD:
            raise ProjectError('upload_too_large','路径文件总量超过50MB。')
        contents.append((file.filename or 'feff.dat',blob))
    paths=import_paths(target,project_id,contents,element=element,edge=edge,absorber_site=absorber_site)
    value=project.model_dump(mode='json');existing={p.id for p in project.paths}
    value['paths'].extend(p.model_dump(mode='json') for p in paths if p.id not in existing)
    return target.save(Project.model_validate(value),expected_revision).model_dump(mode='json')


@router.post('/projects/{project_id}/structures')
def upload_structure(project_id: str,expected_revision: int,element: str,edge: str,absorber_site: str,
                     source_format: Literal['cif','xyz','feff'],file: UploadFile = File(...)):
    from .paths import prepare_structure
    target=store();project=target.read(project_id)
    if project.revision!=expected_revision:
        raise ProjectError('revision_conflict','项目版本已更新。')
    blob=file.file.read(MAX_UPLOAD+1)
    if len(blob)>MAX_UPLOAD:raise ProjectError('upload_too_large','结构文件超过50MB。')
    structure=prepare_structure(target,project_id,blob,file.filename or 'structure',source_format=source_format,element=element,edge=edge,absorber_site=absorber_site)
    value=project.model_dump(mode='json');value['structures'].append(structure.model_dump(mode='json'))
    return target.save(Project.model_validate(value),expected_revision).model_dump(mode='json')


@router.post('/projects/{project_id}/paths/generate',status_code=202)
def submit_path_generation(project_id: str,request: GeneratePathsRequest):
    from .job_api import get_service
    project=store().read(project_id)
    if project.revision!=request.expected_revision:
        raise ProjectError('revision_conflict','项目版本已更新。')
    return get_service().submit(project_id,'feff',{'project_id':project_id,**request.model_dump(mode='json')},300)


@router.post('/projects/{project_id}/calibrations')
def save_calibration(project_id: str,request: CalibrationRequest):
    target=store();project=target.read(project_id)
    if any(c.id==request.calibration.id for c in project.calibrations):
        raise ProjectError('immutable_calibration','该校准记录已存在，请创建新的校准 ID。')
    value=project.model_dump(mode='json');value['calibrations'].append(request.calibration.model_dump(mode='json'))
    return target.save(Project.model_validate(value),request.expected_revision).model_dump(mode='json')


@router.post('/projects/demo')
def create_demo():
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'data/pro-demo.zip'
    if not source.is_file():raise ProjectError('demo_unavailable','演示工程尚未安装。')
    return store().import_archive(source.read_bytes()).model_dump(mode='json')


@router.post('/projects/{project_id}/reports')
def export_report(project_id: str,run_id: str):
    from .reports import build_report
    return Response(build_report(store(),project_id,run_id),media_type='application/zip',
        headers={'Content-Disposition':'attachment; filename="webxas-analysis-report.zip"'})


@router.post('/projects/{project_id}/compare')
def compare_models(project_id: str,request: CompareRequest):
    from .reports import compare_runs
    return compare_runs(store().read(project_id),request.run_ids)


def project_exception_handler(_, exc):
    status = 404 if exc.code == 'project_not_found' else 409 if exc.code in ('revision_conflict', 'plan_stale') else 422
    return JSONResponse(status_code=status, content={'code':exc.code,'message':exc.message,'details':exc.details,'object_id':exc.object_id})
