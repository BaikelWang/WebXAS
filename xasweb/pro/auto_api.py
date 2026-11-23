"""Project-scoped automatic workflows and verified research figure downloads."""
from __future__ import annotations

import io
import hashlib
import mimetypes
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import quote
from zipfile import ZipFile
from uuid import uuid4

from fastapi import APIRouter, Query
from fastapi.responses import Response
from pydantic import Field

from .api import ProfessionalRoute
from .contracts import Contract, Dataset, Project, AssetRef
from .jobs import get_job_service
from .projects import ProjectStore, ProjectError

router = APIRouter(prefix='/api/auto', tags=['automatic-analysis'], route_class=ProfessionalRoute)


class WorkflowRequest(Contract):
    expected_revision: int = Field(ge=0)
    intent: dict
    plan_fingerprint: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    timeout_seconds: float = Field(default=600, gt=0, le=3600)


class PreflightRequest(Contract):
    expected_revision: int = Field(ge=0)
    intent: dict


class DeliveryRequest(Contract):
    expected_revision: int = Field(ge=0)
    workflow_id: str = Field(min_length=1, max_length=100)
    style: dict = Field(default_factory=dict)
    presentation: Literal['legacy','individual'] = 'legacy'
    figure_families: list[Literal['F01','F02','F03','F04','F05','F06','F07','F08','F09','F10','F11','F12','F13']] | None = Field(default=None,min_length=1,max_length=13)
    difference_pair: list[str] | None = Field(default=None, min_length=2, max_length=2)
    timeout_seconds: float = Field(default=600, gt=0, le=3600)


def checked_project(project_id, revision):
    target = ProjectStore()
    project = target.read(project_id)
    if project.revision != revision:
        raise ProjectError('revision_conflict', '工程已更新，请刷新后再执行。', project_id)
    return target, project


@router.post('/projects/{project_id}/workflows', status_code=202)
def start_workflow(project_id: str, request: WorkflowRequest):
    target, project = checked_project(project_id, request.expected_revision)
    from .auto_contracts import normalize_intent, validate_calibration_scope
    intent = normalize_intent(request.intent)
    if not set(intent.dataset_ids + intent.reference_ids) <= {d.id for d in project.datasets}:
        raise ProjectError('dataset_scope', '所选样品或参考谱不属于当前工程。')
    validate_calibration_scope(intent)
    if request.plan_fingerprint is not None:
        from .auto_preflight import verify_plan
        verify_plan(target, project, intent, request.plan_fingerprint)
    return get_job_service().submit(project_id, 'auto', {
        'project_id': project_id, 'expected_revision': project.revision,
        'intent': intent.model_dump(mode='json'), 'plan_fingerprint': request.plan_fingerprint,
    }, request.timeout_seconds)


@router.post('/projects/{project_id}/preflight')
def inspect_workflow(project_id: str, request: PreflightRequest):
    target, project = checked_project(project_id, request.expected_revision)
    from .auto_preflight import preflight
    return preflight(target, project, request.intent)


@router.post('/projects/{project_id}/deliveries', status_code=202)
def start_delivery(project_id: str, request: DeliveryRequest):
    from .figures.individual import validate_presentation
    validate_presentation(request.presentation,request.figure_families)
    _, project = checked_project(project_id, request.expected_revision)
    if not any(w['id'] == request.workflow_id for w in project.metadata.get('automation_runs', [])):
        raise ProjectError('workflow_missing', '找不到所选自动分析记录。')
    return get_job_service().submit(project_id, 'auto_export', {
        'project_id': project_id, 'expected_revision': project.revision,
        'workflow_id': request.workflow_id, 'style': request.style,
        'presentation':request.presentation,'figure_families':request.figure_families,
        'difference_pair':list(request.difference_pair) if request.difference_pair else None,
    }, request.timeout_seconds)


def delivery_asset(project_id, delivery_id):
    target = ProjectStore()
    project = target.read(project_id)
    record = next((x for x in project.metadata.get('automation_deliveries', []) if x['id'] == delivery_id), None)
    if record is None:
        raise ProjectError('delivery_missing', '找不到所选图件包。')
    return record, target.read_asset(project_id, AssetRef.model_validate(record['asset']))


@router.get('/projects/{project_id}/deliveries/{delivery_id}/download')
def download_delivery(project_id: str, delivery_id: str):
    _, payload = delivery_asset(project_id, delivery_id)
    return Response(payload, media_type='application/zip', headers={
        'Content-Disposition': 'attachment; filename="WebXAS-research-figures.zip"'})


@router.get('/projects/{project_id}/deliveries/{delivery_id}/preview')
def preview_delivery(project_id: str, delivery_id: str, filename: str = Query(max_length=300)):
    from .delivery import delivery_manifest,read_delivery_file,safe_delivery_name
    record, payload = delivery_asset(project_id, delivery_id)
    allowed = {x['filename'] for x in record.get('figures', []) if x.get('filename', '').endswith('.png')}
    if filename not in allowed or not safe_delivery_name(filename):
        raise ProjectError('preview_not_available', '只能预览清单内的 PNG 图件。')
    if delivery_manifest(payload) is not None:
        blob,_=read_delivery_file(payload,filename,max_bytes=32*1024*1024)
        return Response(blob,media_type='image/png',headers={'Cache-Control':'private, max-age=3600'})
    with ZipFile(io.BytesIO(payload)) as archive:
        info = archive.getinfo(filename)
        if info.file_size > 32 * 1024 * 1024:
            raise ProjectError('preview_too_large', '预览图过大，请下载图件包查看。')
        return Response(archive.read(info), media_type='image/png', headers={'Cache-Control': 'private, max-age=3600'})


@router.get('/projects/{project_id}/deliveries/{delivery_id}/figures')
def list_delivery_figures(project_id: str, delivery_id: str):
    from .delivery import delivery_manifest,friendly_delivery_name
    record,payload=delivery_asset(project_id,delivery_id)
    manifest=delivery_manifest(payload)
    snapshot=manifest or record
    if snapshot.get('workflow_id')!=record.get('workflow_id'):
        raise ProjectError('delivery_integrity','图件包与记录的工作流归属不一致。')
    figures=[]
    for original in snapshot.get('figures',[]):
        fig=dict(original)
        sources=fig.get('sources',[])
        ids=fig.get('dataset_ids') or list(dict.fromkeys(s['dataset_id'] for s in sources if s.get('dataset_id')))
        if not fig.get('panel_id'):fig['panel_id']=fig.get('id')
        fig.setdefault('dataset_ids',ids);fig.setdefault('sample_names',[])
        if not fig.get('insight'):
            fig['insight']={'observations':['旧版交付未保存结构化逐图解读；这里只展示已保存的图件与来源。'],
                'metrics':[],'limitations':['未用当前可变工程的数据或元信息重新推断历史结果。'],
                'method':{'policy':'legacy-manifest-limited','workflow_id':record['workflow_id']},
                'sources':sources,'dataset_ids':ids,'sample_names':fig['sample_names']}
        from .figures.presentation import brief_insight
        fig['insight']={**fig['insight'],'display':brief_insight(fig['insight'],fig.get('family',''))}
        files=snapshot.get('files',{})
        fig['files']=[{'filename':name,'download_name':friendly_delivery_name(snapshot,name),**files[name]} for name in dict.fromkeys([
            *fig.get('formats',[]),*fig.get('data_files',[]),fig.get('spec_file')]) if name in files]
        figures.append(fig)
    return {'figures':figures,'workflow_id':record['workflow_id'],'delivery_id':delivery_id,
            'missing':snapshot.get('missing',[]),'presentation':snapshot.get('presentation','legacy'),
            'figure_families':snapshot.get('figure_families')}


@router.get('/projects/{project_id}/deliveries/{delivery_id}/files')
def download_delivery_file(project_id: str, delivery_id: str, filename: str = Query(max_length=300)):
    from .delivery import read_delivery_file,delivery_manifest,friendly_delivery_name
    _,payload=delivery_asset(project_id,delivery_id)
    blob,_=read_delivery_file(payload,filename)
    basename=friendly_delivery_name(delivery_manifest(payload),filename)
    ascii_name=''.join(c if 32<ord(c)<127 and c not in '\\"' else '_' for c in basename)
    return Response(blob,media_type=mimetypes.guess_type(basename)[0] or 'application/octet-stream',headers={
        'Content-Disposition':f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(basename,safe="")}',
        'X-Content-SHA256':hashlib.sha256(blob).hexdigest()})


@router.post('/demo')
def create_processing_demo():
    """An explicitly synthetic raw mu(E) example, not experimental validation."""
    import numpy as np
    target = ProjectStore()
    project = target.create('自动处理示例 · 合成 μ(E)')
    energy = np.arange(8780., 9980.1, 2.)
    offset = energy - 8979.
    step = (1 + np.tanh(offset / 3.)) / 2
    k = np.sqrt(np.maximum(offset, 0) / 3.81)
    mu = .2 + .00004 * offset + step * (1 + .035 * np.sin(2 * 2.1 * k) * np.exp(-k / 8))
    text = '# XDI/1.0 WebXAS synthetic example\n# Element.symbol: Cu\n# Element.edge: K\n# Column.1: energy eV\n# Column.2: mu\n# Synthetic mathematical spectrum; not experimental data.\n'
    text += '\n'.join(f'{e:.17g} {y:.17g}' for e, y in zip(energy, mu))
    raw = target.put_asset(project.id, text.encode(), 'synthetic-mu.xdi')
    dataset = Dataset(id='ds_' + uuid4().hex, filename='synthetic-Cu-K.xdi', raw_file=raw,
                      raw_text=text, element='Cu', edge='K', metadata={'not_experimental_data': True, 'synthetic': True})
    value = project.model_dump(mode='json')
    value['datasets'] = [dataset.model_dump(mode='json')]
    value['metadata']['not_experimental_data'] = True
    return target.save(Project.model_validate(value), project.revision).model_dump(mode='json')
