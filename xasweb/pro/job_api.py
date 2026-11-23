"""Project-scoped job routes; the application owns lifecycle shutdown."""
from typing import Literal
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import Field

from .contracts import Contract
from .jobs import JobError, get_job_service
from .projects import ProjectStore

# Stable accessor used by the processing/fit endpoints in the main pro router.
get_service = get_job_service

router = APIRouter(prefix="/api/pro/projects", tags=["professional-jobs"])


class SubmitJob(Contract):
    operation: Literal["preprocess", "fit", "feff", "sensitivity"]
    payload: dict
    timeout_seconds: float = Field(default=300, gt=0, le=3600)


def _response(action):
    try:
        return action()
    except JobError as exc:
        return JSONResponse({"code": exc.code, "message": exc.message, "details": {}, "object_id": None}, status_code=exc.status_code)


@router.post("/{project_id}/jobs", status_code=202)
def submit_job(project_id: str, request: SubmitJob):
    project = ProjectStore().read(project_id)
    def submit_checked():
        if request.payload.get('project_id',project_id) != project_id:
            raise JobError('project_scope_mismatch','任务输入与 URL 项目不一致。',409)
        expected=request.payload.get('expected_revision')
        if isinstance(expected,bool) or not isinstance(expected,int) or expected < 0:
            raise JobError('revision_required','任务必须声明非负整数 expected_revision。')
        if expected != project.revision:
            raise JobError('revision_conflict','任务基于旧项目版本，请刷新后重试。',409)
        payload={**request.payload,'project_id':project_id}
        return get_job_service().submit(project_id,request.operation,payload,request.timeout_seconds)
    return _response(submit_checked)


@router.get("/{project_id}/jobs")
def list_jobs(project_id: str):
    ProjectStore().read(project_id)
    return _response(lambda: get_job_service().list(project_id))


@router.get("/{project_id}/jobs/{job_id}")
def get_job(project_id: str, job_id: str):
    ProjectStore().read(project_id)
    return _response(lambda: get_job_service().get(project_id, job_id))


@router.post("/{project_id}/jobs/{job_id}/cancel")
def cancel_job(project_id: str, job_id: str):
    ProjectStore().read(project_id)
    return _response(lambda: get_job_service().cancel(project_id, job_id))


@router.get("/{project_id}/jobs/{job_id}/result")
def job_result(project_id: str, job_id: str):
    ProjectStore().read(project_id)
    return _response(lambda: get_job_service().result(project_id, job_id))
