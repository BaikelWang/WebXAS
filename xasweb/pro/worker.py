"""Fixed scientific entry points for disposable job processes."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback

from .jobs import JobError, atomic_json, utc_now

# The request supplies only one of these operation names, never an import path.
HANDLERS = {
    "auto": ("xasweb.pro.workflows", "run_job"),
    "auto_export": ("xasweb.pro.delivery", "run_job"),
    "preprocess": ("xasweb.pro.preprocessing", "run_job"),
    "fit": ("xasweb.pro.larch_adapter", "run_job"),
    "feff": ("xasweb.pro.paths", "run_job"),
    "sensitivity": ("xasweb.pro.sensitivity", "run_job"),
}


class JobContext:
    def __init__(self, work_dir: Path, snapshot_sha256: str):
        self.work_dir = work_dir
        self.snapshot_sha256 = snapshot_sha256

    def heartbeat(self, stage: str, progress: float | None = None) -> None:
        if not isinstance(stage, str) or not 0 < len(stage) <= 200:
            raise JobError("invalid_stage", "任务阶段必须为非空短文本。")
        if progress is not None and not 0 <= progress <= 1:
            raise JobError("invalid_progress", "任务进度必须介于 0 和 1。")
        atomic_json(self.work_dir / "heartbeat.json", {"stage": stage, "progress": progress, "updated": utc_now()})


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m xasweb.pro.worker JOB_DIRECTORY SNAPSHOT_SHA256", file=sys.stderr)
        return 2
    directory = Path(argv[0]).resolve()
    try:
        raw = (directory / "input.json").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != argv[1]:
            raise JobError("snapshot_hash_mismatch", "工作进程输入哈希不匹配。")
        snapshot = json.loads(raw)
        operation = snapshot["operation"]
        if operation not in HANDLERS:
            raise JobError("unknown_operation", "任务操作不在固定注册表中。")
        context = JobContext(directory, digest)
        context.heartbeat("starting", 0)
        module_name, function_name = HANDLERS[operation]
        try:
            module = importlib.import_module(module_name)
            function = getattr(module, function_name)
        except (ModuleNotFoundError, AttributeError) as exc:
            raise JobError("operation_unavailable", f"科学操作 {operation} 尚未安装完整。") from exc
        result = function(snapshot["payload"], context)
        if not isinstance(result, dict):
            raise JobError("invalid_worker_output", "科学操作必须返回 JSON 对象。")
        atomic_json(directory / "output.json", result)
        context.heartbeat("completed", 1)
        return 0
    except Exception as exc:
        traceback.print_exc()
        atomic_json(directory / "error.json", {"code": getattr(exc, "code", "science_operation_failed"), "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
