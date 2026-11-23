"""One local scheduler, one disposable scientific process per immutable job.

Workers only write inside their job directory. Publishing a result into a
Project remains the caller's revision-checked responsibility.
"""
from __future__ import annotations

import atexit
from collections import deque
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from xasweb import _runtime
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Callable
from uuid import UUID, uuid4
from .projects import replace_with_retry

OPERATIONS = frozenset({"preprocess", "fit", "feff", "sensitivity", "auto", "auto_export"})
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "timed_out", "interrupted"})
MAX_INPUT_BYTES = 50 * 1024 * 1024
MAX_OUTPUT_BYTES = 50 * 1024 * 1024


class JobError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code, self.message, self.status_code = code, message, status_code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise JobError("invalid_snapshot", "任务输入必须是有限值的 JSON 数据。") from exc


def atomic_json(path: Path, value) -> None:
    # Keep the temporary basename short on Windows. Adding 37 characters to an
    # already valid project/job path can otherwise cross legacy MAX_PATH.
    temporary = path.with_name(uuid4().hex[:8])
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checked_uuid(value: str) -> str:
    try:
        parsed = UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise JobError("invalid_id", "项目和任务 ID 必须是 UUID。") from exc
    if str(parsed) != value:
        raise JobError("invalid_id", "项目和任务 ID 必须是规范 UUID。")
    return value


class _ProcessTree:
    """Keep Windows descendants in a kill-on-close Job Object."""
    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.handle = None
        if os.name == "nt":
            from ctypes import wintypes as wt

            class BasicLimit(ctypes.Structure):
                _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                            ("flags", wt.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                            ("active_processes", wt.DWORD), ("affinity", ctypes.c_size_t),
                            ("priority", wt.DWORD), ("scheduling", wt.DWORD)]

            class IOCounters(ctypes.Structure):
                _fields_ = [(name, ctypes.c_ulonglong) for name in
                            ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

            class ExtendedLimit(ctypes.Structure):
                _fields_ = [("basic", BasicLimit), ("io", IOCounters),
                            ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                            ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wt.LPCWSTR]
            kernel.CreateJobObjectW.restype = wt.HANDLE
            kernel.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
            kernel.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
            kernel.TerminateJobObject.argtypes = [wt.HANDLE, wt.UINT]
            kernel.CloseHandle.argtypes = [wt.HANDLE]
            handle = kernel.CreateJobObjectW(None, None)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            info = ExtendedLimit()
            info.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            try:
                if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not kernel.AssignProcessToJobObject(handle, wt.HANDLE(int(process._handle))):
                    raise ctypes.WinError(ctypes.get_last_error())
            except BaseException:
                kernel.CloseHandle(handle)
                raise
            self.kernel, self.handle = kernel, handle

    def terminate(self) -> None:
        if self.handle:
            self.kernel.TerminateJobObject(self.handle, 1)
        elif os.name != "nt":
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif self.process.poll() is None:
            # Only used if Job Object setup failed before the tree was bound.
            subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                           capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
        elif os.name != "nt":
            # Even a successful worker may have left children behind.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class JobService:
    def __init__(self, root: str | Path | None = None, *, command_factory: Callable | None = None,
                 poll_interval: float = 0.05):
        default = _runtime / 'jobs'
        self.root = Path(root or os.environ.get("XASWEB_JOB_ROOT", default)).resolve()
        if os.name == "nt" and not str(self.root).startswith("\\\\?\\"):
            name = str(self.root)
            self.root = Path("\\\\?\\UNC\\" + name[2:] if name.startswith("\\\\") else "\\\\?\\" + name)
        self.root.mkdir(parents=True, exist_ok=True)
        self._command_factory = command_factory
        self._poll_interval = poll_interval
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._queue = deque()
        self._active: dict[str, _ProcessTree] = {}
        self._recover()
        self._thread = threading.Thread(target=self._dispatch, name="webxas-job-scheduler", daemon=True)
        self._thread.start()

    def _directory(self, project_id: str, job_id: str) -> Path:
        return self.root / checked_uuid(project_id) / "jobs" / checked_uuid(job_id)

    def _read(self, project_id: str, job_id: str) -> dict:
        try:
            value = json.loads((self._directory(project_id, job_id) / "job.json").read_bytes())
        except FileNotFoundError as exc:
            raise JobError("job_not_found", "当前项目中不存在该任务。", 404) from exc
        if value.get("project_id") != project_id or value.get("job_id") != job_id:
            raise JobError("job_identity_mismatch", "任务身份与存储位置不一致。", 409)
        return value

    def _save(self, record: dict) -> None:
        atomic_json(self._directory(record["project_id"], record["job_id"]) / "job.json", record)

    def _recover(self) -> None:
        for path in self.root.glob("*/jobs/*/job.json"):
            try:
                record = self._read(path.parents[2].name, path.parent.name)
            except (JobError, ValueError, OSError):
                continue
            if record["state"] not in TERMINAL_STATES:
                record.update(state="interrupted", finished=utc_now(), result_ref=None,
                              error={"code": "service_restarted", "message": "服务重启，任务未完成；请检查日志后重新提交。"})
                self._save(record)

    def submit(self, project_id: str, operation: str, payload: dict, timeout_seconds: float = 300) -> dict:
        checked_uuid(project_id)
        if operation not in OPERATIONS:
            raise JobError("unknown_operation", "任务操作不在允许列表中。")
        if not isinstance(payload, dict):
            raise JobError("invalid_snapshot", "任务输入必须是 JSON 对象。")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 3600:
            raise JobError("invalid_timeout", "任务超时必须大于 0 且不超过 3600 秒。")
        if payload.get("project_id", project_id) != project_id:
            raise JobError("project_scope_mismatch", "任务输入与提交项目不一致。", 409)
        frozen_payload = {**payload, "project_id": project_id}
        snapshot = {"schema_version": 1, "project_id": project_id, "operation": operation, "payload": frozen_payload}
        raw = canonical_bytes(snapshot)
        if len(raw) > MAX_INPUT_BYTES:
            raise JobError("snapshot_too_large", "任务输入超过 50 MB；大数组应使用校验过的资源引用。")
        with self._lock:
            if self._stop.is_set():
                raise JobError("service_closed", "任务服务已关闭。", 503)
            job_id = str(uuid4())
            directory = self._directory(project_id, job_id)
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "input.json").write_bytes(raw)
            record = {"job_id": job_id, "project_id": project_id, "operation": operation,
                      "state": "queued", "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
                      "timeout_seconds": timeout_seconds, "created": utc_now(), "started": None,
                      "finished": None, "worker_pid": None, "heartbeat": None,
                      "result_ref": None, "error": None}
            self._save(record)
            self._queue.append((project_id, job_id))
            self._wake.set()
            return record.copy()

    def get(self, project_id: str, job_id: str) -> dict:
        with self._lock:
            return self._read(project_id, job_id)

    def list(self, project_id: str) -> list[dict]:
        checked_uuid(project_id)
        with self._lock:
            records = [self._read(project_id, p.parent.name) for p in (self.root / project_id / "jobs").glob("*/job.json")]
        return sorted(records, key=lambda x: x["created"])

    def result(self, project_id: str, job_id: str) -> dict:
        with self._lock:
            record = self._read(project_id, job_id)
            if record["state"] != "succeeded" or not record["result_ref"]:
                raise JobError("result_unavailable", "任务尚无成功结果。", 409)
            path = self._directory(project_id, job_id) / record["result_ref"]["path"]
            if record["result_ref"]["path"] != "output.json":
                raise JobError("invalid_result_reference", "任务结果引用不是受控输出文件。", 409)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["result_ref"]["sha256"]:
                raise JobError("result_hash_mismatch", "任务结果哈希不匹配。", 409)
            return json.loads(raw)

    def input_snapshot(self,project_id: str,job_id: str) -> dict:
        with self._lock:
            record=self._read(project_id,job_id)
            raw=(self._directory(project_id,job_id)/'input.json').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=record['snapshot_sha256']:
                raise JobError('snapshot_hash_mismatch','任务输入快照哈希不符。',409)
            value=json.loads(raw)
            if value.get('project_id')!=project_id or value.get('operation')!=record['operation']:
                raise JobError('snapshot_identity_mismatch','任务输入身份不一致。',409)
            return value

    def cancel(self, project_id: str, job_id: str) -> dict:
        with self._lock:
            record = self._read(project_id, job_id)
            if record["state"] in TERMINAL_STATES:
                return record
            # Revoke permission to publish before killing the worker tree.
            record.update(state="cancelled", finished=utc_now(), result_ref=None,
                          error={"code": "cancelled", "message": "用户取消任务。"})
            self._save(record)
            tree = self._active.get(job_id)
            if tree:
                tree.terminate()
            self._wake.set()
            return record

    def _dispatch(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self._poll_interval)
            with self._lock:
                if not self._queue:
                    self._wake.clear()
                    continue
                project_id, job_id = self._queue.popleft()
                if self._read(project_id, job_id)["state"] != "queued":
                    continue
            self._execute(project_id, job_id)

    def _execute(self, project_id: str, job_id: str) -> None:
        directory = self._directory(project_id, job_id)
        process = tree = None
        stdout = stderr = None
        try:
            with self._lock:
                record = self._read(project_id, job_id)
                if record["state"] != "queued" or self._stop.is_set():
                    if record["state"] == "queued":
                        record.update(state="interrupted", finished=utc_now(), result_ref=None,
                                      error={"code": "service_closed", "message": "任务服务关闭，任务未启动。"})
                        self._save(record)
                    return
                raw = (directory / "input.json").read_bytes()
                if hashlib.sha256(raw).hexdigest() != record["snapshot_sha256"]:
                    raise JobError("snapshot_hash_mismatch", "任务快照已被修改。")
                command = (self._command_factory(directory, record.copy()) if self._command_factory else
                           [sys.executable, "-m", "xasweb.pro.worker", str(directory), record["snapshot_sha256"]])
                stdout = (directory / "stdout.log").open("wb")
                stderr = (directory / "stderr.log").open("wb")
                app_root = str(Path(__file__).resolve().parents[2])
                # Windows CreateProcess still limits the working-directory
                # argument even when file APIs accept extended-length paths.
                # Operations receive an absolute JobContext.work_dir and must
                # use it explicitly for scientific files and FEFF execution.
                launch_cwd = app_root if os.name == "nt" and len(str(directory)) >= 240 else str(directory)
                kwargs = {"stdout": stdout, "stderr": stderr, "stdin": subprocess.DEVNULL,
                          "cwd": launch_cwd, "env": os.environ.copy()}
                inherited_path = kwargs["env"].get("PYTHONPATH", "")
                kwargs["env"]["PYTHONPATH"] = app_root + (os.pathsep + inherited_path if inherited_path else "")
                kwargs["env"]["XRAYLARCH_NO_PLOT"] = "1"
                if self._command_factory is None:
                    kwargs["env"]["WEBXAS_DISPOSABLE_JOB"] = "1"
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    kwargs["start_new_session"] = True
                process = subprocess.Popen(command, **kwargs)
                tree = _ProcessTree(process)
                self._active[job_id] = tree
                record.update(state="running", started=utc_now(), worker_pid=process.pid)
                self._save(record)
            deadline = time.monotonic() + record["timeout_seconds"]
            while process.poll() is None:
                with self._lock:
                    record = self._read(project_id, job_id)
                    if record["state"] != "running":
                        tree.terminate()
                        return
                    if self._stop.is_set() or time.monotonic() >= deadline:
                        timed_out = not self._stop.is_set()
                        record.update(state="timed_out" if timed_out else "interrupted", finished=utc_now(),
                                      result_ref=None, error={"code": "timeout" if timed_out else "service_closed",
                                      "message": "任务超过硬超时。" if timed_out else "任务服务关闭。"})
                        self._save(record)
                        tree.terminate()
                        return
                    heartbeat_path = directory / "heartbeat.json"
                    if heartbeat_path.exists():
                        try:
                            record["heartbeat"] = json.loads(heartbeat_path.read_bytes())
                            self._save(record)
                        except (ValueError, OSError):
                            pass
                self._stop.wait(self._poll_interval)
            with self._lock:
                record = self._read(project_id, job_id)
                if record["state"] != "running":
                    return
                heartbeat_path = directory / "heartbeat.json"
                if heartbeat_path.exists():
                    try:
                        record["heartbeat"] = json.loads(heartbeat_path.read_bytes())
                    except (ValueError, OSError):
                        pass
                if self._stop.is_set() or time.monotonic() >= deadline:
                    record.update(state="timed_out" if not self._stop.is_set() else "interrupted", finished=utc_now(),
                                  result_ref=None, error={"code": "deadline_exceeded", "message": "任务在截止后结束，结果未提交。"})
                elif process.returncode != 0:
                    error_path = directory / "error.json"
                    error = json.loads(error_path.read_bytes()) if error_path.exists() else {
                        "code": "worker_failed", "message": f"工作进程退出码 {process.returncode}；详见 stderr.log。"}
                    record.update(state="failed", finished=utc_now(), result_ref=None, error=error)
                else:
                    result_path = directory / "output.json"
                    if not result_path.exists() or result_path.stat().st_size > MAX_OUTPUT_BYTES:
                        raise JobError("invalid_worker_output", "工作进程没有返回有效的有限大小结果。")
                    result_raw = result_path.read_bytes()
                    result = json.loads(result_raw)
                    if not isinstance(result, dict):
                        raise JobError("invalid_worker_output", "工作进程结果必须是 JSON 对象。")
                    canonical_bytes(result)  # Reject NaN and Infinity even if a custom test worker wrote them.
                    record.update(state="succeeded", finished=utc_now(), error=None,
                                  result_ref={"path": "output.json", "sha256": hashlib.sha256(result_raw).hexdigest()})
                self._save(record)
        except Exception as exc:
            if process and process.poll() is None:
                if tree:
                    tree.terminate()
                else:
                    process.kill()
                    process.wait(timeout=5)
            with self._lock:
                record = self._read(project_id, job_id)
                if record["state"] not in TERMINAL_STATES:
                    record.update(state="failed", finished=utc_now(), result_ref=None,
                                  error={"code": getattr(exc, "code", "worker_exception"), "message": str(exc)})
                    self._save(record)
        finally:
            with self._lock:
                self._active.pop(job_id, None)
                if tree:
                    tree.close()
            if stdout:
                stdout.close()
            if stderr:
                stderr.close()

    def close(self) -> None:
        with self._lock:
            if self._stop.is_set():
                return
            self._stop.set()
            for project_id, job_id in self._queue:
                record = self._read(project_id, job_id)
                if record["state"] == "queued":
                    record.update(state="interrupted", finished=utc_now(), result_ref=None,
                                  error={"code": "service_closed", "message": "任务服务关闭，排队任务未运行。"})
                    self._save(record)
            self._wake.set()
        self._thread.join(timeout=15)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


_service = None
_service_lock = threading.Lock()


def get_job_service() -> JobService:
    global _service
    with _service_lock:
        if _service is None or _service._stop.is_set():
            _service = JobService()
        return _service


def close_job_service() -> None:
    global _service
    with _service_lock:
        if _service is not None:
            _service.close()
            _service = None


atexit.register(close_job_service)
