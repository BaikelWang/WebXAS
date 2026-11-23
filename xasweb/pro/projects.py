"""Versioned local projects and verified, portable project archives."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import threading
import time
from uuid import UUID, uuid4
from zipfile import ZipFile, BadZipFile, ZIP_DEFLATED

from .contracts import AssetRef, Project

MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_EXPANDED_BYTES = 250 * 1024 * 1024
MAX_ENTRIES = 5000
_lock = threading.RLock()


class ProjectError(ValueError):
    def __init__(self, code, message, object_id=None, details=None):
        super().__init__(message)
        self.code, self.message, self.object_id = code, message, object_id
        self.details=details or {}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def safe_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ProjectError('unsafe_path', '资源必须使用项目相对路径。')
    path = PurePosixPath(value)
    if path.is_absolute() or any(x in ('', '.', '..') for x in value.split('/')):
        raise ProjectError('unsafe_path', '资源路径越出项目目录。')
    return path


def asset_refs(value):
    """Find serialized asset references, including refs nested in metadata."""
    if isinstance(value, dict):
        if {'path', 'sha256', 'byte_size'} <= value.keys():
            yield value
        else:
            for child in value.values():
                yield from asset_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from asset_refs(child)


def replace_with_retry(source,destination):
    # Windows readers may briefly hold a destination without FILE_SHARE_DELETE.
    # Retry only these transient sharing/lock errors, with a bounded budget.
    for attempt in range(8):
        try:
            os.replace(source,destination)
            return
        except PermissionError as exc:
            if os.name!='nt' or getattr(exc,'winerror',None) not in (5,32,33) or attempt==7:
                raise
            time.sleep(.002*(2**attempt))


def atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.write-' + uuid4().hex + '.tmp')
    try:
        with temp.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


class ProjectStore:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get('XASWEB_PROJECT_ROOT', Path(__file__).resolve().parents[2] / '.runtime/projects')).resolve()
        # Portable projects can be nested under long Windows workspace paths.
        if os.name == 'nt' and not str(self.root).startswith('\\\\?\\'):
            name = str(self.root)
            self.root = Path('\\\\?\\UNC\\' + name[2:] if name.startswith('\\\\') else '\\\\?\\' + name)
        self.root.mkdir(parents=True, exist_ok=True)

    def folder(self, project_id):
        try:
            name = str(UUID(project_id))
        except (ValueError, TypeError, AttributeError):
            raise ProjectError('invalid_project_id', '项目 ID 必须是 UUID。', str(project_id)) from None
        return self.root / name

    @contextmanager
    def locked(self, project_id):
        folder = self.folder(project_id)
        folder.mkdir(parents=True, exist_ok=True)
        with _lock, (folder / '.lock').open('a+b') as stream:
            stream.seek(0)
            if not stream.read(1):
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield folder
            finally:
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def read(self, project_id):
        path = self.folder(project_id) / 'project.json'
        if not path.is_file():
            raise ProjectError('project_not_found', '未找到项目。', project_id)
        return Project.model_validate_json(path.read_bytes())

    def create(self, title='Untitled'):
        project = Project(title=title)
        return self.save(project)

    def save(self, project, expected_revision=None):
        project = Project.model_validate(project.model_dump(mode='json'))
        with self.locked(project.id) as folder:
            path = folder / 'project.json'
            exists = path.exists()
            if exists:
                current = self.read(project.id)
                if expected_revision is None or current.revision != expected_revision:
                    raise ProjectError('revision_conflict', '项目已更新，请重新读取后保存。', project.id)
                revision = current.revision + 1
            else:
                if expected_revision not in (None, -1):
                    raise ProjectError('revision_conflict', '项目尚不存在。', project.id)
                revision = 0
            value = project.model_dump(mode='json')
            value['revision'] = revision
            project = Project.model_validate(value)
            self.verify_assets(project)
            payload = canonical(project.model_dump(mode='json'))
            history = folder / 'revisions' / f'{revision:08d}.json'
            if history.exists() and history.read_bytes() != payload:
                raise ProjectError('revision_conflict', '版本记录已存在。', project.id)
            if not history.exists():
                atomic_write(history, payload)
            atomic_write(path, payload)
            return project

    def put_asset(self, project_id, payload, filename='asset.bin'):
        if not isinstance(payload, bytes) or len(payload) > MAX_EXPANDED_BYTES:
            raise ProjectError('invalid_asset', '资源大小超出限制。')
        digest = hashlib.sha256(payload).hexdigest()
        suffix = Path(filename).suffix.lower()
        if not re.fullmatch(r'\.[a-z0-9]{1,12}', suffix):
            suffix = '.bin'
        relative = f'assets/{digest}{suffix}'
        path = self.folder(project_id) / relative
        if not path.exists():
            atomic_write(path, payload)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ProjectError('asset_integrity', '已有资源校验失败。', project_id)
        return AssetRef(path=relative, sha256=digest, byte_size=len(payload))

    def read_asset(self, project_id, ref):
        value = ref.model_dump(mode='json') if hasattr(ref, 'model_dump') else ref
        relative = safe_path(value['path'])
        if relative.parts[0] in ('project.json','manifest.json','revisions','.lock'):
            raise ProjectError('unsafe_path', '资源路径与工程内部文件冲突。')
        path = self.folder(project_id).joinpath(*relative.parts)
        if not path.is_file():
            raise ProjectError('missing_asset', '工程附件缺失：' + value['path'], project_id)
        payload = path.read_bytes()
        if len(payload) != value['byte_size'] or hashlib.sha256(payload).hexdigest() != value['sha256']:
            raise ProjectError('asset_integrity', '工程附件哈希不符：' + value['path'], project_id)
        return payload

    def verify_assets(self, project):
        refs = {}
        for ref in asset_refs(project.model_dump(mode='json')):
            safe_path(ref['path'])
            if ref['path'] in refs and refs[ref['path']] != ref:
                raise ProjectError('asset_conflict', '相同资源路径包含不同身份。', project.id)
            refs[ref['path']] = ref
            self.read_asset(project.id, ref)
        return refs

    def export(self, project_id, expected_revision=None):
        with self.locked(project_id):
            project = self.read(project_id)
            if expected_revision is not None and project.revision != expected_revision:
                raise ProjectError('revision_conflict', '导出版本已过期。', project_id)
            refs = self.verify_assets(project)
            entries = {'project.json': canonical(project.model_dump(mode='json'))}
            entries.update({name: self.read_asset(project_id, ref) for name, ref in refs.items()})
            manifest = {'format': 'webxas-project', 'schema_version': 1,
                        'files': {name: {'sha256': hashlib.sha256(blob).hexdigest(), 'byte_size': len(blob)} for name, blob in entries.items()}}
            output = io.BytesIO()
            with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
                for name, blob in entries.items():
                    archive.writestr(name, blob)
                archive.writestr('manifest.json', canonical(manifest))
            return output.getvalue()

    def import_archive(self, payload):
        if len(payload) > MAX_ARCHIVE_BYTES:
            raise ProjectError('archive_too_large', '工程压缩包超过 50 MB。')
        try:
            with ZipFile(io.BytesIO(payload)) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_ENTRIES or sum(x.file_size for x in infos) > MAX_EXPANDED_BYTES:
                    raise ProjectError('archive_too_large', '工程解压体积或文件数量超限。')
                names = [x.filename for x in infos]
                if len(names) != len(set(names)):
                    raise ProjectError('duplicate_entry', '工程包含重复文件名。')
                for name in names:
                    safe_path(name)
                entries = {name: archive.read(name) for name in names}
        except (BadZipFile, RuntimeError, OSError) as exc:
            raise ProjectError('invalid_archive', '无法读取工程压缩包。') from exc
        if 'manifest.json' not in entries or 'project.json' not in entries:
            raise ProjectError('unsupported_archive', '不是新格式 WebXAS 工程。')
        try:
            manifest = json.loads(entries.pop('manifest.json'))
            if manifest.get('format') != 'webxas-project' or manifest.get('schema_version') != 1:
                raise ProjectError('unsupported_version', '工程格式版本不受支持。')
            if set(manifest['files']) != set(entries):
                raise ProjectError('manifest_mismatch', '工程清单和文件不一致。')
            for name, blob in entries.items():
                record = manifest['files'][name]
                if len(blob) != record['byte_size'] or hashlib.sha256(blob).hexdigest() != record['sha256']:
                    raise ProjectError('asset_integrity', '工程文件哈希不符：' + name)
            project = Project.model_validate_json(entries['project.json'])
            refs = {ref['path']: ref for ref in asset_refs(project.model_dump(mode='json'))}
            if set(entries) != {'project.json', *refs}:
                raise ProjectError('unreferenced_asset', '工程资源引用与清单不一致。')
            for name, ref in refs.items():
                blob = entries[name]
                if len(blob) != ref['byte_size'] or hashlib.sha256(blob).hexdigest() != ref['sha256']:
                    raise ProjectError('asset_integrity', '工程引用的资源身份不符：' + name)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ProjectError):
                raise
            raise ProjectError('invalid_manifest', '工程清单或项目数据无效。') from exc
        # Import is always a new project. Existing work is never overwritten.
        old_id = project.id
        value = project.model_dump(mode='json')
        value['id'] = str(uuid4())
        value['metadata']['imported_project_id'] = old_id
        value['metadata']['imported_revision'] = project.revision
        project = Project.model_validate(value)
        folder = self.folder(project.id)
        for name in refs:
            atomic_write(folder.joinpath(*safe_path(name).parts), entries[name])
        return self.save(project)
