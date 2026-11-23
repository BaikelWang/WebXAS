"""Explicit migration of old sessions without inventing missing fit models."""
import base64
import json
from uuid import uuid4

from .contracts import Dataset, MigrationIssue, Project
from .projects import MAX_ARCHIVE_BYTES, ProjectError


def import_session(store, payload):
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ProjectError('archive_too_large', '旧会话文件超过大小限制。')
    try:
        saved = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProjectError('invalid_session', '无法读取旧会话 JSON。') from exc
    if not isinstance(saved, dict) or saved.get('kind') != 'webxas-session' or saved.get('schema_version') != 3:
        raise ProjectError('unsupported_version', '只支持明确标识的 schema 3 旧会话。')
    spectra = saved.get('spectra')
    if not isinstance(spectra, list) or not spectra:
        raise ProjectError('empty_session', '旧会话未包含可恢复的光谱对象。')
    project = Project(title='导入的旧会话', recomputable=False)
    originals = []
    try:
        for item in saved.get('sources', []):
            if item.get('encoding') != 'base64':
                raise ProjectError('unsupported_encoding', '旧会话原件编码不受支持。')
            blob = base64.b64decode(item['data'], validate=True)
            ref = store.put_asset(project.id, blob, item.get('name', 'source.bin'))
            originals.append({'filename':item.get('name', ''), 'asset':ref.model_dump(mode='json')})
        datasets = []
        for index, item in enumerate(spectra):
            raw = item.get('original_raw_text')
            if not isinstance(raw, str) or not raw.strip():
                raise ProjectError('missing_raw_data', f'第 {index+1} 个谱缺少原始数值文本。')
            filename = str(item.get('filename') or f'spectrum-{index+1}.dat')
            ref = store.put_asset(project.id, raw.encode('utf-8'), filename)
            datasets.append(Dataset(id='legacy_' + str(index+1), filename=filename,
                raw_file=ref, raw_text=raw, kind='mu',
                metadata={'legacy_context':item.get('context', {}), 'legacy_manual_context':item.get('manual_context', {}),
                          'legacy_result':item.get('result'), 'legacy_selected_fit_id':item.get('selected_fit_id'),
                          'legacy_last_error':item.get('last_error')}))
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ProjectError):
            raise
        raise ProjectError('invalid_session', '旧会话数据或原件不完整。') from exc
    value = project.model_dump(mode='json')
    value['datasets'] = [d.model_dump(mode='json') for d in datasets]
    value['metadata'] = {'legacy_kind':'webxas-session', 'legacy_schema':3, 'original_sources':originals}
    value['migration_report'] = [MigrationIssue(code='fit_model_missing',
        message='已恢复原始谱和历史结果；旧会话缺少新专业模型、路径绑定及标定合同，需要重新建立模型。').model_dump(mode='json')]
    return store.save(Project.model_validate(value))


def import_old_bundle(store, payload):
    from ..export import read_bundle
    from ..ingest import InputError
    try:
        request, saved = read_bundle(payload)
    except (InputError, ValueError, KeyError) as exc:
        raise ProjectError('invalid_legacy_bundle', '旧 ZIP 校验失败：' + str(exc)) from exc
    project = Project(title=request.sample or request.filename, recomputable=False)
    ref = store.put_asset(project.id, request.raw_text.encode('utf-8'), request.filename)
    dataset = Dataset(id='legacy_1', filename=request.filename, raw_file=ref, raw_text=request.raw_text,
                      mapping=request.mapping.model_dump(mode='json'),
                      metadata={'legacy_recipe':request.model_dump(mode='json',exclude={'raw_text'}), 'legacy_result':saved})
    value = project.model_dump(mode='json')
    value['datasets'] = [dataset.model_dump(mode='json')]
    value['migration_report'] = [MigrationIssue(code='legacy_processing_only',
        message='旧 ZIP 的处理配方与数组已保存；专业拟合路径和参数图需要重新建立。').model_dump(mode='json')]
    return store.save(Project.model_validate(value))
