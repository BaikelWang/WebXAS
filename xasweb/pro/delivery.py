"""Research figures, exact source data, methods and a reopenable project ZIP.

The synchronous builder is read-only. The worker operation publishes one new
immutable delivery record after an optimistic revision check; it never submits
another job or changes analysis settings.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import re
from pathlib import PurePosixPath
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile, BadZipFile

import numpy as np

from .contracts import Project, AssetRef
from .figures import FigureStyle, FigureSpec, collect_figures, render_figure
from .figures.sources import FigureData
from .figures.individual import independent_panels, validate_presentation
from .figures.insights import figure_insight, column_units
from .projects import (MAX_ARCHIVE_BYTES, ProjectError, ProjectStore,
                       asset_refs, canonical)

MAX_DELIVERY_BYTES = 250 * 1024 * 1024


def safe_delivery_name(name):
    return (isinstance(name,str) and 0<len(name)<=300 and '\\' not in name and ':' not in name and
            not any(ord(c)<32 for c in name) and not name.startswith('/') and
            str(PurePosixPath(name))==name and all(p not in ('.','..') for p in name.split('/')))


def friendly_delivery_name(manifest, filename):
    """Friendly external names use only frozen manifest provenance, never live data."""
    def slug(value,limit=32):
        return re.sub(r'[^\w\-]+','-',str(value),flags=re.UNICODE).strip('-_')[:limit] or 'sample'
    owners=[f for f in manifest.get('figures',[]) if filename in [f.get('filename'),f.get('spec_file'),
             *f.get('formats',[]),*f.get('data_files',[])]]
    names=list(dict.fromkeys(name for f in owners for name in f.get('sample_names',[])))
    ids=list(dict.fromkeys(did for f in owners for did in f.get('dataset_ids',[])))
    label='-'.join(slug(PurePosixPath(n).stem,24) for n in names[:2]) or '科研交付'
    if len(names)>2:label+=f'-等{len(names)}谱'
    identity='-'.join(slug(did,10) for did in ids[:2]) or 'snapshot'
    base=PurePosixPath(filename)
    kind=(owners[0].get('panel_id') or owners[0].get('id',base.stem)) if owners else base.stem
    if base.name.endswith('-parameters.csv'):kind='fit-parameters'
    elif base.name.endswith('.metadata.json'):kind+='-data-metadata'
    elif filename.startswith('specs/'):kind+='-geometry' if base.name.endswith('.geometry.json') else '-spec'
    workflow=slug(manifest.get('workflow_id','workflow'),100)[-12:]
    return f'{label}_{identity}_{slug(kind,45)}_wf-{workflow}{base.suffix}'


def delivery_manifest(payload):
    """The outer AssetRef is verified by the caller; ZIP CRC is verified on read."""
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            names=archive.namelist()
            if len(names)!=len(set(names)):
                raise ValueError('duplicate archive member')
            if 'manifest.json' not in names:return None
            if archive.getinfo('manifest.json').file_size>16*1024*1024:raise ValueError('manifest too large')
            manifest=json.loads(archive.read('manifest.json'))
            if manifest.get('format')!='webxas-scientific-delivery' or manifest.get('version')!=1 or not isinstance(manifest.get('files'),dict):
                raise ValueError('unsupported manifest')
            if any(not safe_delivery_name(name) for name in manifest['files']):raise ValueError('unsafe manifest name')
            return manifest
    except (BadZipFile,ValueError,TypeError,KeyError,UnicodeDecodeError) as exc:
        raise ProjectError('delivery_integrity','图件包清单损坏或包含不安全文件名。') from exc


def read_delivery_file(payload, filename, *, max_bytes=MAX_DELIVERY_BYTES):
    if not safe_delivery_name(filename):
        raise ProjectError('delivery_file_not_available','文件名无效，只能下载清单内的文件。')
    manifest=delivery_manifest(payload)
    entry=manifest['files'].get(filename) if manifest else None
    if not isinstance(entry,dict):
        raise ProjectError('delivery_file_not_available','只能下载清单内的文件。')
    try:
        expected=entry['byte_size'];digest=entry['sha256']
        if isinstance(expected,bool) or not isinstance(expected,int) or expected<0 or expected>max_bytes:
            raise ValueError('invalid or oversized file')
        with ZipFile(io.BytesIO(payload)) as archive:
            info=archive.getinfo(filename)
            if info.is_dir() or info.file_size!=expected:raise ValueError('file size mismatch')
            blob=archive.read(info)
        if len(blob)!=expected or hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('file hash mismatch')
        return blob,entry
    except (BadZipFile,ValueError,TypeError,KeyError,RuntimeError) as exc:
        raise ProjectError('delivery_file_integrity','交付文件大小或 SHA256 不符合清单，未提供该文件。') from exc


def _individual_snapshot(store, project, workflow, style, difference_pair=None):
    """Reuse all frozen numerical panels, including initially unselected families."""
    fingerprint=hashlib.sha256(canonical(workflow)).hexdigest()
    for record in reversed(project.metadata.get('automation_deliveries',[])):
        if (record.get('workflow_id')!=workflow['id'] or record.get('presentation')!='individual' or
                record.get('difference_pair') != difference_pair):continue
        payload=store.read_asset(project.id,AssetRef.model_validate(record['asset']))
        manifest=delivery_manifest(payload)
        if not manifest or manifest.get('workflow_sha256')!=fingerprint or 'snapshot_figures' not in manifest:continue
        items=[]
        for frozen in manifest['snapshot_figures']:
            spec=FigureSpec.model_validate(json.loads(read_delivery_file(payload,frozen['spec_file'])[0]))
            blob=read_delivery_file(payload,frozen['arrays_file'])[0]
            with np.load(io.BytesIO(blob),allow_pickle=False) as arrays:
                values={name:arrays[name].copy() for name in arrays.files}
            transforms={**spec.transforms,'offset':style.offset,'display_xlim':style.xlim,'display_ylim':style.ylim}
            insight={**spec.insight,'method':{**spec.insight.get('method',{}),'display_transforms':transforms}}
            from .figures.presentation import brief_insight
            insight['display']=brief_insight(insight,spec.family,spec.method)
            spec=spec.model_copy(update={'style':style,'transforms':transforms,'insight':insight,
                'data_files':[f'data/{spec.id}.csv',f'data/{spec.id}.npz']})
            items.append(FigureData(spec,values))
        return items,manifest.get('snapshot_missing',manifest.get('missing',[]))
    return None


def _workflow(project, workflow_id):
    workflow=next((w for w in project.metadata.get('automation_runs',[]) if w.get('id')==workflow_id),None)
    if workflow is None:
        raise ProjectError('workflow_not_found','未找到所选自动分析运行。',workflow_id)
    if workflow.get('version') != 1:
        raise ProjectError('workflow_version','自动分析运行版本不受支持。',workflow_id)
    if workflow.get('state') in ('queued','running','cancelling'):
        raise ProjectError('workflow_in_progress','自动分析尚未完成，请完成后导出。',workflow_id)
    return workflow


def _csv(headers, rows):
    stream=io.StringIO(newline='')
    writer=csv.writer(stream)
    writer.writerow(headers)
    for row in rows:
        writer.writerow([format(float(x),'.17g') if isinstance(x,(float,np.floating)) else x for x in row])
    return stream.getvalue().encode('utf-8-sig')


def _figure_csv(item):
    if item.spec.kind=='heatmap':
        panel=item.spec.panels[0]
        xs,ys,z=[item.arrays[panel[n]] for n in ('x','y','z')]
        return _csv([panel['x'],panel['y'],panel['z']],
                    ((x,y,z[j,i]) for j,y in enumerate(ys) for i,x in enumerate(xs)))
    names=list(item.arrays)
    if any(np.ndim(item.arrays[n])!=1 for n in names):
        raise ProjectError('figure_array_shape','曲线源数据必须为一维数组。',item.spec.id)
    length=max((len(item.arrays[n]) for n in names),default=0)
    masks={}
    for panel in item.spec.panels:
        for name,mask in zip(panel.get('ys',[]),panel.get('valid_masks',[])):
            if mask:masks[name]=mask
        for name,mask in zip(panel.get('error_keys',[]),panel.get('error_valid_masks',[])):
            if mask:masks[name]=mask
        if panel.get('state_valid'):masks[panel['x']]=panel['state_valid']
    extra=[]
    if item.spec.kind=='trend':extra=['state_label','observation_status','observation_reason']
    if item.spec.kind=='geometry':extra=['element']
    def rows():
        for i in range(length):
            row=[item.arrays[n][i] if i<len(item.arrays[n]) and (n not in masks or item.arrays[masks[n]][i]) else '' for n in names]
            if item.spec.kind=='trend':
                point=item.spec.sources[0]['point_records'][i]
                row.extend([point.get('state_value'),point.get('state'),point.get('reason')])
            if item.spec.kind=='geometry':row.append(item.spec.method['atoms'][i]['element'])
            yield row
    return _csv([*names,*extra],rows())


def _individual_csv(item, workflow_id):
    """One panel's data, declared units and explicit immutable provenance."""
    reader=csv.reader(io.StringIO(_figure_csv(item).decode('utf-8-sig')))
    headers=next(reader)
    units=column_units(item.spec,item.arrays)
    source_hashes=sorted({s.get('snapshot_sha256') or s.get('processing_sha256') or
                          s.get('raw_sha256') for s in item.spec.sources} - {None})
    provenance=[workflow_id,item.spec.panel_id,json.dumps(item.spec.dataset_ids,ensure_ascii=False),
                json.dumps(item.spec.sample_names,ensure_ascii=False),';'.join(source_hashes)]
    labels=[f'{name} [{units.get(name,"not_declared")}]' for name in headers]
    return _csv(labels+['workflow_id','panel_id','dataset_ids','sample_names','source_snapshot_sha256'],
                (row+provenance for row in reader))


def _reopenable_project(store, project):
    # Delivery ZIPs are derived artifacts. Excluding older delivery bundles from
    # the embedded project prevents recursive ZIP growth on every style export.
    # All original data, processing revisions, models, runs and workflows remain.
    value=project.model_dump(mode='json')
    value['metadata'].pop('automation_deliveries',None)
    entries={'project.json':canonical(value)}
    for ref in asset_refs(value):
        if ref['path'] not in entries:
            entries[ref['path']]=store.read_asset(project.id,ref)
        if sum(map(len,entries.values()))>MAX_DELIVERY_BYTES:
            raise ProjectError('delivery_too_large','可重开工程已超过 250 MB；请拆分项目后导出。')
    manifest={'format':'webxas-project','schema_version':1,'files':{
        name:{'sha256':hashlib.sha256(blob).hexdigest(),'byte_size':len(blob)} for name,blob in entries.items()}}
    stream=io.BytesIO()
    with ZipFile(stream,'w',ZIP_DEFLATED, compresslevel=1) as archive:
        for name,blob in entries.items(): archive.writestr(name,blob)
        archive.writestr('manifest.json',canonical(manifest))
    payload=stream.getvalue()
    if len(payload)>MAX_ARCHIVE_BYTES:
        raise ProjectError('delivery_project_too_large','嵌入工程超过当前可导入的 50 MB 限制；请拆分工程，不能生成无法重开的交付包。')
    if store.read(project.id).revision!=project.revision:
        raise ProjectError('revision_conflict','构建交付工程时项目已更新，请重新导出。',project.id)
    return payload


def _methods(project,workflow,figures,missing,style):
    lines=['# 科研图件方法与适用条件 / Figure methods and conditions','',
           '本包读取已存处理与分析快照；修改版式不重新拟合、不改变处理数组。',
           'This package renders stored processing and analysis snapshots; display edits do not refit data.',
           '',f'工程：{project.title}；原工程版本：{project.revision}。',
           f'版式 {style.layout}，标称宽度 {style.size_mm[0]:g} mm；栅格 {style.dpi} dpi。',
           'SVG/PDF 曲线与文字为矢量；密集热图在矢量文件中栅格化。图像不自动裁边，因此画布物理尺寸固定。',
           'R 坐标未相位校正，不直接等同于键长。WT 不独立确定散射元素或结构。',
           'LCF 报告参考组分系数；含量/价态解释依赖参考覆盖与适用条件。',
           '误差、收敛及科学判断沿用原运行；未运行的分析、未声明的实验条件不作补充推断。',
           'The embedded project retains all scientific inputs and results. Earlier derived delivery ZIPs are excluded to prevent recursive archive growth.',
           '', '## 运行判断 / Workflow assessment', '```json',
           json.dumps({key:workflow.get(key) for key in ('task','state','decisions','issues')},ensure_ascii=False,indent=2), '```','']
    for item in figures:
        spec=item.spec
        basis=spec.insight.get('method_basis') or {}
        lines.extend([f'## {spec.id} · {spec.title}', '', spec.title_en, '',
                      '数据 / Source data: '+', '.join(spec.data_files),
                      '展示变换 / Display transforms: '+json.dumps(spec.transforms,ensure_ascii=False),
                      '实际设置 / Recorded method: '+json.dumps(spec.method,ensure_ascii=False,sort_keys=True),
                      *([basis['summary']] if basis else []),
                      *(f'- [{ref["label"]}]({ref["url"]})' for ref in basis.get('references', [])),
                      *spec.notes,''])
    lines.extend(['## 缺项 / Omitted figures',''])
    lines.extend(f'- {m["family"]}: {m["reason"]}' for m in missing)
    lines.extend(['','## 软件与方法引用 / Software and method references',
                  '- Matplotlib savefig: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html',
                  '- Larch Cauchy WT: https://xraypy.github.io/xraylarch/xafs_wavelets.html',
                  '- Muñoz, Argoul & Farges (2003), American Mineralogist 88, 694–700.（仅适用于实际生成的 WT）',
                  '- 具体预处理/拟合版本及调用记录见 project/processing-snapshots.json 和 project/fit-runs.json。'])
    return '\n'.join(lines).encode('utf-8')


def _readme(project, workflow, records, missing):
    esc=lambda x:html.escape(str(x),quote=True)
    sections=[]
    for fig in records:
        links=' · '.join(f'<a href="{esc(path)}">{esc(path.rsplit(".",1)[-1].upper())}</a>' for path in fig['formats'])
        data=' · '.join(f'<a href="{esc(path)}">{esc(path)}</a>' for path in fig['data_files'])
        insight=fig.get('insight') or {}
        from .figures.presentation import brief_insight
        display=brief_insight(insight,fig['family'])
        observations=esc(display['summary'])
        metrics=''.join(f'<tr><th>{esc(m["label"])}</th><td>{esc(m["value"])} {esc(m["unit"])}</td><td>{esc(m["basis"])}</td></tr>'
                        for m in display['metrics'])
        limits=''.join(f'<li>{esc(text)}</li>' for text in display['notices'])
        basis=insight.get('method_basis') or {}
        references=' · '.join(f'<a href="{esc(ref["url"])}">{esc(ref["label"])}</a>' for ref in basis.get('references', [])
                              if ref.get('url','').startswith('https://'))
        sections.append(f'<section><h2>{esc(fig["id"])} · {esc(fig["title"])}</h2>'
                        f'<img src="{esc(fig["filename"])}" alt="{esc(fig["title"])}">'
                        f'<p>{links}</p><p>{data}</p><ul>{observations}</ul><table>{metrics}</table>'
                        f'<ul>{limits}</ul><details><summary>方法与来源</summary><p>{esc(basis.get("summary", ""))}</p><p>{references}</p>'
                        f'<pre>{esc(json.dumps(insight,ensure_ascii=False,indent=2))}</pre></details>'
                        f'<a href="{esc(fig["spec_file"])}">FigureSpec</a></section>')
    omitted=''.join(f'<li>{esc(m["family"])}: {esc(m["reason"])}</li>' for m in missing)
    kinds={s.get('data_kind','not_declared') for f in records for s in f['sources'] if 'dataset_id' in s}
    provenance='包含合成数据，仅用于方法或工程验证。' if 'synthetic' in kinds or 'same_engine_synthetic' in kinds else '数据的实验来源尚未全部声明，请查看工程来源。' if 'not_declared' in kinds else '工程声明为实验数据；声明本身不替代原始测量审核。'
    return (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(project.title)} · WebXAS 科研图件</title><style>'
            'body{font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#243244}'
            'h1,h2{line-height:1.3}section{border-top:1px solid #ddd;padding:24px 0}img{max-width:100%;height:auto;background:white}'
            'a{color:#076c9c}p{overflow-wrap:anywhere}</style>'
            f'<h1>{esc(project.title)} · 科研图件</h1><p>工作流状态：{esc(workflow.get("state"))}。'
            '图件展示已存数据和分析条件；不自动证明唯一结构、含量、价态或机理。</p>'
            f'<p>来源：{esc(provenance)}</p>'
            '<p><a href="project/project.webxas.zip">可重开工程</a> · <a href="methods/methods.md">中英文方法与图注</a> · '
            '<a href="manifest.json">文件哈希与来源清单</a></p>'+''.join(sections)+
            f'<h2>未生成图件与原因</h2><ul>{omitted}</ul></html>').encode('utf-8')


def render_batch(items):
    """Keep the render pool inside the disposable job's process group."""
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import contextmanager
    import multiprocessing
    import os
    import sys

    @contextmanager
    def pool_results():
        allowed = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 1
        default = '4' if sys.platform.startswith('linux') else '1'
        limit = int(os.environ.get('WEBXAS_RENDER_WORKERS', default))
        workers = min(len(items), allowed, max(1, min(4, limit)))
        if workers < 2:
            yield map(render_figure, items)
            return
        method = 'spawn'
        if sys.platform.startswith('linux') and os.environ.get('WEBXAS_DISPOSABLE_JOB') == '1':
            # The worker may already own native scientific-library threads. A
            # clean forkserver shares the read-only font and backend imports
            # without forking a thread-owning application or a live job.
            method = 'forkserver'
            multiprocessing.set_forkserver_preload([
                'matplotlib.pyplot', 'matplotlib.backends.backend_pdf',
                'matplotlib.backends.backend_svg',
            ])
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context(method)) as pool:
            yield pool.map(render_figure, items, chunksize=1)

    return pool_results()


def build_delivery(store, project_id, workflow_id, style=None, presentation='legacy', figure_families=None, difference_pair=None):
    """Return a complete ZIP without mutating projects or launching analysis."""
    project=store.read(project_id)
    workflow=_workflow(project,workflow_id)
    validate_presentation(presentation,figure_families)
    try: style=FigureStyle.model_validate(style or {})
    except ValueError as exc:
        raise ProjectError('invalid_figure_style','图件展示设置无效。',details={'reason':str(exc)}) from exc
    if presentation=='individual':
        style=style.model_copy(update={'formats':['svg','pdf','png','tiff']})
    entries={}
    total=0
    def put(name,payload):
        nonlocal total
        if name in entries:
            raise ProjectError('duplicate_figure_file','交付包包含重复文件名。',name)
        total+=len(payload)
        if total>MAX_DELIVERY_BYTES:
            raise ProjectError('delivery_too_large','图件、数据与工程的展开总体积超过 250 MB；请拆分工作流或调整分辨率后重试。')
        entries[name]=payload
    put('project/project.webxas.zip',_reopenable_project(store,project))
    cached=_individual_snapshot(store,project,workflow,style,difference_pair) if presentation=='individual' else None
    if cached is not None:
        items,missing=cached
    else:
        items,missing=collect_figures(store,project,workflow,style,difference_pair)
        if presentation=='individual':
            items=independent_panels(items)
            missing=[m for m in missing if m['family']!='F12']
            missing.append({'family':'F12','reason':'独立图交付不生成总览拼图；各图均从各自数组单独渲染。','code':'individual_presentation'})
        for item in items:
            insight=figure_insight(project,workflow,item)
            item.spec=item.spec.model_copy(update={'dataset_ids':insight['dataset_ids'],
                                                 'sample_names':insight['sample_names'],'insight':insight})
    snapshot_figures=[]
    snapshot_missing=missing
    if presentation=='individual':
        for item in items:
            spec_file=f'snapshots/figures/{item.spec.id}.json'
            arrays_file=f'snapshots/figures/{item.spec.id}.npz'
            put(spec_file,canonical(item.spec.model_dump(mode='json')))
            buffer=io.BytesIO();np.savez_compressed(buffer,**item.arrays)
            put(arrays_file,buffer.getvalue())
            snapshot_figures.append({'id':item.spec.id,'family':item.spec.family,
                                     'spec_file':spec_file,'arrays_file':arrays_file})
    if figure_families is not None:
        items=[item for item in items if item.spec.family in figure_families]
        missing=[m for m in missing if m['family'] in figure_families]
    if not items and presentation=='legacy':
        raise ProjectError('figures_unavailable','当前工作流没有可绘制的有效源数组。',workflow_id,{'missing':missing})
    records=[]
    with render_batch(items) as results:
        for item, (rendered,geometry) in zip(items, results, strict=True):
            if item.spec.kind!='composite':
                put(item.spec.data_files[0],_individual_csv(item,workflow_id) if presentation=='individual' else _figure_csv(item))
                if item.spec.kind=='heatmap' or presentation=='individual':
                    array_file=f'data/{item.spec.id}.npz'
                    if presentation == 'individual':
                        # These are the same full-precision arrays frozen just above.
                        data = entries[f'snapshots/figures/{item.spec.id}.npz']
                    else:
                        buffer = io.BytesIO()
                        np.savez_compressed(buffer, **item.arrays)
                        data = buffer.getvalue()
                    put(array_file, data)
                    if array_file not in item.spec.data_files:
                        item.spec=item.spec.model_copy(update={'data_files':[*item.spec.data_files,array_file]})
                if presentation=='individual':
                    metadata_name=f'data/{item.spec.id}.metadata.json'
                    put(metadata_name,canonical({'workflow_id':workflow_id,'panel_id':item.spec.panel_id,
                        'dataset_ids':item.spec.dataset_ids,'sample_names':item.spec.sample_names,
                        'column_units':column_units(item.spec,item.arrays),'sources':item.spec.sources,
                        'method':item.spec.method,'display_transforms':item.spec.transforms}))
                    item.spec=item.spec.model_copy(update={'data_files':[*item.spec.data_files,metadata_name]})
            put(f'specs/{item.spec.id}.json',canonical(item.spec.model_dump(mode='json')))
            for name,blob in rendered.items(): put(name,blob)
            records.append({'id':item.spec.id,'family':item.spec.family,
                            'panel_id':item.spec.panel_id,'parent_figure_id':item.spec.parent_figure_id,
                            'panel_index':item.spec.panel_index,'dataset_ids':item.spec.dataset_ids,
                            'sample_names':item.spec.sample_names,'insight':item.spec.insight,
                            'title':item.spec.title if style.language=='zh' else item.spec.title_en,
                            'filename':f'figures/{item.spec.id}.png','spec_file':f'specs/{item.spec.id}.json',
                            'data_files':item.spec.data_files,'sources':item.spec.sources,
                            'formats':[n for n in rendered if n.startswith('figures/')],
                            'width_mm':geometry['width_mm'],'height_mm':geometry['height_mm'],
                            'rasterized_heatmap':geometry['rasterized_heatmap']})
        put('project/workflow.json',canonical(workflow))
        selected=set(workflow.get('processing_ids',[]))
        put('project/processing-snapshots.json',canonical([p.model_dump(mode='json') for p in project.processing if p.id in selected]))
        selected_runs=set(workflow.get('fit_run_ids',[]))
        put('project/fit-runs.json',canonical([r.model_dump(mode='json') for r in project.runs if r.id in selected_runs]))
        put('tables/decisions.json',canonical(workflow.get('decisions',[])))
        put('tables/issues.json',canonical(workflow.get('issues',[])))
        put('tables/features.json',canonical(workflow.get('features',[])))
        put('tables/lcf.json',canonical(workflow.get('lcf_results',[])))
    for index,lcf in enumerate(workflow.get('lcf_results',[])):
        put(f'tables/lcf-{index:03d}.csv',_csv(['dataset_id','reference_id','reference_component_coefficient'],
            ((lcf['dataset_id'],rid,c) for rid,c in zip(lcf.get('reference_ids',[]),lcf.get('coefficients',[])))))
    for fit_index,rid in enumerate(sorted(selected_runs)):
        run=next(r for r in project.runs if r.id==rid)
        if run.result:
            parameter_name=f'tables/fit-{hashlib.sha256(rid.encode()).hexdigest()[:16]}-parameters.csv'
            declarations={p.id:p for p in run.input_snapshot.model.parameters.nodes}
            put(parameter_name,_csv(['run_id','parameter_id','value','unit','stderr','uncertainty_basis','uncertainty_reason','scope','owner_id'],
                ((rid,p.id,p.value,p.unit,p.stderr if p.stderr is not None else '',run.result.uncertainty_basis,
                  p.uncertainty_reason or '',declarations[p.id].scope if p.id in declarations else 'not_declared',
                  declarations[p.id].owner_id or '' if p.id in declarations else '') for p in run.result.parameters)))
            for record in records:
                if any(s.get('run_id')==rid for s in record['sources']):
                    record.setdefault('parameter_tables',[]).append(parameter_name)
                    record['data_files']=[*record['data_files'],parameter_name]
            put(f'tables/fit-{fit_index:03d}.json',canonical({
                'run_id':rid,'parameters':[p.model_dump(mode='json') for p in run.result.parameters],
                'statistics':run.result.statistics.model_dump(mode='json'),
                'uncertainty_basis':run.result.uncertainty_basis,'converged':run.result.converged,
                'warnings':run.result.warnings}))
    from pathlib import Path
    method_page=(Path(__file__).parents[1] / 'static' / 'methods.html').read_text(encoding='utf-8')
    method_page=method_page.replace('<script src="/static/methods-examples.js" defer></script>', '').replace('href="/auto"', 'href="../README.html"').replace('分析工作台 ↗','离线报告 ↗')
    put('methods/algorithms.html',method_page.encode('utf-8'))
    if workflow.get('decomposition'):
        put('tables/decomposition.json', store.read_asset(project.id, workflow['decomposition']))
    put('methods/methods.md',_methods(project,workflow,items,missing,style))
    put('README.html',_readme(project,workflow,records,missing))
    manifest={'format':'webxas-scientific-delivery','version':1,'project_id':project.id,
              'presentation':presentation,'figure_families':figure_families,'difference_pair':difference_pair,
              'snapshot_figures':snapshot_figures,'snapshot_missing':snapshot_missing,
              'source_project_revision':project.revision,'workflow_id':workflow_id,
              'workflow_sha256':hashlib.sha256(canonical(workflow)).hexdigest(),
              'style':style.model_dump(mode='json'),'figures':records,'missing':missing,
              'expanded_byte_size':total,'derived_archives_excluded_from_embedded_project':True,
              'files':{n:{'sha256':hashlib.sha256(b).hexdigest(),'byte_size':len(b)} for n,b in entries.items()}}
    # The manifest cannot contain its own content hash; all other entries have
    # exact hashes and the outer stored AssetRef covers the entire ZIP.
    put('manifest.json',canonical(manifest))
    if store.read(project.id).revision!=project.revision:
        raise ProjectError('revision_conflict','渲染图件期间项目已更新，请重新导出。',project.id)
    stream=io.BytesIO()
    with ZipFile(stream,'w',ZIP_DEFLATED, compresslevel=1) as archive:
        for name,blob in entries.items(): archive.writestr(name,blob)
    if len(stream.getvalue())>MAX_DELIVERY_BYTES:
        raise ProjectError('delivery_too_large','压缩交付包超过 250 MB，未发布不完整包。')
    return stream.getvalue()


def run_job(payload, context):
    """Worker target for auto_export; append a new immutable delivery record."""
    store=ProjectStore()
    project_id=payload['project_id']
    project=store.read(project_id)
    expected=payload['expected_revision']
    if project.revision!=expected:
        raise ProjectError('revision_conflict','任务提交后项目已更新。',project_id)
    context.heartbeat('scientific_figures',.1)
    blob=build_delivery(store,project_id,payload['workflow_id'],payload.get('style'),
                        payload.get('presentation','legacy'),payload.get('figure_families'),payload.get('difference_pair'))
    context.heartbeat('scientific_figures',.9)
    with ZipFile(io.BytesIO(blob)) as archive: manifest=json.loads(archive.read('manifest.json'))
    if manifest['source_project_revision']!=expected:
        raise ProjectError('revision_conflict','任务渲染期间读取了其他项目版本。',project_id)
    asset=store.put_asset(project_id,blob,'scientific-delivery.zip')
    record={'id':'delivery_'+uuid4().hex,'version':1,'workflow_id':payload['workflow_id'],
            'source_project_revision':expected,'workflow_sha256':manifest['workflow_sha256'],
            'created_at':datetime.now(timezone.utc).isoformat(),'asset':asset.model_dump(mode='json'),
            'figures':manifest['figures'],'missing':manifest['missing'],'style':manifest['style'],
            'presentation':manifest.get('presentation','legacy'),'figure_families':manifest.get('figure_families'),
            'difference_pair':manifest.get('difference_pair')}
    value=project.model_dump(mode='json')
    value['metadata'].setdefault('automation_deliveries',[]).append(record)
    store.save(Project.model_validate(value),expected_revision=expected)
    context.heartbeat('scientific_figures',1.)
    return record
