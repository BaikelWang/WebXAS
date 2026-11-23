"""Read-only reports and comparison of immutable professional FitRun records."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
from pathlib import PurePosixPath
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np

from .preprocessing import load_array
from .projects import ProjectError,canonical

MAX_REPORT_BYTES=250*1024*1024


def _run(project,run_id):
    run=next((r for r in project.runs if r.id==run_id),None)
    if run is None:
        raise ProjectError('run_not_found','项目中没有所选运行。',run_id)
    return run


def _provenance(dataset):
    metadata=dataset.metadata
    if metadata.get('synthetic') is True or metadata.get('data_kind') in ('synthetic','same_engine_synthetic'):
        return 'synthetic'
    if metadata.get('data_kind') in ('experimental','real_experimental') or metadata.get('synthetic') is False:
        return 'experimental'
    return 'not_declared'


def _basis(run):
    """Comparison basis excludes model paths/parameters deliberately."""
    spec=run.input_snapshot
    datasets={d.id:d for d in spec.datasets}
    processing={p.id:p for p in spec.processing}
    calibrations={c.id:c.model_dump(mode='json') for c in spec.calibrations}
    entries=[]
    for binding in sorted(spec.model.bindings,key=lambda b:b.dataset_id):
        dataset=datasets.get(binding.dataset_id)
        recipe=processing.get(binding.processing_revision_id)
        if dataset is None or recipe is None or recipe.dataset_id!=binding.dataset_id:
            raise ProjectError('incomplete_snapshot','运行快照缺少所绑定的数据或处理版本。',run.id)
        entries.append({'dataset_id':dataset.id,'raw_sha256':dataset.raw_file.sha256,
            'kind':dataset.kind,'mapping':dataset.mapping.model_dump(mode='json'),
            'element':dataset.element,'edge':dataset.edge,'sample_id':dataset.sample_id,
            'conditions':dataset.conditions,'corrections':dataset.correction_states,
            'lineage':sorted(dataset.lineage),'dataset_mask':dataset.mask.model_dump(mode='json') if dataset.mask else None,
            'processing_revision':recipe.model_dump(mode='json'),
            'calibration':calibrations.get(recipe.calibration_id),
            'transform':binding.transform.model_dump(mode='json'),
            'noise':(recipe.noise if binding.noise.source=='unknown' else binding.noise).model_dump(mode='json')})
    return {'datasets':entries,'engine_versions':spec.engine_versions,'source_sha256':spec.source_sha256,
        'statistics_definition':run.result.statistics.definition if run.result else None,
        'nind_convention':run.result.statistics.nind_convention if run.result else None}


def _differences(a,b,prefix=''):
    if isinstance(a,dict) and isinstance(b,dict):
        output=[]
        for key in sorted(set(a)|set(b)):
            path=f'{prefix}.{key}' if prefix else key
            if key not in a or key not in b:output.append(path)
            else:output.extend(_differences(a[key],b[key],path))
        return output
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):return [prefix+'.length']
        return [x for i,(left,right) in enumerate(zip(a,b)) for x in _differences(left,right,f'{prefix}[{i}]')]
    return [] if a==b else [prefix]


def compare_runs(project,run_ids):
    if len(run_ids)<2 or len(run_ids)!=len(set(run_ids)):
        raise ProjectError('invalid_comparison','至少选择两个不同运行。')
    runs=[_run(project,rid) for rid in run_ids]
    reasons=[];differences={};bases={}
    summaries=[]
    for run in runs:
        result=run.result
        summaries.append({'run_id':run.id,'model_id':run.model_id,'task_state':run.task_state,
            'assessment':result.assessment if result else None,'converged':result.converged if result else False,
            'statistics':result.statistics.model_dump(mode='json') if result else None,
            'warnings':result.warnings if result else []})
        if run.task_state!='succeeded' or result is None or not result.converged:
            reasons.append({'run_id':run.id,'code':'run_not_converged','message':'运行未成功收敛，不能直接统计比较。'})
        try:
            bases[run.id]=_basis(run)
        except ProjectError as exc:
            reasons.append({'run_id':run.id,'code':exc.code,'message':exc.message})
        processing={p.id:p for p in run.input_snapshot.processing}
        if any(b.noise.source=='unknown' and (b.processing_revision_id not in processing or processing[b.processing_revision_id].noise.source=='unknown') for b in run.input_snapshot.model.bindings):
            reasons.append({'run_id':run.id,'code':'unknown_noise','message':'未声明噪声口径，不能确认统计比较条件。'})
    first=runs[0].id
    for run in runs[1:]:
        if first in bases and run.id in bases:
            changed=_differences(bases[first],bases[run.id])
            if changed:
                differences[run.id]=changed
                reasons.append({'run_id':run.id,'code':'different_analysis_basis','message':'数据、处理、掩码、噪声、变换、校准或引擎统计口径不同。'})
    comparable=not reasons
    return {'comparability':comparable,'reference_run_id':first,'run_ids':list(run_ids),
        'runs':summaries,'differences':differences,'reasons':reasons,
        'assessment':'review' if comparable else 'not_comparable','unique_structure':False,
        'conclusion':'同条件下可以审查失配、参数和稳定性；这些指标本身不足以确定唯一结构。' if comparable else '分析条件不同或结果不完整，不能直接按统计指标排名。',
        'comparison_basis_sha256':{rid:hashlib.sha256(canonical(value)).hexdigest() for rid,value in bases.items()}}


def _csv(headers,rows):
    stream=io.StringIO(newline='')
    writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows)
    return stream.getvalue().encode('utf-8-sig')


def _matrix_csv(names,matrix):
    return _csv(['parameter',*names],[[name,*row] for name,row in zip(names,matrix)])


def _table(headers,rows):
    def cell(v):return html.escape('' if v is None else str(v))
    return '<table><thead><tr>'+''.join('<th>'+cell(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def _plot(fig,entries,name):
    import matplotlib.pyplot as plt
    try:
        for extension,kwargs in [('png',{'dpi':160}),('svg',{})]:
            stream=io.BytesIO()
            fig.savefig(stream,format=extension,bbox_inches='tight',metadata={'Description':name} if extension=='svg' else {'Description':name},**kwargs)
            entries[name+'.'+extension]=stream.getvalue()
    finally:plt.close(fig)


def _plots(store,project_id,run,entries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=[];missing=[]
    for result in run.result.datasets:
        values={name:load_array(store,project_id,ref) for name,ref in result.arrays.items()}
        stem=f'figures/{result.dataset_id}'
        title=f'Run {run.id}\nDataset {result.dataset_id}'
        if {'k','data_chi','model_chi'}<=values.keys():
            k,data,model=[np.asarray(values[x]) for x in ('k','data_chi','model_chi')]
            if k.ndim!=1 or data.shape!=k.shape or model.shape!=k.shape:
                raise ProjectError('report_array_shape','k 空间结果数组形状不一致。',result.dataset_id)
            fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,gridspec_kw={'height_ratios':[3,1]},layout='constrained')
            axes[0].plot(k,data,label='data',lw=1.3);axes[0].plot(k,model,label='model',lw=1.2)
            axes[0].set_ylabel('chi(k)');axes[0].legend();axes[0].set_title(title)
            axes[1].plot(k,data-model,label='data - model',lw=1)
            axes[1].axhline(0,color='0.5',lw=.6);axes[1].set_ylabel('residual');axes[1].set_xlabel('k (1/Angstrom)')
            _plot(fig,entries,stem+'-k');figures.append((result.dataset_id+' k 空间拟合与残差',stem+'-k.png'))
            entries[f'arrays/{result.dataset_id}-k.csv']=_csv(['k_inv_A','data_chi','model_chi','residual_chi'],zip(k,data,model,data-model))
        else:missing.append(result.dataset_id+' 的 k 空间数组不完整')
        required={'r','data_re','data_im','model_re','model_im'}
        if required<=values.keys():
            r,dr,di,mr,mi=[np.asarray(values[x]) for x in ('r','data_re','data_im','model_re','model_im')]
            if r.ndim!=1 or any(a.shape!=r.shape for a in [dr,di,mr,mi]):
                raise ProjectError('report_array_shape','R 空间结果数组形状不一致。',result.dataset_id)
            fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True,layout='constrained')
            axes[0].plot(r,dr,label='data real');axes[0].plot(r,mr,label='model real');axes[0].legend();axes[0].set_title(title)
            axes[1].plot(r,di,label='data imaginary');axes[1].plot(r,mi,label='model imaginary');axes[1].legend()
            axes[2].plot(r,dr-mr,label='real residual');axes[2].plot(r,di-mi,label='imaginary residual');axes[2].legend()
            for axis in axes:axis.set_ylabel('chi(R)')
            axes[2].set_xlabel('R (Angstrom, not phase corrected)')
            _plot(fig,entries,stem+'-r');figures.append((result.dataset_id+' R 实虚部与残差',stem+'-r.png'))
            entries[f'arrays/{result.dataset_id}-r.csv']=_csv(['R_A','data_re','data_im','model_re','model_im','residual_re','residual_im'],zip(r,dr,di,mr,mi,dr-mr,di-mi))
        else:missing.append(result.dataset_id+' 的 R 空间实虚部数组不完整')
        for binding_id,refs in result.path_contributions.items():
            if not {'k','chi'}<=refs.keys():
                missing.append(result.dataset_id+'/'+binding_id+' 的逐路径数组不完整');continue
            k=load_array(store,project_id,refs['k']);chi=load_array(store,project_id,refs['chi'])
            if k.ndim!=1 or chi.shape!=k.shape:
                raise ProjectError('report_array_shape','逐路径数组形状不一致。',binding_id)
            fig,axis=plt.subplots(figsize=(9,4),layout='constrained')
            axis.plot(k,chi,lw=1.2);axis.set_xlabel('k (1/Angstrom)');axis.set_ylabel('path chi(k)')
            axis.set_title(title+'\nPath binding '+binding_id)
            filename=f'{stem}-path-{binding_id}'
            _plot(fig,entries,filename);figures.append((result.dataset_id+' 路径 '+binding_id,filename+'.png'))
            entries[f'arrays/{result.dataset_id}-path-{binding_id}.csv']=_csv(['k_inv_A','path_chi'],zip(k,chi))
        if sum(map(len,entries.values()))>MAX_REPORT_BYTES:
            raise ProjectError('report_too_large','报告资源超过 250 MB，请缩小导出范围。')
    return figures,missing


def build_report(store,project_id,run_id):
    """Export stored results without fitting, storing assets, or modifying a project."""
    project=store.read(project_id)
    run=_run(project,run_id)
    if run.result is None:
        raise ProjectError('result_unavailable','此运行没有可导出的拟合结果。',run_id)
    spec=run.input_snapshot
    snapshot=spec.model_dump(mode='json')
    digest=hashlib.sha256(canonical(snapshot)).hexdigest()
    if digest!=run.snapshot_sha256:
        raise ProjectError('snapshot_integrity','运行输入快照哈希与记录不符。',run_id)
    bindings={b.dataset_id:b for b in spec.model.bindings}
    seen=set()
    for dataset_result in run.result.datasets:
        binding=bindings.get(dataset_result.dataset_id)
        if binding is None or dataset_result.dataset_id in seen or binding.processing_revision_id!=dataset_result.processing_revision_id:
            raise ProjectError('result_snapshot_mismatch','结果数组的数据集或处理版本与运行快照不符。',dataset_result.dataset_id)
        seen.add(dataset_result.dataset_id)
    # Freeze the matching project archive before image rendering. An edit while
    # this capture is made produces a revision conflict, not mixed-version data.
    project_zip=store.export(project_id,expected_revision=project.revision)
    entries={'project.webxas.zip':project_zip,'run.json':canonical(run.model_dump(mode='json')),
             'input-snapshot.json':canonical(snapshot),'result.json':canonical(run.result.model_dump(mode='json'))}
    declarations={p.id:p for p in spec.model.parameters.nodes}
    parameter_rows=[]
    for estimate in run.result.parameters:
        declaration=declarations.get(estimate.id)
        parameter_rows.append([estimate.id,declaration.label if declaration else '',
            declaration.scope if declaration else '',declaration.owner_id if declaration else '',
            declaration.kind if declaration else '',estimate.value,estimate.unit,estimate.stderr,
            estimate.uncertainty_reason,estimate.at_bound,declaration.physical_basis if declaration else ''])
    entries['parameters.csv']=_csv(['id','label','scope','owner_id','kind','value','unit','stderr','uncertainty_reason','at_bound','physical_basis'],parameter_rows)
    entries['parameters.json']=canonical({'run_id':run.id,'estimates':[p.model_dump(mode='json') for p in run.result.parameters],
        'declarations':spec.model.parameters.model_dump(mode='json'),'uncertainty_basis':run.result.uncertainty_basis})
    path_assets={p.id:p for p in spec.paths}
    path_rows=[]
    for binding in spec.model.bindings:
        for path in binding.paths:
            asset=path_assets.get(path.path_id)
            if asset is None:
                raise ProjectError('incomplete_snapshot','运行快照缺少路径资产。',path.path_id)
            store.read_asset(project_id,asset.source_file)
            path_rows.append([binding.dataset_id,path.id,asset.id,path.enabled,asset.absorber_element,asset.edge,
                asset.absorber_site,asset.nleg,asset.reff,asset.degen,asset.source_file.sha256,
                path.amplitude_semantics,path.cn_group_id,json.dumps(path.expressions,ensure_ascii=False,sort_keys=True),path.geometry_basis])
    entries['paths.csv']=_csv(['dataset_id','binding_id','path_id','enabled','absorber','edge','site','nleg','effective_half_path_A','degeneracy','source_sha256','amplitude_semantics','cn_group_id','expressions','geometry_basis'],path_rows)
    warnings=list(run.result.warnings)
    covariance=run.result.covariance
    if covariance is not None:
        entries['covariance.json']=canonical(covariance.model_dump(mode='json'))
        if covariance.matrix is not None:
            cov=np.asarray(covariance.matrix,dtype=float)
            entries['covariance.csv']=_matrix_csv(covariance.var_names,cov)
            diagonal=np.diag(cov)
            positive=diagonal>0
            denominator=np.sqrt(np.outer(np.where(positive,diagonal,0),np.where(positive,diagonal,0)))
            correlation=np.full_like(cov,np.nan)
            np.divide(cov,denominator,out=correlation,where=denominator>0)
            if not np.allclose(cov,cov.T,rtol=1e-7,atol=1e-12):
                warnings.append('协方差不对称，相关矩阵不可解释。');correlation[:]=np.nan
            invalid=np.abs(correlation)>1+1e-7
            if invalid.any():warnings.append('协方差导出的部分相关系数超出有效范围，已保留为空值。')
            rows=[[float(np.clip(v,-1,1)) if np.isfinite(v) and not bad else None for v,bad in zip(row,badrow)] for row,badrow in zip(correlation,invalid)]
            entries['correlation.csv']=_matrix_csv(covariance.var_names,rows)
        else:warnings.append('协方差不可用：'+str(covariance.missing_reason))
    else:warnings.append('运行未提供协方差。')
    provenance=[{'dataset_id':d.id,'filename':d.filename,'data_kind':_provenance(d),
        'raw_sha256':d.raw_file.sha256,'conditions':d.conditions,'correction_states':d.correction_states,'metadata':d.metadata} for d in spec.datasets]
    sources={'project_id':project_id,'project_revision':project.revision,'run_id':run.id,
        'snapshot_sha256':digest,'engine_versions':spec.engine_versions,'source_sha256':spec.source_sha256,
        'data':provenance,'paths':[p.model_dump(mode='json') for p in spec.paths],
        'calibrations':[c.model_dump(mode='json') for c in spec.calibrations],
        'result_metadata':run.result.metadata,'scientific_validation':'not_inferred_by_report_export'}
    entries['sources.json']=canonical(sources)
    synthetic=any(d['data_kind']=='synthetic' for d in provenance)
    synthetic_note='包含合成数据，仅用于数值集成或演示，不作为实验科学验收。' if synthetic else '本报告仅整理存储结果，导出操作不新增科学验证。'
    if any(d['data_kind']=='not_declared' for d in provenance):
        warnings.append('部分数据未声明实验或合成属性，请核对 sources.json。')
    figures,missing=_plots(store,project_id,run,entries)
    warnings.extend(missing)
    methods=['# WebXAS 分析报告','',f'项目 {project.title}  项目版本 {project.revision}',
        f'运行 {run.id}  模型 {run.model_id}',f'输入快照 SHA256 {digest}','',synthetic_note,'',
        f'任务状态 {run.task_state}；收敛 {run.result.converged}；评估 {run.result.assessment}。',
        f'终止原因 {run.result.termination_reason}',f'误差依据 {run.result.uncertainty_basis}','',
        '## 方法配置','', '输入、校准、噪声、窗口、路径和约束完整保存在 input-snapshot.json。',
        'R 空间图展示实部、虚部与残差；横坐标未经相移校正，不能直接解释为单根键长。',
        '路径表中的 reff 为有效半程长度，MS 简并度不等于配位数。','',
        '```json',json.dumps({'engine_versions':spec.engine_versions,
            'bindings':[b.model_dump(mode='json') for b in spec.model.bindings],
            'calibrations':[c.model_dump(mode='json') for c in spec.calibrations]},ensure_ascii=False,indent=2),'```','',
        '## 统计与解释','',json.dumps(run.result.statistics.model_dump(mode='json'),ensure_ascii=False,indent=2),'',
        '统计误差、标定条件和模型敏感性应结合解释；最低 R-factor 不足以确定唯一结构。',
        '协方差的尺度与变量顺序见 covariance.json；不可估计的误差保留为空。','',
        '## 检查项','',*['- '+w for w in warnings], '',
        '## 图件','',*[f'![{label}]({path})' for label,path in figures], '',
        '原工程为 project.webxas.zip；全部图件同时提供 PNG 和 SVG，数据表为 UTF-8 CSV。']
    entries['methods.md']='\n'.join(methods).encode('utf-8')
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>WebXAS 分析报告</title>',
        '<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 24px;color:#172b35}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccd5db;padding:7px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f8;padding:12px}img{max-width:100%;height:auto}.notice{background:#eef4f6;padding:12px}a{color:#116a65}</style><body>',
        '<h1>WebXAS 分析报告</h1>',f'<p>项目 {html.escape(project.title)} · 版本 {project.revision} · 运行 {html.escape(run.id)}</p>',
        '<p class="notice">'+html.escape(synthetic_note)+'</p>',
        _table(['任务状态','收敛','结果评估','误差依据'],[[run.task_state,run.result.converged,run.result.assessment,run.result.uncertainty_basis]]),
        '<p>终止原因：'+html.escape(run.result.termination_reason)+'</p>',
        '<h2>参数</h2>',_table(['ID','值','单位','标准误差','误差说明','触界'],[[p.id,p.value,p.unit,p.stderr,p.uncertainty_reason,p.at_bound] for p in run.result.parameters]),
        '<h2>统计与条件</h2><pre>'+html.escape(json.dumps(run.result.statistics.model_dump(mode='json'),ensure_ascii=False,indent=2))+'</pre>',
        '<p>最低 R-factor 不足以确定唯一结构；统计误差需结合标定、模型假设和敏感性解释。</p>',
        '<h2>数据来源</h2>',_table(['数据集','文件','属性','原始SHA256'],[[d['dataset_id'],d['filename'],d['data_kind'],d['raw_sha256']] for d in provenance]),
        '<h2>方法配置</h2><pre>'+html.escape(json.dumps({'engine_versions':spec.engine_versions,'bindings':[b.model_dump(mode='json') for b in spec.model.bindings]},ensure_ascii=False,indent=2))+'</pre>',
        '<h2>检查项</h2><ul>'+''.join('<li>'+html.escape(w)+'</li>' for w in warnings)+'</ul>',
        '<h2>图件</h2><p>R 坐标未经相移校正；MS 有效路径长度和简并度不直接代表单根键长或配位数。</p>']
    for label,path in figures:
        page.append('<figure><img src="'+html.escape(path,quote=True)+'" alt="'+html.escape(label,quote=True)+'"><figcaption>'+html.escape(label)+' · '+html.escape(run.id)+'</figcaption></figure>')
    page.append('<p><a href="project.webxas.zip">原工程</a> · <a href="parameters.csv">参数表</a> · <a href="paths.csv">路径表</a> · <a href="input-snapshot.json">完整输入快照</a> · <a href="sources.json">来源与版本</a></p></body></html>')
    entries['report.html']=''.join(page).encode('utf-8')
    if sum(map(len,entries.values()))>MAX_REPORT_BYTES:
        raise ProjectError('report_too_large','报告资源超过 250 MB。')
    entries['manifest.json']=canonical({'format':'webxas-report','schema_version':1,'run_id':run.id,
        'project_id':project_id,'project_revision':project.revision,
        'files':{name:{'sha256':hashlib.sha256(value).hexdigest(),'byte_size':len(value)} for name,value in entries.items()}})
    stream=io.BytesIO()
    with ZipFile(stream,'w',ZIP_DEFLATED) as archive:
        for name,payload in sorted(entries.items()):
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
                raise ProjectError('unsafe_report_path','报告资源名称无效。')
            archive.writestr(name,payload)
    return stream.getvalue()
