"""Descriptive trends from recorded state values, never gap-filled results."""
from __future__ import annotations

import numpy as np


def add_series_trends(project, workflow, add, missing):
    series=workflow.get('series') or {}
    points=series.get('points') or []
    if len(points)<2:return
    state_key=str(series.get('state_key','State'))
    states=[p.get('state_value') for p in points]
    numeric=all(s is None or (isinstance(s,(int,float)) and not isinstance(s,bool)) for s in states)
    labels=[]
    if numeric:
        xs=np.asarray([float(s) if s is not None else 0. for s in states])
        axis=state_key
    else:
        labels=list(dict.fromkeys(str(s) for s in states if s is not None))
        xs=np.asarray([float(labels.index(str(s))) if s is not None else 0. for s in states])
        axis=state_key+' (categorical)'
    state_valid=np.asarray([s is not None for s in states],bool)
    # A missing state remains in source records/CSV, without a made-up x value.
    source={'state_key':state_key,'point_records':points,'state_labels':states,
            'categorical_display_mapping':{str(i):s for i,s in enumerate(labels)} if labels else None}
    n=0
    def emit(key,title,title_en,ylabel,values,valid,errors=None,error_valid=None,method=None):
        nonlocal n
        valid=np.asarray(valid,bool)&state_valid
        if not np.any(valid):return
        arrays={'state':xs,'state_valid':state_valid,key:np.asarray([float(v) if good else 0. for v,good in zip(values,valid)]),
                'value_valid':valid}
        panel={'x':'state','ys':[key],'labels':[title_en],'xlabel':axis,'ylabel':ylabel,
               'valid_masks':['value_valid'],'state_valid':'state_valid','failed_state_markers':True,
               'series_modes':['trend'],'state_tick_labels':labels}
        if errors is not None:
            mask=np.asarray(error_valid,bool)&valid
            arrays.update(stderr=np.asarray([float(e) if good else 0. for e,good in zip(errors,mask)]),stderr_valid=mask)
            panel.update(error_keys=['stderr'],error_valid_masks=['stderr_valid'])
        add('F10',100+n,title,title_en,arrays,[panel],[source],kind='trend',
            method={'state_key':state_key,'state_axis':'measured numeric values' if numeric else 'explicit categorical labels; integers are display positions only',
                    'missing_policy':'invalid values are blank in CSV and break plotted lines; zero placeholders are masked and never displayed',
                    'connection':'selected observation order, no smoothing, interpolation, or causal inference',
                    'missing_state_count':int((~state_valid).sum()), **(method or {})},
            notes=['失败状态留空并在坐标轴底部以 × 标示；无实际状态坐标的记录仅保留在源记录与CSV。',
                   '误差条仅表示明确存储的统计标准误；缺失误差不绘制，也不等同于零误差。'])
        n+=1
    usable=lambda p:p.get('state','succeeded')=='succeeded' and p.get('processing_id') is not None
    for key,title,title_en,unit in [('edge_position_eV','边位随状态变化','Edge position versus state','Edge position (eV)'),
                                    ('white_line_height','白线高度随状态变化','White-line height versus state','White-line height (1)')]:
        values=[p.get(key) for p in points]
        valid=[usable(p) and isinstance(v,(int,float)) and not isinstance(v,bool) for p,v in zip(points,values)]
        emit(key,title,title_en,unit,values,valid,method={'quantity':key,'feature_definition':'see recorded workflow.features; no universal valence conversion'})
    lcf_by_dataset={r['dataset_id']:r for r in workflow.get('lcf_results',[])}
    references=list(dict.fromkeys(rid for r in lcf_by_dataset.values() for rid in r.get('reference_ids',[])))
    for index,rid in enumerate(references):
        values=[];valid=[]
        for point in points:
            record=lcf_by_dataset.get(point['dataset_id'],{})
            ids=record.get('reference_ids',[])
            coefficients=point.get('reference_coefficients')
            pos=ids.index(rid) if rid in ids else -1
            value=coefficients[pos] if isinstance(coefficients,list) and 0<=pos<len(coefficients) else None
            values.append(value);valid.append(usable(point) and isinstance(value,(int,float)))
        emit('reference_coefficient',f'参考组分 {index+1} 随状态变化',f'Reference component {index+1} versus state',
             'Reference coefficient (1)',values,valid,
             method={'reference_id':rid,'interpretation':'conditional reference-component coefficient; not automatically concentration or valence',
                     'constraints_and_diagnostics':[r for r in workflow.get('lcf_results',[]) if rid in r.get('reference_ids',[])],
                     'uncertainty':'not estimated; no fabricated Gaussian errors at constraint boundaries'})
    runs={r.id:r for r in project.runs}
    rows={}
    for i,point in enumerate(points):
        for fit in point.get('fits',[]):
            run=runs.get(fit.get('fit_run_id'))
            if run is None or run.result is None:continue
            fit={**fit,'converged':run.result.converged,'uncertainty_basis':run.result.uncertainty_basis,
                 'snapshot_sha256':run.snapshot_sha256}
            declarations={p.id:p for p in run.input_snapshot.model.parameters.nodes}
            binding=next((b for b in run.input_snapshot.model.bindings if b.dataset_id==point['dataset_id']),None)
            if binding is None:continue
            owners={p.id for p in binding.paths} if binding else set()
            for estimate in [p.model_dump(mode='json') for p in run.result.parameters]:
                decl=declarations.get(estimate['id'])
                if decl is None:continue
                if decl.scope=='dataset' and decl.owner_id!=point['dataset_id']:continue
                if decl.scope=='path' and decl.owner_id not in owners:continue
                key=(estimate['id'],estimate.get('unit','1'),fit.get('uncertainty_basis','not_estimated'),decl.physical_basis)
                # Multiple stored fits at one state are separate candidate
                # explanations. Refuse an arbitrary 'best' parameter trend.
                rows.setdefault(key,[[] for _ in points])[i].append((fit,estimate,decl))
    for (pid,unit,basis,physical),observations in rows.items():
        global_runs={}
        for options in observations:
            for fit,estimate,decl in options:
                if decl.scope=='global':global_runs.setdefault(fit['fit_run_id'],[]).append(estimate)
        shared={rid:estimates for rid,estimates in global_runs.items() if len(estimates)>1}
        if shared:
            missing.append({'family':'F10','parameter_id':pid,
                'reason':'同一联合运行的全局共享估计不复制为多个独立状态点或误差条；参见单个共享估计与运行记录。',
                'global_shared_estimates':[{'fit_run_id':rid,'estimate':estimates[0],
                                           'uncertainty_basis':basis,'role':'single global shared estimate'} for rid,estimates in shared.items()]})
            continue
        values=[];valid=[];errors=[];error_valid=[]
        for point,options in zip(points,observations):
            good=len(options)==1 and options[0][0].get('converged') is True and usable(point)
            fit,estimate,decl=options[0] if options else ({},{},None)
            value=estimate.get('value');error=estimate.get('stderr')
            values.append(value);valid.append(good and isinstance(value,(int,float)))
            errors.append(error);error_valid.append(good and isinstance(error,(int,float)) and error>=0)
        if sum(valid)<2:
            missing.append({'family':'F10','parameter_id':pid,
                'reason':'少于两个可比较且唯一选定的状态参数估计；未隐式混合独立基准、联合结果或多个候选。'})
            continue
        label=next((entry[2].label for options in observations for entry in options if entry[2].label),pid)
        axis_label=label if len(label)<=20 else 'Parameter value'
        emit('parameter_value',f'{label} 随状态变化',f'{label} versus state',f'{axis_label} ({_unit(unit)})',
             values,valid,errors,error_valid,
             method={'parameter_id':pid,'unit':unit,'uncertainty_basis':basis,'error_bar':'stored statistical standard error only',
                     'physical_basis':physical,'multiple_candidate_policy':'ambiguous states left blank; no arbitrary winner',
                     'conditional_error_notice':'conditional_fixed errors exclude fixed calibration uncertainty' if basis=='conditional_fixed' else None,
                     'parameter_declarations':[entry[2].model_dump(mode='json') for options in observations for entry in options],
                     'selected_fit_runs':[{'dataset_id':point['dataset_id'],'fit_run_ids':[entry[0]['fit_run_id'] for entry in options]}
                                          for point,options in zip(points,observations)],
                     'estimate_sources':[{'fit_run_id':entry[0]['fit_run_id'],'snapshot_sha256':entry[0]['snapshot_sha256'],
                                          'estimate':entry[1]} for options in observations for entry in options],
                     'run_selection_policy':'one applicable stored fit per parameter/state; alternatives remain blank'})
    if not n:missing.append({'family':'F10','reason':'状态记录缺少可用的特征、适用参考系数或收敛拟合参数；不补造趋势。'})


def _unit(unit):
    return {'angstrom':'Å','angstrom2':'Å²','angstrom-1':'Å⁻¹','1':'1','eV':'eV','K':'K'}.get(unit,unit)
