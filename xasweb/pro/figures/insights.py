"""Conservative Chinese descriptions of a bound FigureSpec and stored arrays.

This deterministic adapter does not fit, classify oxidation states, infer bonds,
or attribute run-level joint statistics to individual spectra.
"""
from __future__ import annotations

import re
import numpy as np


def _unique(values):
    return list(dict.fromkeys(v for v in values if v is not None))


def dataset_scope(project, spec):
    ids = [s['dataset_id'] for s in spec.sources if s.get('dataset_id')]
    for source in spec.sources:
        ids.extend(p['dataset_id'] for p in source.get('points', source.get('point_records', [])) if p.get('dataset_id'))
    runs = {r.id: r for r in project.runs}
    snapshots = {}
    for source in spec.sources:
        run = runs.get(source.get('run_id'))
        if run:
            snapshots.update({d.id: d for d in run.input_snapshot.datasets})
            if not source.get('dataset_id'):
                if run.result:ids.extend(d.dataset_id for d in run.result.datasets)
                else:ids.extend(b.dataset_id for b in run.input_snapshot.model.bindings)
    ids = _unique(ids)
    datasets = {d.id: d for d in project.datasets} | snapshots
    names = [str(datasets[i].metadata.get('display_name') or datasets[i].metadata.get('sample_name') or
                 datasets[i].filename) if i in datasets else i for i in ids]
    return ids, names


def column_units(spec, arrays):
    units = {}
    def from_label(label, fallback):
        match = re.search(r' \(([^()]*)\)$', label or '')
        return match.group(1) if match else fallback
    for panel in spec.panels:
        if panel.get('x'):
            units[panel['x']] = from_label(panel.get('xlabel'), 'not_declared')
        for key in panel.get('xkeys', []):
            units[key] = from_label(panel.get('xlabel'), 'not_declared')
        if panel.get('y'):
            units[panel['y']] = from_label(panel.get('ylabel'), 'not_declared')
        if panel.get('z'):
            units[panel['z']] = from_label(panel.get('colorlabel'), 'stored_transform_convention')
        for key in [*panel.get('ys', []), *panel.get('error_keys', [])]:
            units[key] = from_label(panel.get('ylabel'), 'not_declared')
    for key in arrays:
        if key.startswith('energy') or key.endswith('_eV'): units[key] = 'eV'
        elif key in ('k', 'k_inv_A') or key.endswith('_inv_A'): units[key] = 'Å⁻¹'
        elif key in ('r', 'R_A', 'x_A', 'y_A', 'z_A'): units[key] = 'Å'
        elif key == 'weighted_chi' or key.startswith('weighted_chi_'):
            units[key] = f'Å^(-{spec.method["kweight"]})' if 'kweight' in spec.method else 'not_declared'
        elif key == 'derivative_per_eV': units[key] = 'eV⁻¹'
        elif key in ('chi','data_chi','model_chi','residual_chi','path_chi','normalized_mu','data','model','residual','occupancy','atom_index','index','correlation') or key.startswith(('normalized_mu_', 'contribution_')) or key.endswith(('_valid',)):
            units[key] = '1'
        elif key in ('mu','raw_mu','bkg','pre_edge','post_edge','rejected_mu'): units[key] = 'relative_absorbance'
        elif key.startswith('chir_'): units[key] = 'stored_FT_convention'
        elif key in ('state','state_values'):
            state = spec.method.get('state_key', '')
            units[key] = next((unit for suffix, unit in [('_K','K'),('_V','V'),('_s','s')] if state.endswith(suffix)), 'declared_state')
        units.setdefault(key, 'not_declared')
    return units


def figure_insight(project, workflow, item):
    spec, arrays = item.spec, item.arrays
    ids, names = dataset_scope(project, spec)
    observations, metrics = [], []
    limitations = list(spec.notes)
    def metric(label, value, unit, basis, uncertainty=None):
        if isinstance(value, (float, np.floating)) and not np.isfinite(value): return
        row = {'label': label, 'value': float(value) if isinstance(value, np.floating) else value,
               'unit': unit, 'basis': basis}
        if uncertainty is not None: row['uncertainty'] = uncertainty
        metrics.append(row)
    panel = spec.panels[0] if spec.panels else {}
    x = arrays.get(panel.get('x'))
    units = column_units(spec, arrays)
    grids=panel.get('xkeys') or ([panel['x']] if panel.get('x') else [])
    for grid_index,key in enumerate(grids):
        values=arrays.get(key)
        if values is None:continue
        if panel.get('state_valid'):values=values[np.asarray(arrays[panel['state_valid']],bool)]
        if not len(values):continue
        label=(panel.get('labels') or names or ['当前绑定数据'])[grid_index] if len(grids)>1 else '、'.join(names) or '当前绑定数据'
        prefix=label+' · ' if len(grids)>1 else ''
        basis=f'绑定数组 {key}；只统计有效状态坐标（若适用），不是独立重复次数'
        metric(prefix+'横轴采样点数',len(values),'1',basis)
        if panel.get('state_tick_labels'):
            observations.append(f'{label}使用已记录状态标签；横轴整数仅为显示位置，有效状态观测 {len(values)} 条。')
            continue
        unit=units.get(key,'not_declared')
        metric(prefix+'横轴下界',float(np.min(values)),unit,basis+'；不是拟合窗口或置信区间')
        metric(prefix+'横轴上界',float(np.max(values)),unit,basis+'；不是拟合窗口或置信区间')
        observations.append(f'{label}的横轴实际覆盖 {float(np.min(values)):.6g}–{float(np.max(values)):.6g} {unit}，'
                            f'共 {len(values)} 个有效采样坐标；各系列保留各自网格。')
    family = spec.family
    if family == 'F01':
        observations.append('图中展示已存处理输入与可用背景曲线，可核查测量覆盖和处理形态。')
        if 'rejected_rows' in spec.method:
            metric('已记录拒绝点数', spec.method['rejected_rows'], '1', '处理快照的原始行选择掩码')
        limitations.append('背景曲线是该处理快照的结果；图形平滑或背景贴合本身不证明实验或模型正确。')
    elif family in ('F02','F03'):
        observations.append('归一化谱形与导数保留绑定快照的能量坐标，不自动对齐样品化学位移。')
        if family == 'F03' and 'derivative_per_eV' in arrays:
            peak = int(np.argmax(arrays['derivative_per_eV']))
            metric('导数最大值对应能量', float(arrays['energy_eV'][peak]), 'eV', '本图未额外平滑的一阶导数最大值；描述性边位定义')
            observations.append(f'本图一阶导数最大值对应能量为 {float(arrays["energy_eV"][peak]):.6g} eV。')
        processing_ids={s.get('processing_revision_id') for s in spec.sources}
        features=[f for f in workflow.get('features',[]) if f.get('processing_id') in processing_ids]
        processors={p.id:p for p in project.processing}
        calibrations={c.id:c for c in project.calibrations}
        for feature in features:
            processing=processors.get(feature.get('processing_id'))
            dataset_id=feature.get('dataset_id') or (processing.dataset_id if processing else None)
            feature_label=names[ids.index(dataset_id)] if dataset_id in ids else str(dataset_id or '绑定样品')
            calibration=calibrations.get(processing.calibration_id) if processing else None
            calibrated=calibration is not None and calibration.kind=='energy'
            state='已记录标准样能量校准' if calibrated else '未记录标准样绝对能量校准'
            edge=feature.get('edge_position_eV')
            if edge is not None:
                definition=str(feature.get('edge_definition','绑定工作流的已存边位定义'))
                metric('工作流已存边位',edge,'eV',f'样品 {feature_label}；处理 {feature.get("processing_id")}；'+definition+'；'+state)
                observations.append(f'{feature_label}绑定处理快照的已存边位为 {edge:.6g} eV（{definition}）；{state}。')
            if feature.get('white_line_height') is not None:
                height=feature['white_line_height']
                metric('工作流已存白线高度',height,'1',f'样品 {feature_label}；处理 {feature.get("processing_id")}；'+str(feature.get('white_line_definition','绑定工作流已存白线定义')))
                observations.append(f'{feature_label}按该工作流窗口与定义，已存白线高度为 {height:.6g}。')
            if calibration:
                metric('已记录能量校准量',calibration.value,calibration.unit,'处理快照 calibration_id 绑定的校准记录；不是样品化学位移',
                       {'stderr':calibration.stderr,'basis':calibration.uncertainty_basis} if calibration.stderr is not None else None)
        limitations.append('边位和白线等谱形特征不能单独换算为绝对价态、组分含量或唯一局域结构。')
    elif family == 'F04':
        observations.append('图中展示所选处理快照的 EXAFS 振荡；显示加权口径保存在方法中。')
        if 'kweight' in spec.method: metric('显示 k 权重', spec.method['kweight'], '1', 'FigureSpec 记录的显示权重；不改写原始 χ(k)')
        limitations.append('振荡形态本身不确定配位数、键长或散射元素，需要适用路径模型和不确定度依据。')
    elif family == 'F05':
        observations.append('图中展示已存傅里叶变换分量，可描述径向变换信号的分布。')
        if 'chir_mag' in arrays and x is not None:
            metric('最大 FT 幅值对应 R', float(x[np.argmax(arrays['chir_mag'])]), 'Å', '本图覆盖范围内的幅值峰坐标，未相位校正')
            observations.append(f'本图覆盖范围内，最大 FT 幅值对应的未相位校正 R 为 {float(x[np.argmax(arrays["chir_mag"])]):.6g} Å。')
        limitations.append('R 坐标未相位校正，峰位不能直接称为真实键长。')
    elif family == 'F06':
        observations.append('Cauchy WT 展示同一 χ(k) 在 k–R 空间的信号分布，色标和有效范围见方法。')
        if 'magnitude' in arrays:
            row, col = np.unravel_index(np.argmax(arrays['magnitude']), arrays['magnitude'].shape)
            metric('WT 最大值对应 k', float(arrays['k'][col]), 'Å⁻¹', '本图公共有效窗口内的绝对 WT 幅值最大网格单元')
            metric('WT 最大值对应 R', float(arrays['r'][row]), 'Å', '未相位校正的 WT 变换坐标')
            observations.append(f'当前 WT 网格最大幅值位于 k={float(arrays["k"][col]):.6g} Å⁻¹、'
                                f'R={float(arrays["r"][row]):.6g} Å；这是该窗口内的最大网格单元。')
        limitations.append('WT 不是独立重复测量，不能单凭强度团块确定散射元素或结构。')
    elif family == 'F07':
        observations.append('该独立图展示绑定拟合快照的数据/模型或残差，不重新执行优化。')
        for key in panel.get('ys', []):
            if key.startswith('residual') and key in arrays:
                metric(f'本谱 {key} 的未加权 RMS', float(np.sqrt(np.mean(arrays[key]**2))), units.get(key,'not_declared'),
                       '仅当前数据集和图示分量的 data−model 数组；不是联合 χ²、约化 χ² 或独立实验误差')
                observations.append(f'本谱图示分量 {key} 的未加权残差 RMS 为 {float(np.sqrt(np.mean(arrays[key]**2))):.6g}。')
        for run in project.runs:
            if run.id not in {s.get('run_id') for s in spec.sources} or not run.result:continue
            result=run.result
            count=len(result.datasets)
            scope='联合运行' if count>1 else '单谱运行'
            available=sum(p.stderr is not None for p in result.parameters)
            observations.append(f'绑定{scope}{"已收敛" if result.converged else "未收敛"}，覆盖 {count} 条谱；'
                                f'{len(result.parameters)} 个记录参数中 {available} 个带有标准误，误差依据为 {result.uncertainty_basis}。')
            if result.statistics.reduced_chisquare is not None:
                metric(scope+'约化 χ²',result.statistics.reduced_chisquare,'1',
                       f'运行 {run.id} 的完整统计口径；覆盖 {count} 条谱，不归属为当前图单独计算的拟合优度')
        limitations.extend(['联合运行的全局统计量不作为本谱单独统计量；原始统计定义与参数表随包提供。',
                            '收敛和小残差不证明模型唯一或结构已被实验标准样验证。'])
    elif family == 'F08':
        observations.append('该图来自绑定拟合运行的路径贡献或已保存 FEFF 顶点投影，是模型信息。')
        if 'effective_half_path_A' in spec.method:
            metric('有效半路径长', spec.method['effective_half_path_A'], 'Å', '已存 FEFF 路径参数；多重散射时不是单根键长')
        limitations.append('模型投影不等同于独立结构测量；投影重叠不说明原子重合，不补造未保存的几何。')
    elif family == 'F09':
        observations.append('图中展示给定参考和约束下的参考组分拟合或其残差。')
        for index, coefficient in enumerate(spec.method.get('coefficients', [])):
            metric(f'参考组分系数 {index+1}', coefficient, '1', '绑定 LCF 结果的条件式参考系数；参考身份、约束和适用条件见方法')
            observations.append(f'绑定 LCF 的参考组分 {index+1} 系数为 {float(coefficient):.6g}，仅适用于已记录参考和约束。')
        limitations.append('参考组分系数不自动等同于含量或价态；未估计的系数不确定度不补造误差条。')
    elif family == 'F10':
        observations.append('状态图使用工作流中记录的实际状态和有效观测，缺项不会补为新测量值。')
        limitations.extend(['相邻状态连线不构成因果或动力学模型；不同状态不是独立重复样本。',
                            '联合模型的全局共享估计不能复制为多个独立状态点；误差范围沿用已记录定义。'])
    elif family == 'F11':
        observations.append('该图展示绑定运行协方差导出的参数相关系数，涉及该运行的参数集合。')
        matrix = arrays.get('correlation')
        if matrix is not None and len(matrix)>1:
            metric('最大非对角绝对相关系数', float(np.max(np.abs(matrix[~np.eye(len(matrix),dtype=bool)]))), '1', '绑定运行的协方差归一化；属于整个运行，不是单谱的独立拟合优度')
            observations.append(f'该运行包含 {len(matrix)} 个相关性参数，最大非对角绝对相关系数为 '
                                f'{float(np.max(np.abs(matrix[~np.eye(len(matrix),dtype=bool)]))):.6g}。')
        limitations.append('参数相关性不是参数误差大小、物理因果关系或模型充分性的证明。')
    else:
        observations.append('本图呈现绑定工作流的已存数据与方法；解读范围限于清单中记录的来源。')
    limitations.extend(i['message'] for i in workflow.get('issues', []) if isinstance(i,dict) and i.get('message') and
                       (not i.get('dataset_id') or i['dataset_id'] in ids))
    limitations.append('未提供或未估计的实验不确定度保持缺失，不以采样点数代替独立重复数。')
    method = {'policy':'snapshot-figure-insight-v1','workflow_id':workflow['id'],
              'workflow_state':workflow.get('state'),'figure_method':spec.method,
              'display_transforms':spec.transforms,'column_units':units}
    run_ids = _unique(s.get('run_id') for s in spec.sources)
    if run_ids:
        method['fit_statistics_scope'] = [{'run_id':r.id, 'dataset_ids':[d.dataset_id for d in r.result.datasets],
             'scope':'joint_run' if len(r.result.datasets)>1 else 'single_dataset_run','converged':r.result.converged}
             for r in project.runs if r.id in run_ids and r.result]
    result = {'observations':observations,'metrics':metrics,'limitations':_unique(limitations),
            'method':method,'sources':spec.sources,'dataset_ids':ids,'sample_names':names}
    from .presentation import brief_insight
    result['display'] = brief_insight(result, family, spec.method)
    return result
