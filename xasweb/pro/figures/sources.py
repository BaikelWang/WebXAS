"""Adapt immutable processing / fit records to full-precision figure data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np

from ..contracts import ArrayRef
from ..preprocessing import load_array
from ..xanes import bounded_derivative_edge
from ..projects import ProjectError, canonical
from .spec import FigureSpec, FigureStyle

MAX_FIGURES = 160


@dataclass
class FigureData:
    spec: FigureSpec
    arrays: dict
    children: list | None = None


def _arrays(store, pid, refs):
    return {name: load_array(store, pid, ArrayRef.model_validate(ref) if isinstance(ref, dict) else ref)
            for name, ref in refs.items()}


def _pair(x, y, name):
    if np.ndim(x) != 1 or len(x) < 2 or np.shape(y) != np.shape(x) or np.any(np.diff(x) <= 0):
        raise ProjectError('figure_array_shape', '图件横坐标必须递增，信号必须与其等长。', name)


def _label(dataset, index):
    # Machine IDs are provenance, never publication titles.
    label = dataset.metadata.get('display_name') or dataset.metadata.get('sample_name') or Path(dataset.filename).stem
    if len(label) > 44 or (len(label) > 24 and sum(c.isdigit() for c in label) > 12):
        label = f'Sample {index + 1}'
    return label


def _panel(x, ys, xlabel, ylabel, labels=None):
    return {'x': x, 'ys': list(ys), 'xlabel': xlabel, 'ylabel': ylabel,
            'labels': labels or list(ys)}


def _wavelet(k, chi, lo, hi):
    """Same Cauchy convention as spectral_views; no preview downsampling."""
    from larch import Group
    from larch.xafs import cauchy_wavelet, ftwindow
    step = .05
    grid = np.arange(0, np.floor(min(float(k[-1]), hi + .5) / step) + 1) * step
    uniform = np.interp(grid, k, chi, left=0, right=0)
    uniform *= ftwindow(grid, xmin=lo, xmax=hi, dx=1, window='hanning')
    group = Group()
    cauchy_wavelet(grid, uniform, group=group, kweight=2, rmax_out=6, nfft=2048)
    cols = np.where((grid >= lo) & (grid <= hi))[0]
    rows = np.where(group.wcauchy_r >= .3)[0]
    matrix = group.wcauchy_mag[np.ix_(rows, cols)]
    if not np.isfinite(matrix).all():
        raise ProjectError('wavelet_nonfinite', 'Cauchy 小波输出含非有限值。')
    return {'k': grid[cols], 'r': group.wcauchy_r[rows], 'magnitude': matrix,
            'input_k': grid, 'windowed_chi': uniform}, {
        'algorithm': 'Larch continuous Cauchy', 'kweight': 2, 'kstep': step, 'nfft': 2048,
        'window': 'hanning', 'dk': 1, 'kmin': lo, 'kmax': hi, 'rmax_out': 6,
        'interpolation': 'linear; zero outside measured coverage; Hanning taper',
        'normalization': 'absolute Larch wcauchy_mag; no per-image maximum normalization',
        'source': 'https://xraypy.github.io/xraylarch/xafs_wavelets.html'}


def collect_figures(store, project, workflow, style=None, difference_pair=None):
    """No fitting, preprocessing, writes, or mutation of source records."""
    style = style if isinstance(style, FigureStyle) else FigureStyle.model_validate(style or {})
    output = []
    missing = []
    by_dataset = {d.id: d for d in project.datasets}
    by_processing = {p.id: p for p in project.processing}
    selected = []
    for pid in workflow.get('processing_ids', []):
        if pid not in by_processing:
            raise ProjectError('processing_not_found', '工作流关联的处理快照缺失。', pid)
        selected.append(by_processing[pid])
    if len(selected) > 32:
        raise ProjectError('delivery_scope_limit', '单次科研图包最多支持 32 个处理快照，请拆分工作流。')

    def add(family, index, title, title_en, arrays, panels, sources, kind='line', method=None, notes=None):
        fid = f'{family}-{index:03d}'
        spec = FigureSpec(id=fid, family=family, title=title, title_en=title_en, kind=kind,
                          sources=sources, panels=panels, style=style,
                          method=method or {}, notes=notes or [],
                          transforms={'offset': style.offset, 'display_xlim': style.xlim,
                                      'display_ylim': style.ylim, 'smoothing': 'none',
                                      'data_excluded': 0}, data_files=[f'data/{fid}.csv'])
        if kind == 'heatmap':
            spec = spec.model_copy(update={'data_files': [f'data/{fid}.csv', f'data/{fid}.npz']})
        for value in arrays.values():
            if not np.isfinite(value).all():
                raise ProjectError('figure_nonfinite', '图件源数组包含非有限值。', fid)
        output.append(FigureData(spec, arrays))
        if len(output) > MAX_FIGURES:
            raise ProjectError('delivery_scope_limit', '图件数量超过上限，请拆分工作流。')
        return output[-1]

    wavelets = []
    # Comparison wavelets use the common valid k interval. No dataset-specific
    # normalization or silently different WT window is permitted.
    loaded = [(p, _arrays(store, project.id, p.arrays)) for p in selected]
    eligible = [(p, a) for p, a in loaded if {'k', 'chi'} <= a.keys()]
    common_lo = max([max(float(a['k'][0]), p.parameters.kmin) for p, a in eligible], default=0)
    common_hi = min([min(float(a['k'][-1]), p.parameters.kmax) for p, a in eligible], default=0)
    features = {f['processing_id']: f for f in workflow.get('features', []) if 'processing_id' in f}
    for index, (processing, a) in enumerate(loaded):
        begin=len(output)
        dataset = by_dataset[processing.dataset_id]
        label = _label(dataset, index)
        source = {'dataset_id': dataset.id, 'processing_revision_id': processing.id,
                  'data_kind': 'synthetic' if dataset.metadata.get('synthetic') is True or dataset.metadata.get('data_kind') in ('synthetic','same_engine_synthetic') else 'experimental' if dataset.metadata.get('synthetic') is False or dataset.metadata.get('data_kind') in ('experimental','real_experimental') else 'not_declared',
                  'raw_sha256': dataset.raw_file.sha256,
                  'processing_sha256': hashlib.sha256(canonical(processing.model_dump(mode='json'))).hexdigest(),
                  'arrays': {n: r.model_dump(mode='json') for n, r in processing.arrays.items()}}
        sources = [source]
        mu_valid = dataset.kind == 'mu' and {'energy', 'mu'} <= a.keys()
        if mu_valid:
            _pair(a['energy'], a['mu'], processing.id)
            values = {'energy_eV': a['energy'], 'mu': a['mu']}
            ys = ['mu']
            for name in ('pre_edge', 'post_edge', 'bkg'):
                if name in a:
                    _pair(a['energy'], a[name], name)
                    values[name] = a[name]; ys.append(name)
            panel=_panel('energy_eV', ys, 'E (eV)', 'μ(E) (relative absorbance)')
            method={'input':'stored processing input; calibrated energy if recorded',
                    'resolved_calls':processing.resolved_calls,'warnings':processing.warnings}
            if processing.mask is not None:
                try:
                    from ...ingest import construct_mu
                    from ...models import Mapping
                    mapping=Mapping.model_validate(processing.metadata.get('input_mapping',dataset.mapping.model_dump(mode='json')))
                    raw=store.read_asset(project.id,dataset.raw_file).decode('utf-8-sig')
                    energy,mu,conversion=construct_mu(raw,mapping,processing.metadata.get('effective_energy_shift_eV',0.))
                    selected_mask=load_array(store,project.id,processing.mask)
                    if selected_mask.dtype.kind!='b' or selected_mask.shape!=energy.shape:
                        raise ValueError('stored mask length does not match original mapped rows')
                    values.update(raw_energy_eV=energy,raw_mu=mu,selected=selected_mask,rejected=~selected_mask)
                    source['mask']=processing.mask.model_dump(mode='json')
                    method.update(raw_conversion=conversion,input_mapping=mapping.model_dump(mode='json'),
                                  raw_rows=len(energy),retained_rows=int(selected_mask.sum()),rejected_rows=int((~selected_mask).sum()))
                    if np.any(~selected_mask):
                        values['rejected_mu']=mu.copy()
                        panel['ys'].append('rejected_mu');panel['labels'].append('Rejected points')
                        panel.update(xkeys=['energy_eV']*len(ys)+['raw_energy_eV'],
                                     valid_masks=[None]*len(ys)+['rejected'],
                                     series_modes=['line']*len(ys)+['points'])
                except (ValueError,KeyError,UnicodeDecodeError) as exc:
                    method['rejected_point_reconstruction']='unavailable: '+str(exc)
                    missing.append({'family':'F01','dataset_id':dataset.id,'reason':'拒绝点无法按原快照可靠重建：'+str(exc)})
            add('F01', index, f'{label} · 吸收谱与处理检查', f'{label} · Absorption and quality control',
                values, [panel], sources, method=method,
                notes=['原始文件保留在可重开工程；此图显示本处理快照的输入及已存背景。'])
        if dataset.kind == 'mu' and {'energy','norm'} <= a.keys():
            _pair(a['energy'], a['norm'], processing.id)
            add('F02', index, f'{label} · 归一化近边谱', f'{label} · Normalized XANES',
                {'energy_eV': a['energy'], 'normalized_mu': a['norm']},
                [_panel('energy_eV', ['normalized_mu'], 'E (eV)', 'Normalized μ(E) (dimensionless)', [label])], sources,
                method={'resolved_calls':processing.resolved_calls,
                        'external_plot_only':processing.metadata.get('external_plot_only',False),
                        'external_processing_note':dataset.metadata.get('external_processing_note'),
                        'warnings':processing.warnings})
            feature = features.get(processing.id)
            if feature and {'energy', 'derivative'} <= feature.get('arrays', {}).keys():
                values = _arrays(store, project.id, feature['arrays'])
                energy, derivative = values['energy'], values['derivative']
                method = {k: v for k, v in feature.items() if k != 'arrays'}
                source = {**source, 'feature_arrays': feature['arrays']}
            else:
                energy = a['energy']
                derivative = np.gradient(a['norm'], energy, edge_order=2)
                method = {'derivative_method': 'numpy.gradient, nonuniform grid, edge_order=2; no smoothing'}
                e0 = processing.parameters.e0 or processing.metadata.get('metrics', {}).get('e0_eV')
                try:
                    method.update(bounded_derivative_edge(energy, derivative, e0, dataset.id))
                except ProjectError as exc:
                    method['edge_position_eV'] = None
                    method['edge_unavailable_reason'] = str(exc)
            _pair(energy, derivative, 'derivative')
            add('F03', index, f'{label} · 一阶导数', f'{label} · First derivative',
                {'energy_eV': energy, 'derivative_per_eV': derivative},
                [_panel('energy_eV', ['derivative_per_eV'], 'E (eV)', 'dμnorm/dE (eV⁻¹)', [label])], [source], method=method)
        if {'k', 'chi'} <= a.keys():
            _pair(a['k'], a['chi'], processing.id)
            weight = processing.parameters.kweight
            add('F04', index, f'{label} · k 空间谱', f'{label} · k-space EXAFS',
                {'k_inv_A': a['k'], 'chi': a['chi'], 'weighted_chi': a['chi'] * a['k'] ** weight},
                [_panel('k_inv_A', ['weighted_chi'], 'k (Å⁻¹)', f'k{_sup(weight)}χ(k) (Å{_sup(-weight)})' if weight else 'χ(k) (dimensionless)', [label])],
                sources, method={'kweight': weight, 'processing_calls': processing.resolved_calls})
            if common_hi - common_lo > 1:
                values, method = _wavelet(a['k'], a['chi'], common_lo, common_hi)
                wt = add('F06', index, f'{label} · Cauchy 小波', f'{label} · Cauchy wavelet', values,
                         [{'x': 'k', 'y': 'r', 'z': 'magnitude', 'xlabel': 'k (Å⁻¹)',
                           'ylabel': 'R (Å; uncorrected)', 'colorlabel': '|WT[k²χ(k)]| (Larch convention)'}],
                         sources, kind='heatmap', method=method,
                         notes=['R 未相位校正；WT 不单独判定散射元素或结构。', '统一色标与公共 k 范围；完整矩阵未下采样。'])
                wavelets.append(wt)
            else:
                missing.append({'family': 'F06', 'dataset_id': dataset.id, 'reason': '共同有效 k 区间不足 1 Å⁻¹；不能建立一致 WT 对比。'})
        if 'r' in a and any('chir_' + n in a for n in ('mag', 're', 'im')):
            values = {'R_A': a['r']}; ys = []
            for component in ('mag', 're', 'im'):
                name = 'chir_' + component
                if name in a:
                    _pair(a['r'], a[name], name); values[name] = a[name]; ys.append(name)
            add('F05', index, f'{label} · R 空间谱', f'{label} · R-space EXAFS', values,
                [_panel('R_A', ys, 'R (Å; uncorrected)', 'χ(R) (stored FT convention)',
                        [{'chir_mag': '|χ(R)|', 'chir_re': 'Re χ(R)', 'chir_im': 'Im χ(R)'}[y] for y in ys])],
                sources, method={'phase_corrected': False, 'resolved_calls': processing.resolved_calls},
                notes=['R 是未相位校正的变换坐标，不能直接称为真实键长。'])
        for family,reason in {'F01':'此快照没有原始 μ(E) 处理输入。',
                              'F02':'此快照无归一化 E–μ 数据，不能由 χ(k) 伪造。',
                              'F03':'此快照无适用归一化 E–μ 导数输入。',
                              'F04':'此快照无有效 χ(k)。','F05':'此快照无已存 FT 数组。'}.items():
            if not any(f.spec.family==family for f in output[begin:]):
                missing.append({'family':family,'dataset_id':dataset.id,
                                'processing_revision_id':processing.id,'reason':reason})
    if wavelets:
        vmax = max(float(f.arrays['magnitude'].max()) for f in wavelets)
        for f in wavelets:
            f.spec = f.spec.model_copy(update={'method': {**f.spec.method, 'color_min': 0., 'color_max': vmax}})

    normalized=[(p,a) for p,a in loaded if by_dataset[p.dataset_id].kind=='mu' and {'energy','norm'}<=a.keys()]
    if len(normalized)>1:
        values={};xs=[];ys=[];labels=[];sources=[];identities=[]
        for i,(p,a) in enumerate(normalized):
            xkey,ykey=f'energy_{i}_eV',f'normalized_mu_{i}'
            values[xkey]=a['energy'];values[ykey]=a['norm'];xs.append(xkey);ys.append(ykey)
            labels.append(_label(by_dataset[p.dataset_id],i));identities.append(p.dataset_id)
            sources.append({'dataset_id':p.dataset_id,'processing_revision_id':p.id,
                            'arrays':{n:r.model_dump(mode='json') for n,r in p.arrays.items()}})
        panel=_panel(xs[0],ys,'E (eV)','Normalized μ(E)',labels)
        panel.update(xkeys=xs,dataset_ids=identities)
        add('F02',900,'归一化近边谱比较','Normalized XANES comparison',values,[panel],sources,
            method={'grid':'original individual measured grids; no interpolation or sample-edge alignment'},
            notes=['保留每条谱的实际能量坐标，未将真实边移自动对齐。'])
    if difference_pair is not None:
        spectra = {p.id: (p, a) for p, a in normalized}
        if (len(difference_pair) != 2 or difference_pair[0] == difference_pair[1] or
                any(pid not in spectra for pid in difference_pair)):
            raise ProjectError('difference_pair', '请选择同一次运行中的两个不同归一化谱。')
        (left, la), (right, ra) = [spectra[pid] for pid in difference_pair]
        ld, rd = by_dataset[left.dataset_id], by_dataset[right.dataset_id]
        if not ld.element or not ld.edge or (ld.element, ld.edge) != (rd.element, rd.edge):
            raise ProjectError('difference_edge', '差谱要求两条谱明确具有相同吸收元素与边。')
        _pair(la['energy'], la['norm'], left.id)
        _pair(ra['energy'], ra['norm'], right.id)
        keep = (la['energy'] >= ra['energy'][0]) & (la['energy'] <= ra['energy'][-1])
        if np.count_nonzero(keep) < 2:
            raise ProjectError('difference_coverage', '所选近边谱的共同实测能区不足。')
        grid = la['energy'][keep]
        sample, reference = la['norm'][keep], np.interp(grid, ra['energy'], ra['norm'])
        refs = [{'dataset_id': p.dataset_id, 'processing_revision_id': p.id,
                 'arrays': {key: p.arrays[key].model_dump(mode='json') for key in ('energy', 'norm')}}
                for p in (left, right)]
        add('F02', 901, '归一化近边差谱', 'Normalized XANES difference',
            {'energy_eV': grid, 'sample_norm': sample, 'reference_norm': reference, 'difference': sample - reference},
            [{**_panel('energy_eV', ['difference'], 'E (eV)', 'Δ normalized μ(E) (1)', ['Sample − reference']),
              'data_keys': ['sample_norm', 'reference_norm']}], refs,
            method={'quantity': 'normalized_difference', 'difference_pair': list(difference_pair),
                    'definition': 'sample minus reference on sample grid; linear interpolation, no extrapolation',
                    'energy_range_eV': [float(grid[0]), float(grid[-1])]},
            notes=['只在共有能区相减。实验能量校准和归一化条件影响可比性。'])
    if len(eligible)>1:
        weights={p.parameters.kweight for p,a in eligible}
        if len(weights)==1:
            weight=next(iter(weights));values={};xs=[];ys=[];labels=[];identities=[];sources=[]
            for i,(p,a) in enumerate(eligible):
                xkey,ykey=f'k_{i}_inv_A',f'weighted_chi_{i}'
                values[xkey]=a['k'];values[ykey]=a['chi']*a['k']**weight
                values[f'chi_{i}']=a['chi'];xs.append(xkey);ys.append(ykey)
                labels.append(_label(by_dataset[p.dataset_id],i));identities.append(p.dataset_id)
                sources.append({'dataset_id':p.dataset_id,'processing_revision_id':p.id,
                                'arrays':{n:r.model_dump(mode='json') for n,r in p.arrays.items()}})
            panel=_panel(xs[0],ys,'k (Å⁻¹)',f'k{_sup(weight)}χ(k) (Å{_sup(-weight)})' if weight else 'χ(k) (1)',labels)
            panel.update(xkeys=xs,dataset_ids=identities)
            add('F04',900,'k 空间谱比较','k-space comparison',values,[panel],sources,
                method={'kweight':weight,'grid':'original individual k grids; no interpolation or extra normalization'})
        else:missing.append({'family':'F04','reason':'各处理快照的 k 权重不同，未隐式重加权生成叠图；保留单谱。'})
    transformed=[(p,a) for p,a in loaded if {'r','chir_mag'}<=a.keys()]
    if len(transformed)>1:
        bases=[_ft_basis(p) for p,a in transformed]
        if bases[0] is not None and all(b==bases[0] for b in bases[1:]):
            values={};xs=[];ys=[];labels=[];identities=[];sources=[]
            for i,(p,a) in enumerate(transformed):
                xkey,ykey=f'R_{i}_A',f'chir_mag_{i}'
                values[xkey]=a['r'];values[ykey]=a['chir_mag'];xs.append(xkey);ys.append(ykey)
                labels.append(_label(by_dataset[p.dataset_id],i));identities.append(p.dataset_id)
                sources.append({'dataset_id':p.dataset_id,'processing_revision_id':p.id,
                                'arrays':{n:r.model_dump(mode='json') for n,r in p.arrays.items()}})
            weight=bases[0]['arguments']['kweight']
            panel=_panel(xs[0],ys,'R (Å; uncorrected)',f'|χ(R)| (Å{_sup(-weight-1)})',labels)
            panel.update(xkeys=xs,dataset_ids=identities)
            add('F05',900,'R 空间谱比较','R-space comparison',values,[panel],sources,
                method={'transform_basis':bases[0],'grid':'original individual R grids; no interpolation or extra normalization',
                        'phase_corrected':False},notes=['只在记录的 FT 窗口、权重、网格和尺度口径一致时叠加；不据此进行模型排名。'])
        else:missing.append({'family':'F05','reason':'FT 窗口、权重、网格、引擎或尺度记录不同/缺失，未生成不可比的 R 叠图；保留单谱。'})

    # Fits are read only from their verified input snapshots, not current models.
    runs = {r.id: r for r in project.runs}
    for run_index, rid in enumerate(workflow.get('fit_run_ids', [])):
        run = runs.get(rid)
        if run is None or run.result is None or run.task_state != 'succeeded':
            missing.append({'family': 'F07', 'run_id': rid, 'reason': '无成功完成的已存拟合结果。'}); continue
        if hashlib.sha256(canonical(run.input_snapshot.model_dump(mode='json'))).hexdigest() != run.snapshot_sha256:
            raise ProjectError('snapshot_integrity', '拟合输入快照哈希不符。', rid)
        binding = {b.dataset_id: b for b in run.input_snapshot.model.bindings}
        snapshots = {d.id: d for d in run.input_snapshot.datasets}
        for result_index, result in enumerate(run.result.datasets):
            if result.dataset_id not in binding or binding[result.dataset_id].processing_revision_id != result.processing_revision_id:
                raise ProjectError('result_snapshot_mismatch', '拟合结果与输入快照不符。', rid)
            a = _arrays(store, project.id, result.arrays)
            label = _label(snapshots[result.dataset_id], result_index)
            idx = 100 * run_index + result_index
            transform = binding[result.dataset_id].transform
            source = {'dataset_id': result.dataset_id, 'processing_revision_id': result.processing_revision_id,
                      'data_kind': 'synthetic' if snapshots[result.dataset_id].metadata.get('synthetic') is True else 'experimental' if snapshots[result.dataset_id].metadata.get('synthetic') is False else snapshots[result.dataset_id].metadata.get('data_kind','not_declared'),
                      'run_id': rid, 'snapshot_sha256': run.snapshot_sha256,
                      'arrays': {k: v.model_dump(mode='json') for k, v in result.arrays.items()}}
            if {'k', 'data_chi', 'model_chi'} <= a.keys():
                _pair(a['k'], a['data_chi'], rid); _pair(a['k'], a['model_chi'], rid)
                values = {n: a[n] for n in ('k', 'data_chi', 'model_chi')}
                values['residual_chi'] = a['data_chi'] - a['model_chi']
                add('F07', idx, f'{label} · 拟合与残差', f'{label} · Fit and residual', values,
                    [_panel('k', ['data_chi', 'model_chi'], 'k (Å⁻¹)', 'χ(k)', ['Data', 'Model']),
                     _panel('k', ['residual_chi'], 'k (Å⁻¹)', 'Residual χ(k)', ['Data − model'])], [source], kind='residual',
                    method={'weight': 0, 'space': 'k', 'transform': transform.model_dump(mode='json'),
                            'converged': run.result.converged, 'assessment': run.result.assessment,
                            'statistics': run.result.statistics.model_dump(mode='json')})
            if {'r', 'data_re', 'data_im', 'model_re', 'model_im'} <= a.keys():
                values = {n: a[n] for n in ('r', 'data_re', 'data_im', 'model_re', 'model_im')}
                for n in ('data_re', 'data_im', 'model_re', 'model_im'):
                    _pair(a['r'], a[n], rid)
                values.update(residual_re=a['data_re']-a['model_re'], residual_im=a['data_im']-a['model_im'])
                add('F07', 10000+idx, f'{label} · R 空间拟合', f'{label} · R-space fit', values,
                    [_panel('r', ['data_re','model_re','data_im','model_im'], 'R (Å; uncorrected)', 'χ(R)', ['Data Re','Model Re','Data Im','Model Im']),
                     _panel('r', ['residual_re','residual_im'], 'R (Å; uncorrected)', 'Residual χ(R)', ['Re residual','Im residual'])], [source], kind='residual',
                    method={'phase_corrected': False, 'space': 'r', 'weight': transform.kweight,
                            'transform': transform.model_dump(mode='json')})
            if 'weighted_residual' in a:
                add('F07', 20000+idx, f'{label} · 拟合目标残差', f'{label} · Weighted objective residual',
                    {'objective_index':np.arange(len(a['weighted_residual']),dtype=float), 'weighted_residual':a['weighted_residual']},
                    [_panel('objective_index', ['weighted_residual'], 'Objective index (1)', 'Weighted residual (1)')], [source],
                    method={'quantity': 'objective_residual', 'transform': transform.model_dump(mode='json'),
                            'definition': 'exact signed residual vector of this dataset in the optimizer order'},
                    notes=['带符号加权残差保留优化器顺序。横轴索引不是 R。'])
            for path_index, (path_id, refs) in enumerate(result.path_contributions.items()):
                p = _arrays(store, project.id, refs)
                if not {'k', 'chi'} <= p.keys():
                    missing.append({'family': 'F08', 'path_id': path_id, 'reason': '路径贡献数组不完整。'}); continue
                _pair(p['k'], p['chi'], path_id)
                add('F08', 10000*run_index+100*result_index+path_index, f'{label} · 路径贡献 {path_index+1}', f'{label} · Path contribution {path_index+1}',
                    {'k': p['k'], 'path_chi': p['chi']}, [_panel('k', ['path_chi'], 'k (Å⁻¹)', 'Path χ(k)')],
                    [{**source, 'path_binding_id': path_id, 'path_arrays': {n:r.model_dump(mode='json') for n,r in refs.items()}}],
                    notes=['路径贡献为原始 χ(k)；未将各路径 FT 幅值相加。几何信息保留在工程中。'])
            assets={p.id:p for p in run.input_snapshot.paths}
            for path_index,path_binding in enumerate(binding[result.dataset_id].paths):
                asset=assets.get(path_binding.path_id)
                if asset is None or not asset.geometry:
                    missing.append({'family':'F08','dataset_id':result.dataset_id,'path_binding_id':path_binding.id,
                                    'reason':'此路径未保存可核查的几何原子序列，未补造模型示意。'})
                    continue
                coordinates=np.asarray([atom.xyz for atom in asset.geometry],float)
                values={'atom_index':np.arange(len(asset.geometry),dtype=float)+1,
                        'x_A':coordinates[:,0],'y_A':coordinates[:,1],'z_A':coordinates[:,2],
                        'occupancy':np.asarray([atom.occupancy for atom in asset.geometry])}
                atom_labels=[f'{atom.element}{i+1}' for i,atom in enumerate(asset.geometry)]
                add('F08',50000+10000*run_index+100*result_index+path_index,
                    f'{label} · 路径模型投影 {path_index+1}',f'{label} · Path model projections {path_index+1}',values,
                    [{'x':'x_A','y':'y_A','xlabel':'x (Å)','ylabel':'y (Å)','projection':'XY','atom_labels':atom_labels},
                     {'x':'x_A','y':'z_A','xlabel':'x (Å)','ylabel':'z (Å)','projection':'XZ','atom_labels':atom_labels}],
                    [{**source,'path_binding_id':path_binding.id,'path_id':asset.id,
                      'path_source':asset.source_file.model_dump(mode='json'),'geometry':[a.model_dump(mode='json') for a in asset.geometry]}],kind='geometry',
                    method={'projection':'XY and XZ projections of stored FEFF geometry; model schematic, not an independent measurement',
                            'connectivity':'stored FEFF vertex sequence, closed to first vertex; no inferred bonds',
                            'nleg':asset.nleg,'effective_half_path_A':asset.reff,
                            'length_semantics':'effective half path length; never labelled as a single MS bond',
                            'atoms':[a.model_dump(mode='json') for a in asset.geometry]},
                    notes=['模型示意，来自真实保存的 FEFF 顶点坐标；投影重叠不代表原子重合。',
                           'MS 的有效半路径长不标注为单根键长。'])
        covariance = run.result.covariance
        if covariance and covariance.matrix is not None and covariance.var_names:
            cov = np.asarray(covariance.matrix)
            diagonal = np.diag(cov)
            if np.any(diagonal <= 0):
                missing.append({'family':'F11','run_id':rid,'reason':'协方差对角线非正，相关系数不可估计。'})
                continue
            matrix = cov / np.sqrt(np.outer(diagonal, diagonal))
            add('F11', run_index, '参数相关性', 'Parameter correlation',
                {'index': np.arange(len(covariance.var_names), dtype=float), 'correlation': matrix},
                [{'x':'index','y':'index','z':'correlation','xlabel':'Parameter','ylabel':'Parameter',
                  'ticks': covariance.var_names, 'colorlabel':'Correlation'}],
                [{'run_id':rid,'snapshot_sha256':run.snapshot_sha256}], kind='heatmap',
                method={'color_min':-1.,'color_max':1.,'colormap':'RdBu_r'})

    for index, lcf in enumerate(workflow.get('lcf_results', [])):
        a = _arrays(store, project.id, lcf.get('arrays', {}))
        if not {'energy', 'data', 'model', 'residual'} <= a.keys():
            missing.append({'family':'F09','dataset_id':lcf.get('dataset_id'),'reason':'LCF 数组不完整。'}); continue
        ys = ['data', 'model'] + sorted((n for n in a if n.startswith('contribution_')),key=lambda n:int(n.split('_')[-1]))
        for n in [*ys, 'residual']:
            _pair(a['energy'], a[n], 'LCF')
        dataset = by_dataset[lcf['dataset_id']]
        label = _label(dataset, index)
        labels = ['Data', 'Model'] + [f'Reference {int(n.split("_")[-1])+1}' for n in ys[2:]]
        add('F09', index, f'{label} · 参考组分拟合', f'{label} · Reference-component fit', a,
            [_panel('energy',ys,'E (eV)','Normalized μ(E)',labels),
             _panel('energy',['residual'],'E (eV)','Residual μnorm',['Data − model'])],
            [{'dataset_id':dataset.id,'processing_revision_ids':lcf.get('processing_ids',[]),'lcf_arrays':lcf['arrays']}], kind='residual',
            method={k:v for k,v in lcf.items() if k!='arrays'},
            notes=['系数是给定参考与约束下的参考组分系数；不自动等同于含量或价态。'])
        coefficients = lcf.get('coefficients') or []
        reference_ids = lcf.get('reference_ids') or []
        if coefficients and len(coefficients) == len(reference_ids):
            panel = _panel('reference_index', ['coefficient'], 'Reference', 'Coefficient (1)', ['Coefficient'])
            panel.update(state_tick_labels=[f'Ref. {i+1}' for i in range(len(coefficients))], series_modes=['markers'])
            add('F09', 100 + index, f'{label} · 参考组分系数', f'{label} · Reference coefficients',
                {'reference_index': np.arange(len(coefficients), dtype=float), 'coefficient': np.asarray(coefficients)},
                [panel], [{'dataset_id': dataset.id, 'processing_revision_ids': lcf.get('processing_ids', [])}],
                method={'quantity':'reference_coefficients', 'reference_labels':[
                            _label(by_dataset[d], i) if d in by_dataset else f'Reference {i+1}' for i,d in enumerate(reference_ids)],
                        **{k:v for k,v in lcf.items() if k!='arrays'}},
                notes=['滑动分析窗口和移除参考谱是敏感性检查；其波动不是置信区间。'])
    series = workflow.get('series') or {}
    axis='energy' if 'energy' in series.get('arrays',{}) else 'k'
    if {axis, 'matrix', 'state_values'} <= series.get('arrays', {}).keys():
        a = _arrays(store, project.id, series['arrays'])
        if a['matrix'].shape != (len(a['state_values']), len(a[axis])):
            raise ProjectError('series_array_shape', '序列矩阵必须对应实际状态与能量坐标。')
        if len(a['state_values']) >= 2 and np.all(np.diff(a['state_values']) > 0):
            add('F10', 0, '状态序列变化', 'Measured state sequence', a,
                [{'x':axis,'y':'state_values','z':'matrix','xlabel':'E (eV)' if axis=='energy' else 'k (Å⁻¹)',
                  'ylabel':str(series.get('state_key','State')),'colorlabel':'Normalized μ(E)' if axis=='energy' else 'χ(k)'}],
                [{'series_arrays':series['arrays'],'points':series.get('points',[])}], kind='heatmap',
                method={'state_key':series.get('state_key'),'normalization':'stored sequence; no per-row normalization'},
                notes=['仅使用实际状态元数据；不补造失败状态或进行趋势平滑。'])
        else:
            missing.append({'family':'F10','reason':'状态值不足或并非严格递增；不建立会误导的热图。'})
    from .decomposed import add_decomposition, add_preedge
    add_preedge(store, project, workflow, add)
    add_decomposition(store, project, workflow, add)
    from .trends import add_series_trends
    add_series_trends(project,workflow,add,missing)
    # A compact evidence sequence: near-edge shape, EXAFS oscillation,
    # radial transform, then complementary k–R information.
    children = []
    for family in ('F02','F04','F05','F06'):
        match = next((f for f in output if f.spec.family == family), None)
        if match is not None: children.append(match)
    if len(children) >= 2:
        panel_labels={'F02':'Normalized μ(E) (1)','F04':None,'F05':'χ(R) (FT)','F06':'R (Å; uncorrected)'}
        panels=[{'child_id':f.spec.id,'ylabel':panel_labels[f.spec.family] or f.spec.panels[0]['ylabel']} for f in children]
        item = add('F12', 0, '处理结果概览', 'Processing overview', {}, panels,
                   [s for f in children for s in f.spec.sources], kind='composite',
                   method={'panels': [f.spec.id for f in children],
                           'question':'What complementary spectral information is available from this workflow?',
                           'claim':'Stored near-edge and EXAFS views describe the available spectral evidence; they do not establish a unique structure.'},
                   notes=['各面板绑定原快照；跨空间视图不构成独立重复测量。'])
        item.children = children
        item.spec = item.spec.model_copy(update={'data_files': [name for f in children for name in f.spec.data_files]})
    present = {f.spec.family for f in output}
    reasons = {'F01':'无原始 μ(E) 处理输入。','F02':'无归一化 E–μ 数据；不能由 χ(k) 伪造。',
               'F03':'无适用 E–μ 导数输入。','F04':'无有效 χ(k)。','F05':'无已存 FT 数组。',
               'F06':'无适用 χ(k) 或公共 k 区间不足。','F07':'未完成适用拟合。',
               'F08':'无已存路径贡献；模型几何保留在工程。','F09':'未完成适用 LCF。',
               'F10':'未提供适用的实际状态序列。','F11':'无已存相关性矩阵或诊断图输入。',
               'F12':'适用的互补面板少于两项。'}
    for family, reason in reasons.items():
        if family not in present and not any(m['family']==family for m in missing):
            missing.append({'family':family,'reason':reason})
    return output, missing


def _sup(value):
    return str(value).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))


def _ft_basis(processing):
    calls=[c for c in processing.resolved_calls if c.get('operation')=='xftf']
    if len(calls)!=1 or not processing.engine_versions.get('xraylarch'):return None
    args=calls[0].get('arguments',{})
    keys=('kmin','kmax','kweight','dk','window','kstep','nfft')
    if not all(k in args for k in keys):return None
    return {'engine':processing.engine_versions['xraylarch'],'operation':'xftf',
            'arguments':{k:args[k] for k in keys},'normalization':'Larch Δk/sqrt(pi) FFT[k^w χ(k) window]'}
