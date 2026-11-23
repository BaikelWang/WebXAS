"""Statistical panels drawn from one immutable multivariate result."""
import json
import numpy as np


def add_decomposition(store, project, workflow, add):
    ref = workflow.get('decomposition')
    if not ref:
        return
    result = json.loads(store.read_asset(project.id, ref))
    pca, mcr = result['pca'], result['mcr']
    energy = np.asarray(result['energy_eV'])
    ids = result['dataset_ids']
    sample = np.arange(1, len(ids) + 1)
    sources = [dict(dataset_id=did, processing_id=pid, snapshot_sha256=ref['sha256'])
               for did, pid in zip(ids, result['processing_ids'], strict=True)]
    names = {dataset.id:dataset.filename for dataset in project.datasets}
    labels = [names[did] for did in ids]
    method = dict(artifact=ref, sample_order=ids, sample_labels=labels, components=result['n_components'],
                  window_eV=result['window_eV'], relative_error=mcr['relative_error'],
                  stability=mcr['stability']['max_concentration_range'], converged=mcr['converged'])

    def panel(index, title, english, xname, x, xlabel, matrix, ylabel, series_labels):
        ys = [f'curve_{i+1}' for i in range(len(matrix))]
        spec = dict(x=xname, ys=ys, labels=series_labels, xlabel=xlabel, ylabel=ylabel,
                    distinct_curves=len(series_labels)>1)
        if xname == 'sample_index':
            spec['series_modes'] = ['markers'] * len(ys)
        values=np.asarray(matrix)
        indicators={}
        if english=='Explained variance':
            indicators['retained_variance']=float(values[0,:result['n_components']].sum())
        if english.endswith('residual'):
            indicators['residual_rms']=float(np.sqrt(np.mean(values**2)))
        add('F13', index, title, english, {xname:x, **dict(zip(ys, np.asarray(matrix), strict=True))},
            [spec], sources, method={**method, **indicators, 'quantity':english,'ylabel':ylabel},
            notes=['统计组分存在非唯一性，需独立标样确认物种及绝对含量。'])

    components = [f'Component {i+1}' for i in range(result['n_components'])]
    variance = np.asarray(pca['explained_variance_ratio'])
    panel(1, 'PCA 解释方差', 'Explained variance', 'component', np.arange(1,len(variance)+1),
          'Principal component', [variance], 'Variance fraction', ['Variance'])
    panel(2, 'PCA 得分', 'PCA scores', 'sample_index', sample, 'Sample index',
          np.asarray(pca['scores']).T, 'Score', components)
    panel(3, 'PCA 载荷', 'PCA loadings', 'energy_eV', energy, 'Energy (eV)', pca['loadings'], 'Loading', components)
    panel(4, 'MCR 组分谱', 'MCR profiles', 'energy_eV', energy, 'Energy (eV)', mcr['spectra'], 'Normalized absorption', components)
    panel(5, 'MCR 系数', 'MCR coefficients', 'sample_index', sample, 'Sample index',
          np.asarray(mcr['concentrations']).T, 'Coefficient', components)
    for index, (title, block) in enumerate([('PCA',pca),('MCR',mcr)], 6):
        panel(index*2, title+' 重构', title+' reconstruction', 'energy_eV', energy, 'Energy (eV)',
              block['reconstruction'], 'Normalized absorption', labels)
        panel(index*2+1, title+' 残差', title+' residual', 'energy_eV', energy, 'Energy (eV)',
              block['residual'], 'Data - reconstruction', labels)

def add_preedge(store, project, workflow, add):
    for index,record in enumerate(workflow.get('preedge_results',[])):
        ref=record['artifact']
        raw=json.loads(store.read_asset(project.id,ref))
        source=[dict(dataset_id=raw['dataset_id'],processing_id=raw['processing_id'],snapshot_sha256=ref['sha256'])]
        method={k:raw[k] for k in ('area_eV','centroid_eV','baseline_only_area_se_eV','left_eV','peak_eV','right_eV','degree')}
        for offset,title,english,names,labels in [
                (0,'预边与拟合背景','Pre-edge baseline',('energy_eV','observed','baseline'),['Absorption','Baseline']),
                (1,'预边积分','Pre-edge integral',('peak_energy_eV','integrand'),['Baseline-subtracted signal'])]:
            arrays={key:np.asarray(raw[key]) for key in names}
            panel=dict(x=names[0],ys=list(names[1:]),labels=labels,xlabel='Energy (eV)',ylabel='Normalized absorption')
            add('F03',500+index*2+offset,title,english,arrays,[panel],source,method={**method,'quantity':'preedge_integral',
                'artifact':ref,'panel_title':title},notes=['面积标准误仅包含背景拟合的条件不确定度。'])
