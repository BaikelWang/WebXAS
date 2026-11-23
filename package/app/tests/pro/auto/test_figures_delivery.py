"""Scientific source identity, display-only editing and actual export geometry."""
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

from xasweb.pro.contracts import Dataset, ProcessingRevision, Project
from xasweb.pro.delivery import build_delivery, run_job
from xasweb.pro.figures import FigureStyle, collect_figures
from xasweb.pro.preprocessing import save_array
from xasweb.pro.projects import ProjectError, ProjectStore, canonical


def make_project(store, kind='mu'):
    project=store.create('Cu 标准样 <script>')
    energy=np.linspace(8900.,9800.,361)
    norm=1/(1+np.exp(-(energy-8979)/3))+.08*np.exp(-((energy-8990)/8)**2)
    k=np.linspace(.05,14.,280)
    chi=np.sin(2*2.1*k)*np.exp(-k/8)/(1+k)
    r=np.linspace(0.,6.,161)
    arrays={'k':k,'chi':chi,'r':r,'chir_re':np.sin(r)*np.exp(-r),'chir_im':np.cos(r)*np.exp(-r)}
    arrays['chir_mag']=np.hypot(arrays['chir_re'],arrays['chir_im'])
    if kind=='mu': arrays.update(energy=energy,mu=.25+.0001*(energy-8900)+norm,norm=norm)
    raw='\n'.join(f'{x:.17g} {y:.17g}' for x,y in zip(energy,arrays.get('mu',norm)))
    dataset=Dataset(id='dataset1',filename='铜箔.dat',kind=kind,
                    raw_file=store.put_asset(project.id,raw.encode('utf-8'),'input.dat'),
                    sample_id='sample1',element='Cu',edge='K',metadata={'synthetic':True})
    processing=ProcessingRevision(id='processing1',dataset_id=dataset.id,
        arrays={n:save_array(store,project.id,a) for n,a in arrays.items()},
        engine_versions={'xraylarch':'2026.3.1'},resolved_calls=[{'operation':'fixture','arguments':{}}])
    value=project.model_dump(mode='json')
    value.update(datasets=[dataset.model_dump(mode='json')],processing=[processing.model_dump(mode='json')])
    value['metadata']['automation_runs']=[{'id':'workflow1','version':1,'task':'process','state':'succeeded',
        'processing_ids':[processing.id],'decisions':[],'issues':[],'lcf_results':[],'fit_run_ids':[]}]
    project=store.save(Project.model_validate(value),expected_revision=project.revision)
    return project,arrays


def test_difference_freezes_both_operands_and_matches_export(tmp_path):
    from xasweb.pro.figures.individual import independent_panels
    store = ProjectStore(tmp_path)
    project, arrays = make_project(store)
    original = project.processing[0]
    reference = project.datasets[0].model_copy(update={'id':'reference'})
    alternate_energy = arrays['energy'][10:-8] + .3
    alternate_norm = arrays['norm'][10:-8] * .95
    other = ProcessingRevision(id='reference_pp',dataset_id=reference.id,
        arrays={'energy':save_array(store,project.id,alternate_energy),
                'norm':save_array(store,project.id,alternate_norm)})
    project.datasets.append(reference)
    project.processing.append(other)
    workflow = project.metadata['automation_runs'][0]
    workflow['processing_ids'].append(other.id)
    pair = [original.id,other.id]
    items, _ = collect_figures(store, project, workflow, difference_pair=pair)
    figure = next(item for item in independent_panels(items) if item.spec.method.get('quantity') == 'normalized_difference')
    a = figure.arrays
    assert a['energy_eV'][0] >= alternate_energy[0] and a['energy_eV'][-1] <= alternate_energy[-1]
    np.testing.assert_array_equal(a['reference_norm'],np.interp(a['energy_eV'],alternate_energy,alternate_norm))
    np.testing.assert_array_equal(a['difference'],a['sample_norm']-a['reference_norm'])
    with pytest.raises(ProjectError,match='两个不同归一化谱'):
        collect_figures(store,project,workflow,difference_pair=[original.id,original.id])


@pytest.fixture(scope='module')
def delivered(tmp_path_factory):
    store=ProjectStore(tmp_path_factory.mktemp('figures')/'projects')
    project,arrays=make_project(store)
    before=(store.folder(project.id)/'project.json').read_bytes()
    with patch.object(store,'save',side_effect=AssertionError('builder is read only')), \
         patch.object(store,'put_asset',side_effect=AssertionError('builder cannot add assets')):
        payload=build_delivery(store,project.id,'workflow1')
    assert (store.folder(project.id)/'project.json').read_bytes()==before
    return store,project,arrays,payload


def test_delivery_no_fit_has_exact_sources_formats_hashes_and_fixed_size(delivered):
    store,project,arrays,payload=delivered
    with ZipFile(io.BytesIO(payload)) as archive:
        assert archive.testzip() is None
        manifest=json.loads(archive.read('manifest.json'))
        families={f['family'] for f in manifest['figures']}
        assert {'F01','F02','F03','F04','F05','F06','F12'}==families
        assert {'F07','F08','F09','F10','F11'}<={m['family'] for m in manifest['missing']}
        assert not project.runs
        for name,record in manifest['files'].items():
            blob=archive.read(name)
            assert hashlib.sha256(blob).hexdigest()==record['sha256']
            assert len(blob)==record['byte_size']
        spec=json.loads(archive.read('specs/F04-000.json'))
        assert spec['sources'][0]['processing_revision_id']=='processing1'
        assert spec['sources'][0]['arrays']['chi']['sha256']==project.processing[0].arrays['chi'].sha256
        csv=archive.read('data/F04-000.csv').decode('utf-8-sig')
        values=np.loadtxt(io.StringIO(csv),delimiter=',',skiprows=1)
        np.testing.assert_array_equal(values[:,0],arrays['k'])
        np.testing.assert_array_equal(values[:,1],arrays['chi'])
        np.testing.assert_array_equal(values[:,2],arrays['chi']*arrays['k']**2)
        for extension in ('svg','pdf','png','tiff'):
            assert f'figures/F04-000.{extension}' in archive.namelist()
        for extension in ('png','tiff'):
            im=Image.open(io.BytesIO(archive.read(f'figures/F04-000.{extension}')))
            assert abs(im.width-85/25.4*300)<=1
            assert abs(im.height-68/25.4*300)<=1
            assert abs(im.info['dpi'][0]-300)<1
        svg=archive.read('figures/F04-000.svg').decode('utf-8')
        assert '<text' in svg and 'Å' in svg and 'Times New Roman' in svg
        assert '铜箔' not in svg  # module/source title remains in FigureSpec, never in the plot
        assert 'processing1' not in svg
        assert '&lt;script&gt;' in archive.read('README.html').decode('utf-8')
        assert '<script>' not in archive.read('README.html').decode('utf-8')
        wt=np.load(io.BytesIO(archive.read('data/F06-000.npz')))
        assert wt['magnitude'].shape==(len(wt['r']),len(wt['k']))
        assert len(wt['k'])>180  # original engine preview subsamples every 2 points
        wtcsv=np.loadtxt(io.StringIO(archive.read('data/F06-000.csv').decode('utf-8-sig')),delimiter=',',skiprows=1)
        np.testing.assert_array_equal(wtcsv[:,2].reshape(wt['magnitude'].shape),wt['magnitude'])
        assert json.loads(archive.read('specs/F06-000.json'))['method']['normalization'].startswith('absolute')
        reopened=store.import_archive(archive.read('project/project.webxas.zip'))
        assert reopened.processing[0].arrays==project.processing[0].arrays
        assert not reopened.runs


def test_display_style_changes_never_modify_data_or_fit(delivered):
    store,project,arrays,original=delivered
    before=canonical(project.model_dump(mode='json'))
    with patch('xasweb.pro.preprocessing.analyze',side_effect=AssertionError('display may not reprocess')):
        styled=build_delivery(store,project.id,'workflow1',{
            'layout':'double','dpi':600,'language':'en','formats':['png'],'palette':'muted',
            'offset':.2,'xlim':[2.,10.]})
    assert canonical(store.read(project.id).model_dump(mode='json'))==before
    with ZipFile(io.BytesIO(original)) as first,ZipFile(io.BytesIO(styled)) as second:
        for name in first.namelist():
            if name.startswith('data/') and name.endswith('.csv'):
                assert first.read(name)==second.read(name)
        for name in ('F04-000','F06-000'):
            a=json.loads(first.read(f'specs/{name}.json'));b=json.loads(second.read(f'specs/{name}.json'))
            assert a['sources']==b['sources'] and a['method']==b['method']
        im=Image.open(io.BytesIO(second.read('figures/F04-000.png')))
        assert abs(im.width-180/25.4*600)<=1
        assert abs(im.height-115/25.4*600)<=1


def test_chi_snapshot_does_not_manufacture_energy_figures(tmp_path):
    store=ProjectStore(tmp_path/'projects');project,_=make_project(store,'chi')
    figures,missing=collect_figures(store,project,project.metadata['automation_runs'][0])
    assert {'F01','F02','F03'}.isdisjoint({f.spec.family for f in figures})
    assert {'F01','F02','F03'}<={m['family'] for m in missing}
    assert {'F04','F05','F06'}<={f.spec.family for f in figures}


def test_repeated_worker_exports_reopen_without_recursive_archives(delivered):
    store,project,_,_=delivered
    class Context:
        def heartbeat(self,*args):pass
    records=[]
    for layout in ('single','double'):
        current=store.read(project.id)
        with patch('xasweb.pro.delivery.ProjectStore',return_value=store):
            record=run_job({'project_id':project.id,'expected_revision':current.revision,
                           'workflow_id':'workflow1','style':{'layout':layout,'formats':['png']}},Context())
        records.append(record)
        with ZipFile(io.BytesIO(store.read_asset(project.id,record['asset']))) as archive:
            inner=archive.read('project/project.webxas.zip')
            reopened=store.import_archive(inner)
            assert 'automation_deliveries' not in reopened.metadata
            assert len(reopened.metadata['automation_runs'])==1
            with ZipFile(io.BytesIO(inner)) as project_archive:
                assert not any(n.endswith('.zip') for n in project_archive.namelist())
    current=store.read(project.id)
    assert current.metadata['automation_deliveries']==records
    assert records[0]['id']!=records[1]['id']
    assert records[0]['asset']['sha256']!=records[1]['asset']['sha256']


def test_missing_snapshot_invalid_style_stale_and_oversize_are_explicit(tmp_path):
    store=ProjectStore(tmp_path/'projects');project,_=make_project(store)
    with pytest.raises(ProjectError,match='展示设置'):
        build_delivery(store,project.id,'workflow1',{'dpi':150})
    with pytest.raises(ProjectError) as error:
        build_delivery(store,project.id,'missing')
    assert error.value.code=='workflow_not_found'
    with patch('xasweb.pro.delivery.MAX_DELIVERY_BYTES',10):
        with pytest.raises(ProjectError) as error: build_delivery(store,project.id,'workflow1')
    assert error.value.code=='delivery_too_large'
    value=project.model_dump(mode='json')
    value['metadata']['automation_runs'][0]['processing_ids']=['missing']
    broken=Project.model_validate(value)
    with pytest.raises(ProjectError) as error: collect_figures(store,broken,broken.metadata['automation_runs'][0])
    assert error.value.code=='processing_not_found'
    with patch('xasweb.pro.delivery.ProjectStore',return_value=store):
        with pytest.raises(ProjectError) as error:
            run_job({'project_id':project.id,'expected_revision':-1,'workflow_id':'workflow1'},None)
    assert error.value.code=='revision_conflict'


def test_presentation_is_fixed_sixteen_by_nine():
    style=FigureStyle(layout='presentation')
    assert style.size_mm[0]/style.size_mm[1]==16/9
    with pytest.raises(ValueError): FigureStyle(formats=['pdf'])


def test_individual_panels_are_rendered_from_distinct_arrays_and_axes():
    from xasweb.pro.figures import FigureSpec
    from xasweb.pro.figures.sources import FigureData
    from xasweb.pro.figures.individual import independent_panels
    from xasweb.pro.figures.render import make_figure
    import matplotlib.pyplot as plt
    x=np.arange(5,dtype=float)
    residual=FigureData(FigureSpec(id='F07-000',family='F07',title='拟合',title_en='Fit',kind='residual',
        panels=[{'x':'k','ys':['data','model'],'labels':['Data','Model'],'xlabel':'k (Å⁻¹)','ylabel':'χ(k)'},
                {'x':'k','ys':['residual'],'labels':['Residual'],'xlabel':'k (Å⁻¹)','ylabel':'Residual χ(k)'}]),
        {'k':x,'data':x+1,'model':x,'residual':np.ones(5)})
    panels=independent_panels([residual])
    assert [p.spec.id for p in panels]==['F07-000-p01','F07-000-p02']
    assert set(panels[0].arrays)=={'k','data','model'}
    assert set(panels[1].arrays)=={'k','residual'}
    for item in panels:
        figure,geometry=make_figure(item)
        try:
            assert len(figure.axes)==len(geometry['axes'])==1
            assert len(item.spec.panels)==1 and item.spec.kind=='line'
        finally:plt.close(figure)
    geometry_item=FigureData(FigureSpec(id='F08-50000',family='F08',title='路径模型',title_en='Geometry',kind='geometry',
        panels=[{'x':'x_A','y':'y_A','projection':'XY','atom_labels':['Cu1','O2'],'xlabel':'x (Å)','ylabel':'y (Å)'},
                {'x':'x_A','y':'z_A','projection':'XZ','atom_labels':['Cu1','O2'],'xlabel':'x (Å)','ylabel':'z (Å)'}],
        method={'atoms':[{'element':'Cu'},{'element':'O'}]}),
        {'x_A':np.array([0.,1.]),'y_A':np.array([0.,.5]),'z_A':np.array([0.,2.]),'atom_index':np.array([1.,2.])})
    for index,item in enumerate(independent_panels([geometry_item])):
        assert ('z_A' in item.arrays)==(index==1)
        assert ('y_A' in item.arrays)==(index==0)
        figure,geometry=make_figure(item)
        try:assert len(figure.axes)==len(geometry['axes'])==1
        finally:plt.close(figure)


def test_individual_delivery_freezes_all_families_and_restyles_without_scientific_recompute(tmp_path):
    store=ProjectStore(tmp_path/'projects');project,arrays=make_project(store)
    before=canonical(project.model_dump(mode='json'))
    payload=build_delivery(store,project.id,'workflow1',{'formats':['png']},'individual',['F04'])
    assert canonical(store.read(project.id).model_dump(mode='json'))==before
    with ZipFile(io.BytesIO(payload)) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        assert manifest['presentation']=='individual' and manifest['figure_families']==['F04']
        assert [f['id'] for f in manifest['figures']]==['F04-000-p01']
        assert 'F12' not in {f['family'] for f in manifest['snapshot_figures']}
        assert {'F01','F02','F03','F04','F05','F06'}=={f['family'] for f in manifest['snapshot_figures']}
        figure=manifest['figures'][0]
        assert len(figure['formats'])==4
        for extension in ('svg','pdf','png','tiff'):
            assert f'figures/F04-000-p01.{extension}' in figure['formats']
        data=archive.read(figure['data_files'][0]).decode('utf-8-sig')
        assert 'k_inv_A [Å⁻¹]' in data and 'workflow_id,panel_id,dataset_ids,sample_names,source_snapshot_sha256' in data
        assert 'workflow1,F04-000-p01' in data
        insight=figure['insight']
        assert insight['observations'] and insight['limitations'] and insight['sources']
        assert insight['dataset_ids']==['dataset1'] and insight['sample_names']==['铜箔.dat']
        assert all({'label','value','unit','basis'}<=m.keys() for m in insight['metrics'])
        assert '配位数' in ''.join(insight['limitations']) and '键长' in ''.join(insight['limitations'])
        import html
        assert html.escape(insight['limitations'][0],quote=True) in archive.read('README.html').decode('utf-8')
        assert '<script>' not in archive.read('README.html').decode('utf-8')
    asset=store.put_asset(project.id,payload,'delivery.zip')
    value=project.model_dump(mode='json')
    value['datasets'][0]['filename']='current mutable renamed sample.dat'
    value['metadata']['automation_deliveries']=[{'id':'delivery1','workflow_id':'workflow1',
        'presentation':'individual','asset':asset.model_dump(mode='json')}]
    current=store.save(Project.model_validate(value),project.revision)
    with patch('xasweb.pro.delivery.collect_figures',side_effect=AssertionError('style/family edits cannot recompute scientific transforms')), \
         patch('xasweb.pro.preprocessing.analyze',side_effect=AssertionError('cannot analyze')):
        second=build_delivery(store,project.id,'workflow1',{'layout':'double','language':'en'},'individual',['F02','F04'])
    with ZipFile(io.BytesIO(second)) as archive:
        manifest2=json.loads(archive.read('manifest.json'))
        assert {f['family'] for f in manifest2['figures']}=={'F02','F04'}
        assert all(f['sample_names']==['铜箔.dat'] for f in manifest2['figures'])
        with np.load(io.BytesIO(archive.read('data/F04-000-p01.npz'))) as source:
            np.testing.assert_array_equal(source['chi'],arrays['chi']) if 'chi' in source else np.testing.assert_array_equal(source['weighted_chi'],arrays['chi']*arrays['k']**2)
    unchanged=store.read(project.id)
    assert unchanged.revision==current.revision and not unchanged.runs
    assert len(unchanged.metadata['automation_runs'])==1


def test_individual_missing_families_are_explicit_without_fabricated_figures(tmp_path):
    store=ProjectStore(tmp_path/'projects');project,_=make_project(store,'chi')
    payload=build_delivery(store,project.id,'workflow1',presentation='individual',figure_families=['F01','F12'])
    with ZipFile(io.BytesIO(payload)) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        assert manifest['figures']==[]
        assert {'F01','F12'}=={m['family'] for m in manifest['missing']}
        assert not any(n.startswith('figures/') for n in archive.namelist())
    with pytest.raises(ProjectError):build_delivery(store,project.id,'workflow1',presentation='invalid')
    with pytest.raises(ProjectError):build_delivery(store,project.id,'workflow1',figure_families=['F04','F04'])


def test_insight_uses_bound_features_and_keeps_joint_statistics_at_run_scope(tmp_path):
    from types import SimpleNamespace
    from xasweb.pro.figures import FigureSpec
    from xasweb.pro.figures.sources import FigureData
    from xasweb.pro.figures.insights import figure_insight
    store=ProjectStore(tmp_path/'projects');project,a=make_project(store)
    workflow={**project.metadata['automation_runs'][0],'features':[
        {'processing_id':'foreign','edge_position_eV':99999},
        {'processing_id':'processing1','edge_position_eV':8982.,'edge_definition':'fixture recorded derivative maximum',
         'white_line_height':1.05,'white_line_definition':'fixture recorded window'}]}
    item=FigureData(FigureSpec(id='F02-000-p01',family='F02',title='近边谱',title_en='XANES',
        sources=[{'dataset_id':'dataset1','processing_revision_id':'processing1'}],
        panels=[{'x':'energy_eV','ys':['normalized_mu'],'xlabel':'E (eV)','ylabel':'Normalized μ(E) (1)'}]),
        {'energy_eV':a['energy'],'normalized_mu':a['norm']})
    insight=figure_insight(project,workflow,item)
    assert next(m for m in insight['metrics'] if m['label']=='工作流已存边位')['value']==8982.
    assert '8982' in ''.join(insight['observations']) and '未记录标准样绝对能量校准' in ''.join(insight['observations'])
    assert '99999' not in json.dumps(insight)
    other=project.datasets[0].model_copy(update={'id':'dataset2','filename':'second sample.dat'})
    run=SimpleNamespace(id='joint1',input_snapshot=SimpleNamespace(datasets=[project.datasets[0],other]),
        result=SimpleNamespace(datasets=[SimpleNamespace(dataset_id='dataset1'),SimpleNamespace(dataset_id='dataset2')],
            parameters=[SimpleNamespace(stderr=.02),SimpleNamespace(stderr=None)],converged=True,
            uncertainty_basis='joint_reference',statistics=SimpleNamespace(reduced_chisquare=123.)))
    joint=SimpleNamespace(datasets=project.datasets,processing=project.processing,calibrations=[],runs=[run])
    item=FigureData(FigureSpec(id='F07-000-p02',family='F07',title='残差',title_en='Residual',
        sources=[{'dataset_id':'dataset1','run_id':'joint1','snapshot_sha256':'a'*64}],
        panels=[{'x':'k','ys':['residual_chi'],'xlabel':'k (Å⁻¹)','ylabel':'Residual χ(k)'}]),
        {'k':np.arange(5,dtype=float),'residual_chi':np.full(5,.2)})
    insight=figure_insight(joint,workflow,item)
    assert insight['dataset_ids']==['dataset1']
    stat=next(m for m in insight['metrics'] if '约化' in m['label'])
    assert stat['label']=='联合运行约化 χ²' and '覆盖 2 条谱' in stat['basis']
    assert '2 个记录参数中 1 个带有标准误' in ''.join(insight['observations'])
    assert insight['method']['fit_statistics_scope'][0]['scope']=='joint_run'


def test_insight_different_grids_keep_series_ownership_and_mask_missing_states(tmp_path):
    from types import SimpleNamespace
    from xasweb.pro.figures import FigureSpec
    from xasweb.pro.figures.sources import FigureData
    from xasweb.pro.figures.insights import figure_insight
    store=ProjectStore(tmp_path/'projects');project,a=make_project(store)
    other=project.datasets[0].model_copy(update={'id':'dataset2','filename':'second sample.dat'})
    project=SimpleNamespace(datasets=[project.datasets[0],other],processing=project.processing,calibrations=[],runs=[])
    item=FigureData(FigureSpec(id='F02-900-p01',family='F02',title='叠图',title_en='Comparison',
        sources=[{'dataset_id':'dataset1'},{'dataset_id':'dataset2'}],
        panels=[{'x':'energy_0_eV','xkeys':['energy_0_eV','energy_1_eV'],'ys':['normalized_mu_0','normalized_mu_1'],
                 'labels':['first','second'],'xlabel':'E (eV)','ylabel':'Normalized μ(E) (1)'}]),
        {'energy_0_eV':a['energy'],'normalized_mu_0':a['norm'],'energy_1_eV':np.array([8910.,9000.,9100.]),'normalized_mu_1':np.array([.1,1.,.9])})
    insight=figure_insight(project,{'id':'workflow1'},item)
    assert any('first' in s and '8900–9800' in s for s in insight['observations'])
    assert any('second' in s and '8910–9100' in s for s in insight['observations'])
    assert next(m for m in insight['metrics'] if m['label']=='second · 横轴采样点数')['value']==3
    trend=FigureData(FigureSpec(id='F10-100-p01',family='F10',title='状态',title_en='State',kind='trend',
        panels=[{'x':'state','ys':['value'],'state_valid':'state_valid','xlabel':'temperature_K','ylabel':'Value (1)'}],
        method={'state_key':'temperature_K'}),
        {'state':np.array([300.,0.,500.]),'state_valid':np.array([True,False,True]),'value':np.array([1.,0.,2.])})
    insight=figure_insight(project,{'id':'workflow1'},trend)
    assert next(m for m in insight['metrics'] if m['label']=='横轴下界')['value']==300.
    assert next(m for m in insight['metrics'] if m['label']=='横轴采样点数')['value']==2


def test_stored_fit_lcf_series_residuals_and_snapshot_integrity(tmp_path):
    from xasweb.pro.contracts import (CovarianceRecord,DatasetBinding,DatasetFitResult,
        FitResult,FitRun,FitSpec,FitStatistics,ModelRevision,Parameter,ParameterEstimate,
        ParameterGraph,PathAsset,PathBinding)
    store=ProjectStore(tmp_path/'projects');project,a=make_project(store)
    path=PathAsset(id='path1',source_file=store.put_asset(project.id,b'fixture path; no fitting','path.dat'),
        absorber_element='Cu',edge='K',absorber_site='1',nleg=2,reff=2.1,degen=12.,k_max=20.)
    model=ModelRevision(id='model1',parameters=ParameterGraph(nodes=[Parameter(id='amp',value=1.)]),
        bindings=[DatasetBinding(dataset_id='dataset1',processing_revision_id='processing1',
            paths=[PathBinding(id='path_binding',path_id='path1',expressions={'s02':'amp'})])])
    spec=FitSpec(model=model,datasets=project.datasets,processing=project.processing,paths=[path])
    refs={n:save_array(store,project.id,v) for n,v in {
        'k':a['k'],'data_chi':a['chi'],'model_chi':a['chi']*.98,'r':a['r'],
        'data_re':a['chir_re'],'data_im':a['chir_im'],
        'model_re':a['chir_re']*.98,'model_im':a['chir_im']*.98}.items()}
    stats=FitStatistics(residual_count=100,n_independent=12.,n_variables=1,degrees_of_freedom=11.)
    result=FitResult(run_id='run1',converged=True,termination_reason='synthetic stored result fixture',statistics=stats,
        parameters=[ParameterEstimate(id='amp',value=.98,unit='1',stderr=.02,uncertainty_reason=None)],
        covariance=CovarianceRecord(var_names=['amp'],matrix=[[.0004]],scaling_description='fixture'),
        datasets=[DatasetFitResult(dataset_id='dataset1',processing_revision_id='processing1',arrays=refs,
            path_contributions={'path_binding':{'k':refs['k'],'chi':refs['model_chi']}})])
    run=FitRun(id='run1',model_id='model1',input_snapshot=spec,
        snapshot_sha256=hashlib.sha256(canonical(spec.model_dump(mode='json'))).hexdigest(),task_state='succeeded',result=result)
    lcf_arrays={n:save_array(store,project.id,v).model_dump(mode='json') for n,v in {
        'energy':a['energy'],'data':a['norm'],'model':a['norm']*.99,
        'residual':a['norm']*.01,'contribution_0':a['norm']*.99}.items()}
    state_arrays={n:save_array(store,project.id,v).model_dump(mode='json') for n,v in {
        'energy':a['energy'],'matrix':np.vstack([a['norm'],a['norm']*.99]),'state_values':np.array([0.,10.])}.items()}
    value=project.model_dump(mode='json')
    value.update(paths=[path.model_dump(mode='json')],models=[model.model_dump(mode='json')],runs=[run.model_dump(mode='json')])
    workflow=value['metadata']['automation_runs'][0]
    workflow.update(fit_run_ids=['run1'],lcf_results=[{'dataset_id':'dataset1','reference_ids':['dataset1'],
        'processing_ids':['processing1'],'coefficients':[.99],'arrays':lcf_arrays,
        'diagnostics':{'synthetic_fixture':True}}],series={'state_key':'time_s','arrays':state_arrays,
        'points':[{'dataset_id':'dataset1','state_value':0.},{'dataset_id':'dataset1','state_value':10.}]})
    project=store.save(Project.model_validate(value),expected_revision=project.revision)
    payload=build_delivery(store,project.id,'workflow1',{'formats':['png','pdf'],'language':'en'})
    with ZipFile(io.BytesIO(payload)) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        assert {f'F{n:02}' for n in range(1,13)}=={f['family'] for f in manifest['figures']}
        k=np.loadtxt(io.StringIO(archive.read('data/F07-000.csv').decode('utf-8-sig')),delimiter=',',skiprows=1)
        np.testing.assert_array_equal(k[:,-1],a['chi']-a['chi']*.98)
        lcf=np.genfromtxt(io.StringIO(archive.read('data/F09-000.csv').decode('utf-8-sig')),delimiter=',',names=True)
        np.testing.assert_array_equal(lcf['residual'],a['norm']*.01)
        assert json.loads(archive.read('specs/F11-000.json'))['method']['color_min']==-1.
        assert json.loads(archive.read('tables/fit-000.json'))['parameters'][0]['id']=='amp'
        spec=json.loads(archive.read('specs/F09-000.json'))
        assert spec['method']['coefficients']==[.99]
    value=project.model_dump(mode='json');value['runs'][0]['snapshot_sha256']='0'*64
    store.save(Project.model_validate(value),expected_revision=project.revision)
    with pytest.raises(ProjectError) as error:
        collect_figures(store,store.read(project.id),workflow,{'formats':['png']})
    assert error.value.code=='snapshot_integrity'


def test_comparison_retains_individual_grids_and_common_wt_scale(tmp_path):
    store=ProjectStore(tmp_path/'projects');project,a=make_project(store)
    value=project.model_dump(mode='json')
    dataset=dict(value['datasets'][0]);dataset['id']='dataset2';dataset['filename']='Cu sample 2.dat'
    processing=dict(value['processing'][0]);processing['id']='processing2';processing['dataset_id']='dataset2'
    processing['arrays']=dict(processing['arrays'])
    processing['arrays']['energy']=save_array(store,project.id,a['energy']+1.25).model_dump(mode='json')
    processing['arrays']['chi']=save_array(store,project.id,a['chi']*.5).model_dump(mode='json')
    value['datasets'].append(dataset);value['processing'].append(processing)
    value['metadata']['automation_runs'][0]['processing_ids'].append('processing2')
    project=store.save(Project.model_validate(value),expected_revision=project.revision)
    figures,missing=collect_figures(store,project,project.metadata['automation_runs'][0])
    compare=next(f for f in figures if f.spec.id=='F02-900')
    np.testing.assert_array_equal(compare.arrays['energy_1_eV'],a['energy']+1.25)
    assert compare.spec.panels[0]['xkeys']==['energy_0_eV','energy_1_eV']
    wavelets=[f for f in figures if f.spec.family=='F06']
    assert wavelets[0].spec.method==wavelets[1].spec.method
    np.testing.assert_array_equal(wavelets[0].arrays['k'],wavelets[1].arrays['k'])
    np.testing.assert_allclose(wavelets[0].arrays['magnitude']*.5,wavelets[1].arrays['magnitude'],rtol=1e-12)
    compare_k=next(f for f in figures if f.spec.id=='F04-900')
    np.testing.assert_array_equal(compare_k.arrays['weighted_chi_1'],a['chi']*.5*a['k']**2)
    assert not any(f.spec.id=='F05-900' for f in figures)
    assert any(m['family']=='F05' and '尺度' in m['reason'] for m in missing)
    value=project.model_dump(mode='json')
    settings={'kmin':2.,'kmax':12.,'kweight':2,'dk':1.,'window':'hanning','kstep':.05,'nfft':2048}
    for pp in value['processing']:pp['resolved_calls']=[{'operation':'xftf','arguments':dict(settings)}]
    comparable=Project.model_validate(value)
    figures,_=collect_figures(store,comparable,comparable.metadata['automation_runs'][0])
    compare_r=next(f for f in figures if f.spec.id=='F05-900')
    np.testing.assert_array_equal(compare_r.arrays['chir_mag_1'],a['chir_mag'])
    value['processing'][1]['resolved_calls'][0]['arguments']['kmax']=10.
    incomparable=Project.model_validate(value)
    figures,missing=collect_figures(store,incomparable,incomparable.metadata['automation_runs'][0])
    assert not any(f.spec.id=='F05-900' for f in figures)
    assert any(m['family']=='F05' and '尺度' in m['reason'] for m in missing)


def test_external_normalized_plot_has_no_invented_raw_mu_or_exafs(tmp_path):
    store=ProjectStore(tmp_path/'projects');project,a=make_project(store)
    value=project.model_dump(mode='json')
    value['datasets'][0]['metadata'].update(import_processing_state='normalized',external_processing_note='Provided normalized spectrum; external provenance declared.')
    processing=value['processing'][0]
    processing['arrays']={n:r for n,r in processing['arrays'].items() if n in ('energy','norm')}
    processing['metadata']['external_plot_only']=True
    processing['resolved_calls']=[{'operation':'external_plot_import','arguments':{'processing_state':'normalized'}}]
    project=store.save(Project.model_validate(value),expected_revision=project.revision)
    figures,missing=collect_figures(store,project,project.metadata['automation_runs'][0])
    assert {f.spec.family for f in figures}=={'F02','F03'}
    assert {'F01','F04','F05','F06'}<={m['family'] for m in missing}
    normalized=next(f for f in figures if f.spec.family=='F02')
    np.testing.assert_array_equal(normalized.arrays['normalized_mu'],a['norm'])
    assert normalized.spec.method['external_plot_only'] is True
    assert normalized.spec.method['external_processing_note'].startswith('Provided normalized')


def test_single_parameter_correlation_cell_has_visible_area_and_states_do_not_extrapolate():
    from xasweb.pro.figures.spec import FigureSpec
    from xasweb.pro.figures.sources import FigureData
    from xasweb.pro.figures.render import make_figure
    import matplotlib.pyplot as plt
    spec=FigureSpec(id='F11-test',family='F11',title='相关性',title_en='Correlation',kind='heatmap',
        style=FigureStyle(language='en'),method={'color_min':-1.,'color_max':1.,'colormap':'RdBu_r'},
        panels=[{'x':'index','y':'index','z':'correlation','ticks':['amplitude'],'xlabel':'Parameter','ylabel':'Parameter'}])
    fig,_=make_figure(FigureData(spec,{'index':np.array([0.]),'correlation':np.array([[1.]])}))
    try:
        coords=fig.axes[0].collections[0].get_coordinates()
        assert np.ptp(coords[:,:,0])>0 and np.ptp(coords[:,:,1])>0
        color=fig.axes[0].collections[0].get_facecolors()[0]
        assert color[0]>color[2]  # positive correlation is the red end, not empty/white
    finally:plt.close(fig)
    series=spec.model_copy(update={'family':'F10','panels':[{'x':'energy','y':'state','z':'data','xlabel':'E (eV)','ylabel':'time (s)'}]})
    fig,_=make_figure(FigureData(series,{'energy':np.array([8970.,8990.]),'state':np.array([0.,10.]),'data':np.array([[0.,1.],[0.,.99]])}))
    try:
        assert fig.axes[0].get_xlim()==(8970.,8990.)
        assert fig.axes[0].get_ylim()==(0.,10.)
    finally:plt.close(fig)


def make_trend_project(store):
    from xasweb.pro.contracts import (DatasetBinding,DatasetFitResult,FitResult,FitRun,FitSpec,
        FitStatistics,GeometryAtom,ModelRevision,Parameter,ParameterEstimate,ParameterGraph,PathAsset,PathBinding)
    project,a=make_project(store)
    value=project.model_dump(mode='json')
    mask=np.ones(len(a['energy']),bool);mask[100:105]=False
    p1=value['processing'][0]
    p1['mask']=save_array(store,project.id,mask).model_dump(mode='json')
    for name in ('energy','mu','norm'):
        p1['arrays'][name]=save_array(store,project.id,a[name][mask]).model_dump(mode='json')
    p1['metadata'].update(input_mapping=value['datasets'][0]['mapping'],effective_energy_shift_eV=0.)
    dataset3=dict(value['datasets'][0]);dataset3['id']='dataset3';dataset3['filename']='Cu state 20.dat'
    pp3=dict(p1);pp3['id']='processing3';pp3['dataset_id']='dataset3'
    value['datasets'].append(dataset3);value['processing'].append(pp3)
    path=PathAsset(id='path1',source_file=store.put_asset(project.id,b'fixture FEFF path with preserved geometry','path.dat'),
        absorber_element='Cu',edge='K',absorber_site='1',nleg=3,reff=3.2,degen=4.,k_max=20.,
        geometry=[GeometryAtom(element='Cu',xyz=[0.,0.,0.]),GeometryAtom(element='O',xyz=[2.1,0.,.5]),
                  GeometryAtom(element='Cu',xyz=[0.,1.8,1.1])])
    value['paths']=[path.model_dump(mode='json')];points=[];lcf=[]
    for pos,(did,pid,state,amplitude) in enumerate([('dataset1','processing1',0.,.9),('dataset3','processing3',20.,1.1)]):
        dataset=Dataset.model_validate(next(d for d in value['datasets'] if d['id']==did))
        pp=ProcessingRevision.model_validate(next(p for p in value['processing'] if p['id']==pid))
        model=ModelRevision(id='model'+str(pos),parameters=ParameterGraph(nodes=[Parameter(id='amp',label='Amplitude',value=1.,physical_basis='FEFF scale; not coordination number')]),
            bindings=[DatasetBinding(dataset_id=did,processing_revision_id=pid,paths=[PathBinding(id='binding'+str(pos),path_id='path1',expressions={'s02':'amp'})])])
        spec=FitSpec(model=model,datasets=[dataset],processing=[pp],paths=[path])
        refs={name:save_array(store,project.id,v) for name,v in {'k':a['k'],'data_chi':a['chi'],'model_chi':a['chi']*.98}.items()}
        estimate=ParameterEstimate(id='amp',value=amplitude,unit='1',stderr=.05,uncertainty_reason=None)
        rid='run'+str(pos)
        result=FitResult(run_id=rid,converged=True,termination_reason='synthetic stored result',
            statistics=FitStatistics(residual_count=100,n_independent=12.,n_variables=1,degrees_of_freedom=11.),
            parameters=[estimate],uncertainty_basis='conditional_fixed',
            datasets=[DatasetFitResult(dataset_id=did,processing_revision_id=pid,arrays=refs,
                path_contributions={'binding'+str(pos):{'k':refs['k'],'chi':refs['model_chi']}})])
        run=FitRun(id=rid,model_id=model.id,input_snapshot=spec,snapshot_sha256=hashlib.sha256(canonical(spec.model_dump(mode='json'))).hexdigest(),task_state='succeeded',result=result)
        value['models'].append(model.model_dump(mode='json'));value['runs'].append(run.model_dump(mode='json'))
        coefficient=.2+pos*.1
        lcf_arrays={n:save_array(store,project.id,v).model_dump(mode='json') for n,v in {
            'energy':a['energy'],'data':a['norm'],'model':a['norm']*coefficient,
            'residual':a['norm']*(1-coefficient),'contribution_0':a['norm']*coefficient}.items()}
        lcf.append({'dataset_id':did,'reference_ids':['reference1'],'processing_ids':[pid],
                    'coefficients':[coefficient],'arrays':lcf_arrays,'diagnostics':{'synthetic_fixture':True}})
        points.append({'dataset_id':did,'processing_id':pid,'state_value':state,'state':'succeeded','reason':None,
            'edge_position_eV':8979.+pos,'white_line_height':1.05+.01*pos,'reference_coefficients':[coefficient],
            'fits':[{'fit_run_id':rid,'converged':True,'parameters':[estimate.model_dump(mode='json')],'uncertainty_basis':'conditional_fixed'}]})
    points.insert(1,{'dataset_id':'failed_observation','processing_id':None,'state_value':10.,'state':'failed',
                     'reason':'synthetic failure retained for gap check','edge_position_eV':None,'white_line_height':None,
                     'reference_coefficients':None,'fits':[]})
    workflow=value['metadata']['automation_runs'][0]
    workflow.update(processing_ids=['processing1','processing3'],fit_run_ids=['run0','run1'],lcf_results=lcf,
        series={'state_key':'time_s','points':points,'arrays':{}})
    return store.save(Project.model_validate(value),expected_revision=project.revision)


def test_state_trends_geometry_and_rejected_points_are_source_bound(tmp_path):
    store=ProjectStore(tmp_path/'projects');project=make_trend_project(store)
    figures,missing=collect_figures(store,project,project.metadata['automation_runs'][0],{'formats':['png','pdf']})
    trends=[f for f in figures if f.spec.kind=='trend']
    assert len(trends)==4  # edge, white line, reference component, amplitude
    for item in trends:
        assert item.arrays['state'].tolist()==[0.,10.,20.]
        assert item.arrays['value_valid'].tolist()==[True,False,True]
    parameter=next(f for f in trends if f.spec.method.get('parameter_id')=='amp')
    assert parameter.spec.method['uncertainty_basis']=='conditional_fixed'
    np.testing.assert_array_equal(parameter.arrays['stderr'],[.05,0.,.05])
    assert parameter.spec.method['conditional_error_notice']
    geometry=next(f for f in figures if f.spec.kind=='geometry')
    assert geometry.spec.method['nleg']==3
    np.testing.assert_array_equal(geometry.arrays['z_A'],[0.,.5,1.1])
    assert geometry.spec.sources[0]['path_source']['sha256']==project.paths[0].source_file.sha256
    assert 'single MS bond' in geometry.spec.method['length_semantics']
    raw=next(f for f in figures if f.spec.id=='F01-000')
    assert raw.spec.method['rejected_rows']==5
    assert raw.arrays['rejected'].sum()==5
    assert raw.spec.panels[0]['series_modes'][-1]=='points'
    from xasweb.pro.delivery import _figure_csv
    import csv
    rows=list(csv.DictReader(io.StringIO(_figure_csv(parameter).decode('utf-8-sig'))))
    assert rows[1]['parameter_value']=='' and rows[1]['stderr']==''
    assert rows[1]['state']=='10' and rows[1]['observation_status']=='failed'
    assert rows[0]['stderr']=='0.050000000000000003'
    geometry_rows=list(csv.DictReader(io.StringIO(_figure_csv(geometry).decode('utf-8-sig'))))
    assert [r['element'] for r in geometry_rows]==['Cu','O','Cu']
    from xasweb.pro.figures.render import make_figure,render_figure
    import matplotlib.pyplot as plt
    fig,_=make_figure(parameter)
    try:
        line=fig.axes[0].lines[0]
        assert np.isnan(line.get_ydata()[1])  # actual line break, not a joined interpolation
    finally:plt.close(fig)
    for item in [raw,geometry,*trends]:
        rendered,_=render_figure(item)
        assert f'figures/{item.spec.id}.pdf' in rendered
    payload=build_delivery(store,project.id,'workflow1',{'formats':['png']})
    with ZipFile(io.BytesIO(payload)) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        for item in [raw,geometry,*trends]:
            assert archive.read(item.spec.data_files[0])==_figure_csv(item)
            record=next(r for r in manifest['figures'] if r['id']==item.spec.id)
            assert record['sources']==item.spec.sources
        reopened=store.import_archive(archive.read('project/project.webxas.zip'))
        assert reopened.paths[0].geometry==project.paths[0].geometry
        assert reopened.metadata['automation_runs'][0]['series']['points'][1]['state']=='failed'


def test_shared_joint_estimate_not_copied_as_independent_state_trend(tmp_path):
    from xasweb.pro.figures.trends import add_series_trends
    store=ProjectStore(tmp_path/'projects');project=make_trend_project(store)
    value=project.model_dump(mode='json')
    left,right=value['runs']
    left['input_snapshot']['datasets']+=right['input_snapshot']['datasets']
    left['input_snapshot']['processing']+=right['input_snapshot']['processing']
    left['input_snapshot']['model']['bindings']+=right['input_snapshot']['model']['bindings']
    left['result']['datasets']+=right['result']['datasets']
    left['snapshot_sha256']=hashlib.sha256(canonical(left['input_snapshot'])).hexdigest()
    value['models'][0]=left['input_snapshot']['model']
    project=Project.model_validate(value)
    workflow=project.metadata['automation_runs'][0]
    for point in workflow['series']['points']:
        for fit in point['fits']:fit['fit_run_id']='run0'
    emitted=[];missing=[]
    def capture(*args,**kwargs):emitted.append(kwargs.get('method',{}))
    add_series_trends(project,workflow,capture,missing)
    assert not any(m.get('parameter_id')=='amp' for m in emitted)
    shared=next(m for m in missing if m.get('parameter_id')=='amp')
    assert len(shared['global_shared_estimates'])==1
    assert shared['global_shared_estimates'][0]['fit_run_id']=='run0'
