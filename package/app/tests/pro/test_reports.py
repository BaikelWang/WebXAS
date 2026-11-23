import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest

from xasweb.pro.contracts import (CovarianceRecord,Dataset,DatasetBinding,DatasetFitResult,
    FitResult,FitRun,FitSpec,FitStatistics,ModelRevision,NoiseSpec,Parameter,ParameterEstimate,
    ParameterGraph,PathAsset,PathBinding,ProcessingRevision,Project)
from xasweb.pro.preprocessing import save_array
from xasweb.pro.projects import ProjectError,ProjectStore,canonical
from xasweb.pro.reports import build_report,compare_runs

BENCH=Path(__file__).resolve().parents[4]/'benchmarks'


@pytest.fixture
def sample(tmp_path):
    store=ProjectStore(tmp_path/'projects')
    project=store.create('Report <script>alert(1)</script>')
    k=np.linspace(0,15,101);r=np.linspace(0,6,81)
    data=np.sin(k)*np.exp(-k/8);model=data*.98
    refs={name:save_array(store,project.id,value) for name,value in {
        'k':k,'data_chi':data,'model_chi':model,'r':r,'data_re':np.sin(r),
        'data_im':np.cos(r),'model_re':np.sin(r)*.98,'model_im':np.cos(r)*.98}.items()}
    raw=b'0 0\n1 1\n2 0\n'
    dataset=Dataset(id='dataset1',filename='synthetic.dat',kind='chi',raw_file=store.put_asset(project.id,raw,'synthetic.dat'),
                    element='Ce',edge='L3',sample_id='sample1',metadata={'data_kind':'synthetic'})
    recipe=ProcessingRevision(id='processing1',dataset_id=dataset.id,arrays={'k':refs['k'],'chi':refs['data_chi']})
    pathfile=BENCH/'inputs'/'parseq_xas'/'parseq_XAS'/'data'/'CeRu2'/'feff'/'CeRu2'/'feff0003.dat'
    asset=PathAsset(id='path1',source_file=store.put_asset(project.id,pathfile.read_bytes(),'feff0003.dat'),
        absorber_element='Ce',edge='L3',absorber_site='unique',nleg=3,reff=4.4588,degen=36.,k_max=20.)
    binding=PathBinding(id='binding1',path_id=asset.id,expressions={'s02':'amp','e0':'0','deltar':'0','sigma2':'0.004'})
    model_record=ModelRevision(id='model1',bindings=[DatasetBinding(dataset_id=dataset.id,processing_revision_id=recipe.id,
                paths=[binding],noise=NoiseSpec(source='explicit_scalar',value=.001))],
                parameters=ParameterGraph(nodes=[Parameter(id='amp',value=1.1,unit='1')]))
    spec=FitSpec(model=model_record,datasets=[dataset],processing=[recipe],paths=[asset],engine_versions={'xraylarch':'2026.3.1'})
    stats=FitStatistics(residual_count=160,n_independent=15.,n_variables=1,degrees_of_freedom=14.,
                        raw_chisquare=1.,chisquare=.09375,reduced_chisquare=.006696,rfactor=.002)
    result=FitResult(run_id='run1',converged=True,termination_reason='manual legal test fixture',statistics=stats,
        parameters=[ParameterEstimate(id='amp',value=1.1,unit='1',stderr=.1,uncertainty_reason=None)],
        covariance=CovarianceRecord(var_names=['amp'],matrix=[[.01]],scaling_description='test fixture'),
        datasets=[DatasetFitResult(dataset_id=dataset.id,processing_revision_id=recipe.id,arrays=refs,
            path_contributions={'binding1':{'k':refs['k'],'chi':refs['model_chi']}})],uncertainty_basis='conditional_fixed')
    run=FitRun(id='run1',model_id=model_record.id,input_snapshot=spec,
        snapshot_sha256=hashlib.sha256(canonical(spec.model_dump(mode='json'))).hexdigest(),task_state='succeeded',result=result)
    record=project.model_dump(mode='json')
    record.update(datasets=[dataset.model_dump(mode='json')],processing=[recipe.model_dump(mode='json')],
        paths=[asset.model_dump(mode='json')],models=[model_record.model_dump(mode='json')],runs=[run.model_dump(mode='json')])
    project=store.save(Project.model_validate(record),expected_revision=project.revision)
    return store,project


def add_alternative(project,mutate=None):
    value=project.model_dump(mode='json')
    run=json.loads(json.dumps(value['runs'][0]))
    run['id']='run2';run['model_id']='model2';run['input_snapshot']['model']['id']='model2';run['result']['run_id']='run2'
    run['input_snapshot']['model']['parameters']['nodes'][0]['value']=.9
    run['result']['statistics']['rfactor']=.001
    if mutate:mutate(run)
    run['snapshot_sha256']=hashlib.sha256(canonical(run['input_snapshot'])).hexdigest()
    value['models'].append(run['input_snapshot']['model']);value['runs'].append(run)
    return Project.model_validate(value)


def test_report_zip_is_readonly_complete_and_source_identified(sample):
    store,project=sample
    before=(store.folder(project.id)/'project.json').read_bytes()
    with (patch.object(store,'save',side_effect=AssertionError('report may not save')),
          patch.object(store,'put_asset',side_effect=AssertionError('report may not add assets'))):
        payload=build_report(store,project.id,'run1')
    assert (store.folder(project.id)/'project.json').read_bytes()==before
    with ZipFile(io.BytesIO(payload)) as archive:
        assert archive.testzip() is None
        names=set(archive.namelist())
        assert {'report.html','methods.md','parameters.csv','parameters.json','paths.csv','covariance.csv','correlation.csv',
                'project.webxas.zip','input-snapshot.json','sources.json','result.json','run.json','manifest.json'}<=names
        assert len([x for x in names if x.endswith('.png')])==3
        assert len([x for x in names if x.endswith('.svg')])==3
        manifest=json.loads(archive.read('manifest.json'))
        assert manifest['run_id']=='run1'
        for name,record in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==record['sha256']
        page=archive.read('report.html').decode()
        assert '包含合成数据' in page and '&lt;script&gt;' in page and '<script>' not in page
        assert 'not_declared' not in page
        assert 'run1' in page
        sources=json.loads(archive.read('sources.json'))
        assert sources['data'][0]['data_kind']=='synthetic'
        assert sources['snapshot_sha256']==project.runs[0].snapshot_sha256
        assert '1.0' in archive.read('correlation.csv').decode('utf-8-sig')
        k_rows=archive.read('arrays/dataset1-k.csv').decode('utf-8-sig')
        assert 'residual_chi' in k_rows
        with ZipFile(io.BytesIO(archive.read('project.webxas.zip'))) as inner:
            assert json.loads(inner.read('project.json'))['runs'][0]['id']=='run1'


def test_comparison_keeps_ambiguity_and_uses_same_basis(sample):
    _,project=sample
    project=add_alternative(project)
    comparison=compare_runs(project,['run1','run2'])
    assert comparison['comparability'] is True
    assert comparison['unique_structure'] is False
    assert comparison['assessment']=='review'
    assert 'winner' not in comparison
    assert comparison['runs'][1]['statistics']['rfactor']<comparison['runs'][0]['statistics']['rfactor']


@pytest.mark.parametrize('change',[
    lambda r:r['input_snapshot']['model']['bindings'][0]['transform'].update(rmax=3.5),
    lambda r:r['input_snapshot']['model']['bindings'][0]['noise'].update(value=.002),
    lambda r:r['input_snapshot']['datasets'][0]['correction_states'].update(self_absorption='uncorrected'),
    lambda r:r['input_snapshot']['processing'][0]['parameters'].update(energy_shift=1.),
    lambda r:r['input_snapshot']['engine_versions'].update(xraylarch='different'),
    lambda r:r['input_snapshot'].update(source_sha256='a'*64),
    lambda r:r.update(task_state='failed')])
def test_comparison_rejects_different_conditions_or_failed_runs(sample,change):
    _,project=sample
    output=compare_runs(add_alternative(project,change),['run1','run2'])
    assert output['comparability'] is False and output['reasons']


def test_unknown_noise_prevents_false_statistical_comparison(sample):
    _,project=sample
    def change(run):run['input_snapshot']['model']['bindings'][0]['noise'].update(source='unknown',value=None)
    result=compare_runs(add_alternative(project,change),['run1','run2'])
    assert result['comparability'] is False
    assert any(reason['code']=='unknown_noise' for reason in result['reasons'])


def test_mask_change_prevents_direct_comparison(sample):
    store,project=sample
    mask=save_array(store,project.id,np.ones(101,dtype=bool)).model_dump(mode='json')
    def change(run):run['input_snapshot']['datasets'][0]['mask']=mask
    result=compare_runs(add_alternative(project,change),['run1','run2'])
    assert result['comparability'] is False
    assert any('dataset_mask' in field for fields in result['differences'].values() for field in fields)


def test_comparison_inherits_defined_processing_noise(sample):
    _,project=sample
    value=project.model_dump(mode='json')
    snapshot=value['runs'][0]['input_snapshot']
    snapshot['processing'][0]['noise']=NoiseSpec(source='explicit_scalar',value=.001).model_dump(mode='json')
    snapshot['model']['bindings'][0]['noise']=NoiseSpec().model_dump(mode='json')
    value['runs'][0]['snapshot_sha256']=hashlib.sha256(canonical(snapshot)).hexdigest()
    result=compare_runs(add_alternative(Project.model_validate(value)),['run1','run2'])
    assert result['comparability'] is True
    assert not any(reason['code']=='unknown_noise' for reason in result['reasons'])


def test_bad_snapshot_or_result_identity_refused(sample):
    store,project=sample
    broken=project.model_dump(mode='json');broken['runs'][0]['snapshot_sha256']='0'*64
    store.save(Project.model_validate(broken),expected_revision=project.revision)
    with pytest.raises(ProjectError) as error:build_report(store,project.id,'run1')
    assert error.value.code=='snapshot_integrity'
    with pytest.raises(ProjectError) as error:compare_runs(project,['run1','missing'])
    assert error.value.code=='run_not_found'
