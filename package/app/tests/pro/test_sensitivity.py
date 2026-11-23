"""Bounded sensitivity presets retain successful, failed and inapplicable cases."""
from copy import deepcopy

import numpy as np
import pytest

from test_fitting import fixture
from xasweb.models import Parameters
from xasweb.pro.contracts import Dataset,FitSpec,NoiseSpec,Project
from xasweb.pro.larch_adapter import build_fit_spec
from xasweb.pro.preprocessing import process_dataset
from xasweb.pro.projects import ProjectError
from xasweb.pro.sensitivity import run_sensitivity


def test_chi_presets_are_fixed_and_keep_complete_immutable_runs(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    result=run_sensitivity(store,p.id,spec,p.revision)
    assert [c['case_id'] for c in result['cases']]==['baseline','window','background']
    assert result['cases'][0]['status']=='succeeded'
    assert result['cases'][1]['status']=='succeeded'
    assert result['cases'][2]['status']=='not_applicable'
    assert result['cases'][2]['details'][0]['dataset_id']=='d0'
    assert len(result['runs'])==2 and len(result['models'])==2 and not result['processing_revisions']
    models={m['id']:m for m in result['models']}
    for run in result['runs']:
        assert run['input_snapshot']['model']==models[run['model_id']]
        assert run['result']['metadata']['sensitivity']['source_model_id']==spec.model.id
    assert result['models'][1]['bindings'][0]['transform']['kmax']==spec.model.bindings[0].transform.kmax-.5
    assert not result['runs'][1]['result']['metadata']['sensitivity']['direct_statistical_comparison']
    assert store.read(p.id)==p  # No Project commit, even though immutable assets were added.


def test_failed_and_nonconverged_presets_are_not_discarded(tmp_path,monkeypatch):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    def failed(*args,**kwargs):raise RuntimeError('controlled solver failure')
    monkeypatch.setattr('xasweb.pro.sensitivity.fit_snapshot',failed)
    result=run_sensitivity(store,p.id,spec,p.revision)
    assert [c['status'] for c in result['cases']]==['failed','failed','not_applicable']
    assert 'controlled solver failure' in result['cases'][0]['reason']
    assert len(result['models'])==2 and not result['runs']


def test_actual_evaluation_limit_is_reported_as_nonconverged(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    spec=FitSpec.model_validate({**spec.model_dump(mode='json'),'max_evaluations':1})
    result=run_sensitivity(store,p.id,spec,p.revision)
    assert [c['status'] for c in result['cases']]==['nonconverged','nonconverged','not_applicable']
    assert len(result['runs'])==2 and not any(r['result']['converged'] for r in result['runs'])


def test_stale_or_wrong_saved_model_rejected_before_presets(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    with pytest.raises(ProjectError) as exc:run_sensitivity(store,p.id,spec,p.revision+1)
    assert exc.value.code=='revision_conflict'
    value=spec.model_dump(mode='json');value['model']['title']='changed after save'
    with pytest.raises(ProjectError) as exc:run_sensitivity(store,p.id,FitSpec.model_validate(value),p.revision)
    assert exc.value.code=='sensitivity_snapshot'


def test_invalid_narrower_window_is_not_silently_clipped(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    value=p.model_dump(mode='json')
    value['models'][0]['bindings'][0]['transform'].update(kmin=3.,kmax=4.4,dk=1.,rmin=.5,rmax=20.)
    p=store.save(Project.model_validate(value),p.revision)
    spec=build_fit_spec(p,spec.model.id,max_evaluations=1)
    result=run_sensitivity(store,p.id,spec,p.revision)
    assert result['cases'][1]['status']=='not_applicable'
    assert 'taper' in result['cases'][1]['reason']


def test_total_budget_keeps_unattempted_case_records(tmp_path,monkeypatch):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    ticks=iter([0.,10.,11.,12.])
    monkeypatch.setattr('xasweb.pro.sensitivity.time.monotonic',lambda:next(ticks))
    # Restore the shared time function after the test; no optimizer can start.
    value=spec.model_dump(mode='json');value['timeout_seconds']=1.
    result=run_sensitivity(store,p.id,FitSpec.model_validate(value),p.revision)
    assert len(result['cases'])==3 and all(c['status']=='failed' for c in result['cases'])
    assert all('budget exhausted' in c['reason'] for c in result['cases'])
    assert not result['runs']


def test_mu_background_preset_reprocesses_original_and_returns_new_revision(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    energy=np.arange(5400.,7100.1,1.)
    k=np.sqrt(np.maximum(energy-5723.,0)/3.80998212)
    oscillation=np.interp(k,arrays['k'],arrays['chi_0'],left=0,right=0)
    step=(1+np.tanh((energy-5723.)/3))/2
    mu=.0001*(energy-5400)+step*(1+oscillation)
    raw='\n'.join(f'{x:.17g} {y:.17g}' for x,y in zip(energy,mu))+'\n'
    data=p.datasets[0].model_dump(mode='json')
    data.update(kind='mu',raw_text=raw,raw_file=store.put_asset(p.id,raw.encode(),'synthetic-mu.dat').model_dump(mode='json'))
    value=p.model_dump(mode='json');value['datasets']=[data]
    p=store.save(Project.model_validate(value),p.revision)
    pp=process_dataset(store,p.id,'d0',Parameters(e0=5723.,kmin=3.,kmax=13.),noise=NoiseSpec(source='explicit_scalar',value=1e-4))
    value=p.model_dump(mode='json');value['processing'].append(pp.model_dump(mode='json'))
    value['models'][0]['bindings'][0]['processing_revision_id']=pp.id
    value['models'][0]['bindings'][0]['noise']=pp.noise.model_dump(mode='json')
    p=store.save(Project.model_validate(value),p.revision)
    spec=build_fit_spec(p,spec.model.id,max_evaluations=20)
    result=run_sensitivity(store,p.id,spec,p.revision)
    case=result['cases'][2]
    assert case['status'] in ('succeeded','nonconverged'),case
    assert len(result['processing_revisions'])==1
    new=result['processing_revisions'][0]
    assert new['parameters']['rbkg']==pytest.approx(pp.parameters.rbkg+.1)
    assert new['metadata']['sensitivity']['source_processing_id']==pp.id
    assert new['input_revision']==p.revision
    assert new['arrays']['chi']['sha256']!=pp.arrays['chi'].sha256
    run=next(r for r in result['runs'] if r['id']==case['run_id'])
    assert run['input_snapshot']['model']['bindings'][0]['processing_revision_id']==new['id']
    assert run['input_snapshot']['model']['bindings'][0]['noise']['processing_revision_id']==new['id']
    assert not run['result']['metadata']['sensitivity']['direct_statistical_comparison']
    assert store.read(p.id)==p
