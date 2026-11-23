import io
from pathlib import Path
import numpy as np
import pytest

from xasweb.models import Parameters
from xasweb.pro.contracts import Calibration, Dataset, NoiseSpec, Project
from xasweb.pro.preprocessing import process_dataset,commit_processing,load_array,save_array,validate_noise,merge_scan_arrays,merge_project_scans
from xasweb.pro.projects import ProjectStore,ProjectError


def make_project(tmp_path,kind='mu',offset=0):
    store=ProjectStore(tmp_path);project=store.create('processing')
    if kind=='mu':
        raw=(Path(__file__).resolve().parents[3]/'validation/sample-spectrum.xdi').read_text()
        rows=[]
        for line in raw.splitlines():
            if line.startswith('#'):rows.append(line)
            elif line.strip():
                x,y=map(float,line.split());rows.append(f'{x+offset:.17g} {y:.17g}')
        raw='\n'.join(rows)+'\n'
    else:
        k=np.linspace(0,15,301);raw='\n'.join(f'{x:.17g} {np.sin(2*x)*np.exp(-x/8):.17g}' for x in k)+'\n'
    ref=store.put_asset(project.id,raw.encode(),'input.dat')
    dataset=Dataset(id='d1',filename='input.dat',raw_file=ref,raw_text=raw,kind=kind,element='Fe',edge='K',sample_id='sample1')
    value=project.model_dump(mode='json');value['datasets']=[dataset.model_dump(mode='json')]
    return store,store.save(Project.model_validate(value),0),raw


def test_processing_freezes_arrays_and_raw(tmp_path):
    from xasweb.science import analyze
    from xasweb.models import AnalysisRequest
    store,project,raw=make_project(tmp_path)
    settings=Parameters(kmax=10)
    reference=analyze(AnalysisRequest(raw_text=raw,parameters=settings))
    processing=process_dataset(store,project.id,'d1',settings)
    for name in ['energy','norm','k','chi','chir_re','chir_im']:
        np.testing.assert_allclose(load_array(store,project.id,processing.arrays[name]),reference['arrays'][name],rtol=1e-7,atol=1e-9)
    saved=commit_processing(store,project.id,processing,project.revision)
    assert store.read_asset(project.id,project.datasets[0].raw_file)==raw.encode()
    assert saved.processing[0].parameters.kmax==10
    assert processing.noise.processing_revision_id==processing.id


def test_normalization_only_bypasses_autobk_and_preserves_normalized_values(tmp_path, monkeypatch):
    store, project, _ = make_project(tmp_path)
    settings = Parameters(kmax=10)
    full = process_dataset(store, project.id, 'd1', settings)
    def forbidden(*args, **kwargs):
        raise AssertionError('XANES-only processing must not call AUTOBK')
    monkeypatch.setattr('xasweb.science.autobk', forbidden)
    near_edge = process_dataset(store, project.id, 'd1', settings, normalization_only=True)
    assert [call['operation'] for call in near_edge.resolved_calls] == ['pre_edge']
    assert near_edge.arrays['norm'] == full.arrays['norm']
    assert 'chi' not in near_edge.arrays and near_edge.noise.source == 'unknown'


def test_energy_calibration_applies_once(tmp_path):
    store,p,raw=make_project(tmp_path,offset=3)
    calibration=Calibration(id='cal1',kind='energy',value=-3.0,unit='eV',reference_dataset_ids=['d1'],element='Fe',edge='K')
    value=p.model_dump(mode='json');value['calibrations']=[calibration.model_dump(mode='json')]
    p=store.save(Project.model_validate(value),p.revision)
    shifted=process_dataset(store,p.id,'d1',Parameters(kmax=10),calibration_id='cal1')
    unshifted=process_dataset(store,p.id,'d1',Parameters(kmax=10))
    np.testing.assert_allclose(load_array(store,p.id,unshifted.arrays['energy'])-load_array(store,p.id,shifted.arrays['energy']),3,rtol=0,atol=1e-10)
    assert shifted.metadata['effective_energy_shift_eV']==-3
    assert unshifted.metadata['effective_energy_shift_eV']==0


def test_mask_and_duplicate_input_rejected(tmp_path):
    store,p,_=make_project(tmp_path)
    with pytest.raises(ProjectError,match='掩码'):
        process_dataset(store,p.id,'d1',mask=[True,False])
    p.datasets[0].raw_text='100 1\n'*50
    p=store.save(p,p.revision)
    with pytest.raises(ProjectError,match='严格递增'):
        process_dataset(store,p.id,'d1')


def test_chi_processing_and_noise_contract(tmp_path):
    store,p,_=make_project(tmp_path,kind='chi')
    processing=process_dataset(store,p.id,'d1',Parameters(kmax=12))
    assert len(load_array(store,p.id,processing.arrays['k']))==301
    good=save_array(store,p.id,np.ones(301)*0.01)
    explicit=NoiseSpec(source='explicit_array',array=good)
    assert validate_noise(store,p.id,explicit,processing.id,301).processing_revision_id==processing.id
    with pytest.raises(ProjectError,match='等长'):
        validate_noise(store,p.id,explicit,processing.id,99)
    bad=save_array(store,p.id,np.zeros(301))
    with pytest.raises(ProjectError,match='为正'):
        validate_noise(store,p.id,NoiseSpec(source='explicit_array',array=bad),processing.id,301)
    with pytest.raises(ProjectError,match='能量偏移'):
        process_dataset(store,p.id,'d1',Parameters(energy_shift=2))


def test_merge_intersection_weighting_and_no_extrapolation():
    result=merge_scan_arrays([np.arange(5.),np.arange(1.,6.)],[np.ones(5),np.ones(5)*3],[1.,3.])
    np.testing.assert_array_equal(result['energy'],[1,2,3,4])
    np.testing.assert_allclose(result['mu'],1.5)
    np.testing.assert_allclose(result['variance'],0.75)
    assert result['method']=='inverse_variance'
    with pytest.raises(ProjectError,match='共同'):
        merge_scan_arrays([[0,1],[2,3]],[[1,1],[1,1]])


def test_merge_project_requires_same_state_and_keeps_lineage(tmp_path):
    store,p,raw=make_project(tmp_path)
    raw2='\n'.join(line if line.startswith('#') else f'{line.split()[0]} {float(line.split()[1])*1.01:.17g}' for line in raw.splitlines() if line.strip())+'\n'
    second=p.datasets[0].model_copy(update={'id':'d2','raw_text':raw2,'raw_file':store.put_asset(p.id,raw2.encode(),'scan2.dat')})
    p.datasets.append(second);p=store.save(p,p.revision)
    with pytest.raises(ProjectError,match='不同'):
        merge_project_scans(store,p.id,['d1','d1'],p.revision)
    p.datasets[1].conditions={'potential_V':1.0};p=store.save(p,p.revision)
    with pytest.raises(ProjectError,match='状态'):
        merge_project_scans(store,p.id,['d1','d2'],p.revision)
    p.datasets[1].conditions={};p=store.save(p,p.revision)
    merged=merge_project_scans(store,p.id,['d1','d2'],p.revision)
    assert merged.datasets[-1].lineage==['d1','d2']
    assert store.read_asset(p.id,p.datasets[0].raw_file)==raw.encode()
