"""Application adapter against references frozen before its implementation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from xasweb.pro.contracts import (
    Calibration,CoordinationGroup,Dataset,DatasetBinding,FitSpec,ModelRevision,NoiseSpec,Parameter,
    ParameterGraph,PathAsset,PathBinding,ProcessingRevision,Project,TransformSpec,
)
from xasweb.pro.larch_adapter import FitError,build_fit_spec,fit_snapshot,validate_snapshot,_prepare
from xasweb.pro.preprocessing import load_array,save_array
from xasweb.pro.projects import ProjectStore,canonical
from xasweb.pro.diagnostics import joint_reference_basis,make_s02_calibration
from xasweb.pro.parameters import compile_model

BENCH=Path(__file__).resolve().parents[4]/'benchmarks'
REFERENCE=BENCH/'references'/'synthetic_joint_ss_ms'


def fixture(tmp_path,indices=(0,1,2),noise_seed=None):
    from larch.xafs import feffpath
    metadata=json.loads((REFERENCE/'reference.json').read_text(encoding='utf-8'))
    arrays=np.load(REFERENCE/'arrays.npz',allow_pickle=False)
    store=ProjectStore(tmp_path/'projects');p=store.create('Frozen synthetic SS/MS adapter case')
    datasets=[];processed=[];paths=[];bindings=[]
    for j,(name,digest) in enumerate(zip(metadata['path_files'],metadata['path_sha256'])):
        source=BENCH/name.replace(chr(92), '/')
        payload=source.read_bytes();assert hashlib.sha256(payload).hexdigest()==digest
        raw=store.put_asset(p.id,payload,source.name)
        fp=feffpath(str(source))
        paths.append(PathAsset(id=f'path{j}',source_file=raw,absorber_element='Ce',edge='L3',absorber_site='site0',nleg=int(fp.nleg),reff=float(fp.reff),degen=float(fp.degen),k_max=float(fp._feffdat.k[-1])))
    nodes=[Parameter(id='amp',value=.95,minimum=.5,maximum=1.2),Parameter(id='dr',unit='angstrom',value=0.,minimum=-.08,maximum=.08)]
    for i in indices:
        did=f'd{i}';pp=f'pp{i}'
        chi = arrays[f'chi_{i}']
        if noise_seed is not None:
            rng = np.random.default_rng(noise_seed + i)
            chi = chi + rng.normal(0., metadata['epsilon_k'], size=chi.shape)
        raw_payload='\n'.join(f'{x:.17g} {y:.17g}' for x,y in zip(arrays['k'],chi)).encode()
        raw=store.put_asset(p.id,raw_payload,f'synthetic_{i}.dat')
        datasets.append(Dataset(id=did,raw_file=raw,filename=f'synthetic_{i}.dat',kind='chi',element='Ce',edge='L3',metadata={'data_kind':'same_engine_synthetic','not_experimental_data':True}))
        noise=NoiseSpec(source='explicit_scalar',value=metadata['epsilon_k'],processing_revision_id=pp)
        processed.append(ProcessingRevision(id=pp,dataset_id=did,arrays={'k':save_array(store,p.id,arrays['k']),'chi':save_array(store,p.id,chi)},noise=noise))
        nodes.extend([Parameter(id=f'e0_{i}',value=0.,unit='eV',minimum=-10,maximum=10,scope='dataset',owner_id=did),Parameter(id=f'sig_{i}',value=.004,unit='angstrom2',minimum=.0001,maximum=.02,scope='dataset',owner_id=did)])
        bindings.append(DatasetBinding(dataset_id=did,processing_revision_id=pp,noise=noise,transform=TransformSpec.model_validate(metadata['transform']),paths=[PathBinding(id=f'b{i}_{j}',path_id=path.id,expressions={'s02':'amp','deltar':'dr','e0':f'e0_{i}','sigma2':f'sig_{i}'}) for j,path in enumerate(paths)]))
    model=ModelRevision(id='joint_model',parameters=ParameterGraph(nodes=nodes),bindings=bindings)
    value=p.model_dump(mode='json');value.update(datasets=[d.model_dump(mode='json') for d in datasets],processing=[r.model_dump(mode='json') for r in processed],paths=[r.model_dump(mode='json') for r in paths],models=[model.model_dump(mode='json')])
    p=store.save(Project.model_validate(value),p.revision)
    spec=build_fit_spec(p,model.id,max_evaluations=500)
    return store,p,spec,metadata,arrays


@pytest.mark.parametrize('indices',[(0,),(0,1,2)])
def test_single_and_true_joint_adapter_recover_frozen_reference(tmp_path,indices,monkeypatch):
    store,p,spec,metadata,arrays=fixture(tmp_path,indices)
    from larch.xafs import feffit as native_feffit
    calls=[]
    def recorded(parameters,datasets,**options):
        calls.append(len(datasets))
        return native_feffit(parameters,datasets,**options)
    monkeypatch.setattr('larch.xafs.feffit',recorded)
    run=fit_snapshot(store,p.id,spec)
    assert calls==[len(indices)]  # One native solve, including all spectra.
    assert run.result.converged
    assert run.snapshot_sha256==hashlib.sha256(canonical(spec.model_dump(mode='json'))).hexdigest()
    values={x.id:x.value for x in run.result.parameters};truth=metadata['truth'];tol=metadata['tolerances']
    for name,key in [('amp','amp'),('dr','dr')]:
        np.testing.assert_allclose(values[name],truth[key],**tol[key])
    for i in indices:
        np.testing.assert_allclose(values[f'e0_{i}'],truth['e0'][i],**tol['e0'])
        np.testing.assert_allclose(values[f'sig_{i}'],truth['sigma2'][i],**tol['sigma2'])
        result=next(d for d in run.result.datasets if d.dataset_id==f'd{i}')
        model_chi=load_array(store,p.id,result.arrays['model_chi'])
        np.testing.assert_allclose(model_chi,arrays[f'model_0_{i}'],**tol['model_chi'])
        contributions=[load_array(store,p.id,x['chi']) for x in result.path_contributions.values()]
        np.testing.assert_allclose(sum(contributions),model_chi,rtol=1e-10,atol=1e-12)
        assert len(contributions)==3
    covariance=run.result.covariance
    assert covariance.matrix is not None
    errors={x.id:x.stderr for x in run.result.parameters}
    np.testing.assert_allclose(np.diag(covariance.matrix),[errors[name]**2 for name in covariance.var_names],rtol=1e-6,atol=np.finfo(float).tiny)
    assert run.result.statistics.n_variables==2+2*len(indices)
    assert sum(d.statistics.raw_chisquare for d in run.result.datasets)==pytest.approx(run.result.statistics.raw_chisquare)
    assert sum(d.statistics.chisquare for d in run.result.datasets)==pytest.approx(run.result.statistics.chisquare)
    assert all(d.statistics.reduced_chisquare is None for d in run.result.datasets)
    assert run.result.metadata['attempts'][0]['success']
    assert len(run.result.metadata['residual_layout'])==len(indices)
    # The worker may add content-addressed outputs but never changes Project.
    assert store.read(p.id).revision==p.revision and not store.read(p.id).runs


def test_order_permutation_preserves_shared_and_local_solution(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path)
    payload=spec.model_dump(mode='json');payload['model']['bindings'].reverse();payload['model']['parameters']['nodes'].reverse()
    run=fit_snapshot(store,p.id,FitSpec.model_validate(payload))
    values={x.id:x.value for x in run.result.parameters}
    np.testing.assert_allclose(values['amp'],metadata['truth']['amp'],**metadata['tolerances']['amp'])
    np.testing.assert_allclose(values['dr'],metadata['truth']['dr'],**metadata['tolerances']['dr'])
    assert [d.dataset_id for d in run.result.datasets]==['d2','d1','d0']


def test_same_observation_rejected_before_native_optimizer(tmp_path,monkeypatch):
    store,p,spec,metadata,arrays=fixture(tmp_path)
    payload=spec.model_dump(mode='json');payload['datasets'][1]['raw_file']=payload['datasets'][0]['raw_file']
    def forbidden(*args,**kwargs):raise AssertionError('Optimizer must not run for repeated observations')
    monkeypatch.setattr('larch.xafs.feffit',forbidden)
    with pytest.raises(FitError,match='share original observations') as exc:
        fit_snapshot(store,p.id,FitSpec.model_validate(payload))
    assert exc.value.code=='duplicate_observation'


def test_derived_common_lineage_is_not_an_independent_measurement(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path)
    payload=spec.model_dump(mode='json');payload['datasets'][1]['lineage']=['d0']
    with pytest.raises(FitError) as exc:validate_snapshot(FitSpec.model_validate(payload))
    assert exc.value.code=='duplicate_observation'


def test_information_gate_precedes_fitting(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    payload=spec.model_dump(mode='json');payload['model']['bindings'][0]['transform'].update(rmin=2.,rmax=2.001)
    with pytest.raises(FitError) as exc:validate_snapshot(FitSpec.model_validate(payload))
    assert exc.value.code=='insufficient_information'


def test_wrong_path_metadata_is_not_silently_used(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    payload=spec.model_dump(mode='json');payload['paths'][0]['degen']=99.
    with pytest.raises(FitError) as exc:fit_snapshot(store,p.id,FitSpec.model_validate(payload))
    assert exc.value.code=='path_metadata'


def test_absorption_edge_and_processing_identity_checked(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    payload=spec.model_dump(mode='json');payload['datasets'][0]['edge']='K'
    with pytest.raises(FitError) as exc:validate_snapshot(FitSpec.model_validate(payload))
    assert exc.value.code=='path_edge'
    payload=spec.model_dump(mode='json');payload['processing'][0]['dataset_id']='other'
    with pytest.raises(FitError) as exc:validate_snapshot(FitSpec.model_validate(payload))
    assert exc.value.code=='processing_dataset'


def test_multi_initial_runs_keep_all_attempts(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    payload=spec.model_dump(mode='json');payload['initial_sets']=[{'amp':.75,'dr':.02},{'amp':2.}]
    run=fit_snapshot(store,p.id,FitSpec.model_validate(payload))
    attempts=run.result.metadata['attempts']
    assert len(attempts)==3 and attempts[0]['success'] and attempts[1]['success']
    assert not attempts[2]['success'] and attempts[2]['code']=='PARAMETER_BOUND'


def test_stale_mapping_source_and_weight_rejected(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    payload=spec.model_dump(mode='json')
    d=payload['datasets'][0]
    payload['processing'][0]['metadata']={'input_mapping':d['mapping'],'input_kind':d['kind'],'input_kweight':0,'source_sha256':d['raw_file']['sha256']}
    validate_snapshot(FitSpec.model_validate(payload))
    for key,new_value in [('input_mapping',{'energy':1,'signal':0,'reference':2,'mode':'mu','energy_unit':'eV'}),('input_kweight',2),('source_sha256','f'*64)]:
        changed=deepcopy(payload);changed['processing'][0]['metadata'][key]=new_value
        with pytest.raises(FitError) as exc:validate_snapshot(FitSpec.model_validate(changed))
        assert exc.value.code=='stale_processing'


def test_missing_processing_provenance_is_only_allowed_in_labelled_synthetic_fixtures(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    payload=spec.model_dump(mode='json');payload['datasets'][0]['metadata']={}
    with pytest.raises(FitError) as exc:validate_snapshot(FitSpec.model_validate(payload))
    assert exc.value.code=='missing_processing_provenance'


def test_fluorescence_empty_correction_state_remains_reviewable(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    payload=spec.model_dump(mode='json');payload['datasets'][0]['mapping']['mode']='fluorescence'
    run=fit_snapshot(store,p.id,FitSpec.model_validate(payload))
    assert run.result.converged and run.result.assessment=='needs_review'
    assert any('fluorescence' in w for w in run.result.warnings)
    assert run.result.metadata['experimental_correction_states']['d0']=={}


def test_metadata_flag_alone_cannot_promote_joint_calibration(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,1))
    payload=spec.model_dump(mode='json');payload['calibrations']=[Calibration(id='claimed',kind='s02',value=.86,unit='1',reference_dataset_ids=['d0'],element='Ce',edge='L3',uncertainty_basis='joint_reference',applicability_note='This is only an unverified claim.',conditions={'parameter_id':'amp'}).model_dump(mode='json')]
    spec=FitSpec.model_validate(payload)
    basis,links=joint_reference_basis(spec,compile_model(spec.model,spec.paths))
    assert basis=='conditional_fixed' and not links


@pytest.mark.parametrize("noise_seed", [None, 411], ids=["noise_free", "declared_noise"])
def test_standard_fixed_cn_and_target_free_cn_share_joint_amplitude(tmp_path, monkeypatch, noise_seed):
    from larch.xafs import feffit as native_feffit
    native_results = []

    def record_native(*args, **kwargs):
        fit = native_feffit(*args, **kwargs)
        native_results.append(fit)
        return fit

    monkeypatch.setattr("larch.xafs.feffit", record_native)
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,1),noise_seed=noise_seed)
    payload=spec.model_dump(mode='json');d=spec.paths[0].degen
    nodes=payload['model']['parameters']['nodes']
    nodes.extend([Parameter(id='known_N',kind='fixed',value=d).model_dump(mode='json'),Parameter(id='target_N',value=.9*d,minimum=.5*d,maximum=1.5*d).model_dump(mode='json')])
    groups=[]
    for i,nid in [(0,'known_N'),(1,'target_N')]:
        binding=payload['model']['bindings'][i]['paths'][0]
        binding['amplitude_semantics']='single_path_cn'
        groups.append(CoordinationGroup(id=f'group{i}',path_binding_ids=[binding['id']],coordination_parameter_id=nid,reference_degeneracy=d,counting_basis='Fixed single-scattering reference count in this synthetic contract case.').model_dump(mode='json'))
    payload['model']['coordination_groups']=groups
    payload['calibrations']=[Calibration(id='cal',kind='s02',value=.86,unit='1',reference_dataset_ids=['d0'],element='Ce',edge='L3',reference_model_id='joint_model',applicability_note='Same synthetic edge/path system only; no experimental transfer claim.',conditions={'parameter_id':'amp'}).model_dump(mode='json')]
    spec=FitSpec.model_validate(payload)
    run=fit_snapshot(store,p.id,spec)
    assert run.result.converged and run.result.uncertainty_basis=='joint_reference'
    estimate=next(v for v in run.result.parameters if v.id=='target_N')
    amp = next(v for v in run.result.parameters if v.id == 'amp')
    (native,) = native_results
    covariance = run.result.covariance

    if noise_seed is None:
        # Native solvers may omit covariance at a numerical zero residual.
        assert estimate.value == pytest.approx(d, abs=1e-5)
        assert amp.value == pytest.approx(.86, abs=1e-6)
        assert native.fit_details.chisqr < 1e-10
    else:
        # Independent noise draws at the declared epsilon_k exercise propagation.
        assert estimate.stderr is not None and estimate.stderr > 0
        assert amp.stderr is not None and amp.stderr > 0
        assert abs(estimate.value - d) < 4 * estimate.stderr
        assert abs(amp.value - .86) < 4 * amp.stderr

    native_covariance = getattr(native.fit_details, 'scaled_covar', None)
    if native_covariance is None:
        assert noise_seed is None
        assert covariance.matrix is None
        assert covariance.missing_reason == 'optimizer_covariance_unavailable'
        assert estimate.stderr is None and amp.stderr is None
        assert estimate.uncertainty_reason == 'optimizer_did_not_estimate_uncertainty'
    else:
        np.testing.assert_allclose(covariance.matrix, native_covariance, rtol=1e-9, atol=0)
        i = covariance.var_names.index('target_N')
        j = covariance.var_names.index('amp')
        assert estimate.stderr == pytest.approx(np.sqrt(native_covariance[i, i]), rel=1e-6, abs=0)
        assert amp.stderr == pytest.approx(np.sqrt(native_covariance[j, j]), rel=1e-6, abs=0)
        if noise_seed is not None:
            assert native_covariance[i, j] != 0
    assert run.result.metadata['calibration_links'][0]['fixed_reference_cn_ids']==['known_N']
    cal=make_s02_calibration(run,'amp',['d0'],'Ce','L3','Same synthetic source only; conditional on known reference N.')
    assert cal.value == amp.value
    assert cal.stderr == amp.stderr
    assert cal.uncertainty_basis == 'conditional_fixed'


def test_evaluation_budget_preserves_nonconvergence_and_missing_covariance(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    payload=spec.model_dump(mode='json');payload['max_evaluations']=1
    run=fit_snapshot(store,p.id,FitSpec.model_validate(payload))
    assert not run.result.converged and run.result.assessment=='needs_review'
    assert run.result.covariance.matrix is None
    assert run.result.covariance.missing_reason
    assert all(parameter.stderr is None for parameter in run.result.parameters)
    assert any('did not converge' in w for w in run.result.warnings)


def test_queued_snapshot_rejects_changed_application_source(tmp_path,monkeypatch):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    monkeypatch.setattr('xasweb.pro.larch_adapter.current_source_sha256',lambda:'f'*64)
    with pytest.raises(FitError) as exc:validate_snapshot(spec)
    assert exc.value.code=='engine_source_changed'


def test_snapshot_rejects_changed_scientific_dependency_version(tmp_path):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    payload=spec.model_dump(mode='json');payload['engine_versions']['numpy']='0.0'
    with pytest.raises(FitError) as exc:validate_snapshot(FitSpec.model_validate(payload))
    assert exc.value.code=='engine_version'


def estimated_noise_fixture(tmp_path,limits):
    store,p,spec,metadata,arrays=fixture(tmp_path,(0,))
    k=arrays['k'];chi=arrays['chi_0']+1e-4*np.sin(35*k)
    raw='\n'.join(f'{x:.17g} {y:.17g}' for x,y in zip(k,chi)).encode()
    payload=spec.model_dump(mode='json')
    payload['datasets'][0]['raw_file']=store.put_asset(p.id,raw,'noise-contract.dat').model_dump(mode='json')
    payload['processing'][0]['arrays']['chi']=save_array(store,p.id,chi).model_dump(mode='json')
    noise=NoiseSpec(source='engine_estimate',estimation_range=limits).model_dump(mode='json')
    payload['processing'][0]['noise']=noise;payload['model']['bindings'][0]['noise']=noise
    return store,p,FitSpec.model_validate(payload),k,chi


@pytest.mark.parametrize('limits',[[15.,30.],[14.,28.]])
def test_estimated_r_noise_preserves_native_parseval_scaling(tmp_path,limits):
    from larch import Group
    from larch.fitting import group2params
    from larch.xafs import feffit_dataset,feffit_transform
    store,p,spec,k,chi=estimated_noise_fixture(tmp_path,limits)
    compiled=compile_model(spec.model,spec.paths)
    pars,datasets,_=_prepare(store,p.id,spec,compiled)
    datasets[0].prepare_fit(group2params(pars))
    native=feffit_dataset(data=Group(k=k,chi=chi),paths=[],transform=feffit_transform(**spec.model.bindings[0].transform.model_dump(mode='json')))
    native.prepare_fit(group2params(pars))
    native.estimate_noise(chi=native._chi,rmin=limits[0],rmax=limits[1])
    assert datasets[0].data.epsilon_k is None  # Never turn an estimate into an explicit sigma.
    assert datasets[0].epsilon_k==pytest.approx(native.epsilon_k,rel=1e-12)
    assert datasets[0].epsilon_r==pytest.approx(native.epsilon_r,rel=1e-12)


def test_default_estimated_noise_statistics_equal_native_feffit(tmp_path):
    from larch import Group
    from larch.xafs import feffpath,feffit,feffit_dataset,feffit_transform
    store,p,spec,k,chi=estimated_noise_fixture(tmp_path,[15.,30.])
    compiled=compile_model(spec.model,spec.paths)
    parameters,expressions=compiled.to_larch()
    binding=spec.model.bindings[0];assets={x.id:x for x in spec.paths}
    paths=[feffpath(str(store.folder(p.id)/assets[b.path_id].source_file.path),label=b.id,**expressions[b.id]) for b in binding.paths]
    dataset=feffit_dataset(data=Group(k=k,chi=chi),paths=paths,transform=feffit_transform(**binding.transform.model_dump(mode='json')))
    reference=feffit(parameters,[dataset],max_nfev=spec.max_evaluations,path_outputs=True,fix_unused_variables=False)
    run=fit_snapshot(store,p.id,spec)
    assert run.result.statistics.chisquare==pytest.approx(reference.chi_square,rel=1e-8,abs=1e-12)
    assert run.result.statistics.reduced_chisquare==pytest.approx(reference.chi2_reduced,rel=1e-8,abs=1e-12)
    assert run.result.metadata['resolved_noise'][0]['epsilon_r']==pytest.approx(dataset.epsilon_r,rel=1e-12)
