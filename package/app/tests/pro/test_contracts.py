"""Serialization and scientific-record integrity, independent of fitting."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from xasweb.pro.contracts import (
    ArrayRef, AssetRef, Calibration, CovarianceRecord, Dataset, DatasetBinding,
    FitResult, FitRun, FitSpec, FitStatistics, ModelRevision, NoiseSpec, Parameter,
    ParameterEstimate, ParameterGraph, PathAsset, PathBinding, ProcessingRevision,
    Project, TransformSpec,
)


def asset(path="raw/cu.dat"):
    return AssetRef(path=path,sha256="a"*64,byte_size=100)


def project():
    dataset=Dataset(id="cu",raw_file=asset(),filename="cu.dat",raw_text="1 2\n2 3",element="Cu",edge="K")
    processed=ProcessingRevision(id="pp_cu",dataset_id="cu")
    path=PathAsset(id="cu_path",source_file=asset("paths/feff0001.dat"),absorber_element="Cu",edge="K",absorber_site="site0",nleg=2,reff=2.55,degen=12,k_max=20)
    model=ModelRevision(id="model1",bindings=[DatasetBinding(dataset_id="cu",processing_revision_id="pp_cu",paths=[PathBinding(id="p1",path_id="cu_path",expressions={"s02":"s02"})])],parameters=ParameterGraph(nodes=[Parameter(id="s02",kind="fixed",value=1)]))
    return Project(title="铜箔合同样例",datasets=[dataset],processing=[processed],paths=[path],models=[model],metadata={"contract_example":True})


def test_empty_project_defaults_and_uuid():
    p=Project(title="new")
    assert p.format=="webxas-project" and p.schema_version==1
    assert len(p.id)==36 and p.revision==0 and p.recomputable


def test_project_round_trip_preserves_unicode_and_nested_records():
    p=project()
    as_json=p.model_dump_json()
    assert Project.model_validate_json(as_json)==p
    assert Project.model_validate(json.loads(as_json))==p


@pytest.mark.parametrize("update",[{"schema_version":2},{"unexpected":1},{"revision":"1"},{"id":"not-a-uuid"}])
def test_project_rejects_unknown_version_fields_and_implicit_coercion(update):
    with pytest.raises(ValidationError):
        Project.model_validate({**project().model_dump(),**update})


@pytest.mark.parametrize("path",["../raw/a","/absolute","C:/a",r"raw\a","a//b","a/./b","a/../b"])
def test_assets_are_portable_relative_files(path):
    with pytest.raises(ValidationError):
        asset(path)


@pytest.mark.parametrize("change",[{"sha256":"g"*64},{"sha256":"A"*64},{"byte_size":-1}])
def test_asset_identity_is_well_formed(change):
    with pytest.raises(ValidationError):
        AssetRef.model_validate({**asset().model_dump(),**change})


def test_non_finite_values_rejected_even_in_nested_metadata():
    for value in [float("nan"),float("inf"),-float("inf")]:
        with pytest.raises(ValidationError):
            Project(metadata={"nested":{"value":[value]}})
        with pytest.raises(ValidationError):
            Parameter(id="bad",value=value)


def test_array_shape_and_boolean_mask():
    with pytest.raises(ValidationError):
        ArrayRef(**asset("arrays/x.npy").model_dump(),dtype="float64",shape=[0])
    array=ArrayRef(**asset("arrays/x.npy").model_dump(),dtype="float64",shape=[12])
    with pytest.raises(ValidationError):
        Dataset(id="d",raw_file=asset(),filename="x",mask=array)


@pytest.mark.parametrize("kwargs",[
    {"source":"explicit_scalar"},
    {"source":"explicit_scalar","value":0},
    {"source":"explicit_array"},
    {"source":"unknown","value":1},
    {"source":"engine_estimate","estimation_range":[5.,4.]},
])
def test_noise_source_has_explicit_contract(kwargs):
    with pytest.raises(ValidationError):
        NoiseSpec(**kwargs)


def test_noise_valid_scalar_and_array():
    assert NoiseSpec(source="explicit_scalar",value=0.1).value==0.1
    arr=ArrayRef(**asset("arrays/noise.npy").model_dump(),dtype="float64",shape=[12])
    assert NoiseSpec(source="explicit_array",array=arr).array.shape==[12]


@pytest.mark.parametrize("kwargs",[
    {"kind":"free","value":1,"expression":"other"},
    {"kind":"expr","expression":"other","value":1},
    {"kind":"fixed","value":1,"minimum":0},
    {"kind":"free","value":3,"minimum":0,"maximum":2},
    {"kind":"free","value":1,"minimum":2,"maximum":1},
    {"kind":"free","value":1,"scope":"dataset"},
    {"kind":"free","value":1,"scope":"global","owner_id":"d1"},
])
def test_parameter_declarations_are_unambiguous(kwargs):
    with pytest.raises(ValidationError):
        Parameter(id="p",**kwargs)


def test_unique_parameter_ids():
    p=Parameter(id="same",value=1)
    with pytest.raises(ValidationError):
        ParameterGraph(nodes=[p,p])


def test_model_rejects_unbound_local_parameter_owner():
    payload=project().models[0].model_dump()
    payload["parameters"]["nodes"]=[Parameter(id="x",value=1,scope="dataset",owner_id="other").model_dump()]
    with pytest.raises(ValidationError,match="owner"):
        ModelRevision.model_validate(payload)


@pytest.mark.parametrize("section,field,value",[
    ("processing","dataset_id","other"),
    ("models","dataset_id","other"),
    ("models","processing_revision_id","other"),
    ("models","path_id","other"),
])
def test_project_rejects_missing_or_cross_project_references(section,field,value):
    data=project().model_dump()
    if section=="processing":
        data[section][0][field]=value
    elif field=="path_id":
        data[section][0]["bindings"][0]["paths"][0][field]=value
    else:
        data[section][0]["bindings"][0][field]=value
    with pytest.raises(ValidationError):
        Project.model_validate(data)


def test_duplicate_entities_rejected():
    data=project().model_dump()
    data["datasets"].append(deepcopy(data["datasets"][0]))
    with pytest.raises(ValidationError,match="Duplicate"):
        Project.model_validate(data)


def test_joint_fft_density_cannot_accidentally_reweight_data():
    binding=project().models[0].bindings[0].model_dump()
    other=deepcopy(binding)
    other.update(dataset_id="other",processing_revision_id="other_pp")
    other["paths"][0]["id"]="other_p"
    other["transform"]["nfft"]=4096
    with pytest.raises(ValidationError,match="kstep/nfft"):
        ModelRevision(id="joint",bindings=[DatasetBinding.model_validate(binding),DatasetBinding.model_validate(other)])


def test_r_space_is_the_only_initially_verified_mode():
    with pytest.raises(ValidationError):
        TransformSpec(fitspace="k")
    with pytest.raises(ValidationError):
        TransformSpec(nfft=1000)


def test_calibration_unit_is_distinct_from_its_type():
    base={"id":"cal","kind":"energy","value":1.,"reference_dataset_ids":["cu"],"element":"Cu","edge":"K"}
    with pytest.raises(ValidationError):
        Calibration(**base,unit="1")
    assert Calibration(**base,unit="eV").kind=="energy"


def test_missing_uncertainty_is_null_with_reason():
    e=ParameterEstimate(id="n",value=10.,unit="1")
    assert json.loads(e.model_dump_json())["stderr"] is None
    with pytest.raises(ValidationError):
        ParameterEstimate(id="n",value=10.,unit="1",uncertainty_reason=None)


def test_covariance_dimensions_and_missing_reason():
    with pytest.raises(ValidationError):
        CovarianceRecord(var_names=["a","b"],matrix=[[1.]])
    with pytest.raises(ValidationError):
        CovarianceRecord(var_names=["a"])
    assert CovarianceRecord(var_names=["a"],missing_reason="rank deficient").matrix is None


def test_succeeded_run_requires_matching_model_and_result():
    p=project()
    spec=FitSpec(model=p.models[0])
    with pytest.raises(ValidationError):
        FitRun(id="r",model_id="wrong",input_snapshot=spec,snapshot_sha256="a"*64)
    with pytest.raises(ValidationError):
        FitRun(id="r",model_id=spec.model.id,input_snapshot=spec,snapshot_sha256="a"*64,task_state="succeeded")
    stats=FitStatistics(residual_count=100,n_independent=10.,n_variables=2,degrees_of_freedom=8.)
    result=FitResult(run_id="another",converged=True,termination_reason="converged",statistics=stats)
    with pytest.raises(ValidationError):
        FitRun(id="r",model_id=spec.model.id,input_snapshot=spec,snapshot_sha256="a"*64,result=result)


@pytest.mark.parametrize("name,count,elements",[("single",1,{"Cu"}),("joint",2,{"Cu"}),("cross_edge",2,{"Cu","Zn"})])
def test_examples_are_valid_and_do_not_claim_experimental_data(name,count,elements):
    example=Project.model_validate_json((Path(__file__).parent/'examples'/f'{name}.json').read_text(encoding='utf-8'))
    assert not example.recomputable and example.metadata['not_experimental_data']
    assert len(example.datasets)==count and {x.element for x in example.datasets}==elements
    assert Project.model_validate_json(example.model_dump_json())==example
