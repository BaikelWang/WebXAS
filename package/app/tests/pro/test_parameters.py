"""Parameter semantics checked before any scientific optimizer is invoked."""
import math

import pytest

from xasweb.pro.contracts import (
    AssetRef,CoordinationGroup,DatasetBinding,ModelRevision,Parameter,
    ParameterGraph,PathAsset,PathBinding,
)
from xasweb.pro.parameters import ParameterGraphError,compile_model


def asset(index=0,degen=12,nleg=2):
    return PathAsset(id=f"asset{index}",source_file=AssetRef(path=f"paths/p{index}.dat",sha256=f"{index+1:064x}",byte_size=100),absorber_element="Cu",edge="K",absorber_site="site0",nleg=nleg,reff=2.55,degen=degen,k_max=20)


def model(nodes,expressions=None,*,more_paths=None,groups=None):
    paths=[PathBinding(id="b0",path_id="asset0",expressions=expressions or {"deltar":"R"})]
    paths.extend(more_paths or [])
    return ModelRevision(id="m",bindings=[DatasetBinding(dataset_id="d0",processing_revision_id="pp0",paths=paths)],parameters=ParameterGraph(nodes=nodes),coordination_groups=groups or [])


def assert_error(code,fn):
    with pytest.raises(ParameterGraphError) as exc:fn()
    assert exc.value.code==code
    assert exc.value.object_id


def test_shared_and_derived_values_with_deterministic_names():
    nodes=[Parameter(id="R",unit="angstrom",value=.05),Parameter(id="twice",unit="angstrom",kind="expr",expression="2 * R")]
    compiled=compile_model(model(nodes,{"deltar":"twice"}),[asset()])
    assert compiled.name_map=={"R":"p_R","twice":"p_twice"}
    assert compiled.free_ids==("R",)
    assert compiled.dependency_order.index("R")<compiled.dependency_order.index("twice")
    assert compiled.values({"R":.1})["twice"]==pytest.approx(.2)
    assert compiled.path_values("b0",{"R":.1})["deltar"]==pytest.approx(.2)


def test_additive_constants_use_context_but_factors_are_dimensionless():
    compiled=compile_model(model([Parameter(id="R",unit="angstrom",value=2.6)],{"deltar":"R - 2.55","sigma2":"2 * 0.003"}),[asset()])
    assert compiled.path_values("b0")["deltar"]==pytest.approx(.05)
    assert compiled.path_values("b0")["sigma2"]==pytest.approx(.006)


@pytest.mark.parametrize("expression",["R + energy","R * R","energy/R"])
def test_path_units_cannot_be_mixed(expression):
    nodes=[Parameter(id="R",unit="angstrom",value=.1),Parameter(id="energy",unit="eV",kind="fixed",value=1)]
    assert_error("DIMENSION_MISMATCH",lambda:compile_model(model(nodes,{"deltar":expression}),[asset()]))


def test_declared_expression_parameter_unit_checked():
    nodes=[Parameter(id="R",unit="angstrom",value=.1),Parameter(id="derived",unit="eV",kind="expr",expression="R")]
    assert_error("DIMENSION_MISMATCH",lambda:compile_model(model(nodes,{"e0":"derived"}),[asset()]))


@pytest.mark.parametrize("expression",["abs(R)","R.real","R[0]","[R]","(x for x in [R])","True","R << 2","__import__('os')"])
def test_non_arithmetic_ast_rejected(expression):
    assert_error("EXPRESSION_AST" if expression!="True" else "EXPRESSION_LITERAL",lambda:compile_model(model([Parameter(id="R",unit="angstrom",value=.1)],{"deltar":expression}),[asset()]))


@pytest.mark.parametrize("expression",["R**R","R**1000"])
def test_unbounded_or_variable_powers_rejected(expression):
    assert_error("EXPRESSION_POWER",lambda:compile_model(model([Parameter(id="R",unit="angstrom",value=.1)],{"deltar":expression}),[asset()]))


def test_undefined_cycle_and_unused_free_are_not_silently_fixed():
    assert_error("UNDEFINED_PARAMETER",lambda:compile_model(model([],{"deltar":"missing"}),[asset()]))
    cycle=[Parameter(id="a",unit="angstrom",kind="expr",expression="b"),Parameter(id="b",unit="angstrom",kind="expr",expression="a")]
    assert_error("PARAMETER_CYCLE",lambda:compile_model(model(cycle,{"deltar":"a"}),[asset()]))
    assert_error("UNUSED_FREE_PARAMETER",lambda:compile_model(model([Parameter(id="unused",value=1)],{"s02":"1"}),[asset()]))


def test_closing_last_referencing_path_rechecks_free_variables():
    disabled=PathBinding(id="b1",path_id="asset1",enabled=False,expressions={"deltar":"R"})
    assert_error("UNUSED_FREE_PARAMETER",lambda:compile_model(model([Parameter(id="R",unit="angstrom",value=.1)],{"s02":"1"},more_paths=[disabled]),[asset(),asset(1)]))


def test_dataset_local_parameter_cannot_leak_to_another_spectrum():
    p=Parameter(id="local0",unit="angstrom",value=.1,scope="dataset",owner_id="d0")
    m=model([p],{"deltar":"local0"})
    other=DatasetBinding(dataset_id="d1",processing_revision_id="pp1",paths=[PathBinding(id="b1",path_id="asset1",expressions={"deltar":"local0"})])
    m=ModelRevision.model_validate({**m.model_dump(),"bindings":[*m.bindings,other]})
    assert_error("PARAMETER_SCOPE",lambda:compile_model(m,[asset(),asset(1)]))


def test_global_derived_parameter_cannot_depend_on_local_node():
    nodes=[Parameter(id="local0",unit="angstrom",value=.1,scope="dataset",owner_id="d0"),Parameter(id="g",unit="angstrom",kind="expr",expression="local0")]
    assert_error("PARAMETER_SCOPE",lambda:compile_model(model(nodes,{"deltar":"g"}),[asset()]))


@pytest.mark.parametrize("expression",["1/0","(-1)**0.5"])
def test_numeric_domain_errors_are_localized(expression):
    assert_error("EXPRESSION_DOMAIN",lambda:compile_model(model([],{"deltar":expression}),[asset()]))


def test_overrides_only_free_bounded_finite_parameters():
    c=compile_model(model([Parameter(id="R",unit="angstrom",value=.1,minimum=0,maximum=1)]),[asset()])
    assert_error("INVALID_OVERRIDE",lambda:c.values({"other":1}))
    assert_error("PARAMETER_BOUND",lambda:c.values({"R":2}))
    assert_error("PARAMETER_VALUE",lambda:c.values({"R":float("nan")}))


def cn_model(degeneracies,*,single=False,base_expression="s02"):
    nodes=[Parameter(id="s02",kind="fixed",value=.9),Parameter(id="N",value=10.,minimum=0,maximum=20)]
    paths=[PathBinding(id=f"b{i}",path_id=f"asset{i}",amplitude_semantics="single_path_cn" if single else "group_cn",cn_group_id=None if single else "shell",expressions={"s02":base_expression}) for i in range(len(degeneracies))]
    group=CoordinationGroup(id="shell",path_binding_ids=[p.id for p in paths],coordination_parameter_id="N",reference_degeneracy=sum(degeneracies),counting_basis="Non-overlapping single-scattering representatives in one explicitly declared shell.")
    m=ModelRevision(id="m",bindings=[DatasetBinding(dataset_id="d0",processing_revision_id="pp0",paths=paths)],parameters=ParameterGraph(nodes=nodes),coordination_groups=[group])
    return m,[asset(i,d) for i,d in enumerate(degeneracies)]


def test_single_and_group_cn_multiply_degeneracy_once():
    single=compile_model(*cn_model([12],single=True))
    grouped=compile_model(*cn_model([4,8]))
    assert single.path_values("b0")["s02"]*12==pytest.approx(.9*10)
    assert sum(grouped.path_values(f"b{i}")["s02"]*d for i,d in enumerate([4,8]))==pytest.approx(.9*10)
    occupancy=compile_model(model([Parameter(id="occ",value=10/12)],{"s02":"0.9*occ"}),[asset()])
    assert occupancy.path_values("b0")["s02"]==pytest.approx(single.path_values("b0")["s02"])


def test_cn_factor_in_base_expression_cannot_be_applied_twice():
    assert_error("CN_DOUBLE_COUNT",lambda:compile_model(*cn_model([12],base_expression="s02*N/12")))


def test_cn_group_sum_and_physical_identity_checked():
    m,paths=cn_model([4,8])
    data=m.model_dump();data["coordination_groups"][0]["reference_degeneracy"]=99.
    assert_error("CN_DEGENERACY",lambda:compile_model(ModelRevision.model_validate(data),paths))
    copies=[paths[0],paths[1].model_copy(update={"source_file":paths[0].source_file})]
    assert_error("CN_DUPLICATE_PATH",lambda:compile_model(m,copies))
    ms=[paths[0],paths[1].model_copy(update={"nleg":3})]
    assert_error("CN_MULTIPLE_SCATTERING",lambda:compile_model(m,ms))


def test_amplitude_product_degeneracy_identified_before_optimizer():
    nodes=[Parameter(id="N",value=10),Parameter(id="s02",value=.9)]
    c=compile_model(model(nodes,{"s02":"s02*N/12"}),[asset()])
    assert c.diagnostics[0]["code"]=="AMPLITUDE_DEGENERACY"
    assert_error("AMPLITUDE_DEGENERACY",lambda:c.to_larch())


def test_standard_sample_reference_can_identify_shared_s02_and_sample_n():
    nodes=[Parameter(id="N",value=10),Parameter(id="s02",value=.9)]
    m=model(nodes,{"s02":"s02"})
    other=DatasetBinding(dataset_id="d1",processing_revision_id="pp1",paths=[PathBinding(id="b1",path_id="asset1",expressions={"s02":"s02*N/12"})])
    m=ModelRevision.model_validate({**m.model_dump(),"bindings":[*m.bindings,other]})
    c=compile_model(m,[asset(),asset(1)])
    assert not c.diagnostics
    changed=c.path_values(overrides={"N":11,"s02":.8})
    assert changed["b0"]["s02"]==.8
    assert changed["b1"]["s02"]==pytest.approx(.8*11/12)


def test_larch_safe_parameter_compilation_preserves_values():
    from larch.fitting import group2params
    nodes=[Parameter(id="R",unit="angstrom",value=.1),Parameter(id="dr2",unit="angstrom",kind="expr",expression="R/2")]
    c=compile_model(model(nodes,{"deltar":"dr2"}),[asset()])
    group,expressions=c.to_larch()
    params=group2params(group)
    params.update_constraints()
    assert params[c.name_map["R"]].vary
    assert params[c.name_map["dr2"]].value==pytest.approx(.05)
    assert expressions["b0"]["deltar"]=="p_dr2"
    assert expressions["b0"]["s02"]=="1"
