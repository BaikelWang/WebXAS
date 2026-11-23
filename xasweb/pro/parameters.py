"""Compile an explicitly scoped parameter graph without executing user code.

Constants in an additive expression adopt that expression's unit. Numeric
factors in products/quotients are dimensionless; an entirely numeric expression
adopts its destination unit. All values use angstrom, eV and kelvin conventions.

``scale`` bindings give FEFF's full amplitude multiplier. For ``group_cn`` and
``single_path_cn``, expressions.s02 is intrinsic S02 only: this compiler applies
N/reference_degeneracy exactly once. Adapters must not apply that factor again.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Mapping

from .contracts import ModelRevision, PathAsset

Dim = tuple[Fraction, Fraction, Fraction]
ZERO: Dim = (Fraction(0), Fraction(0), Fraction(0))
UNITS = {"1": ZERO, "angstrom": (1,0,0), "angstrom2": (2,0,0),
         "angstrom-1": (-1,0,0), "eV": (0,1,0), "K": (0,0,1)}
PATH_UNITS = {"s02": ZERO, "e0": UNITS["eV"], "deltar": UNITS["angstrom"],
              "sigma2": UNITS["angstrom2"], "third": (3,0,0), "fourth": (4,0,0)}
DEFAULTS = {"s02":"1", "e0":"0", "deltar":"0", "sigma2":"0", "third":"0", "fourth":"0"}


class ParameterGraphError(ValueError):
    def __init__(self, code, message, object_id=None):
        self.code, self.object_id = code, object_id
        super().__init__(message)


def _fail(code, text, owner):
    raise ParameterGraphError(code,text,owner)


@dataclass(frozen=True)
class Expression:
    source: str
    tree: ast.Expression
    names: frozenset[str]


def _parse(source, owner):
    try:
        tree=ast.parse(source,mode="eval")
    except (SyntaxError,ValueError,RecursionError):
        _fail("EXPRESSION_SYNTAX","Expression is not valid arithmetic.",owner)
    nodes=list(ast.walk(tree))
    if len(nodes)>200:
        _fail("EXPRESSION_COMPLEXITY","Expression exceeds the 200-node arithmetic limit.",owner)
    allowed=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Name,ast.Load,ast.Constant,
             ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.UAdd,ast.USub)
    for node in nodes:
        if not isinstance(node,allowed):
            _fail("EXPRESSION_AST",f"Unsupported expression operation: {type(node).__name__}.",owner)
        if isinstance(node,ast.Constant):
            if isinstance(node.value,bool) or not isinstance(node.value,(int,float)):
                _fail("EXPRESSION_LITERAL","Only finite real numeric constants are allowed.",owner)
            try: value=float(node.value)
            except (OverflowError,ValueError): value=float("inf")
            if not math.isfinite(value):
                _fail("EXPRESSION_LITERAL","Only finite real numeric constants are allowed.",owner)
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow):
            power=node.right
            if isinstance(power,ast.UnaryOp) and isinstance(power.op,(ast.UAdd,ast.USub)):
                power=power.operand
            if not isinstance(power,ast.Constant) or isinstance(power.value,bool):
                _fail("EXPRESSION_POWER","Powers require a literal numeric exponent.",owner)
            if abs(float(power.value))>8:
                _fail("EXPRESSION_POWER","Power magnitude cannot exceed 8.",owner)
    return Expression(source,tree,frozenset(n.id for n in nodes if isinstance(n,ast.Name)))


def _constant(node):
    if isinstance(node,ast.Constant): return float(node.value)
    if isinstance(node,ast.UnaryOp):
        return (-1 if isinstance(node.op,ast.USub) else 1)*_constant(node.operand)
    raise ValueError("not a constant")


def _dimension(node, dimensions, owner):
    if isinstance(node,ast.Constant): return None
    if isinstance(node,ast.Name): return dimensions[node.id]
    if isinstance(node,ast.UnaryOp): return _dimension(node.operand,dimensions,owner)
    left=_dimension(node.left,dimensions,owner)
    right=_dimension(node.right,dimensions,owner)
    if isinstance(node.op,(ast.Add,ast.Sub)):
        if left is None: return right
        if right is None: return left
        if left!=right: _fail("DIMENSION_MISMATCH","Addition/subtraction requires compatible units.",owner)
        return left
    # A purely numeric subtree retains the destination's contextual unit.
    if left is None and right is None: return None
    left=left or ZERO; right=right or ZERO
    if isinstance(node.op,ast.Mult): return tuple(a+b for a,b in zip(left,right))
    if isinstance(node.op,ast.Div): return tuple(a-b for a,b in zip(left,right))
    return tuple(a*Fraction(str(_constant(node.right))) for a in left)


@dataclass
class _Dual:
    value: float
    gradient: dict[str,float]


def _combine(a,b,af,bf):
    return {key:af*a.get(key,0.)+bf*b.get(key,0.) for key in a.keys()|b.keys()}


def _evaluate(node, values, owner):
    try:
        if isinstance(node,ast.Constant): return _Dual(float(node.value),{})
        if isinstance(node,ast.Name): return values[node.id]
        if isinstance(node,ast.UnaryOp):
            a=_evaluate(node.operand,values,owner)
            sign=-1 if isinstance(node.op,ast.USub) else 1
            return _Dual(sign*a.value,{k:sign*v for k,v in a.gradient.items()})
        a=_evaluate(node.left,values,owner); b=_evaluate(node.right,values,owner)
        if isinstance(node.op,ast.Add): value,grad=a.value+b.value,_combine(a.gradient,b.gradient,1,1)
        elif isinstance(node.op,ast.Sub): value,grad=a.value-b.value,_combine(a.gradient,b.gradient,1,-1)
        elif isinstance(node.op,ast.Mult): value,grad=a.value*b.value,_combine(a.gradient,b.gradient,b.value,a.value)
        elif isinstance(node.op,ast.Div): value,grad=a.value/b.value,_combine(a.gradient,b.gradient,1/b.value,-a.value/(b.value*b.value))
        else:
            power=b.value; value=a.value**power
            factor=0. if power==0 else power*(a.value**(power-1))
            grad={k:factor*v for k,v in a.gradient.items()}
        if isinstance(value,complex) or not math.isfinite(value) or any(not math.isfinite(v) for v in grad.values()):
            _fail("EXPRESSION_DOMAIN","Expression produced a non-finite or complex result.",owner)
        return _Dual(value,grad)
    except (ZeroDivisionError,OverflowError,ValueError) as exc:
        if isinstance(exc,ParameterGraphError): raise
        _fail("EXPRESSION_DOMAIN",f"Arithmetic expression is outside its domain: {exc}.",owner)


class _Rename(ast.NodeTransformer):
    def __init__(self,names): self.names=names
    def visit_Name(self,node): return ast.copy_location(ast.Name(id=self.names[node.id],ctx=ast.Load()),node)


class CompiledModel:
    def __init__(self,model,assets):
        self.model=model.model_copy(deep=True)
        self.assets={p.id:p for p in assets}
        if len(self.assets)!=len(assets): _fail("DUPLICATE_PATH","Path asset IDs must be unique.",model.id)
        self.nodes={p.id:p for p in self.model.parameters.nodes}
        self.name_map={pid:"p_"+pid for pid in sorted(self.nodes)}
        self.free_ids=tuple(sorted(p.id for p in self.nodes.values() if p.kind=="free"))
        self.dependency_order=[]
        self.expressions={}
        self.path_expressions={}
        self.path_owners={p.id:d.dataset_id for d in self.model.bindings for p in d.paths}
        self.binding_assets={}
        self.diagnostics=[]
        dimensions={pid:UNITS[p.unit] for pid,p in self.nodes.items()}
        for pid,p in self.nodes.items():
            if p.kind=="expr":
                expression=self._expression(p.expression,pid,p.scope,p.owner_id)
                self.expressions[pid]=expression
                actual=_dimension(expression.tree.body,dimensions,pid)
                if actual is not None and actual!=dimensions[pid]:
                    _fail("DIMENSION_MISMATCH",f"Expression for {pid} does not match declared {p.unit}.",pid)
        self._topology()
        groups={g.id:g for g in self.model.coordination_groups}
        all_bindings={p.id:p for d in self.model.bindings for p in d.paths}
        for group in groups.values():
            members=[all_bindings[b] for b in group.path_binding_ids]
            member_assets=[]
            for b in members:
                if b.path_id not in self.assets: _fail("UNKNOWN_PATH",f"Unknown asset {b.path_id}.",b.id)
                member_assets.append(self.assets[b.path_id])
            if any(p.nleg!=2 for p in member_assets): _fail("CN_MULTIPLE_SCATTERING","MS paths cannot be counted as single-scattering CN.",group.id)
            if len({self.path_owners[b.id] for b in members})!=1:
                _fail("CN_SCOPE","A coordination group belongs to one dataset; share its parameter explicitly across groups.",group.id)
            contexts={(p.absorber_element,p.edge,p.absorber_site) for p in member_assets}
            if len(contexts)!=1: _fail("CN_SCOPE","CN group members require one absorber site and edge.",group.id)
            identities=[(p.source_file.sha256,p.absorber_site,p.absorber_element,p.edge) for p in member_assets]
            if len(identities)!=len(set(identities)): _fail("CN_DUPLICATE_PATH","CN group contains a repeated physical path asset.",group.id)
            expected=sum(p.degen for p in member_assets)
            if not math.isclose(group.reference_degeneracy,expected,rel_tol=1e-12,abs_tol=1e-12):
                _fail("CN_DEGENERACY","CN reference degeneracy must match the non-duplicated member path degeneracies.",group.id)
            if UNITS[self.nodes[group.coordination_parameter_id].unit]!=ZERO:
                _fail("DIMENSION_MISMATCH","Coordination number must be dimensionless.",group.id)
        used=set()
        for dataset in self.model.bindings:
            if not any(p.enabled for p in dataset.paths): _fail("NO_ENABLED_PATH","Every fitted dataset requires an enabled path.",dataset.dataset_id)
            for binding in dataset.paths:
                if binding.path_id not in self.assets: _fail("UNKNOWN_PATH",f"Unknown asset {binding.path_id}.",binding.id)
                asset=self.assets[binding.path_id]
                self.binding_assets[binding.id]=asset
                if not binding.enabled: continue
                expressions={**DEFAULTS,**binding.expressions}
                if binding.amplitude_semantics!="scale":
                    candidates=[g for g in groups.values() if binding.id in g.path_binding_ids]
                    if binding.amplitude_semantics=="group_cn":
                        candidates=[g for g in candidates if g.id==binding.cn_group_id]
                    if len(candidates)!=1: _fail("CN_GROUP_REQUIRED","CN amplitude needs one explicit counting group.",binding.id)
                    group=candidates[0]
                    if binding.amplitude_semantics=="single_path_cn" and len(group.path_binding_ids)!=1:
                        _fail("CN_GROUP_REQUIRED","single_path_cn needs a single-member group.",binding.id)
                    if asset.nleg!=2: _fail("CN_MULTIPLE_SCATTERING","MS paths cannot be converted to CN.",binding.id)
                    if any(not all_bindings[x].enabled for x in group.path_binding_ids):
                        _fail("CN_DISABLED_MEMBER","Revise the CN counting model before disabling a counted path.",group.id)
                    base=self._expression(expressions["s02"],binding.id,"path",binding.id)
                    if group.coordination_parameter_id in self._closure(base.names):
                        _fail("CN_DOUBLE_COUNT","Intrinsic S02 expression must not already contain the CN factor.",binding.id)
                    expressions["s02"]=f"({expressions['s02']})*{group.coordination_parameter_id}/{group.reference_degeneracy!r}"
                self.path_expressions[binding.id]={}
                for field,source in expressions.items():
                    expression=self._expression(source,binding.id,"path",binding.id)
                    actual=_dimension(expression.tree.body,dimensions,binding.id)
                    if actual is not None and actual!=PATH_UNITS[field]:
                        _fail("DIMENSION_MISMATCH",f"Path {field} expression has incompatible units.",binding.id)
                    self.path_expressions[binding.id][field]=expression
                    used.update(self._closure(expression.names))
        unused=set(self.free_ids)-used
        if unused: _fail("UNUSED_FREE_PARAMETER",f"Unused free parameters: {', '.join(sorted(unused))}.",self.model.id)
        self.path_values()  # Domain and physical checks on declared initial values.
        self._amplitude_diagnostic()

    def _expression(self,source,owner,scope,scope_owner):
        expression=_parse(source,owner)
        missing=expression.names-self.nodes.keys()
        if missing: _fail("UNDEFINED_PARAMETER",f"Undefined parameters: {', '.join(sorted(missing))}.",owner)
        dataset_id=(scope_owner if scope=="dataset" else self.path_owners.get(scope_owner))
        for name in expression.names:
            node=self.nodes[name]
            allowed=(node.scope=="global" or
                     (scope in ("dataset","path") and node.scope=="dataset" and node.owner_id==dataset_id) or
                     (scope=="path" and node.scope=="path" and node.owner_id==scope_owner))
            if not allowed: _fail("PARAMETER_SCOPE",f"{owner} cannot reference local parameter {name}.",owner)
        return expression

    def _topology(self):
        visiting=set(); done=set()
        def visit(pid):
            if pid in visiting: _fail("PARAMETER_CYCLE",f"Cyclic parameter dependency at {pid}.",pid)
            if pid in done:return
            visiting.add(pid)
            for dependency in sorted(self.expressions[pid].names if pid in self.expressions else []): visit(dependency)
            visiting.remove(pid);done.add(pid);self.dependency_order.append(pid)
        for pid in sorted(self.nodes):visit(pid)

    def _closure(self,names):
        result=set(names)
        pending=list(names)
        while pending:
            pid=pending.pop()
            for item in self.expressions[pid].names if pid in self.expressions else []:
                if item not in result:result.add(item);pending.append(item)
        return result

    def _values(self,overrides=None):
        overrides=dict(overrides or {})
        invalid=set(overrides)-set(self.free_ids)
        if invalid: _fail("INVALID_OVERRIDE",f"Only free parameters can be overridden: {', '.join(sorted(invalid))}.",self.model.id)
        values={}
        for pid in self.dependency_order:
            p=self.nodes[pid]
            if p.kind=="expr":values[pid]=_evaluate(self.expressions[pid].tree.body,values,pid)
            else:
                value=overrides.get(pid,p.value)
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                    _fail("PARAMETER_VALUE","Parameter override must be finite and numeric.",pid)
                if (p.minimum is not None and value<p.minimum) or (p.maximum is not None and value>p.maximum):
                    _fail("PARAMETER_BOUND","Parameter override lies outside its bounds.",pid)
                values[pid]=_Dual(float(value),{pid:1.} if p.kind=="free" else {})
        return values

    def values(self,overrides=None):
        return {pid:x.value for pid,x in self._values(overrides).items()}

    def path_values(self,binding_id=None,overrides=None):
        values=self._values(overrides)
        selected=self.path_expressions if binding_id is None else {binding_id:self.path_expressions[binding_id]}
        output={}
        for bid,fields in selected.items():
            row={field:_evaluate(expression.tree.body,values,bid).value for field,expression in fields.items()}
            if row["s02"]<0 or row["sigma2"]<0 or self.binding_assets[bid].reff+row["deltar"]<=0:
                _fail("PATH_PHYSICAL_VALUE","Path requires non-negative amplitude/sigma2 and positive effective length.",bid)
            output[bid]=row
        return output if binding_id is None else output[binding_id]

    def _amplitude_diagnostic(self):
        amplitude=set();other=set()
        for fields in self.path_expressions.values():
            for field,expression in fields.items():
                (amplitude if field=="s02" else other).update(self._closure(expression.names))
        candidates=sorted(set(self.free_ids)&amplitude-other)
        if len(candidates)<2:return
        import numpy as np
        values=self._values()
        jac=np.array([[_evaluate(fields["s02"].tree.body,values,bid).gradient.get(pid,0.) for pid in candidates] for bid,fields in self.path_expressions.items()])
        norms=np.linalg.norm(jac,axis=0)
        scaled=jac/np.where(norms>0,norms,1)
        rank=int(np.linalg.matrix_rank(scaled))
        if rank<len(candidates):
            self.diagnostics.append({"code":"AMPLITUDE_DEGENERACY","severity":"error","parameter_ids":candidates,"rank":rank,"message":"Independent amplitude parameters cannot be separated at the declared model point; fix one or supply a reference constraint."})

    def to_larch(self,*,allow_degenerate=False):
        if not allow_degenerate and any(x["severity"]=="error" for x in self.diagnostics):
            _fail("AMPLITUDE_DEGENERACY",self.diagnostics[0]["message"],self.model.id)
        from larch.fitting import param,param_group
        group=param_group()
        values=self.values()
        for pid in self.dependency_order:
            node=self.nodes[pid]
            options={"value":values[pid],"vary":node.kind=="free"}
            if node.minimum is not None:options["min"]=node.minimum
            if node.maximum is not None:options["max"]=node.maximum
            if node.kind=="expr":options["expr"]=self._larch_expression(self.expressions[pid])
            setattr(group,self.name_map[pid],param(**options))
        paths={bid:{field:self._larch_expression(expr) for field,expr in fields.items()} for bid,fields in self.path_expressions.items()}
        return group,paths

    def _larch_expression(self,expression):
        # Parse a fresh tree: never mutate the evaluation AST stored above.
        tree=_Rename(self.name_map).visit(ast.parse(expression.source,mode="eval"))
        return ast.unparse(ast.fix_missing_locations(tree))


def compile_model(model: ModelRevision,paths: list[PathAsset]) -> CompiledModel:
    """Return a checked, inspectable graph and lazy Larch compiler."""
    return CompiledModel(model,paths)
