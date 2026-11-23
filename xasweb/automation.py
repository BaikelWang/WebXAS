"""Bounded recipe selection. Diagnostics are heuristics, never scientific truth."""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import time
from typing import Literal

import numpy as np
from larch.xafs import find_e0
from larch.xafs.xafsutils import ETOK
from pydantic import Field, field_validator
from .edges import normalize_edge

from .ingest import InputError, construct_mu, parse_ascii
from .models import AnalysisRequest, CheckedModel, Mapping, Parameters
from .science import ROOT, analyze, canonical_json


class AutoRequest(CheckedModel):
    object_id: str | None = Field(default=None, max_length=80)
    assessment_token: str | None = Field(default=None, max_length=16000)
    filename: str = Field(default="spectrum.dat", max_length=200)
    raw_text: str = Field(max_length=2_000_000)
    mapping: Mapping | None = None
    sample: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=5000)
    method: Literal["fixed", "rules", "random", "agent", "a2c", "physics"] = "rules"
    budget: int = Field(default=8, ge=1, le=32)
    seed: int = Field(default=0, ge=0)
    experimental_search: bool = False
    element: str | None = Field(default=None,max_length=3)
    edge: str | None = Field(default=None,max_length=3)

    @field_validator('edge', mode='before')
    @classmethod
    def canonical_edge(cls, value):
        return normalize_edge(value) if isinstance(value, str) else value


def declared_columns(header, width):
    """Read actual Column fields; Describe.Column is descriptive metadata."""
    columns, positions, units = {}, {}, {}
    for line in header:
        match = re.match(r"^\s*(?:[#;%]\s*)?Column\.(\d+):\s*(\S+)(?:\s+(\S+))?", line, re.I)
        if not match:
            continue
        index, name = int(match[1]) - 1, match[2].lower()
        if not 0 <= index < width:
            return {}, None, 'invalid_column_declarations'
        if name == 'energy' and match[3] and match[3].lower() not in ('ev', 'kev'):
            return {}, None, 'invalid_column_declarations'
        unit = match[3].lower() if match[3] and match[3].lower() in ('ev', 'kev') else None
        if ((name in columns and columns[name] != index)
                or (index in positions and positions[index] != name)
                or (unit and name in units and units[name] != unit)):
            return {}, None, 'ambiguous_column_declarations'
        columns[name], positions[index] = index, name
        if unit:
            units[name] = unit
    unit = {'ev': 'eV', 'kev': 'keV'}.get(units.get('energy'))
    return columns, unit, None


def signal_mapping(columns, energy, unit, header):
    """Choose a named sample signal; reference-only mur/Ir is never a sample."""
    def first(*names):
        return next((columns[name] for name in names if name in columns), None)

    modes = set()
    for line in header:
        modes.update(m.lower() for m in re.findall(
            r'(?:measurement[\s._-]*mode|detection[\s._-]*mode|scan\.mode)\s*[:=]\s*(transmission|fluorescence)\b', line, re.I))
    if len(modes) > 1:
        return None, 'ambiguous_signal_mode'
    mode = next(iter(modes), None)
    direct = first('mu', 'xmu', 'absorption', 'mu(e)', 'xmu(e)', 'absorption(e)')
    mut = first('mut', 'mutrans', 'mutrans(e)')
    muf = first('muf', 'mufluor', 'mufluor(e)')
    i0 = first('i0', 'i_0', 'io')
    it = first('it', 'i1', 'itrans', 'i_trans')
    fluor = first('if', 'ifluor', 'fluorescence', 'ifl')
    transmission = mut is not None or (i0 is not None and it is not None)
    fluorescence = muf is not None or (i0 is not None and fluor is not None)
    base = {'energy': energy, 'energy_unit': unit}

    if mode is None and direct is not None:
        return Mapping(**base, signal=direct, reference=0), 'xdi_header'
    if mode == 'fluorescence' or (mode is None and fluorescence):
        source = 'fluorescence_preferred_over_transmission' if mode is None and transmission else 'declared_fluorescence'
        if muf is not None:
            return Mapping(**base, signal=muf, reference=0), source
        if i0 is not None and fluor is not None:
            return Mapping(**base, signal=fluor, reference=i0, mode='fluorescence'), source
    elif mode == 'transmission' or transmission:
        if i0 is not None and it is not None:
            return Mapping(**base, signal=it, reference=i0, mode='transmission'), 'xdi_header'
        if mut is not None:
            return Mapping(**base, signal=mut, reference=0), 'xdi_header'
    if mode and direct is not None:
        return Mapping(**base, signal=direct, reference=0), 'xdi_header'
    if mode is None and all(value is None for value in (direct, mut, muf, it, fluor)) and first('mur', 'ir', 'i2') is not None:
        return None, 'reference_signal_only'
    return None, 'declared_signal_mode_unavailable' if mode else 'confirm_signal_mode'


def mapping_assumptions(source):
    if source == 'fluorescence_preferred_over_transmission':
        return ['文件同时含透射与荧光通道；初筛默认采用已命名的荧光样品信号。此通道选择属于假设，未进行自吸收校正。']
    if source == 'declared_fluorescence':
        return ['使用已声明的荧光样品信号；未进行自吸收校正。']
    return []


def infer_mapping(raw_text: str):
    """Accept declared columns and units without specimen or example bindings."""
    table, header = parse_ascii(raw_text)
    columns, unit, reason = declared_columns(header, table.shape[1])
    if reason:
        return None, reason
    if 'energy' not in columns or unit is None:
        return None, 'confirm_columns_and_energy_unit'
    return signal_mapping(columns, columns['energy'], unit, header)


def recipe_candidates(request: AnalysisRequest, metadata=None) -> list[Parameters]:
    """Shared finite action space; all ranges are supported by measured samples."""
    energy, mu, _ = construct_mu(request.raw_text, request.mapping)
    e0 = float(metadata["detected_edge_eV"] if metadata and "detected_edge_eV" in metadata else find_e0(energy, mu))
    usable_end = min(energy[-1], (metadata or {}).get("usable_energy_max_eV", energy[-1]))
    before, after = e0 - energy[0], usable_end - e0
    if before < 45 or after < 180:
        raise InputError("Insufficient pre/post-edge coverage for automatic EXAFS. Supply a spectrum with a wider energy range.")
    available = np.floor(np.sqrt(after * ETOK) / .05) * .05
    kmax = min(12., np.floor((available - .55) * 10) / 10)
    if kmax < 5:
        raise InputError("The scan does not support the automatic EXAFS k range.")
    base = Parameters(e0=e0, pre1=max(-150., -before + 2), pre2=-30,
                      norm1=min(150., after * .25), norm2=min(700., after - 3),
                      nnorm=2, kmax=kmax,
                      bkg_kmax=float(available) if usable_end < energy[-1] else None)
    candidates = [base]
    # A bounded, deterministic library; candidate 0 is identical across policies.
    for rbkg, weight, degree, edge_delta in itertools.product((.8, 1., 1.2), (1, 2), (1, 2), (0., -1., 1.)):
        values = base.model_dump()
        values.update(rbkg=rbkg, bkg_kweight=weight, nnorm=degree, e0=e0 + edge_delta)
        candidates.append(Parameters(**values))
    for fraction, pre2, kmin in itertools.product((.6, .85), (-20., -40.), (2., 3.)):
        values = base.model_dump()
        values.update(pre1=max(-200., -before + 3), pre2=pre2,
                      norm1=max(50., min(150., after * .2)), norm2=min(after - 3, after * fraction), kmin=kmin)
        candidates.append(Parameters(**values))
    for reduction in (1., 2., 3.):
        if kmax - reduction > 4:
            candidates.append(base.model_copy(update={"kmax": kmax - reduction}))
    unique = {canonical_json(p.model_dump()): p for p in candidates}
    return list(unique.values())


def diagnostics(result):
    a, p = result["arrays"], result["request"]["parameters"]
    e = np.asarray(a["energy"])
    mu, norm = np.asarray(a["mu"]), np.asarray(a["norm"])
    k, chi = np.asarray(a["k"]), np.asarray(a["chi"])
    edge, step = result["metrics"]["e0_eV"], result["metrics"]["edge_step"]
    pre = (e >= edge + p["pre1"]) & (e <= edge + p["pre2"])
    pre_noise = float(np.sqrt(np.mean(((mu - a["pre_edge"])[pre] / step) ** 2)))
    post = (e > edge + p["norm1"]) & (e < edge + p["norm2"])
    post_offset = float(abs(np.median(norm[post]) - 1))
    band = (k >= 2) & (k <= min(12, k[-1]))
    amplitude = float(np.sqrt(np.mean(chi[band] ** 2)))
    tail = chi[(k >= max(2, k[-1] - 2))]
    roughness = float(np.median(np.abs(np.diff(tail, n=2))) / max(amplitude, 1e-9))
    # Excludes low-R energy: minimizing it alone can erase real signal.
    score = pre_noise + .1 * post_offset + .02 * roughness
    flags = []
    if pre_noise > .03:
        flags.append("noisy_pre_edge")
    if result["metrics"]["k_grid_max"] < 12.5:
        flags.append("limited_k_coverage")
    if amplitude < 1e-5:
        flags.append("weak_exafs_signal")
    if any("Sparse near-edge" in w for w in result["warnings"]):
        flags.append("sparse_near_edge")
    return {"pre_edge_noise": pre_noise, "post_edge_offset": post_offset,
            "chi_rms": amplitude, "tail_roughness": roughness,
            "heuristic_score": float(score), "flags": flags}


def observation(result, count, budget):
    """No target, reference error, reward or held-out identity in policy state."""
    d, p, m = diagnostics(result), result["request"]["parameters"], result["metrics"]
    return np.asarray([d["pre_edge_noise"], d["post_edge_offset"], d["chi_rms"],
                       min(d["tail_roughness"], 10), p["rbkg"], p["bkg_kweight"] / 3,
                       p["nnorm"] / 3, p["kmin"] / 12, p["kmax"] / 15,
                       m["k_available"] / 20, count / budget, float(bool(d["flags"])),
                       p["pre1"] / 300, p["pre2"] / 100, p["norm1"] / 1000,
                       p["norm2"] / 1000, m["e0_eV"] / 10000,
                       p["energy_shift"] / 1000, p["dk"] / 5, p["kweight"] / 3], dtype=np.float32)


class SelectionSession:
    def __init__(self, request, budget=8):
        self.request, self.budget = request, budget
        self.trace, self.results = [], []
        self.started = time.perf_counter()
        self.metadata = {}
        self.hard_stopped = False

    def evaluate(self, parameters):
        if self.hard_stopped:
            raise InputError("Nonfinite computation stopped this session; inspect the cause before retrying.")
        if len(self.trace) >= self.budget:
            raise InputError("Evaluation budget exhausted.")
        start = time.perf_counter()
        record = {"attempt": len(self.trace) + 1, "parameters": parameters.model_dump() if isinstance(parameters, Parameters) else parameters}
        try:
            # Revalidate defaults too: Pydantic's unvalidated integer defaults
            # otherwise change JSON identity (e.g. 2 versus 2.0) after replay.
            p = Parameters.model_validate(parameters.model_dump() if isinstance(parameters, Parameters) else parameters)
            request = self.request.model_copy(update={"parameters": p})
            from .assessment import assess_request
            assessment = assess_request(request)
            if not assessment['analysis_eligible']:
                raise InputError('; '.join(s['message'] for s in assessment['stop_reasons']))
            result = analyze(request)
            assessment.update(stage='S4', completed_stages=['S2','S3','S4'],
                completed_operations=['construct_or_retain_mu','normalize','extract_exafs','fourier_transform'],
                permitted_operations=['preview','construct_or_retain_mu','normalize','extract_exafs','fourier_transform'])
            assessment['allowed_operations'] = assessment['permitted_operations']
            result['assessment'] = assessment
            result['provenance'] = assessment['provenance']
            d = diagnostics(result)
            record.update(status="completed", run_id=result["run_id"], diagnostics=d)
            self.results.append((d["heuristic_score"], result))
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            record.update(status="failed", reason=str(exc)[:500])
            if isinstance(exc, FloatingPointError) or "nonfinite" in str(exc).lower() or "non-finite" in str(exc).lower():
                self.hard_stopped = True
                record["hard_stop"] = True
            result = None
        finally:
            record["wall_seconds"] = time.perf_counter() - start
            self.trace.append(record)
        return result

    def finish(self, method, mapping_source, extra=None, experimental_search=False):
        result = (min(self.results, key=lambda pair: pair[0])[1] if experimental_search else self.results[0][1]) if self.results else None
        result = getattr(self, "selected_result", result)
        quality = {"status": "review_required", "validated": False,
                   "reason": "Automatic recipe validation and expert blind review are not yet complete.",
                   "diagnostics": diagnostics(result) if result else None}
        if result and self.results:
            baseline_amplitude = diagnostics(self.results[0][1])["chi_rms"]
            if quality["diagnostics"]["chi_rms"] < .5 * baseline_amplitude:
                quality["diagnostics"]["flags"].append("signal_loss_relative_to_initial_recipe")
        provenance = {"schema_version": 1, "method": method, "mapping_source": mapping_source,
                      "selection": "experimental_heuristic" if experimental_search else "conservative_initial_valid_recipe",
                      "quality": quality, "trace": self.trace, "scientific_metadata": self.metadata, "hard_stopped": self.hard_stopped,
                      "cost": {"evaluations": len(self.trace), "budget": self.budget,
                               "wall_seconds": time.perf_counter() - self.started,
                               "numerical_wall_seconds": sum(t["wall_seconds"] for t in self.trace),
                               "gpu_hours": (extra or {}).get("model_cost", {}).get("local_gpu_hours", 0.)},
                      **(extra or {})}
        if result:
            provenance["selected_run_id"] = result["run_id"]
        return {"status": "needs_review" if result else "failed", "result": result,
                "request": {**result["request"], "raw_text": self.request.raw_text} if result else None,
                "automation": provenance}


def automatic_analysis(request: AutoRequest, *, selector=None):
    from .assessment import automatic_assessed
    assessed = automatic_assessed(request)
    if assessed is not None:
        return assessed
    mapping, mapping_source = (request.mapping, "user_confirmed") if request.mapping else infer_mapping(request.raw_text)
    if mapping is None:
        return {"status": "needs_input", "result": None, "request": None,
                "reason": mapping_source, "required": ["mapping"]}
    from .metadata import inspect_metadata
    scientific_metadata = inspect_metadata(request.raw_text,mapping,request.element,request.edge)
    if scientific_metadata["status"] == "needs_review":
        return {"status":"needs_review", "result":None, "request":None,
                "reason":scientific_metadata["reason"], "scientific_metadata":scientific_metadata}
    manual = AnalysisRequest(raw_text=request.raw_text, filename=request.filename, assessment_token=request.assessment_token,
                             mapping=mapping, sample=request.sample, notes=request.notes)
    session = SelectionSession(manual, request.budget)
    session.metadata = scientific_metadata
    if request.method == "fixed" and scientific_metadata.get("next_edge"):
        return {"status":"needs_review","result":None,"request":None,"scientific_metadata":scientific_metadata,
                "reason":"Fixed recipe may cross the next absorption edge. Use adaptive processing or specify a manual background cutoff."}
    if request.method == "fixed":
        session.evaluate(manual.parameters)
    else:
        try:
            candidates = recipe_candidates(manual, scientific_metadata)
        except InputError as exc:
            from .assessment import adaptive_partial
            try:
                if scientific_metadata.get('next_edge'):
                    # A generic fallback must not cross a following absorption edge.
                    return {'status': 'needs_review', 'result': None, 'request': None,
                            'reason': str(exc), 'scientific_metadata': scientific_metadata}
                return adaptive_partial(manual, str(exc), scientific_metadata)
            except InputError:
                return {"status": "needs_review", "result": None, "request": None, "reason": str(exc)}
        if request.method in ("agent", "a2c") and selector is None:
            return {"status": "unavailable", "result": None, "request": None,
                    "reason": "Run this method through the research CLI with a pinned local model/checkpoint."}
        session.evaluate(candidates[0])
        if request.method == "physics" and selector is None:
            from .physics_selection import search
            metadata = search(session, candidates)
            return session.finish(request.method, mapping_source, metadata)
        if selector is not None:
            try:
                metadata = selector(session, candidates)
            except (ValueError, RuntimeError) as exc:
                metadata = {"policy_status": "failed", "policy_error": str(exc),
                            "model_calls": getattr(selector, "last_calls", [])}
            completed = session.finish(request.method, mapping_source, metadata, request.experimental_search)
            if metadata.get("policy_status") == "failed":
                completed.update(status="failed", reason=metadata.get("policy_error", "No valid policy tool call."),
                                 diagnostic_result=completed["result"], result=None, request=None)
            return completed
        order = np.arange(1, len(candidates))
        if request.method == "random":
            np.random.default_rng(request.seed).shuffle(order)
        else:
            # Spread the deterministic search across background and normalization families.
            order = np.asarray(sorted(order, key=lambda i: ((i * 17) % len(candidates), i)))
        for index in order[:request.budget - 1]:
            session.evaluate(candidates[int(index)])
            if session.hard_stopped:break
    return session.finish(request.method, mapping_source, {"seed": request.seed}, request.experimental_search)
