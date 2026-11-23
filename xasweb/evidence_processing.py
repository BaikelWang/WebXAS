"""Deterministic staged processing of evidence-assessed spectra.

This production adapter consumes source facts and numeric inputs, never benchmark
expectations or reference output arrays. Each successful stage survives a later
stop. The application must hold its numerical lock while calling these functions.
"""
from __future__ import annotations

import importlib.metadata
import math
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
from larch import Group
from larch.xafs import autobk, find_e0, pre_edge, xftf
from larch.xafs.xafsutils import TINY_ENERGY
from xraydb import xray_edges

RULE_VERSION = "reference-processing-1.2.0"
PINNED_LARCH_VERSION = "2026.3.1"
AUTOBK_DEFAULTS = dict(rbkg=1.0, kmin=0.0, kweight=2, dk=1.0,
                       win="hanning", nfft=2048, kstep=0.05, nclamp=3,
                       clamp_lo=0, clamp_hi=1, calc_uncertainties=False)
FFT_DEFAULTS = dict(kmin=2.0, dk=1.0, dk2=1.0, window="hanning", kweight=2,
                    kstep=0.05, nfft=2048, rmax_out=10.0)


class ProcessingStop(Exception):
    """An expected evidence or coverage boundary; earlier stages remain usable."""


def _require(condition, reason):
    if not condition:
        raise ProcessingStop(reason)


@dataclass
class _Run:
    record: dict
    arrays: dict = field(default_factory=dict)
    group: Group = field(default_factory=Group)
    energy: np.ndarray | None = None
    mu: np.ndarray | None = None
    e0: float | None = None
    measured_k: float | None = None
    ftmax: float | None = None


def _start(object_id, initial_stage):
    return _Run(dict(object_id=object_id, rule_version=RULE_VERSION,
                     initial_stage=initial_stage, achieved_stage=initial_stage,
                     completed_operations=[], stop_reasons=[], parameters={},
                     checks={}, adjustments=[], status="stopped"))


def _runtime_guard():
    _require(importlib.metadata.version("xraylarch") == PINNED_LARCH_VERSION,
             "pinned_larch_version_mismatch")


def _finish(run, callback):
    try:
        _runtime_guard()
        callback()
    except ProcessingStop as exc:
        run.record["stop_reasons"].append(str(exc))
        if run.record["achieved_stage"] == "S3":
            run.record["status"] = "partial_computational_support"
    except Exception as exc:
        run.record["stop_reasons"].append("processing_execution_error:" + type(exc).__name__ + ":" + str(exc))
        run.record["status"] = "execution_failed"
    return {"record": run.record, "arrays": run.arrays}


def _input_array(data):
    data = np.asarray(data, dtype=float)
    _require(data.ndim == 2 and data.shape[1] >= 2, "invalid_numeric_table")
    _require(len(data) >= 20, "too_few_samples")
    _require(np.isfinite(data).all(), "nonfinite_input")
    _require(np.all(np.diff(data[:, 0]) > 0), "nonmonotonic_grid_requires_review")
    _require(np.all(np.diff(data[:, 0]) >= TINY_ENERGY), "energy_steps_below_larch_resolution")
    return data


def _signal_stage(run, obj, data):
    _require(obj["kind"] in {"athena_object", "artemis_snapshot", "standalone_spectrum"}, "not_experimental_mu_object")
    _require(obj["example"] != "Fe foil", "intentional_miscalibration_requires_reference_rule")
    _require(not obj.get("conflicts") and not obj.get("metadata_conflicts"), "source_metadata_conflict")
    _require(not obj.get("correction_history"), "self_absorption_already_applied")
    _require(obj["processing_state"] in {"mu", "raw_channels"}, "processing_state_not_unprocessed_mu")
    _require(obj["axis_unit"] == "eV", "axis_unit_unresolved")
    cr = obj["example"] == "Cr2O3" and obj["kind"] == "standalone_spectrum"
    _require(cr or obj.get("formula_status") != "proposed_interpretation", "signal_formula_unverified")
    data = _input_array(data)
    e = data[:, 0].copy()
    if cr:
        _diagnostic_ratio_stage(run, obj, data)
    formula = obj["signal_formula"]
    if obj["processing_state"] == "mu" and formula in {"stored_y", "stored_mu"}:
        mu = data[:, 1].copy()
    else:
        _require(bool(obj["quality"].get("raw_project_match")), "raw_formula_not_independently_matched")
        if formula == "ln(column_2 / column_3)":
            _require(data.shape[1] >= 3, "invalid_numeric_table")
            _require(np.all(data[:, 1:3] > 0), "nonpositive_transmission_channel")
            mu = np.log(data[:, 1]) - np.log(data[:, 2])
        elif formula == "column_9 / column_4":
            _require(data.shape[1] >= 9, "invalid_numeric_table")
            _require(np.all(data[:, 3] > 0), "nonpositive_fluorescence_denominator")
            mu = data[:, 8] / data[:, 3]
        else:
            raise ProcessingStop("unsupported_signal_formula")
    _require(np.isfinite(mu).all(), "nonfinite_signal")
    run.energy, run.mu = e, mu
    run.arrays.update(energy=e.tolist(), mu=mu.tolist())
    run.record["achieved_stage"] = "S2"
    run.record["completed_operations"].append("construct_or_retain_mu")


def _diagnostic_ratio_stage(run, obj, data):
    header = "\n".join(h["text"] for h in obj["headers"])
    _require(data.shape[1] == 17 and re.search(r"Energy\s+I0\s+It\s+Ifch1\s+Ifch2\s+Ifch3\s+Ifch4", header) is not None, "cr_channel_evidence_mismatch")
    _require(any(ev["path"] == "Cr2O3/megamerge.pl" for ev in obj["source_evidence"]), "cr_commented_formula_evidence_missing")
    invalid = np.where(data[:, 1] <= 0)[0]
    keep = np.ones(len(data), dtype=bool)
    if len(invalid):
        final_zero = len(invalid) == 1 and int(invalid[0]) == len(data)-1 and data[-1, 1] == 0 and np.all(data[-1, 3:7] == 0)
        _require(final_zero, "cr_invalid_denominator_requires_review")
        keep[-1] = False
        run.record["adjustments"].append(dict(type="diagnostic_ratio_mask", excluded_original_indices=[len(data)-1], reason="final_I0_and_four_fluorescence_counts_zero", scope="derived_ratio_only", rule_version=RULE_VERSION))
    ratio = np.sum(data[keep, 3:7], axis=1) / data[keep, 1]
    _require(np.isfinite(ratio).all(), "nonfinite_uncorrected_ratio")
    run.arrays.update(energy=data[keep, 0].tolist(), uncorrected_signal_ratio=ratio.tolist(), original_row_indices=np.where(keep)[0].tolist())
    run.record["parameters"]["diagnostic_ratio"] = dict(signal_formula="sum(column_4:column_7)/column_2", column_index_base=1, correction_applied=False, interpretation="uncorrected diagnostic fluorescence ratio, not corrected mu(E)")
    run.record["checks"]["ratio_finite"] = True
    run.record["achieved_stage"] = "S2"
    run.record["completed_operations"].append("construct_uncorrected_signal_ratio")
    raise ProcessingStop("multichannel_correction_unverified")


def _source_edge_stage(run, obj, objects):
    e, mu = run.energy, run.mu
    seed, identity = None, None
    candidates = [obj] + [objects[i] for i in obj["quality"].get("raw_project_match", {}).get("object_ids", []) if i in objects]
    for candidate in candidates:
        if identity is None:
            symbol = str(candidate["metadata"].get("bkg_z", "")).strip()
            edge = str(candidate["metadata"].get("fft_edge", "")).strip().upper()
            try:
                edges = xray_edges(symbol)
                if edge in edges:
                    identity = symbol, edge, edges, candidate["id"]
            except (ValueError, TypeError):
                pass
        try:
            val = float(candidate["metadata"]["bkg_e0"]) - float(candidate["metadata"].get("bkg_eshift", 0))
            if seed is None and e[0] < val < e[-1]:
                seed = val
                run.record["parameters"]["e0_seed_evidence_object"] = candidate["id"]
        except (KeyError, ValueError, TypeError):
            pass
    _require(identity is not None, "declared_edge_identity_unresolved")
    symbol, edge, edges, identity_source = identity
    tabulated = float(edges[edge].energy)
    run.record["parameters"]["edge_identity"] = dict(declared_element=symbol, declared_edge=edge, evidence_object=identity_source, tabulated_energy_eV=tabulated, database="xraydb", database_version=importlib.metadata.version("xraydb"), claim="source declaration checked for numerical consistency; no phase or calibration proof")
    _require(seed is None or abs(seed-tabulated) <= 40.0, "saved_e0_conflicts_with_declared_edge")
    mask = np.ones(len(e), dtype=bool) if seed is None else abs(e-seed) <= 50
    _require(np.count_nonzero(mask) >= 8, "edge_search_insufficient_points")
    run.e0 = float(find_e0(e[mask], mu[mask]))
    run.record["parameters"].update(e0=run.e0, e0_seed=seed, energy_shift_applied=0.0)
    _require(abs(run.e0-tabulated) <= 40.0, "detected_e0_conflicts_with_declared_edge")
    higher = sorted((float(v.energy), k) for k, v in edges.items() if float(v.energy) > tabulated)
    if higher and e[-1] >= higher[0][0]-30.0:
        next_energy, next_name = higher[0]
        cap = next_energy - 30.0
        keep = e < cap
        run.record["adjustments"].append(dict(type="subsequent_edge_range_restriction", next_edge=next_name, next_edge_energy_eV=next_energy, maximum_processing_energy_exclusive_eV=cap, excluded_original_indices=np.where(~keep)[0].tolist(), rule_version=RULE_VERSION, evidence_object=identity_source, database="xraydb"))
        run.arrays.update(original_energy=e.tolist(), original_mu=mu.tolist())
        run.energy, run.mu = e[keep], mu[keep]
        run.arrays.update(energy=run.energy.tolist(), mu=run.mu.tolist())
        _require(len(run.energy) >= 20, "subsequent_edge_leaves_insufficient_samples")


def _normalization_stage(run, *, fes2=False, manual=None):
    e, mu, e0, g = run.energy, run.mu, run.e0, run.group
    available = float(e[-1]-e0)
    pre1 = max(-150.0, float(e[0]-e0)+1e-7)
    if fes2:
        pre1 = max(pre1, float(e[5]-e0)+1e-7)
        run.record["adjustments"].append(dict(type="pre_fit_mask", excluded_original_indices=[0, 1, 2, 3, 4], scope="pre_fit_only", evidence="FeS2/README.org anomalous initial pre-edge points", rule_version=RULE_VERSION))
    pre2 = -30.0
    norm2 = min(800.0, available-1e-7)
    norm1 = min(150.0, norm2/3.0)
    nnorm = 2 if norm2-norm1 > 300 else 1
    if manual is not None:
        pre1, pre2 = float(manual.get("pre1", pre1)), float(manual.get("pre2", pre2))
        norm1, norm2 = float(manual.get("norm1", norm1)), float(manual.get("norm2", norm2))
        nnorm = int(manual.get("nnorm", nnorm))
        _require(pre1 >= e[0]-e0 and norm2 <= available, "manual_fit_window_exceeds_coverage")
        _require(pre1 < pre2 < 0 < norm1 < norm2 and 0 <= nnorm <= 3, "invalid_manual_fit_parameters")
    pre = (e >= e0+pre1) & (e <= e0+pre2)
    post = (e >= e0+norm1) & (e <= e0+norm2)
    kws = dict(pre1=pre1, pre2=pre2, norm1=norm1, norm2=norm2, npre=1, nvict=0, nnorm=nnorm, make_flat=True)
    run.record["parameters"]["normalization"] = kws
    run.record["checks"].update(post_edge_coverage_eV=available, pre_fit_points=int(pre.sum()), post_fit_points=int(post.sum()))
    _require(pre2-pre1 >= 30 and pre.sum() >= 5, "insufficient_pre_edge_fit_support")
    _require(available >= 60 and norm2-norm1 >= 30 and post.sum() >= 8, "insufficient_normalization_fit_support")
    precoef = np.polyfit(e[pre]-e0, mu[pre], 1)
    postcoef = np.polyfit(e[post]-e0, mu[post], nnorm)
    signed = float(postcoef[-1]-precoef[-1])
    noise = float(np.std(mu[pre]-np.polyval(precoef, e[pre]-e0)))
    second = post & (e >= e0+norm1+0.05*(norm2-norm1))
    _require(second.sum() >= nnorm+3, "insufficient_post_window_stability_support")
    signed2 = float(np.polyfit(e[second]-e0, mu[second], nnorm)[-1]-precoef[-1])
    sensitivity = abs(signed2-signed)/max(abs(signed), 1e-300)
    run.record["checks"].update(independent_signed_edge_jump=signed, pre_fit_residual_std=noise, post_window_sensitivity_relative=sensitivity)
    _require(signed > max(1e-12, 10*noise), "nonpositive_or_unresolved_edge_jump")
    _require(sensitivity <= 0.20, "normalization_window_unstable")
    with warnings.catch_warnings(record=True) as caught:
        pre_edge(e, mu, group=g, e0=e0, **kws)
    run.record["checks"]["normalization_warnings"] = [str(w.message) for w in caught]
    _require(all(np.isfinite(getattr(g, k)).all() for k in ["norm", "flat", "pre_edge", "post_edge"]), "nonfinite_normalization")
    diff = abs(float(g.edge_step)-signed)/signed
    run.record["checks"].update(larch_edge_step=float(g.edge_step), independent_jump_relative_difference=diff)
    _require(float(g.edge_step) > 0 and diff <= 0.05, "independent_normalization_jump_disagreement")
    run.arrays.update({k: np.asarray(getattr(g, k)).tolist() for k in ["norm", "flat", "pre_edge", "post_edge"]})
    run.record["achieved_stage"] = "S3"
    run.record["completed_operations"].append("normalize")


def _exafs_stage(run, manual=None):
    available = float(run.energy[-1]-run.e0)
    _require(available >= 180, "insufficient_post_edge_for_exafs")
    run.measured_k = math.sqrt(available/3.80998211615486)
    run.ftmax = math.floor(min(12.0, run.measured_k-0.5-0.05)/0.05)*0.05
    _require(run.ftmax >= 5.0, "insufficient_full_taper_k_range")
    kws = dict(AUTOBK_DEFAULTS)
    kws["kmax"] = run.measured_k
    if manual is not None:
        kws.update(rbkg=float(manual.get("rbkg", 1.0)), kweight=int(manual.get("bkg_kweight", 1)), dk=float(manual.get("dk", 1.0)))
        if manual.get("bkg_kmax") is not None:
            kws["kmax"] = float(manual["bkg_kmax"])
        run.ftmax = float(manual.get("kmax", run.ftmax))
        _require(0.3 <= kws["rbkg"] <= 2.5 and 0 <= kws["kweight"] <= 3 and 0.1 <= kws["dk"] <= 5.0 and 0 < kws["kmax"] <= run.measured_k, "invalid_manual_background_parameters")
        _require(run.ftmax+kws["dk"]/2 <= min(run.measured_k, kws["kmax"]), "manual_ft_window_exceeds_coverage")
    run.record["parameters"]["autobk"] = kws
    with warnings.catch_warnings(record=True) as caught:
        autobk(run.energy, run.mu, group=run.group, ek0=run.e0, edge_step=float(run.group.edge_step), **kws)
    run.record["checks"]["autobk_warnings"] = [str(w.message) for w in caught]
    g = run.group
    _require(np.isfinite(g.chi).all() and np.isfinite(g.bkg).all(), "nonfinite_background_or_chi")
    _require(len(g.k) > 20 and np.allclose(np.diff(g.k), 0.05, rtol=0, atol=1e-10), "unexpected_chi_sampling")
    _require(float(g.k[-1]) >= run.ftmax+kws["dk"]/2, "full_taper_exceeds_chi_coverage")
    run.arrays.update({k: np.asarray(getattr(g, k)).tolist() for k in ["k", "chi", "bkg"]})
    run.record["completed_operations"].append("extract_exafs")


def _fourier_stage(run, manual=None):
    kws = dict(FFT_DEFAULTS)
    kws["kmax"] = run.ftmax
    if manual is not None:
        kws.update(kmin=float(manual.get("kmin", 2.0)), kweight=int(manual.get("kweight", 2)), dk=float(manual.get("dk", 1.0)), dk2=float(manual.get("dk", 1.0)))
        _require(0 <= kws["kweight"] <= 3 and kws["kmin"] >= kws["dk"]/2 and kws["kmax"]-kws["kmin"] > kws["dk"], "invalid_manual_ft_parameters")
    run.record["parameters"]["fft"] = kws
    g = run.group
    xftf(g.k, g.chi, group=g, **kws)
    _require(all(np.isfinite(getattr(g, k)).all() for k in ["r", "chir_re", "chir_im", "chir_mag", "kwin"]), "nonfinite_fourier_transform")
    nz = np.where(np.asarray(g.kwin) > 1e-12)[0]
    _require(len(nz) > 0 and float(g.k[nz[-1]]) <= run.measured_k, "window_uses_unmeasured_k")
    weighted = np.asarray(g.chi)*np.asarray(g.k)**kws["kweight"]*np.asarray(g.kwin)
    independent = np.fft.fft(weighted, n=2048)*(0.05/math.sqrt(math.pi))
    actual = np.asarray(g.chir_re) + 1j*np.asarray(g.chir_im)
    err = float(np.max(np.abs(independent[:len(actual)]-actual)))
    _require(np.allclose(independent[:len(actual)], actual, rtol=1e-10, atol=1e-12), "independent_fft_disagreement")
    run.record["checks"].update(full_taper_coverage_pass=True, numpy_fft_max_abs_error=err, measured_k_max=run.measured_k)
    run.arrays.update({k: np.asarray(getattr(g, k)).tolist() for k in ["r", "chir_re", "chir_im", "chir_mag", "kwin"]})
    run.record["completed_operations"].append("fourier_transform")
    run.record["achieved_stage"] = "S4"
    run.record["status"] = "computationally_supported"


def process_object(obj: dict, objects: dict, data) -> dict:
    """Recompute an assessed object's exact accepted profile from supplied arrays.

    Caller verifies object/array provenance. ``objects`` contains related source
    facts for raw/project semantic cross-checks; computational_support is ignored.
    Returns JSON-compatible ``record`` and ``arrays`` without writing files.
    """
    run = _start(obj["id"], obj["confirmed_stage"])

    def execute():
        _signal_stage(run, obj, data)
        _source_edge_stage(run, obj, objects)
        _normalization_stage(run, fes2=obj["example"] == "FeS2")
        _exafs_stage(run)
        _fourier_stage(run)

    return _finish(run, execute)


def process_declared_mu(energy, mu, parameters=None) -> dict:
    """Staged manual processing of user-declared eV/mu arrays.

    Accepts the existing Parameters model or a dict. Explicit e0, energy shift,
    pre/norm/background/FT windows are honored; no element/edge is required or
    inferred. Unlike a known source this carries user-declared signal semantics.
    Earlier arrays remain present if a later requested window is unsupported.
    """
    settings = parameters.model_dump() if hasattr(parameters, "model_dump") else dict(parameters or {})
    run = _start("user_declared_mu", "S1")

    def execute():
        original = _input_array(np.column_stack([energy, mu]))
        shift = float(settings.get("energy_shift", 0.0))
        _require(np.isfinite(shift) and -1000 <= shift <= 1000, "invalid_manual_energy_shift")
        run.energy, run.mu = original[:, 0]+shift, original[:, 1].copy()
        run.arrays.update(energy=run.energy.tolist(), mu=run.mu.tolist())
        run.record["achieved_stage"] = "S2"
        run.record["completed_operations"].append("construct_or_retain_mu")
        if shift:
            run.arrays["original_energy"] = original[:, 0].tolist()
            run.record["adjustments"].append(dict(type="user_declared_energy_shift", energy_shift_eV=shift))
        supplied_e0 = settings.get("e0")
        run.e0 = float(find_e0(run.energy, run.mu)) if supplied_e0 is None else float(supplied_e0)
        _require(np.isfinite(run.e0) and run.energy[0] < run.e0 < run.energy[-1], "manual_e0_outside_coverage")
        run.record["parameters"].update(e0=run.e0, energy_shift_applied=shift, declaration="user_declared_mu_eV")
        _normalization_stage(run, manual=settings)
        _exafs_stage(run, manual=settings)
        _fourier_stage(run, manual=settings)

    return _finish(run, execute)
