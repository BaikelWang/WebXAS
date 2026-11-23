"""Thin Larch adapter. All scientific processing parameters are recorded."""
import hashlib
import importlib.metadata
import json
import os
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import larch
from larch import Group
from larch.math import index_nearest, index_of
from larch.xafs import autobk, find_e0, pre_edge, xftf
from larch.xafs.xafsutils import ETOK, TINY_ENERGY

from . import __version__
from .ingest import InputError, construct_mu
from .models import AnalysisRequest

ROOT = Path(__file__).resolve().parents[1]
KSTEP, NFFT = 0.05, 2048


def source_hashes():
    paths = sorted((ROOT / "xasweb").rglob("*.py")) + sorted((ROOT / "xasweb/static").rglob("*"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


STARTUP_SOURCES = source_hashes()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def software_identity():
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    sources = source_hashes()
    if sources != STARTUP_SOURCES:
        raise InputError("Application source changed since this process started. Restart the server before analysis.")
    # Larch's decorators live in larchlib.py: hashing inspect.getfile(wrapper)
    # alone would miss the actual scientific implementation.
    library_root = Path(larch.__file__).parent
    library_sources = {name: hashlib.sha256((library_root / name).read_bytes()).hexdigest()
                       for name in ("xafs/pre_edge.py", "xafs/autobk.py", "xafs/xafsft.py",
                                    "xafs/xafsutils.py", "math/utils.py", "larchlib.py")}
    return {"application": __version__, "python": platform.python_version(),
            "platform": platform.platform(), "packages": dict(sorted(packages.items())),
            "numpy_build": np.show_config(mode="dicts"),
            "thread_settings": {key: os.environ.get(key) for key in
                                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "source_sha256": sources, "larch_source_sha256": library_sources}


def analyze(request: AnalysisRequest):
    p = request.parameters
    energy, mu, source = construct_mu(request.raw_text, request.mapping, p.energy_shift)
    if np.any(np.diff(energy) < TINY_ENERGY):
        raise InputError(f"Energy steps smaller than {TINY_ENERGY:g} eV would be adjusted by Larch. "
                         "Resolve the sampling explicitly before analysis.")
    notices = [
        "R is phase uncorrected. Fourier peak positions are not bond lengths.",
        "No detector, self-absorption or reference calibration correction is applied.",
        "No experimental uncertainty propagation; parameter sensitivity is not a confidence interval.",
        "AUTOBK endpoint clamps in this Larch version depend on absolute mu scale. Amplitude-dependent background sensitivity is not included in reported fit uncertainties.",
    ]
    e0 = float(find_e0(energy, mu)) if p.e0 is None else p.e0
    if not energy[0] < e0 < energy[-1]:
        raise InputError("E0 must lie strictly inside the measured energy range.")
    counts = {}
    for name, a, b, minimum in [("pre-edge", p.pre1, p.pre2, 5),
                                ("normalization", p.norm1, p.norm2, max(5, p.nnorm + 2))]:
        lo, hi = e0 + a, e0 + b
        if lo < energy[0] or hi > energy[-1]:
            raise InputError(f"The {name} window [{lo:.2f}, {hi:.2f}] eV exceeds the measured "
                             f"range [{energy[0]:.2f}, {energy[-1]:.2f}] eV. Adjust its offsets or E0.")
        # Match the library's half-open fit slice, including its endpoint snapping.
        counts[name] = index_nearest(energy, hi) - index_of(energy, lo)
        if counts[name] < minimum:
            raise InputError(f"The {name} window needs at least {minimum} measured points.")
    near_edge = energy[(energy >= e0 - 15) & (energy <= e0 + 30)]
    if len(near_edge) < 8 or np.max(np.diff(near_edge), initial=0) > 5:
        notices.append("Sparse near-edge sampling: inspect E0 and normalization carefully.")
    k_available = float(np.sqrt((energy[-1] - e0) * ETOK))
    k_grid_max = np.floor(k_available / KSTEP) * KSTEP
    if p.kmax + p.dk / 2 > k_grid_max + 1e-10:
        raise InputError(f"The complete FT taper ends at {p.kmax + p.dk / 2:.2f} Å⁻¹, "
                         f"but data support only {k_grid_max:.2f} Å⁻¹. Lower kmax or dk.")
    bkg_kmax = min(k_available, p.bkg_kmax) if p.bkg_kmax is not None else k_available
    if p.bkg_kmax is not None and p.bkg_kmax < k_available:
        notices.append("Background processing is capped before a following edge or declared cutoff; chi beyond this range is extrapolated and requires exclusion.")
    calls = {
        "pre_edge": {"e0": e0, "pre1": p.pre1, "pre2": p.pre2, "norm1": p.norm1,
                     "norm2": p.norm2, "nnorm": p.nnorm, "nvict": 0, "npre": 1,
                     "step": None, "iscalc": False, "make_flat": True},
        "autobk": {"e0": e0, "rbkg": p.rbkg, "kmin": 0, "kmax": bkg_kmax,
                   "kweight": p.bkg_kweight, "dk": 0.1, "win": "hanning",
                   "kstep": KSTEP, "nfft": NFFT, "nknots": None,
                   "nclamp": 2, "clamp_lo": 1, "clamp_hi": 1,
                   "calc_uncertainties": False, "k_std": None, "chi_std": None,
                   "pre_edge_kws": None, "err_sigma": 1},
        "xftf": {"kmin": p.kmin, "kmax": p.kmax, "dk": p.dk, "dk2": p.dk,
                 "kweight": p.kweight, "window": "hanning", "kstep": KSTEP,
                 "nfft": NFFT, "rmax_out": 6, "with_phase": False},
    }
    g = Group(energy=energy, mu=mu)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pre_edge(energy, mu, group=g, **calls["pre_edge"])
        # Larch takes abs(edge_step), so check the signed fitted jump independently.
        ie0 = int(np.argmin(np.abs(energy - e0)))
        signed_step = float(g.post_edge[ie0] - g.pre_edge[ie0])
        if not np.isfinite(g.edge_step) or signed_step <= max(1e-12, np.ptp(mu) * 1e-8):
            raise InputError("The fitted edge step is not positive and well resolved. Check columns, E0 and windows.")
        if not np.isclose(g.e0, e0, atol=1e-8, rtol=0):
            raise InputError("Larch changed the requested E0; choose a supported value explicitly.")
        calls["autobk"]["edge_step"] = float(g.edge_step)
        autobk(energy, mu, group=g, **calls["autobk"])
        xftf(g.k, g.chi, group=g, **calls["xftf"])
        notices += list(dict.fromkeys(str(w.message) for w in caught))
    names = ("energy", "mu", "pre_edge", "post_edge", "norm", "flat", "bkg", "chie", "k", "chi",
             "kwin", "r", "chir_re", "chir_im", "chir_mag")
    arrays = {}
    for name in names:
        value = np.asarray(getattr(g, name))
        if not np.isfinite(value).all():
            raise InputError(f"Larch returned non-finite {name}; no analysis was exported.")
        arrays[name] = value.tolist()
    if len(g.k) < 2 or p.kmax + p.dk / 2 > g.k[-1] + 1e-10:
        raise InputError("The actual AUTOBK output does not support the complete FT taper.")
    details = getattr(g, "autobk_details", Group())
    background_details = {}
    for key in ("nknots", "kmin", "kmax", "irbkg", "nspl", "iek0", "iemax", "ek0"):
        value = getattr(details, key, None)
        if value is not None:
            background_details[key] = float(value)
    normalization_details = {}
    for key, value in vars(g.pre_edge_details).items():
        if isinstance(value, (int, float, np.number)):
            normalization_details[key] = float(value)
    # Only identify vendored data by exact bytes, never by user-supplied filenames.
    manifest_path = ROOT / "data/sources.json"
    if manifest_path.exists():
        source["public_reference"] = next((s for s in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
                                           if s["sha256"] == source["sha256"]), None)
    request_recipe = request.model_dump(exclude={"raw_text", "assessment_token"})
    identity = software_identity()
    run_id = hashlib.sha256(canonical_json({"source": source["sha256"], "recipe": request_recipe,
                                           "software": identity})).hexdigest()
    return {"schema_version": 1, "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
            "request": request_recipe, "source": source, "software": identity,
            "resolved_calls": calls, "normalization_details": normalization_details,
            "background_details": background_details,
            "sampling_policy": {"kstep": KSTEP, "nfft": NFFT, "energy_to_k_factor": float(ETOK),
                                "normalization_endpoints": "sample at/below lower bound; nearest upper sample excluded",
                                "edge_step_evaluation": "nearest measured energy to E0",
                                "ft_window_endpoints": "Larch snaps the Hanning endpoints to the uniform k grid",
                                "chi_interpolation": "Larch AUTOBK spline evaluation to uniform k grid",
                                "background_domain": "bkg below E0 is Larch's placeholder equal to mu; EXAFS interpretation is post-edge only",
                                "ft_convention": "(dk/sqrt(pi)) * sum(chi * k^weight * window * exp(-2 i k R))"},
            "arrays": arrays, "warnings": notices,
            "metrics": {"e0_eV": e0, "e0_method": "Larch derivative maximum" if p.e0 is None else "user supplied",
                        "edge_step": float(g.edge_step), "edge_step_evaluation_energy_eV": float(energy[ie0]),
                        "k_available": k_available,
                        "k_grid_max": float(g.k[-1]), "energy_min": float(energy[0]),
                        "energy_max": float(energy[-1]), "fit_window_points": counts,
                        "r_grid_spacing": float(g.r[1] - g.r[0]),
                        "nominal_delta_r": float(np.pi / (2 * (p.kmax - p.kmin)))},
            "array_units": {"energy": "eV", "mu": "relative absorbance", "norm": "dimensionless",
                            "k": "angstrom^-1", "chi": "dimensionless", "r": "angstrom (phase uncorrected)",
                            "chir": f"angstrom^-{p.kweight + 1}"}}
