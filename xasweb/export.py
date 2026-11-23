"""Portable evidence bundles; replay treats contents as data, never executable code."""
import hashlib
import io
import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ingest import InputError
from .models import AnalysisRequest
from .science import ROOT, analyze, canonical_json


def json_bytes(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


def figure_bytes(result, fmt="svg"):
    a, p, m = result["arrays"], result["request"]["parameters"], result["metrics"]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.16, "lines.linewidth": 1.25}):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), layout="constrained")
        ax = axes[0, 0]
        ax.plot(a["energy"], a["mu"], color="#146f82", label="Measured μ(E)")
        ax.plot(a["energy"], a["bkg"], color="#c57335", label="AUTOBK μ₀(E)")
        for key, color in [("pre_edge", "#768394"), ("post_edge", "#a085bd")]:
            ax.plot(a["energy"], a[key], "--", color=color, label=key.replace("_", " "))
        for lo, hi, color in [(p["pre1"], p["pre2"], "#146f82"),
                              (p["norm1"], p["norm2"], "#c57335")]:
            ax.axvspan(m["e0_eV"] + lo, m["e0_eV"] + hi, color=color, alpha=0.08)
        ax.set(xlabel="Energy (eV)", ylabel="μ(E) (relative absorbance)", title="A · Absorption and background")
        ax.legend(fontsize=8)
        ax = axes[0, 1]
        ax.plot(a["energy"], a["flat"], color="#146f82")
        ax.axvline(m["e0_eV"], color="#768394", linestyle="--", linewidth=0.8)
        ax.set(xlabel="Energy (eV)", ylabel="Normalized μ(E)", title="B · Near-edge spectrum (flattened)",
               xlim=(m["e0_eV"] - 40, m["e0_eV"] + 100))
        ax = axes[1, 0]
        k, chi = np.array(a["k"]), np.array(a["chi"])
        ax.plot(k, chi * k ** p["kweight"], color="#146f82")
        ax.set(xlabel="k (Å⁻¹)", ylabel=f"k{['⁰', '¹', '²', '³'][p['kweight']]} χ(k) (Å⁻{p['kweight']})",
               title="C · EXAFS and transform window", xlim=(0, k[-1]))
        right = ax.twinx()
        right.plot(k, a["kwin"], "--", color="#c57335", linewidth=0.8)
        right.set(ylabel="Window", ylim=(0, 1.15))
        right.grid(False)
        ax = axes[1, 1]
        ax.plot(a["r"], a["chir_mag"], color="#146f82", label="Magnitude")
        ax.plot(a["r"], a["chir_im"], color="#c57335", label="Imaginary", linewidth=0.85)
        ax.set(xlabel="R (Å), phase uncorrected", ylabel=f"χ(R) (Å⁻{p['kweight'] + 1})",
               title="D · Fourier transform", xlim=(0, 6))
        ax.legend(fontsize=8)
        title = result["request"]["sample"] or result["request"]["filename"]
        fig.suptitle(title[:100], fontsize=13, fontweight="bold", parse_math=False)
        fig.supxlabel(f"E₀ = {m['e0_eV']:.3f} eV · Rbkg = {p['rbkg']:g} Å · "
                      f"FT {p['kmin']:g}–{p['kmax']:g} Å⁻¹, Hanning dk = {p['dk']:g} Å⁻¹ · "
                      f"input SHA-256 {result['source']['sha256'][:12]}", fontsize=8)
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=300)
        plt.close(fig)
        return buf.getvalue()


def csv_bytes(arrays, names):
    out = io.StringIO()
    np.savetxt(out, np.column_stack([arrays[n] for n in names]), delimiter=",",
               header=",".join(names), comments="", fmt="%.17g")
    return out.getvalue().encode()


def build_bundle(request: AnalysisRequest, result, automation=None):
    if result.get('schema_version') == 2:
        return build_staged_bundle(request, result, automation)
    a = result["arrays"]
    entries = {
        "input.txt": request.raw_text.encode("utf-8"),
        "recipe.json": json_bytes(result["request"]),
        "analysis.json": json_bytes(result),
        "energy.csv": csv_bytes(a, ["energy", "mu", "pre_edge", "post_edge", "norm", "flat", "bkg", "chie"]),
        "chi.csv": csv_bytes(a, ["k", "chi", "kwin"]),
        "chir.csv": csv_bytes(a, ["r", "chir_re", "chir_im", "chir_mag"]),
        "figure.svg": figure_bytes(result, "svg"),
        "figure.png": figure_bytes(result, "png"),
        "environment.txt": "\n".join(f"{k}=={v}" for k, v in result["software"]["packages"].items()).encode(),
        "README.txt": (
            "Web XAS pilot — reproducibility bundle, schema 1\n\n"
            "Replay from the matching project checkout:\n"
            "  python -m xasweb.replay path/to/analysis.zip\n\n"
            "input.txt retains the uploaded UTF-8 bytes; its original name is in recipe.json.\n"
            "recipe.json uses zero-based columns and offsets relative to the resolved E0.\n"
            "analysis.json includes all arrays, resolved calls, provenance and software hashes.\n"
            "energy.csv energy is eV; norm, flat and chi are dimensionless. norm retains the post-edge trend; flat removes the fitted post-edge trend for display.\n"
            "chi.csv k is Å^-1; kwin is dimensionless.\n"
            "chir.csv R is Å, phase uncorrected; chi(R) units are Å^-(kweight+1).\n"
            "Pre/post/background curves use the same absorbance units as mu.\n"
            "Larch bkg below E0 copies mu as a placeholder; interpret EXAFS only above E0.\n"
            "No structural fit, detector correction or experimental uncertainty propagation.\n"
            "Manifest hashes detect corruption, not authorship or authenticity.\n"
            "A passing replay is same-engine reproducibility, not independent scientific validation.\n"
            "source_snapshot/ retains the application source for manual audit and recovery.\n"
            "The replay command NEVER executes code from the archive; it uses your local checkout.\n"
        ).encode("utf-8"),
    }
    if automation is not None:
        if automation.get("selected_run_id") != result["run_id"]:
            raise InputError("Automation record belongs to another numerical run.")
        entries["automation.json"] = json_bytes(automation)
    for relative, expected in result["software"]["source_sha256"].items():
        blob = (ROOT / relative).read_bytes()
        if hashlib.sha256(blob).hexdigest() != expected:
            raise InputError("Source changed during export. Restart the server and rerun the analysis.")
        entries["source_snapshot/" + relative] = blob
    for relative in ("pyproject.toml", "requirements-lock.txt", "data/sources.json",
                     "data/LARCH-LICENSE.txt", "docs/SOURCES.md"):
        path = ROOT / relative
        if path.exists():
            entries["source_snapshot/" + relative] = path.read_bytes()
    entries["manifest.json"] = json_bytes({"schema_version": 1, "run_id": result["run_id"],
        "files": {name: hashlib.sha256(value).hexdigest() for name, value in entries.items()}})
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return output.getvalue()


def read_bundle(blob: bytes):
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            if len(z.infolist()) > 128 or sum(i.file_size for i in z.infolist()) > 100_000_000:
                raise InputError("Bundle exceeds the pilot size limit.")
            names = z.namelist()
            if len(set(names)) != len(names):
                raise InputError("Bundle contains duplicate filenames.")
            manifest = json.loads(z.read("manifest.json"))
            if manifest["schema_version"] not in (1, 2):
                raise InputError("Unsupported bundle schema.")
            if set(manifest["files"]) != set(names) - {"manifest.json"}:
                raise InputError("Bundle manifest does not cover every file.")
            for name, expected in manifest["files"].items():
                if hashlib.sha256(z.read(name)).hexdigest() != expected:
                    raise InputError(f"Bundle checksum mismatch: {name}.")
            recipe = json.loads(z.read("recipe.json"))
            request = AnalysisRequest(**recipe, raw_text=z.read("input.txt").decode("utf-8"))
            saved = json.loads(z.read("analysis.json"))
            if saved.get("schema_version") not in (1, 2):
                raise InputError("Unsupported analysis schema.")
            if saved["source"]["sha256"] != hashlib.sha256(z.read("input.txt")).hexdigest():
                raise InputError("Saved analysis does not match input checksum.")
            if canonical_json(saved["request"]) != canonical_json(recipe):
                raise InputError("Saved analysis and recipe disagree.")
            if saved["run_id"] != manifest["run_id"]:
                raise InputError("Saved analysis and manifest run IDs disagree.")
            return request, saved
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise InputError("Invalid or incomplete analysis bundle.") from exc


def replay_bundle(blob: bytes):
    request, saved = read_bundle(blob)
    from .assessment import analyze_assessed
    fresh = analyze_assessed(request) if saved.get('schema_version') == 2 or saved.get('assessment') else analyze(request)
    # Older unflattened bundles did not export the display-only flat array.
    # Continue checking every archived array; only this documented addition is allowed.
    legacy_unflattened = (
        saved.get("resolved_calls", {}).get("pre_edge", {}).get("make_flat") is False
        and set(fresh["arrays"]) - set(saved["arrays"]) == {"flat"}
        and not set(saved["arrays"]) - set(fresh["arrays"])
    )
    if set(saved["arrays"]) != set(fresh["arrays"]) and not legacy_unflattened:
        raise InputError("Saved analysis has missing or unexpected numerical arrays.")
    checks = {}
    for key, old in saved["arrays"].items():
        x, y = np.asarray(old), np.asarray(fresh["arrays"][key])
        same_shape = x.shape == y.shape
        checks[key] = {"passed": bool(same_shape and np.allclose(x, y, rtol=1e-7, atol=1e-9)),
                       "max_abs_difference": float(np.max(np.abs(x - y))) if same_shape else None}
    same_software = saved["software"] == fresh["software"]
    for key in ("e0_eV", "edge_step", "k_available", "k_grid_max", "r_grid_spacing"):
        x, y = saved["metrics"].get(key), fresh["metrics"].get(key)
        if x is None or y is None:
            checks[key] = {'passed':x is None and y is None, 'max_abs_difference':None}
            continue
        checks[key] = {"passed": bool(np.isclose(x, y, rtol=1e-7, atol=1e-9)),
                       "max_abs_difference": float(abs(x - y))}
    if saved.get('assessment'):
        for key in ('stage','completed_operations','adjustments','stop_reasons','provenance'):
            checks['assessment_'+key] = {'passed':saved['assessment'].get(key)==fresh.get('assessment',{}).get(key)}
    return request, fresh, {"passed": all(c["passed"] for c in checks.values()),
                            "same_software": same_software, "added_display_arrays": ["flat"] if legacy_unflattened else [], "rtol": 1e-7, "atol": 1e-9,
                            "checks": checks, "saved_run_id": saved["run_id"],
                            "replayed_run_id": fresh["run_id"]}


def build_staged_bundle(request, result, automation=None):
    """Version 2 archives every retained stage, including stopped/S0 assessments."""
    entries = {'input.txt':request.raw_text.encode('utf-8'),
               'recipe.json':json_bytes(result['request']), 'analysis.json':json_bytes(result),
               'assessment.json':json_bytes(result['assessment']),
               'provenance.json':json_bytes(result['provenance']),
               'arrays.json':json_bytes(result['arrays']),
               'environment.txt':'\n'.join(f'{k}=={v}' for k,v in result['software']['packages'].items()).encode(),
               'README.txt':('XASweb staged replay bundle, schema 2.\n'
                   'Completed operations and stopping reasons are authoritative in assessment.json.\n'
                   'Only completed arrays are exported. No absent stage is fabricated.\n'
                   'Original benchmark bytes remain in the hash-bound benchmark release; input.txt is its identified array representation.\n'
                   'Replay uses installed application code; archive code is never executed.\n'
                   'Computational replay does not certify sample phase or calibration.\n').encode()}
    groups = [('energy.csv', ['energy','mu','uncorrected_signal_ratio','norm','flat','pre_edge','post_edge','bkg']),
              ('chi.csv',['k','chi','kwin']), ('chir.csv',['r','chir_re','chir_im','chir_mag'])]
    for filename, fields in groups:
        present = [k for k in fields if k in result['arrays']]
        if len(present)>=2:
            entries[filename] = csv_bytes(result['arrays'], present)
    for fmt in ('svg','png'):
        entries['figure.'+fmt] = staged_figure(result, fmt)
    if automation is not None:
        if automation.get('selected_run_id') != result['run_id']:
            raise InputError('Automation record belongs to another run.')
        entries['automation.json'] = json_bytes(automation)
    for relative, expected in result['software']['source_sha256'].items():
        content = (ROOT/relative).read_bytes()
        if hashlib.sha256(content).hexdigest()!=expected:
            raise InputError('Application changed during export. Restart and rerun.')
        entries['source_snapshot/'+relative] = content
    entries['manifest.json'] = json_bytes({'schema_version':2,'run_id':result['run_id'],
        'files':{name:hashlib.sha256(value).hexdigest() for name,value in entries.items()}})
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():archive.writestr(name,value)
    return stream.getvalue()


def staged_figure(result, fmt):
    arrays = result['arrays']
    pairs = [('energy','mu','Absorption'),('energy','norm','Normalized absorption'),
             ('k','chi','EXAFS'),('r','chir_mag','Fourier magnitude')]
    if 'uncorrected_signal_ratio' in arrays:
        pairs[0] = ('energy','uncorrected_signal_ratio','Uncorrected diagnostic ratio')
    fig, axes = plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,(x,y,title) in zip(axes.flat,pairs):
        ax.set_title(title)
        if x in arrays and y in arrays:
            ax.plot(arrays[x],arrays[y]);ax.set_xlabel(x);ax.set_ylabel(y)
        else:
            ax.text(.5,.5,'Stage not completed',ha='center',va='center',transform=ax.transAxes)
    fig.suptitle('Completed stage: '+result['assessment']['stage'])
    fig.supxlabel('; '.join(s['code'] for s in result['assessment']['stop_reasons'])[:150] or 'Fixed-profile computational processing')
    stream=io.BytesIO();fig.savefig(stream,format=fmt,dpi=180);plt.close(fig)
    return stream.getvalue()
