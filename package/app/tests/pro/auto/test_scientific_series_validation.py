import numpy as np
import pytest

from xasweb.models import Parameters
from xasweb.pro.contracts import Dataset, ProcessingRevision
from xasweb.pro.decomposition import decompose_series
from xasweb.pro.figures.insights import figure_insight
from xasweb.pro.figures.sources import collect_figures
from xasweb.pro.preprocessing import save_array
from xasweb.pro.projects import ProjectError, ProjectStore
from xasweb.pro.xanes import (
    extract_features,
    preedge_integral,
    fit_reference_combination,
)


def make_project(tmp_path, energy, rows, anchor):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Analytic test")
    for i, y in enumerate(rows):
        label = f"s{i}"
        raw = np.column_stack((energy, y)).tobytes()
        d = Dataset(
            id=label,
            filename=label + ".dat",
            element="Fe",
            edge="K",
            raw_file=store.put_asset(project.id, raw, label + ".dat"),
        )
        project.datasets.append(d)
        project.processing.append(
            ProcessingRevision(
                id="pp_" + label,
                dataset_id=label,
                parameters=Parameters(e0=anchor),
                arrays={
                    "energy": save_array(store, project.id, energy),
                    "norm": save_array(store, project.id, y),
                },
            )
        )
    return store, store.save(project, project.revision)


def test_edge_annotation_uses_declared_window_or_reports_missing_anchor(tmp_path):
    energy = np.arange(6450.221, 8260, 1.0)
    y = np.tanh((energy - 7124.221) / 2)
    y[(energy >= 6461) & (energy <= 6463)] += 15
    y[(energy >= 8256) & (energy <= 8258)] += 18
    store, p = make_project(tmp_path, energy, [y], 7119.221)
    pp = p.processing[0]
    wf = {"id": "edges", "processing_ids": [pp.id], "fit_run_ids": []}
    for feats in ([], [extract_features(store, p.id, pp)]):
        wf["features"] = feats
        fig = next(
            x for x in collect_figures(store, p, wf)[0] if x.spec.family == "F03"
        )
        assert fig.spec.method["edge_position_eV"] == pytest.approx(7124.221)
        labels = [
            m for m in figure_insight(p, wf, fig)["metrics"] if "边位" in m["label"]
        ]
        assert all(m["value"] == pytest.approx(7124.221) for m in labels) and labels
    pp.parameters.e0 = None
    wf["features"] = []
    fig = next(x for x in collect_figures(store, p, wf)[0] if x.spec.family == "F03")
    assert fig.spec.method["edge_position_eV"] is None
    assert not [
        m for m in figure_insight(p, wf, fig)["metrics"] if "边位" in m["label"]
    ]


def test_affine_pca_and_nonnegative_mcr_on_known_mixture(tmp_path):
    e = np.linspace(7100, 7160, 121)
    a = 0.1 + np.exp(-(((e - 7118) / 4) ** 2))
    b = 0.1 + np.exp(-(((e - 7134) / 4) ** 2))
    store, p = make_project(tmp_path, e, [a, 0.35 * a + 0.65 * b, b], 7119.0)
    result = decompose_series(store, p, p.processing, n_components=2, n_starts=3)
    matrix = np.stack([a, 0.35 * a + 0.65 * b, b])
    model = np.array(result["mcr"]["reconstruction"])
    np.testing.assert_allclose(result["pca"]["reconstruction"], matrix, atol=1e-12)
    assert result["pca"]["explained_variance_ratio"][0] == pytest.approx(1)
    np.testing.assert_allclose(
        np.sum(result["mcr"]["concentrations"], axis=1), 1, atol=1e-10
    )
    assert (
        np.min(result["mcr"]["spectra"]) >= 0
        and np.min(result["mcr"]["concentrations"]) >= 0
    )
    assert np.linalg.norm(matrix - model) / np.linalg.norm(matrix) < 1e-5
    assert len(result["mcr"]["starts"]) == 3
    p.datasets[0].edge = "L3"
    with pytest.raises(ProjectError, match="confirmed element"):
        decompose_series(store, p, p.processing, n_components=2)


def test_preedge_area_centroid_and_background_uncertainty(tmp_path):
    e = np.linspace(7090, 7140, 1001)
    y = 0.12 + 0.002 * (e - 7112) + 0.15 * np.exp(-0.5 * ((e - 7112) / 0.8) ** 2)
    store, p = make_project(tmp_path, e, [y], 7124)
    q = preedge_integral(
        store,
        p.id,
        p.processing[0],
        left_eV=[7094, 7100],
        peak_eV=[7106, 7118],
        right_eV=[7123, 7129],
    )
    assert q["area_eV"] == pytest.approx(0.15 * 0.8 * np.sqrt(2 * np.pi), abs=1e-9)
    assert q["centroid_eV"] == pytest.approx(7112, abs=1e-8)
    assert q["baseline_only_area_se_eV"] < 1e-9
    with pytest.raises(ProjectError, match="ordered"):
        preedge_integral(
            store,
            p.id,
            p.processing[0],
            left_eV=[7100, 7101],
            peak_eV=[7090, 7108],
            right_eV=[7115, 7119],
        )


def test_lcf_uses_one_explicit_ordinate_for_all_references(tmp_path):
    energy = np.linspace(7100, 7140, 101)
    a = 0.4 + np.sin((energy - 7100) / 17)
    b = 0.3 + np.exp(-(((energy - 7125) / 8) ** 2))
    store, p = make_project(tmp_path, energy, [0.3 * a + 0.7 * b, a, b], 7119)
    for pp, flat in zip(p.processing, [0.75 * a + 0.25 * b, a, b]):
        pp.arrays["flat"] = save_array(store, p.id, flat)
    cond = {
        "reference_applicability_confirmed": True,
        "reference_applicability_note": "Controlled known-series input.",
    }
    for key, expect in [("norm", 0.3), ("flat", 0.75)]:
        f = fit_reference_combination(
            store,
            p,
            p.processing[0],
            p.processing[1:],
            {"window_eV": [7100, 7140], "normalization_ordinate": key},
            cond,
        )
        assert f["normalization_ordinate"] == key
        assert f["coefficients"][0] == pytest.approx(expect, abs=1e-6)
    with pytest.raises(ProjectError, match="flattened"):
        fit_reference_combination(
            store,
            p,
            p.processing[0],
            p.processing[1:],
            {"normalization_ordinate": "typo"},
            cond,
        )
