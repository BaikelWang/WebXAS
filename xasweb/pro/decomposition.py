"""PCA and constrained MCR-ALS for comparable measured XANES series."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .projects import ProjectError
from .xanes import _spectrum


def _simplex(rows):
    """Euclidean projection of each row onto the unit simplex."""
    ordered = np.sort(rows, axis=1)[:, ::-1]
    excess = np.cumsum(ordered, axis=1) - 1
    width = np.arange(1, ordered.shape[1] + 1)
    active = (ordered > excess / width).sum(axis=1)
    cutoff = excess[np.arange(len(rows)), active - 1] / active
    return np.maximum(rows - cutoff[:, None], 0)


def _projected_least_squares(values, gram, cross, project):
    # Solves independent constrained row problems by accelerated projected gradient.
    lipschitz = np.linalg.norm(gram, 2)
    if lipschitz <= np.finfo(float).eps:
        return values
    current = previous = values
    for step in range(1, 61):
        proposal = current + (step - 1) / (step + 2) * (current - previous)
        updated = project(proposal - (proposal @ gram - cross) / lipschitz)
        delta = np.linalg.norm(updated - current)
        previous, current = current, updated
        if delta <= 1e-10 * (1 + np.linalg.norm(current)):
            break
    return current


def _mcr_trial(matrix, initial, max_iter):
    profiles = initial.copy()
    coefficients = np.full((len(matrix), len(initial)), 1 / len(initial))
    previous_loss = np.inf
    history = []
    for iteration in range(1, max_iter + 1):
        coefficients = _projected_least_squares(
            coefficients, profiles @ profiles.T, matrix @ profiles.T, _simplex
        )
        spectra_t = _projected_least_squares(
            profiles.T,
            coefficients.T @ coefficients,
            matrix.T @ coefficients,
            lambda a: np.maximum(a, 0),
        )
        profiles = spectra_t.T
        residual = matrix - coefficients @ profiles
        loss = float(np.linalg.norm(residual))
        history.append(loss)
        if np.isfinite(previous_loss) and abs(previous_loss - loss) <= 1e-8 * (
            1 + previous_loss
        ):
            break
        previous_loss = loss
    return {
        "relative_error": loss / np.linalg.norm(matrix),
        "iterations": iteration,
        "converged": len(history) > 1
        and abs(history[-2] - history[-1]) <= 1e-8 * (1 + history[-2]),
        "objective_history": history,
        "C": coefficients,
        "S": profiles,
    }


def decompose_series(
    store,
    project,
    processing,
    *,
    n_components=2,
    window_eV=None,
    n_starts=5,
    max_iter=200,
):
    """Decompose only within measured support, with no sample labels used in numerical fitting."""
    count = len(processing)
    if not 3 <= count <= 128 or not 1 <= n_components < count or n_components > 8:
        raise ProjectError(
            "decomposition_dimensions",
            "Use 3–128 spectra and 1–8 components, fewer than the number of spectra.",
        )
    if not 2 <= n_starts <= 12 or not 2 <= max_iter <= 1000:
        raise ProjectError(
            "decomposition_iterations",
            "MCR requires 2–12 starts and 2–1000 iterations.",
        )
    datasets = {d.id: d for d in project.datasets}
    ids = [p.dataset_id for p in processing]
    if len(set(ids)) != count or not set(ids) <= datasets.keys():
        raise ProjectError(
            "decomposition_samples",
            "Each selected processing record must belong to a different project sample.",
        )
    identities = {(datasets[i].element, datasets[i].edge) for i in ids}
    if len(identities) != 1 or not all(next(iter(identities))):
        raise ProjectError(
            "decomposition_identity",
            "Series decomposition requires one confirmed element and absorption edge.",
        )
    if any(p.metadata.get("external_plot_only") for p in processing):
        raise ProjectError(
            "decomposition_normalization",
            "Select recorded normalization revisions for every spectrum.",
        )
    axes, signals = zip(*[_spectrum(store, project.id, p) for p in processing])
    if not all(
        np.isfinite(e).all() and np.isfinite(y).all() for e, y in zip(axes, signals)
    ):
        raise ProjectError(
            "decomposition_finite",
            "Spectral matrices require finite energies and normalized absorption.",
        )
    lo, hi = max(x[0] for x in axes), min(x[-1] for x in axes)
    if window_eV is not None:
        if (
            len(window_eV) != 2
            or not np.isfinite(window_eV).all()
            or not lo <= window_eV[0] < window_eV[1] <= hi
        ):
            raise ProjectError(
                "decomposition_window",
                "The ordered energy window must lie within all measured spectra.",
            )
        lo, hi = window_eV
    energy = axes[0][(axes[0] >= lo) & (axes[0] <= hi)]
    if not max(8, n_components + 2) <= len(energy) <= 6000:
        raise ProjectError(
            "decomposition_grid",
            "Select a shared window with 8–6000 measured energy points.",
        )
    for axis in axes:
        left = max(0, int(np.searchsorted(axis, energy[0], side="right")) - 1)
        right = min(len(axis), int(np.searchsorted(axis, energy[-1])) + 1)
        spacing = np.diff(axis[left:right])
        if not len(spacing) or np.max(spacing) > 5 * np.median(spacing):
            raise ProjectError(
                "decomposition_gap",
                "A common-energy interval contains an internal measurement gap. Select a continuous interval.",
            )
    matrix = np.vstack(
        [np.interp(energy, axis, signal) for axis, signal in zip(axes, signals)]
    )
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    left, singular, right = np.linalg.svd(centered, full_matrices=False)
    if singular[0] <= np.finfo(float).eps * (1 + np.linalg.norm(matrix)):
        raise ProjectError(
            "decomposition_constant",
            "No between-sample variation is measurable in this window.",
        )
    if np.linalg.matrix_rank(matrix) < n_components:
        raise ProjectError(
            "decomposition_rank",
            "Requested MCR components exceed the numerical rank of the input matrix.",
        )
    loadings = right[:n_components].copy()
    signs = np.sign(loadings[np.arange(n_components), np.argmax(abs(loadings), axis=1)])
    loadings *= signs[:, None]
    scores = left[:, :n_components] * singular[:n_components] * signs
    pca_model = mean + scores @ loadings
    # Independent unknown endmembers. Initial spectra depend on measured values, not labels.
    first = int(np.argmax(np.linalg.norm(centered, axis=1)))
    selected = [first]
    while len(selected) < n_components:
        nearest = np.min(
            [np.linalg.norm(matrix - matrix[i], axis=1) for i in selected], axis=0
        )
        nearest[selected] = -1
        selected.append(int(np.argmax(nearest)))
    rng = np.random.default_rng(271828)
    initial = [
        matrix[selected],
        *[
            rng.dirichlet(np.ones(count), n_components) @ matrix
            for _ in range(n_starts - 1)
        ],
    ]
    trials = [_mcr_trial(matrix, np.maximum(s, 0), max_iter) for s in initial]
    best = min(trials, key=lambda trial: trial["relative_error"])
    summaries, aligned = [], []
    for index, trial in enumerate(trials):
        costs = np.linalg.norm(best["S"][:, None, :] - trial["S"][None, :, :], axis=2)
        _, permutation = linear_sum_assignment(costs)
        aligned.append(trial["C"][:, permutation])
        summaries.append(
            {
                "start": index,
                "initialization": "farthest measured spectra"
                if index == 0
                else "random convex mixture of measured spectra",
                "relative_error": float(trial["relative_error"]),
                "converged": trial["converged"],
                "iterations": trial["iterations"],
                "objective_history": trial["objective_history"],
            }
        )
    concentrations, profiles = best["C"], best["S"]
    model = concentrations @ profiles
    spread = np.ptp(np.stack(aligned), axis=0)
    normalization_fields = ("pre1", "pre2", "norm1", "norm2", "nnorm")
    settings = [
        {k: getattr(pp.parameters, k) for k in normalization_fields}
        for pp in processing
    ]
    same_settings = all(row == settings[0] for row in settings)
    return {
        "dataset_ids": ids,
        "normalization": {
            "ordinate": "norm",
            "settings_in_sample_order": settings,
            "shared_windows_and_order": same_settings,
            "warning": None
            if same_settings
            else "Input normalization windows or polynomial orders differ. Review their effect before physical interpretation.",
        },
        "processing_ids": [p.id for p in processing],
        "energy_eV": energy.tolist(),
        "window_eV": [float(lo), float(hi)],
        "n_components": n_components,
        "input_matrix": matrix.tolist(),
        "gridding": "first measured sample grid on shared support, linear interpolation within continuous measured intervals; no extrapolation",
        "pca": {
            "scores": scores.tolist(),
            "loadings": loadings.tolist(),
            "mean": mean.tolist(),
            "singular_values": singular.tolist(),
            "explained_variance_ratio": (singular**2 / (singular @ singular)).tolist(),
            "reconstruction": pca_model.tolist(),
            "residual": (matrix - pca_model).tolist(),
            "preprocessing": "energy-column mean centered, no variance scaling; loading sign fixed by largest absolute coefficient",
        },
        "mcr": {
            "concentrations": concentrations.tolist(),
            "spectra": profiles.tolist(),
            "reconstruction": model.tolist(),
            "residual": (matrix - model).tolist(),
            "relative_error": float(best["relative_error"]),
            "converged": best["converged"],
            "constraints": [
                "nonnegative concentration profiles with row sums of one",
                "nonnegative spectra",
            ],
            "algorithm": "alternating projected least squares with deterministic multistart",
            "seed": 271828,
            "chosen_start": next(i for i, t in enumerate(trials) if t is best),
            "starts": summaries,
            "stability": {
                "aligned_concentration_range": spread.tolist(),
                "max_concentration_range": float(spread.max()),
            },
        },
        "limitations": [
            "Statistical dimensions do not identify chemical species. The centered affine rank can be one for two endmembers.",
            "Nonnegative MCR solutions retain rotational ambiguity; profile names and absolute abundances require external standards.",
            "Residuals are data minus reconstruction, in normalized absorption units. Multistart ranges are not confidence intervals.",
        ],
    }
