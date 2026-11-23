"""Inspectable ASCII importer; never repairs or discards spectrum rows."""
import hashlib
import re

import numpy as np

from .models import Mapping


class InputError(ValueError):
    pass


def parse_ascii(raw_text: str, *, preview: bool = False):
    if len(raw_text.encode("utf-8")) > 2_000_000:
        raise InputError("Input exceeds the pilot limit of 2 MB.")
    rows, header = [], []
    width = None
    for number, line in enumerate(raw_text.splitlines(), 1):
        stripped = line.strip().lstrip("\ufeff")
        if not stripped:
            continue
        if stripped.startswith(("#", ";", "%")):
            header.append(line)
            continue
        # An inline # comment is retained through the original input, not parsed.
        body = stripped.split("#", 1)[0].strip()
        values = [v.strip() for v in body.split(",")] if "," in body else ([v.strip() for v in body.split(";")] if ";" in body else body.split())
        try:
            numeric = [float(v.replace("D", "E").replace("d", "e")) for v in values]
        except ValueError:
            if not rows and (re.fullmatch(r"[-=]{3,}", stripped) or not re.match(r"^[+\-.0-9]", stripped)):
                header.append(line)
                continue
            raise InputError(f"Non-numeric or missing value on line {number}; no rows were discarded.")
        if width is None:
            width = len(numeric)
        if len(numeric) != width:
            raise InputError(f"Line {number} has {len(numeric)} columns; expected {width}.")
        if width < 2 or width > 64:
            raise InputError("Use a numeric table with 2–64 columns.")
        rows.append(numeric)
        if len(rows) > 12000:
            raise InputError("The pilot accepts at most 12,000 data points per scan.")
    if not rows:
        raise InputError("No numeric rows are available for preview.")
    if len(rows) < 40 and not preview:
        raise InputError("At least 40 numeric rows are required for this XANES/EXAFS workflow.")
    table = np.asarray(rows, dtype=float)
    if not preview and not np.isfinite(table).all():
        raise InputError("NaN or infinite values found; correct the input explicitly before analysis.")
    return table, header


def construct_mu(raw_text: str, mapping: Mapping, energy_shift=0):
    table, header = parse_ascii(raw_text)
    from .import_formats import column_labels
    labels = column_labels(header, table.shape[1])
    if mapping.energy < len(labels) and labels[mapping.energy].lower() == 'k':
        raise InputError("This is chi(k), not mu(E). Use the chi Fourier-transform workflow.")
    required = [mapping.energy, mapping.signal]
    if mapping.mode != "mu":
        required.append(mapping.reference)
    if max(required) >= table.shape[1]:
        raise InputError(f"Selected column does not exist. This file has {table.shape[1]} columns.")
    if len(set(required)) != len(required):
        raise InputError("Energy and selected detector columns must be different.")
    # Recognize the common XDI Column.n: energy unit declaration, without claiming
    # full XDI conformance. A declared unit must agree with the explicit mapping.
    for line in header:
        unit = re.search(rf"Column\.{mapping.energy + 1}:\s*energy\s+(keV|eV)\b", line, re.I)
        if unit and unit.group(1).lower() != mapping.energy_unit.lower():
            raise InputError(f"Header declares energy in {unit.group(1)}; change the selected energy unit.")
    energy = table[:, mapping.energy] * (1000 if mapping.energy_unit == "keV" else 1) + energy_shift
    if np.any(energy <= 0) or np.any(energy > 200000):
        raise InputError("Energy must be between 0 and 200,000 eV after conversion and shift.")
    if np.any(np.diff(energy) <= 0):
        raise InputError("Energy must be strictly increasing; duplicates and reversed scans are not silently repaired.")
    signal = table[:, mapping.signal]
    if mapping.mode == "mu":
        mu, formula = signal.copy(), "mu = signal"
    else:
        reference = table[:, mapping.reference]
        if np.any(reference <= 0) or np.any(signal <= 0):
            raise InputError("I0 and detector intensity must be positive for ratio/log processing.")
        if mapping.mode == "transmission":
            mu = np.log(reference) - np.log(signal)
            formula = "mu = ln(I0 / It); signal=It, reference=I0"
        else:
            mu = signal / reference
            formula = "mu = If / I0; signal=If, reference=I0"
    if not np.isfinite(mu).all():
        raise InputError("The selected intensity conversion produced non-finite mu.")
    return energy, mu, {
        "sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "encoding": "UTF-8 (original text including line endings)",
        "rows": len(table), "columns": table.shape[1], "header": header,
        "formula": formula, "energy_unit_after_conversion": "eV",
        "column_indices": "zero-based in recipe, one-based in UI",
    }
