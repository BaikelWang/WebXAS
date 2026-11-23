"""Canonical absorption-edge labels at input and FEFF boundaries."""


def normalize_edge(value):
    """Unify L-shell notation without interpreting FEFF HOLE numbers as edges."""
    if value is None:
        return None
    label = str(value).strip().upper().translate(str.maketrans('₁₂₃', '123'))
    return {'LI':'L1', 'LII':'L2', 'LIII':'L3'}.get(label, label)
