"""Bounded descriptive recipes; no unvalidated structure template registry."""
from copy import deepcopy


_HARD_XAS = {
    'id': 'hard-xas-descriptive', 'version': 1,
    'validation_level': 'engineering_preview',
    'scope': 'Conventional hard-X-ray K/L-edge mu(E), or explicitly declared chi(k).',
    'unsupported': ['soft-X-ray quantitative automation', 'automatic unique structure discovery',
                    'unverified fluorescence/self-absorption correction'],
    'parameters': {'rbkg': 1.0, 'kweight': 2, 'dk': 1.0, 'kmin': 2.0, 'maximum_k': 12.0},
    'budgets': {'preprocessing_candidates': 3, 'initial_sets': 4,
                'max_references': 8, 'max_lcf_alternatives': 9},
    'selection_rule': 'Coverage and signed edge-step checks, deterministic first valid candidate; never final fit R-factor.',
    'scientific_validation': 'Requires independent real standard and expert acceptance; synthetic regression is not that acceptance.',
}


def descriptive_recipe():
    return deepcopy(_HARD_XAS)
