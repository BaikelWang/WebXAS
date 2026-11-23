"""Fill missing absorber metadata from an already parsed energy spectrum."""
import math

from .edges import normalize_edge


def infer_missing_context(context, energy=None, mu=None):
    """Keep declarations; record and apply the library guess without a score gate."""
    missing = [key for key in ('element', 'edge') if not context.get(key)]
    if not missing:
        return
    record = dict(method='larch.xafs.find_e0 -> xraydb.guess_edge',
                  status='failed', missing_fields=missing, filled_fields=[],
                  energy_unit='eV', confidence_gate=False)
    context['edge_inference'] = record
    try:
        if any(context.get('provenance', {}).get(key, {}).get('type') == 'manual' for key in missing):
            raise ValueError('Invalid manual value must be corrected manually; inference cannot replace it.')
        if energy is None or mu is None:
            raise ValueError('No parsed energy-domain mu(E) is available for E0 detection.')
        from larch.xafs import find_e0
        from xraydb import guess_edge
        e0 = float(find_e0(energy, mu))
        if not math.isfinite(e0):
            raise ValueError('find_e0 returned a non-finite energy.')
        record['detected_e0_eV'] = e0
        element, edge = guess_edge(e0)
        guessed = dict(element=element, edge=normalize_edge(edge))
        record['guess'] = guessed
        for key in missing:
            if guessed[key]:
                context[key] = guessed[key]
                context.setdefault('provenance', {})[key] = dict(
                    type='automatic', method=record['method'], detected_e0_eV=e0)
                record['filled_fields'].append(key)
        record['status'] = 'applied' if all(context.get(key) for key in missing) else 'failed'
        if record['status'] == 'failed':
            record['reason'] = 'guess_edge did not return all missing fields.'
    except Exception as exc:
        # Failed inference is a recoverable metadata step, not a lost spectrum.
        record['reason'] = str(exc)
