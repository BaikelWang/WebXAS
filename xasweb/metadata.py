"""Validate declared scientific context without guessing specimen structure."""
import re
import numpy as np
from larch.xafs.pre_edge import _finde0
from xraydb import xray_edge, xray_edges
from .ingest import InputError, construct_mu
from .edges import normalize_edge


def confirmed_edge_position(energy, mu, expected):
    """Find a supported local absorption rise, not the largest whole-scan tail."""
    half_width = max(30., min(100., .005 * expected))
    local = (energy >= expected-half_width) & (energy <= expected+half_width)
    if local.sum() < 12:
        raise InputError('Insufficient measured points near the declared edge for automatic E0 detection.')
    local_energy = energy[local]
    # The public two-pass finder estimates a step with an empty 0:-0 slice
    # for short arrays in the pinned Larch version. Supply the measured step.
    measured_step = float(np.median(np.diff(local_energy)))
    smoothing_step = .5*(max(.01, min(1., measured_step)) + max(.01, min(1., expected/25000.)))
    observed = float(_finde0(local_energy, mu[local], estep=smoothing_step, use_smooth=True)[0])
    left = (energy >= observed-half_width) & (energy <= observed-.35*half_width)
    right = (energy >= observed+.35*half_width) & (energy <= observed+half_width)
    if left.sum() < 4 or right.sum() < 4:
        raise InputError('The declared edge lacks measured sidebands for validating an absorption rise.')
    # A smooth linear baseline or a narrow isolated spike is not an edge.
    pre = np.polyfit(energy[left]-observed, mu[left], 1)
    post = np.polyfit(energy[right]-observed, mu[right], 1)
    rise = float(np.median(mu[right]-np.polyval(pre, energy[right]-observed)))
    residual = np.concatenate((mu[left]-np.polyval(pre, energy[left]-observed),
                               mu[right]-np.polyval(post, energy[right]-observed)))
    noise = float(1.4826*np.median(np.abs(residual-np.median(residual))))
    scale = max(float(np.max(np.abs(mu[local]))), 1.)
    floor = max(128*np.finfo(float).eps*scale, float(np.ptp(mu[local]))*1e-8)
    if not np.isfinite(rise) or rise <= max(5*noise, floor):
        raise InputError('No resolved positive absorption rise near the declared edge; confirm signal columns, units and element/edge.')
    return observed, half_width, {'method':'confirmed_edge_local_derivative',
        'search_range_eV':[float(energy[local][0]),float(energy[local][-1])],
        'local_absorption_rise':rise,'sideband_residual_noise':noise,
        'validation':'positive baseline-corrected rise exceeds five robust sideband noise scales'}


def inspect_metadata(raw_text,mapping,element=None,edge=None):
    header={}
    conflicts=[]
    for line in raw_text.splitlines():
        if not line.lstrip().startswith('#'):continue
        match=re.match(r'\s*#\s*(Element\.(?:symbol|edge)|Scan\.(?:element|edge)):\s*(\S+)',line,re.I)
        if match:
            name=match[1].lower().replace('scan.element','element.symbol').replace('scan.edge','element.edge');value=match[2]
            if name=='element.edge':value=normalize_edge(value)
            if name in header and header[name].lower()!=value.lower():conflicts.append(name)
            header[name]=value
    if conflicts and not (element and edge):
        return {'status':'needs_review','reason':'Conflicting element/edge declarations; correct the file header before importing again.'}
    element=element or header.get('element.symbol');edge=normalize_edge(edge or header.get('element.edge'))
    info={'element':element,'edge':edge,
          'source':'explicit_or_header' if element or edge else 'not_supplied',
          'status':'incomplete' if not (element and edge) else 'declared',
          'limitations':['Element/edge metadata do not identify a compatible structural model.']}
    if not (element and edge):return info
    try:
        declared=xray_edge(element,edge)
        if declared is None:raise ValueError('Unknown declared element/edge.')
        energy,mu,_=construct_mu(raw_text,mapping)
        expected=float(declared.energy)
        if not energy[0]<expected<energy[-1]:
            return {**info,'status':'needs_review','reason':'Declared absorption edge is outside the scan. Confirm element, edge and energy units.','tabulated_edge_eV':expected}
        observed,tolerance,detection=confirmed_edge_position(energy,mu,expected)
        if abs(observed-expected)>tolerance:
            return {**info,'status':'needs_review','reason':'Local detected edge conflicts with the declared edge. Confirm element, edge and signal columns.',
                    'tabulated_edge_eV':expected,'detected_edge_eV':observed,'review_tolerance_eV':tolerance}
        higher=[(name,float(value.energy)) for name,value in xray_edges(element).items() if value.energy>expected+1]
        higher.sort(key=lambda item:item[1])
        next_edge=higher[0] if higher else None
        info.update(status='compatible_energy_range',tabulated_edge_eV=expected,detected_edge_eV=observed,
                    review_tolerance_eV=tolerance,usable_energy_max_eV=float(energy[-1]),
                    edge_detection=detection,next_edge=None,flags=[])
        if next_edge and next_edge[1]<=energy[-1]+30:
            info.update(next_edge={'name':next_edge[0],'energy_eV':next_edge[1]},
                        usable_energy_max_eV=min(float(energy[-1]),next_edge[1]-30))
            info['flags'].append('following_edge_limits_valid_processing_range')
        return info
    except (ValueError,KeyError,TypeError) as exc:
        return {**info,'status':'needs_review','reason':str(exc)}
