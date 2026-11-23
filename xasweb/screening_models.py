"""Finite, explicitly hypothetical FEFF environments for first-shell screening.

The QFS-style ligand reference is a notional rocksalt environment, not an
isolated octahedron.  Only its nearest-neighbour path is a fitting standard.
Surrounding atoms determine FEFF potentials; their higher-shell paths must not
be used as extra evidence about the specimen.  Mixed/split first-shell models
are separate, disclosed environment assumptions, not expanded QFS second shells.
"""
import copy
import hashlib
import itertools
import math
import re

from .edges import normalize_edge
from .ingest import InputError


MODES = [
    {'id': 'light', 'name': '轻元素配位'},
    {'id': 'heavy', 'name': '硫 / 卤素配位'},
    {'id': 'metal', 'name': '金属近邻'},
    {'id': 'combined', 'name': '组合模型'},
]

QFS_REFERENCE = 'https://bruceravel.github.io/demeter/artug/extended/qfs.html'
REFERENCE_POLICY = 'qfs-first-shell-reference-1'
DELTA_R_LIMIT_A = 0.20
CLUSTER_RADIUS_A = 5.9
_TEMPLATE = {'C': 'carbon', 'N': 'nitrogen', 'O': 'oxygen',
             'S': 'sulfur', 'Cl': 'halogen', 'F': 'halogen', 'Br': 'halogen'}


def mode_for(template):
    if template in ('oxygen', 'nitrogen', 'carbon'):
        return 'light'
    if template in ('sulfur', 'halogen'):
        return 'heavy'
    return 'metal' if template == 'metal' else 'combined'


def _distance_prior(nominal):
    """Bounded screening prior, not an assertion of the specimen bond length."""
    margin = min(0.45, 0.20 * nominal)
    return (round(max(0.8, nominal - margin), 6),
            round(min(5.25, nominal + margin), 6))


def _search(nominal, bounds, quick):
    lower, upper = bounds
    if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
        raise InputError('首壳参考距离区间无效。')
    nominal = min(upper, max(lower, nominal))
    # These three overlapping local fit neighbourhoods cover the finite prior.
    grid = sorted({round(min(upper, max(lower, x)), 6)
                   for x in (lower + DELTA_R_LIMIT_A, nominal,
                             upper - DELTA_R_LIMIT_A)})
    initial = nominal
    peak = quick.get('peak_R')
    if isinstance(peak, (int, float)) and math.isfinite(peak) and peak > 0:
        # Never present this empirical phase offset as a measured bond length.
        initial = min(grid, key=lambda x: (abs(x - (peak + 0.4)), abs(x - nominal)))
    return dict(nominal_distances=grid, nominal_distance_A=round(initial, 6),
                minimum_A=lower, maximum_A=upper, delta_r_limit_A=DELTA_R_LIMIT_A,
                max_rereferences=1,
                initial_distance_assumption='未相位校正 FT 第一峰 + 0.4 Å 仅用于选择有限网格初值；不是键长或相位校正结果。',
                range_assumption='元素半径给出的中心距离两侧最多 0.45 Å、且最多 20% 的有限初筛先验；不是所有化学环境的保证。')


def _rocksalt_atoms(element, scatterer, distance, geometry, offsets):
    """A bounded simple-cubic grid with alternating rocksalt sublattices."""
    reach = math.ceil(CLUSTER_RADIUS_A / distance)
    atoms = []
    for xyz in itertools.product(range(-reach, reach + 1), repeat=3):
        norm2 = sum(v * v for v in xyz)
        if not norm2:
            continue
        odd = sum(xyz) % 2
        species = scatterer if odd else element
        scale = distance
        if geometry == 'mixed_rocksalt_first_shell' and odd:
            # An ordered substitution hypothesis in ONE FEFF potential context.
            species = element if xyz[2] % 2 else scatterer
            scale += offsets[1 if species == element else 0]
        elif geometry == 'split_rocksalt_first_shell' and norm2 == 1:
            # The entire reference remains available to establish the potentials.
            scale += offsets[0 if xyz[0] else 1]
        if math.sqrt(norm2) * scale <= CLUSTER_RADIUS_A + 1e-9:
            atoms.append((species, *(float(v * scale) for v in xyz)))
    return sorted(atoms, key=lambda atom: (sum(v * v for v in atom[1:]), atom))


def _fcc_atoms(element, distance):
    spacing = distance / math.sqrt(2)
    reach = math.ceil(CLUSTER_RADIUS_A / spacing)
    atoms = []
    for xyz in itertools.product(range(-reach, reach + 1), repeat=3):
        norm2 = sum(v * v for v in xyz)
        if norm2 and sum(xyz) % 2 == 0 and math.sqrt(norm2) * spacing <= CLUSTER_RADIUS_A + 1e-9:
            atoms.append((element, *(float(v * spacing) for v in xyz)))
    return sorted(atoms, key=lambda atom: (sum(v * v for v in atom[1:]), atom))


def _candidate(spec, search):
    from xraydb import atomic_number

    element, scatterer, edge = spec['element'], spec['scatterer'], spec['edge']
    distance = search['nominal_distance_A']
    geometry, offsets = spec['geometry'], spec['offsets_A']
    if geometry == 'fcc_metal_first_shell':
        atoms = _fcc_atoms(element, distance)
    else:
        atoms = _rocksalt_atoms(element, scatterer, distance, geometry, offsets)
    if not atoms or len(atoms) + 1 > 600:
        raise InputError('首壳 FEFF 参考环境原子数不满足限制。')

    # The absorber is a distinct potential from surrounding atoms of its element.
    species = list(dict.fromkeys(atom[0] for atom in atoms))
    potentials = {symbol: index + 1 for index, symbol in enumerate(species)}
    hole = {'K': 1, 'L1': 2, 'L2': 3, 'L3': 4}[edge]
    text = (f'TITLE Hypothetical first-shell reference {element} {edge}; {geometry}\n'
            f'HOLE {hole} 1.0\nPOTENTIALS\n0 {atomic_number(element)}\n')
    text += '\n'.join(f'{index} {atomic_number(symbol)}' for symbol, index in potentials.items())
    text += '\nATOMS\n0 0 0 0\n'
    text += '\n'.join(f'{x:.8f} {y:.8f} {z:.8f} {potentials[symbol]}'
                      for symbol, x, y, z in atoms) + '\nEND\n'

    shell_paths = []
    for definition in spec['shell_definitions']:
        shell_paths.append(dict(scatterer=definition['scatterer'],
                                nominal_distance_A=round(distance + definition['offset_A'], 6),
                                tolerance_A=0.035,
                                minimum_A=definition['minimum_A'], maximum_A=definition['maximum_A']))
    if geometry == 'rocksalt_first_shell':
        description = (f'假想岩盐局域环境，{element}–{scatterer} 最近邻 {distance:g} Å；'
                       '六个第一近邻及周围两类原子共同建立势，只使用最近邻单散射路径。')
        environment_assumption = '岩盐环境仅为 QFS 首壳散射势近似，不代表样品为岩盐结构；高壳路径不作为拟合标准。'
    elif geometry == 'fcc_metal_first_shell':
        description = f'有限 fcc 金属环境，十二个最近邻 {distance:g} Å；仅使用第一近邻单散射路径。'
        environment_assumption = 'fcc 是金属近邻的参考势环境假设，不是物相或晶体对称性判断。'
    elif geometry == 'mixed_rocksalt_first_shell':
        description = (f'同一岩盐母簇的最近邻位点用四个 {scatterer} 和两个 {element} 替代；'
                       f'参考距离 {shell_paths[0]["nominal_distance_A"]:g} / {shell_paths[1]["nominal_distance_A"]:g} Å。')
        environment_assumption = '配体与金属同属第一配位峰的混合近邻假设；不使用外层金属路径，也不表示已知配体几何或金属团簇。'
    else:
        description = (f'同一岩盐母簇中两个 {scatterer} 短近邻、四个长近邻；'
                       f'参考距离 {shell_paths[0]["nominal_distance_A"]:g} / {shell_paths[1]["nominal_distance_A"]:g} Å。')
        environment_assumption = '分裂同一第一配位壳的参考假设，共享 FEFF 势；需满足径向分辨率和独立信息量才可拟合。'
    source = dict(type='hypothetical_reference', policy=REFERENCE_POLICY,
                  absorber=element, edge=edge, geometry=description, geometry_type=geometry,
                  fit_scope='first_coordination_peak', qfs_first_shell_only=True,
                  first_shell_paths=shell_paths, reference_search=copy.deepcopy(search),
                  generator=copy.deepcopy(spec), qfs_reference=QFS_REFERENCE,
                  input_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                  atom_count=len(atoms) + 1,
                  radius_source='pymatgen: Cordero covalent radii; atomic radius for metallic reference',
                  assumptions=[environment_assumption,
                               '实际 FEFF 计算使用此吸收元素、吸收边、散射元素和完整参考坐标；不同元素或参考距离不共用重标记路径。',
                               '参考簇是初筛假设，不代表样品结构、价态或配位数；球形势、环境及有限半径搜索仍有系统误差。',
                               'N 由实际路径简并度与自由振幅得到；参考簇中固定的原子数量不是样品配位数。'])
    if len(shell_paths) > 1:
        source['assumptions'].append('组合仅解释同一第一配位峰；不扩大 R 拟合窗来容纳更高壳，也不将低残差视为唯一结构。')
    return dict(name=spec['name'], role='feff_input', text=text,
                allowed_templates=spec['allowed_templates'], source=source)


def reference_candidates(element, edge, context):
    """Return nine base environments (plus explicitly named F/Br if supplied).

    ``context['_quick_first_shell']['peak_R']`` is an internal, optional FT
    locator used only to select a reference-distance starting point.  This
    function does not retrieve structures, identify known specimens, or reuse a
    spectrum, a fit, a saved amplitude, or a precomputed FEFF path.
    """
    from pymatgen.core import Element
    from pymatgen.core.molecule_structure_comparator import CovalentRadius
    from xraydb import atomic_number

    edge = normalize_edge(edge)
    if edge not in ('K', 'L1', 'L2', 'L3'):
        raise InputError('参考簇路径生成目前支持 K、L1、L2、L3 边；此边需要匹配的用户路径。')
    try:
        absorber = Element(element)
        radius = CovalentRadius.radius.get(element)
        if radius is None or not absorber.is_metal or atomic_number(element) > 98:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise InputError('通用参考簇目前面向金属吸收中心；此元素需要用户提供有依据的局域模型。') from None
    context = context or {}
    quick = context.get('_quick_first_shell') or {}
    metal_distance = 2 * float(absorber.atomic_radius or radius)
    metal_bounds = _distance_prior(metal_distance)
    candidates = []

    def add(name, scatterer, geometry, center, center_bounds, definitions, allowed):
        spec = dict(element=element, edge=edge, scatterer=scatterer, geometry=geometry,
                    name=name, shell_definitions=definitions, allowed_templates=allowed,
                    offsets_A=[row['offset_A'] for row in definitions])
        candidates.append(_candidate(spec, _search(center, center_bounds, quick)))

    def shell(scatterer, bounds, offset=0.0):
        return dict(scatterer=scatterer, offset_A=offset,
                    minimum_A=bounds[0], maximum_A=bounds[1])

    ligands = ['C', 'N', 'O', 'S', 'Cl']
    chemistry = ' '.join(str(context.get(key) or '') for key in ('formula', 'sample'))
    # Literal formula tokens avoid treating "Fe" as fluorine or "Brown" as Br.
    for symbol in ('F', 'Br'):
        if re.search(symbol + r'(?![a-z])', chemistry):
            ligands.append(symbol)
    for scatterer in ligands:
        distance = round(radius + CovalentRadius.radius[scatterer], 6)
        bounds = _distance_prior(distance)
        add(f'{element}–{scatterer} 首壳参考环境', scatterer, 'rocksalt_first_shell',
            distance, bounds, [shell(scatterer, bounds)], [_TEMPLATE[scatterer]])
    add(f'{element}–{element} 金属首壳参考环境', element, 'fcc_metal_first_shell',
        metal_distance, metal_bounds, [shell(element, metal_bounds)], ['metal'])

    for scatterer in ('O', 'N'):
        ligand_distance = round(radius + CovalentRadius.radius[scatterer], 6)
        ligand_bounds = _distance_prior(ligand_distance)
        overlap_lower = max(ligand_bounds[0], metal_bounds[0])
        overlap_upper = min(ligand_bounds[1], metal_bounds[1])
        if overlap_lower <= overlap_upper:
            center = (overlap_lower + overlap_upper) / 2
            ligand_ref = metal_ref = center
            center_bounds = (overlap_lower, overlap_upper)
        else:
            # Keep the nearest allowed distances in the same first-shell band.
            # These remain hypotheses; the fitter must check the measured peak.
            ligand_ref, metal_ref = ligand_bounds[1], metal_bounds[0]
            center = (ligand_ref + metal_ref) / 2
            center_bounds = (center, center)
        offsets = (round(ligand_ref - center, 6), round(metal_ref - center, 6))
        add(f'{element}–{scatterer} / {element} 混合首壳参考环境', scatterer,
            'mixed_rocksalt_first_shell', center, center_bounds,
            [shell(scatterer, ligand_bounds, offsets[0]), shell(element, metal_bounds, offsets[1])],
            ['ligand_metal'])
        # A large gap is retained as an explicit unavailable model, never widened
        # into a second-shell fit by an automatically expanded fitting window.
        candidates[-1]['source']['requires_same_first_peak'] = True
        candidates[-1]['source']['maximum_first_shell_span_A'] = 0.35
        candidates[-1]['source']['first_shell_span_A'] = round(abs(metal_ref - ligand_ref), 6)

    distance = round(radius + CovalentRadius.radius['O'], 6)
    bounds = _distance_prior(distance)
    half_split = min(0.15, (bounds[1] - bounds[0]) / 4)
    add(f'{element}–O 分裂首壳参考环境', 'O', 'split_rocksalt_first_shell',
        distance, (round(bounds[0] + half_split, 6), round(bounds[1] - half_split, 6)),
        [shell('O', bounds, -half_split), shell('O', bounds, half_split)], ['split_shell'])
    candidates[-1]['source']['requires_same_first_peak'] = True
    candidates[-1]['source']['maximum_first_shell_span_A'] = 0.35
    candidates[-1]['source']['first_shell_span_A'] = round(2 * half_split, 6)
    return candidates


def rereference_candidate(candidate, distance):
    """Rebuild a real FEFF input at a finite reference distance, at most once.

    This does not stretch/relabel saved scattering functions.  The caller must
    execute FEFF and validate the newly generated paths before fitting them.
    """
    source = candidate.get('source') or {}
    if source.get('policy') != REFERENCE_POLICY or not source.get('generator'):
        raise InputError('此路径环境不支持自动首壳重参考。')
    search = copy.deepcopy(source['reference_search'])
    try:
        distance = float(distance)
    except (TypeError, ValueError):
        raise InputError('首壳重参考距离必须是有限数值。') from None
    if not math.isfinite(distance) or not search['minimum_A'] - 1e-8 <= distance <= search['maximum_A'] + 1e-8:
        raise InputError('首壳重参考距离超出声明的有限物理搜索区间。')
    if search.get('rereference_count', 0) >= search['max_rereferences']:
        raise InputError('已达到每个参考环境的重计算次数上限。')
    search['nominal_distance_A'] = round(min(search['maximum_A'], max(search['minimum_A'], distance)), 6)
    search['rereference_count'] = search.get('rereference_count', 0) + 1
    search['previous_nominal_distance_A'] = source['reference_search']['nominal_distance_A']
    result = _candidate(source['generator'], search)
    for key in ('requires_same_first_peak', 'maximum_first_shell_span_A', 'first_shell_span_A'):
        if key in source:
            result['source'][key] = source[key]
    result['source']['previous_input_sha256'] = source['input_sha256']
    return result
