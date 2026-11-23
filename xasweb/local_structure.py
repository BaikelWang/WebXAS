"""Upload-driven first-shell model discovery, FEFF calculation and checked fitting.

No specimen catalog, saved fit, or example identity is consulted.
"""
from pathlib import Path
import hashlib
import json
import math
import platform
import re
import subprocess
import tempfile

import numpy as np
from .ingest import InputError
from .edges import normalize_edge

POLICY = 'quick-first-shell-3.0'


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def classify(name, text):
    """Inspect content before suffix; unknown text stays supporting evidence."""
    suffix = Path(name).suffix.lower()
    numeric_pair=r'(?m)^\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?[,;\t ]+\s*[-+]?(?:\d|\.)'
    if re.search(r'real\[p\]@#', text) and 'nleg' in text.lower():
        return 'feff_path'
    if '_atom_site_' in text and '_cell_length_' in text:
        return 'cif'
    if re.match(r'^\s*\d+\s*\n', text) and re.search(r'(?m)^\s*[A-Z][a-z]?\s+[-\d.]', text):
        return 'xyz'
    if re.search(r'(?im)^\s*POTENTIALS\s*$', text) and re.search(r'(?im)^\s*ATOMS\s*$', text):
        return 'feff_input'
    if suffix in ('.cif', '.xyz', '.pdb', '.inp', '.json', '.md'):
        return {'.cif':'cif','.xyz':'xyz','.pdb':'pdb','.inp':'structure_notes'}.get(suffix, 'sample_info')
    if re.search(r'(?im)^\s*#?\s*(?:Sample\.|Element\.|formula\s*:|mineral\s*:|s02\s*:)', text) and not re.search(numeric_pair, text):
        return 'sample_info'
    if text.strip() and not re.search(numeric_pair, text) and suffix in ('.txt','.text',''):
        return 'sample_info'
    return None


def sample_context(raw, supplied, supports):
    """Resolve explicit values first; inference may later fill only missing fields."""
    out = dict(supplied)
    provenance = {key:dict(type='manual', declared_value=supplied[key]) for key in ('element','edge') if supplied.get(key)}
    if out.get('edge'):
        out['edge'] = normalize_edge(out['edge'])
    conflicts = []
    aliases = {'scan.element':'element','scan.edge':'edge','element.symbol':'element','element.edge':'edge','sample.name':'sample',
               'sample.formula':'formula','sample.mineral':'mineral', 's02':'s02',
               'formula':'formula','mineral':'mineral','element':'element','edge':'edge','sample':'sample', 'environment_id':'environment_id'}
    declarations = []
    for line in raw.splitlines():
        m = re.match(r'\s*#\s*([\w.]+):\s*(.*?)\s*$', line)
        if m and m[1].lower() in aliases:
            declarations.append((aliases[m[1].lower()], m[2], dict(type='file_header', field=m[1])))
    for support in supports:
        if support['role'] != 'sample_info':
            continue
        try:
            data = json.loads(support['text'])
        except (ValueError, TypeError):
            data = dict(re.findall(r'(?im)^\s*([\w.]+):\s*([^\n]+)',support['text']))
        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower() in aliases and isinstance(value, (str,int,float)):
                    declarations.append((aliases[key.lower()], value, dict(type='sample_info', filename=support.get('name'))))
    for key, value, source in declarations:
        if key == 'edge':
            value = normalize_edge(value)
        if key not in out or out[key] in (None, ''):
            out[key] = value
            if key in ('element','edge'):
                provenance[key] = source
        elif str(out[key]).lower() != str(value).lower() and not supplied.get(key):
            conflicts.append(key)
    if out.get('element'):
        from xraydb import atomic_symbol, atomic_number
        try:
            out['element'] = atomic_symbol(atomic_number(str(out['element']).capitalize()))
        except (ValueError, KeyError):
            out['element'] = None
    if out.get('edge'):
        out['edge'] = normalize_edge(out['edge'])
    out['conflicts'] = sorted(set(conflicts))
    out['provenance'] = provenance
    return out


TEMPLATES = [
    {'id':'metal','name':'金属近邻','paths':'M–M'},
    {'id':'oxygen','name':'氧配位','paths':'M–O'},
    {'id':'nitrogen','name':'氮配位','paths':'M–N'},
    {'id':'sulfur','name':'硫配位','paths':'M–S'},
    {'id':'carbon','name':'碳配位','paths':'M–C'},
    {'id':'halogen','name':'卤素配位','paths':'M–Cl / M–F / M–Br'},
    {'id':'ligand_metal','name':'配体与金属双壳','paths':'M–O + M–M / M–N + M–M'},
    {'id':'split_shell','name':'同类配位分裂双壳','paths':'M–配体短键 + M–配体长键'},
]
LIBRARY = Path(__file__).resolve().parents[1]/'.runtime/path-library'
FEFF_CACHE = Path(__file__).resolve().parents[1]/'.runtime/feff-screening-cache'


def template_for(element, scatterer):
    return 'metal' if scatterer==element else {'O':'oxygen','N':'nitrogen','S':'sulfur','C':'carbon','Cl':'halogen','F':'halogen','Br':'halogen'}.get(scatterer)


def template_models(paths, element, kspan):
    """Compose candidates from actual paths in ONE reference environment."""
    from xraydb import atomic_symbol
    groups={}
    for file,path in sorted(paths,key=lambda row:row[1].reff):
        scatterer=atomic_symbol(path.geom[1][1])
        category=template_for(element,scatterer)
        if not category:continue
        shells=groups.setdefault(scatterer,[])
        if not shells or path.reff-shells[-1][0][1].reff>.12:
            shells.append([])
        shells[-1].append((file,path))
    models=[]
    for scatterer,shells in groups.items():
        models.append(dict(template=template_for(element,scatterer),name=f'{element}–{scatterer}',groups=[shells[0]]))
        if len(shells)>1 and kspan>0 and shells[1][0][1].reff-shells[0][0][1].reff>np.pi/(2*kspan):
            models.append(dict(template='split_shell',name=f'{element}–{scatterer} 短键 + 长键',groups=shells[:2]))
    for ligand in ('O','N'):
        if ligand in groups and element in groups:
            models.append(dict(template='ligand_metal',name=f'{element}–{ligand} + {element}–{element}',groups=[groups[ligand][0],groups[element][0]]))
    return models


def atomic_input(text, fmt, element, edge):
    from larch.xrd.structure2feff import read_structure, structure2feffinp
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    edge = normalize_edge(edge)
    struct = read_structure(text, fmt=fmt)
    if not struct.is_ordered:
        raise InputError('结构存在混合占位；自动单壳模型不进行随机占位替代。')
    indices = [i for i,site in enumerate(struct) if site.specie.symbol == element]
    if not indices:
        raise InputError('结构不含当前吸收元素。')
    if len(struct) > 600:
        raise InputError('结构超过 600 个原子，需提供较小的局域坐标模型。')
    if hasattr(struct,'lattice'):
        equivalent = SpacegroupAnalyzer(struct, symprec=.01).get_symmetrized_structure().equivalent_indices
        sites = [group[0] for group in equivalent if group[0] in indices]
    else:
        sites = indices
    declared_site = re.search(r'(?im)absorber_index\s*[:=]\s*(\d+)', text) if fmt=='xyz' else None
    if declared_site:
        index=int(declared_site[1])
        if index not in indices:raise InputError('absorber_index 未指向声明的吸收元素。')
        sites=[index]
    if len(sites) != 1:
        raise InputError(f'结构包含 {len(sites)} 个不等价吸收位点；需给出位点占比或单吸收中心局域坐标，不能任选一位点冒充整体。')
    return structure2feffinp(text, element, edge=edge, site_index=sites[0]+1,
                             fmt=fmt, version8=False, cluster_size=6., rng_seed=0)


def sanitize_feff(text, element, edge):
    """Rebuild a bounded data-only FEFF6 input, ignoring arbitrary control cards."""
    from xraydb import atomic_number
    edge = normalize_edge(edge)
    edge_holes = {'K':1,'L1':2,'L2':3,'L3':4}
    if edge not in edge_holes:
        raise InputError('当前路径生成支持 K、L1、L2、L3 边；其他边请上传已计算且匹配的 FEFF 路径。')
    pots, atoms, mode = [], [], None
    source_edge = re.search(r'(?im)^\s*(?:HOLE|EDGE)\s+(\S+)', text)
    if source_edge and source_edge[1] != str(edge_holes[edge]) and normalize_edge(source_edge[1]) != edge:
        raise InputError('FEFF 输入吸收边与实验声明不一致。')
    for line in text.splitlines():
        words = line.split()
        if not words or words[0].startswith(('*','#','!')):
            continue
        if words[0].upper() in ('POTENTIALS','ATOMS','END'):
            mode = words[0].upper(); continue
        try:
            if mode == 'POTENTIALS':
                ipot, z = int(words[0]), int(words[1])
                if not 0<=ipot<=7 or not 1<=z<=98:
                    raise InputError('FEFF 原子势编号或元素超出支持范围。')
                pots.append((ipot,z))
            if mode == 'ATOMS':
                x,y,z = map(float, words[:3]); ipot=int(words[3])
                if not np.isfinite([x,y,z]).all():
                    raise InputError('原子坐标含非有限数值。')
                atoms.append((x,y,z,ipot))
        except (ValueError,IndexError):
            mode = None
    if not pots or not atoms or len(atoms)>600 or dict(pots).get(0)!=atomic_number(element):
        raise InputError('FEFF 原子势、吸收中心或原子数不满足模型要求。')
    centers = [a for a in atoms if a[3]==0]
    if len(centers)!=1:
        raise InputError('FEFF 模型必须恰有一个吸收中心。')
    origin=np.array(centers[0][:3]); shifted=[]
    for atom in atoms:
        xyz=np.array(atom[:3])-origin
        if atom[3] not in dict(pots):
            raise InputError('原子势引用不存在。')
        if np.linalg.norm(xyz)<=6.01:
            shifted.append((*xyz,atom[3]))
    distances=[np.linalg.norm(a[:3]) for a in shifted if a[3]!=0]
    if not distances or min(distances)<.7:
        raise InputError('近邻坐标缺失或原子间距不合理。')
    # Include the outer shell despite coordinate and FEFF rounding.
    cutoff=min(5.5,max(distances)+.02)
    header=f'TITLE Uploaded candidate; {element} {edge}\nHOLE {edge_holes[edge]} 1.0\nCONTROL 1 1 1 1\nPRINT 1 0 0 0\nRMAX {cutoff:.6f}\nNLEG 2\nPOTENTIALS\n'
    return header+'\n'.join(f'{i} {z}' for i,z in pots)+'\nATOMS\n'+'\n'.join(f'{x:.8f} {y:.8f} {z:.8f} {i}' for x,y,z,i in shifted)+'\nEND\n'


def make_paths(candidate, folder, element, edge):
    import larch
    edge = normalize_edge(edge)
    text=candidate['text']
    if candidate['role'] in ('cif','xyz'):
        text=atomic_input(text,candidate['role'],element,edge)
    text=sanitize_feff(text,element,edge)
    system={'Windows':'win64','Linux':'linux64','Darwin':'darwin64'}[platform.system()]
    exe=Path(larch.__file__).parent/'bin'/system/('feff6l.exe' if system=='win64' else 'feff6l')
    if not exe.exists():
        raise InputError('本机 FEFF 可执行文件不可用，已保留光谱处理结果。')
    identity=dict(input=text,input_sha256=sha(text), executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    cache=FEFF_CACHE/sha(text+identity['executable_sha256'])
    cached=sorted(cache.glob('feff[0-9][0-9][0-9][0-9].dat'))
    if (cache/'complete.json').exists() and cached:
        manifest=json.loads((cache/'complete.json').read_text(encoding='utf-8'))
        if all(sha(p.read_text())==manifest.get(p.name) for p in cached) and len(cached)==len(manifest):
            for path in cached:(folder/path.name).write_text(path.read_text(),encoding='utf-8',newline='\n')
            return sorted(folder.glob('feff[0-9][0-9][0-9][0-9].dat')),identity
    (folder/'feff.inp').write_text(text,encoding='ascii')
    run=subprocess.run([str(exe)],cwd=folder,capture_output=True,timeout=75,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    paths=sorted(folder.glob('feff[0-9][0-9][0-9][0-9].dat'))
    if run.returncode or not paths:
        raise InputError('FEFF 未生成可用散射路径。'+run.stderr.decode(errors='replace')[-300:])
    cache.mkdir(parents=True,exist_ok=True)
    for path in paths:(cache/path.name).write_text(path.read_text(),encoding='utf-8',newline='\n')
    (cache/'complete.json').write_text(json.dumps({p.name:sha(p.read_text()) for p in paths}),encoding='utf-8')
    return paths, identity


def validate_paths(files, element, edge):
    from larch.xafs import feffpath
    from xraydb import atomic_number
    edge = normalize_edge(edge)
    accepted=[]
    for file in files:
        p=feffpath(str(file))
        if not hasattr(p._feffdat,'k') or len(p._feffdat.k)<20:
            raise InputError('FEFF 路径数据不完整。')
        if p.nleg!=2:
            continue
        if p.geom[0][1]!=atomic_number(element):
            raise InputError('FEFF 路径的吸收元素不匹配。')
        if normalize_edge(p._feffdat.shell)!=edge:
            raise InputError(f'FEFF 路径的吸收边不匹配：目标 {edge}，文件 {p._feffdat.shell}（{normalize_edge(p._feffdat.shell)}）。')
        if not (0<p.degen<=100 and .7<p.reff<6):
            raise InputError('路径简并度或距离无效。')
        accepted.append((file,p))
    if not accepted:
        raise InputError('没有匹配的单散射路径。')
    if len(accepted)>64:
        raise InputError('局域单散射路径超过 64 条，请提供更小的参考局域环境。')
    return accepted


from .quick_fit import fit_once, checked_fit, first_shell_window, quick_model_fit, restrict_assumption_ranges


def evaluate_structure(k, chi, context, supports, fft):
    import time
    from xraydb import atomic_symbol
    from .screening_models import MODES, mode_for, reference_candidates, rereference_candidate
    context = {**context, 'edge':normalize_edge(context.get('edge'))}
    base = dict(policy=POLICY, status='unresolved', context=context, candidates=[], fit=None, fits=[],
        modes=[dict(mode, status='pending') for mode in MODES],
        template_assessment=[dict(t, status='missing_qualified_paths') for t in TEMPLATES],
        unresolved_parameters=['N','R','sigma_squared','energy_shift','uncertainties'],
        assumptions=['仅评估实际第一协调峰中的单散射局域模型；不是物相鉴定、高壳层分析或完整结构精修。'],
        stop_reasons=[], model_comparison=[])
    element, edge = context.get('element'), context.get('edge')
    if not element or not edge or context.get('conflicts'):
        base['stop_reasons'] = ['需要明确的吸收元素和吸收边，或样品声明存在冲突。']
        base['required'] = ['element','edge']
        return base
    if k is None or chi is None:
        base['stop_reasons'] = ['尚未得到有效 χ(k) 和 FT，暂不能数值拟合。']
        return base
    k, chi = np.asarray(k, float), np.asarray(chi, float)
    if len(k) < 40 or len(k) != len(chi) or not np.isfinite([k,chi]).all() or np.any(np.diff(k) <= 0):
        base['stop_reasons'] = ['χ(k) 点数、有限性或单调性未通过检查。']
        return base
    s02 = float(context.get('s02') if context.get('s02') is not None else 1.)
    if not .1 <= s02 <= 1.5:
        base['stop_reasons'] = ['S₀² 声明超出支持范围，请核对。']
        return base
    kmin = max(2.5, float(k[0])+.5, float(fft.get('kmin',2)))
    kmax = min(12., float(k[-1])-.5, float(fft.get('kmax',12)))
    try:
        window, decision = first_shell_window(k, chi, kmin, kmax, fft.get('rbkg',1.))
        base['first_shell_window'] = decision
    except InputError as exc:
        base['first_shell_window'] = dict(confidence='insufficient', reason=str(exc))
        base['stop_reasons'] = [str(exc)]
        return base
    base['assumptions'] += [
        f'S₀² 固定为 {s02:g}（'+('用户提供，未独立验证标定' if context.get('s02') else '假设值，未标定')+'）；N 为模型与振幅假设下的条件结果。',
        '先尝试自由 N、R、σ²、ΔE₀；信息量或稳定性不足时使用明确固定 σ²/ΔE₀ 的第一壳条件估计。固定参数不计入自由度，也不是实验测得值。',
        '所有候选比较同一数据驱动第一壳窗口；不能为某模型扩大到更远壳层。',
        '统计标准误差不包含背景、S₀²、自吸收、参考势、峰选择和模型选择误差；假设敏感性范围另列，非置信区间。',
        'C / N / O 等轻元素可能无法区分；排名不能独立确定散射元素、价态或唯一结构。']
    candidates = [dict(s, source={'type':'user_upload','filename':s['name']})
                  for s in supports if s['role'] in ('cif','xyz','feff_input')]
    pathsets = {}
    for support in supports:
        if support['role'] == 'feff_path':
            header = re.split(r'-{5,}', support['text'], maxsplit=1)[0]
            pathsets.setdefault(sha(header), []).append(support)
    for key, items in pathsets.items():
        candidates.append(dict(name=items[0]['name']+' 路径环境', role='feff_paths', text='', items=items,
                               source={'type':'user_paths','environment_sha256':key}))
    environment = context.get('environment_id')
    if not candidates and environment:
        if not re.fullmatch('[0-9a-f]{64}', str(environment)):
            base['stop_reasons'] = ['参考环境标识格式无效。']
            return base
        manifest = LIBRARY/environment/'manifest.json'
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding='utf-8'))
            if saved['element'] == element and normalize_edge(saved['edge']) == edge:
                candidates.append(dict(name='已生成参考环境 '+environment[:12], role='feff_paths', text='',
                    items=[dict(name=p['filename'],text=p['text']) for p in saved['paths']], source=saved['source']))
    if len(candidates) > 4:
        base['assumptions'].append('本轮最多使用 4 个用户参考环境；其余材料保留在导出中。')
    candidates = candidates[:4]
    try:
        candidates += reference_candidates(element, edge, {**context, '_quick_first_shell':decision})
    except InputError as exc:
        base['stop_reasons'].append(str(exc))
    with tempfile.TemporaryDirectory(prefix='webxas-qfs-') as temp:
        folder_root = Path(temp)
        def prepare(candidate):
            source = candidate['source']
            record = dict(name=candidate['name'], source=source, status='evaluating', source_sha256=sha(candidate['text']))
            base['candidates'].append(record)
            folder = folder_root/str(len(base['candidates']))
            folder.mkdir()
            try:
                if source.get('first_shell_span_A',0) > source.get('maximum_first_shell_span_A',float('inf')):
                    raise InputError('组合近邻先验不能位于同一第一协调峰；不扩大窗口强拟合。')
                generation = None
                if candidate['role'] == 'feff_paths':
                    files = []
                    for j,item in enumerate(candidate['items']):
                        path = folder/f'path{j}.dat'
                        path.write_text(item['text'], encoding='utf-8', newline='\n')
                        files.append(path)
                else:
                    files, generation = make_paths(candidate, folder, element, edge)
                paths = validate_paths(files, element, edge)
                for _,p in paths:
                    if p._feffdat.k[-1] < kmax+.5 or p._feffdat.k[0] > kmin-.5:
                        raise InputError('FEFF 路径 k 区间未覆盖实验拟合区间。')
                records = [dict(filename=f.name, sha256=sha(f.read_text()), text=f.read_text(),
                           reff=float(p.reff), degen=float(p.degen), scatterer=atomic_symbol(p.geom[1][1])) for f,p in paths]
                key = sha(json.dumps(dict(element=element, edge=edge, source=source, generation=generation,
                                         paths=[p['sha256'] for p in records]), sort_keys=True))
                record.update(status='compatible_candidate', generation=generation, paths=records, environment_id=key,
                    library_keys=[dict(absorber=element,edge=edge,scatterer=p['scatterer'],reference_environment=key) for p in records])
                cache = LIBRARY/key
                cache.mkdir(parents=True,exist_ok=True)
                (cache/'manifest.json').write_text(json.dumps(dict(element=element,edge=edge,source=source,paths=records),
                                                             ensure_ascii=False), encoding='utf-8')
                definitions = source.get('first_shell_paths')
                if definitions:
                    paths = [(f,p) for f,p in paths if any(atomic_symbol(p.geom[1][1]) == d['scatterer']
                        and abs(p.reff-d['nominal_distance_A']) <= d['tolerance_A'] for d in definitions)]
                    if not paths:
                        raise InputError('FEFF 未生成声明的第一近邻路径；不使用更远路径替代。')
                models = template_models(paths, element, kmax-kmin)
                if candidate.get('allowed_templates'):
                    models = [m for m in models if m['template'] in candidate['allowed_templates']]
                qualified = []
                for model in models:
                    if not all(window['rmin']-.15 <= p.reff <= window['rmax']+.65
                               for group in model['groups'] for _,p in group):
                        continue
                    model.update(environment=record, candidate=candidate)
                    qualified.append(model)
                    next(t for t in base['template_assessment'] if t['id']==model['template'])['status']='qualified_candidate'
                if not qualified:
                    record.update(status='rejected', reason='路径不属于当前第一协调峰范围；未把高壳层当作 QFS 第一壳。')
                return qualified
            except Exception as exc:
                record.update(status='rejected', reason=str(exc))
                return []
        prepared = [model for candidate in candidates for model in prepare(candidate)]
        if not prepared:
            for mode in base['modes']:
                mode.update(status='stopped', reason='无可用匹配第一壳路径。')
            base['stop_reasons'].append('没有模板获得可用且匹配的第一壳路径，详见路径环境评估。')
            return base
        reserved = []
        for mode in MODES:
            model = next((m for m in prepared if mode_for(m['template'])==mode['id']), None)
            if model is not None:
                reserved.append(model)
        scheduled = (reserved+[m for m in prepared if all(m is not r for r in reserved)])[:16]
        started = time.monotonic()
        base['fit_budget'] = dict(max_models=16, warm_fit_seconds=30, timed_out=False,
            budget_kind='soft_target', minimum_detailed_seconds_per_candidate=2.5,
            first_pass=[], max_optimizer_evaluations=120,
            strategy='Each candidate receives one bounded preliminary fit before shared-budget refinement; timeout is not chemical rejection.')
        deadline = started+30
        # Give every candidate a comparable first pass before expensive
        # sensitivity checks. Enumeration order must not exclude O or N.
        for model in scheduled:
            first = dict(template=model['template'], mode=mode_for(model['template']),
                         environment_id=model['environment']['environment_id'])
            try:
                limit = model['environment']['source'].get('reference_search',{}).get('delta_r_limit_A',.25)
                model['initial_attempt'] = fit_once(model['groups'],k,chi,window,s02,
                    delta_r_limit=limit,max_nfev=60)
                first.update(status='completed',statistics=model['initial_attempt']['statistics'])
            except Exception as exc:
                first.update(status='unresolved',reason=str(exc))
            base['fit_budget']['first_pass'].append(first)
        successful = []
        for model_index, model in enumerate(scheduled):
            model_started = time.monotonic()
            env, source = model['environment'], model['environment']['source']
            mode = mode_for(model['template'])
            row = dict(name=model['name'], template=model['template'], mode=mode, environment_id=env['environment_id'])
            try:
                limit = source.get('reference_search',{}).get('delta_r_limit_A',.25)
                assumptions = base['assumptions']+source.get('assumptions',[])
                remaining = max(0.,deadline-time.monotonic())
                candidate_deadline = time.monotonic()+max(2.5,remaining/(len(scheduled)-model_index))
                fit = quick_model_fit(model['groups'], k, chi, window, s02, assumptions, limit,
                                      candidate_deadline, model.get('initial_attempt'))
                search = source.get('reference_search',{})
                reference_trials = []
                if search and search.get('max_rereferences',0) and time.monotonic() < candidate_deadline:
                    shifts = [p['value'] for n,p in fit['parameters'].items() if n.startswith('dr')]
                    if shifts and max(abs(x) for x in shifts) >= .8*limit:
                        distance = float(np.clip(search['nominal_distance_A']+np.median(shifts),
                                                 search['minimum_A'],search['maximum_A']))
                        if abs(distance-search['nominal_distance_A']) > .02:
                            reference_trials.append(dict(environment_id=env['environment_id'], nominal_distance_A=search['nominal_distance_A'],
                                                         statistics=fit['statistics'], parameters=fit['parameters']))
                            try:
                                revised = rereference_candidate(model['candidate'],distance)
                                revised_models = prepare(revised)
                                alternative = next((m for m in revised_models if m['template']==model['template'] and len(m['groups'])==len(model['groups'])),None)
                                if alternative:
                                    rerun = quick_model_fit(alternative['groups'],k,chi,window,s02,
                                        base['assumptions']+alternative['environment']['source'].get('assumptions',[]),limit,candidate_deadline)
                                    reference_trials.append(dict(environment_id=alternative['environment']['environment_id'],
                                        nominal_distance_A=distance, statistics=rerun['statistics'],parameters=rerun['parameters']))
                                    if (not rerun['nr_available'],rerun['statistics']['rfactor']) < (not fit['nr_available'],fit['statistics']['rfactor']):
                                        fit, model = rerun, alternative
                                        env, source = model['environment'], model['environment']['source']
                            except Exception as exc:
                                reference_trials.append(dict(nominal_distance_A=distance, status='stopped', reason=str(exc)))
                                fit['warnings'].append('重新生成参考距离未完成，保留先前已完成的拟合。')
                fit['reference_distance_trials'] = reference_trials
                restrict_assumption_ranges(fit,model['groups'],source.get('first_shell_paths',[]))
                # A moved path must remain inside the element-only reference
                # prior; boundary escape is never repaired by relabelling it.
                for shell in fit['shells']:
                    definitions = [d for d in source.get('first_shell_paths',[]) if shell['pair'].endswith('–'+d['scatterer'])]
                    estimate = shell['R']['estimate']
                    if definitions and estimate is not None and not any(d['minimum_A'] <= estimate <= d['maximum_A'] for d in definitions):
                        for quantity in ('N','R'):
                            shell[quantity].update(value=None,stderr=None,status='unresolved')
                        fit['nr_available'] = False
                        fit['warnings'].append('拟合距离超出明确记录的物理参考范围；N、R 未确定。')
                        fit['unresolved_parameters'] = sorted(set(fit['unresolved_parameters'])|{'distance_prior'})
                identifier = sha(env['environment_id']+model['name']+model['template'])[:20]
                fit.update(id=identifier,model=model['name'],template=model['template'],mode=mode,
                    environment_id=env['environment_id'],source=source,
                    status='quick_estimate' if fit['nr_available'] else 'unresolved')
                row.update(id=identifier,environment_id=env['environment_id'],status='fitted',
                           statistics=fit['statistics'],unresolved_parameters=fit['unresolved_parameters'])
                successful.append((fit,model))
            except Exception as exc:
                row.update(status='stopped', reason=str(exc))
                row.update(getattr(exc,'fit_attempts',{}))
            row['elapsed_seconds'] = time.monotonic()-model_started
            base['model_comparison'].append(row)
        base['fit_budget']['elapsed_fit_seconds'] = time.monotonic()-started
        base['fit_budget']['timed_out'] = time.monotonic() > deadline
        for mode in base['modes']:
            rows = [r for r in base['model_comparison'] if r['mode']==mode['id']]
            mode.update(status='fitted' if any(r['status']=='fitted' for r in rows) else 'stopped',
                        attempted=len(rows), fitted=sum(r['status']=='fitted' for r in rows))
        if not successful:
            base['stop_reasons'].append('第一壳数据或数值检查不足以支持当前候选；已保留图谱与停止原因。')
            return base
        singles = sorted([x for x in successful if len(x[1]['groups'])==1],key=lambda x:x[0]['statistics']['aicc'])
        combos = sorted([x for x in successful if len(x[1]['groups'])==2],key=lambda x:x[0]['statistics']['aicc'])
        if not singles:
            base['stop_reasons'].append('缺少可检查的单壳基线；不自动声称复杂模型得到支持。')
            return base
        selected = next((s for s in singles if s[0]['nr_available']),singles[0])
        for combo in combos:
            if combo[0]['nr_available'] and combo[0]['statistics']['aicc'] < selected[0]['statistics']['aicc']-6 \
                    and combo[0]['statistics']['rfactor'] < selected[0]['statistics']['rfactor']*.8:
                selected = combo
                break
        fit, model = selected
        alternatives = [f for f,m in successful if m is not model and abs(f['statistics']['aicc']-fit['statistics']['aicc'])<6]
        if alternatives:
            fit['warnings'].append('存在近似等价候选；元素与局域解释仍不唯一。')
        base.update(fit=fit, fits=sorted([f for f,m in successful],key=lambda f:f['statistics']['aicc']),
            status='quick_estimate' if fit['nr_available'] else 'needs_review', selected_fit_id=fit['id'],
            unresolved_parameters=fit['unresolved_parameters'],
            selection_rule='同一第一协调峰窗口中优先可报告 N、R 的最简模型；固定假设逐项记录，AICc 仅在这些条件下解释。双壳需 ΔAICc>6、残差改善>20% 和参数检查通过。',
            interpretation='第一壳条件初筛估计；固定假设、范围敏感性与统计误差分开报告，不确定唯一结构。')
    return base
