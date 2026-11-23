"""Bounded, data-only import of beamline tables and Demeter containers."""
import gzip
import hashlib
import io
import json
import re
import zipfile

import yaml

import numpy as np

from .ingest import InputError, parse_ascii
from .edges import normalize_edge

MAX_UPLOAD = 50_000_000
MAX_EXPANDED = 100_000_000


def column_labels(header, width):
    labels = [''] * width
    for line in header:
        match = re.search(r'Column\.(\d+):\s*(\S+)', line, re.I)
        if match and 0 < int(match[1]) <= width:
            labels[int(match[1])-1] = match[2]
        numbered = line if re.match(r'^\s*#?\s*\d+\)', line) else ''
        for match in re.finditer(r'(\d+)\)\s*(.*?)(?=\s+\d+\)|$)', numbered):
            if 0 < int(match[1]) <= width:
                labels[int(match[1])-1] = match[2].strip()
    if any(labels):
        return labels
    for line in reversed(header):
        text = line.strip().lstrip('#;%').strip()
        tokens = re.split(r'[,;\s]+', text)
        if len(tokens) == width and tokens[0].lower() in ('energy', 'e', 'k'):
            return tokens
    return labels


def describe(raw, name, kind=None):
    table, header = parse_ascii(raw, preview=True)
    labels = column_labels(header, table.shape[1])
    kind = kind or ('chi' if labels and labels[0].lower() == 'k' else 'mu')
    issues = []
    if len(table) < 40:
        issues.append('short_scan')
    if not np.isfinite(table).all():
        issues.append('nonfinite_values')
    if np.any(np.diff(table[:, 0]) <= 0):
        issues.append('nonmonotonic_grid')
    state = 'chi' if kind == 'chi' else 'unknown'
    if any(label.lower() in ('normtrans', 'normfluor', 'normrefer', 'norm', 'normalized') for label in labels):
        state = 'normalized'; issues.append('processing_history_requires_review')
    declaration = next((m[1].strip() for line in header if (m := re.search(r'Processing\.state:\s*(\S+)', line, re.I))), None)
    if declaration in ('normalized', 'corrected_mu', 'corrected_chi', 'difference', 'background', 'chi_fit_table'):
        state = declaration
        if 'processing_history_requires_review' not in issues: issues.append('processing_history_requires_review')
    safe = [[float(v) if np.isfinite(v) else None for v in row] for row in table]
    return dict(filename=name, label=name, raw_text=raw, kind=kind,
                rows=len(table), columns=table.shape[1], labels=labels,
                processing_state=state, role='spectrum',
                preview=dict(eligible=True, issues=issues), analysis_blockers=issues,
                chi_data=[row[:2] for row in safe] if kind == 'chi' else None)


def _array(text):
    try:
        return [float(v.strip().strip("'\"").replace('D', 'E').replace('d', 'e')) for v in text.split(',')]
    except ValueError as exc:
        raise InputError('Athena 工程中的数值数组损坏。') from exc


def _athena(text, filename, digest):
    records = []
    if text.lstrip().startswith('{'):
        try:
            data = json.loads(text)
            for key in data.get('_____order', []):
                item = data[key]
                records.append((key, item.get('args', {}), item['x'], item['y']))
        except (ValueError, KeyError, TypeError) as exc:
            raise InputError('Athena JSON 工程损坏。') from exc
    else:
        for block in re.split(r'^\$old_group\s*=', text, flags=re.M)[1:]:
            key = block.split(';', 1)[0].strip().strip("'")
            arrays = dict(re.findall(r'^@(x|y)\s*=\s*\((.*?)\);', block, re.M | re.S))
            if 'x' not in arrays or 'y' not in arrays:
                raise InputError(f'Athena 数据组 {key} 缺少 x/y 数组。')
            # Read only literal scalar metadata. Never execute project expressions.
            args = {}
            for field in ('is_raw', 'is_xmu', 'is_ref', 'is_merge', 'is_bkg', 'source', 'original_label', 'datatype', 'label', 'is_chi', 'bkg_z', 'fft_edge', 'is_rsp', 'is_qsp', 'is_nor', 'is_diff', 'bkg_e0', 'bkg_eshift', 'bkg_pre1', 'bkg_pre2', 'bkg_nor1', 'bkg_nor2', 'bkg_rbkg', 'fft_kmin', 'fft_kmax', 'fft_kw', 'fft_rmin', 'fft_rmax'):
                m = re.search(r"'"+field+r"'\s*,\s*(?:'((?:\\.|[^'])*)'|([^,\)\s]+))", block)
                if m:
                    args[field] = (m[1] if m[1] is not None else m[2]).replace("\\'", "'")
            records.append((key, args, _array(arrays['x']), _array(arrays['y'])))
    datasets, warnings = [], []
    for key, args, x, y in records:
        label = str(args.get('label', key))
        flag = lambda name: str(args.get(name, 0)) in ('1', '1.0')
        kind = 'chi' if flag('is_chi') or args.get('datatype') == 'chi' else 'mu'
        processing = ('r_space' if flag('is_rsp') else 'q_space' if flag('is_qsp') else
                      'difference' if flag('is_diff') else 'background' if flag('is_bkg') else
                      'normalized' if flag('is_nor') else 'chi' if kind == 'chi' else
                      'mu' if flag('is_xmu') or flag('is_raw') else 'unknown')
        group_block = next((b for b in re.split(r'^\$old_group\s*=', text, flags=re.M)[1:]
                            if b.split(';', 1)[0].strip().strip("'") == key), '')
        if 'Self absorption correction of' in group_block or any('Self absorption correction of' in str(t) for t in args.get('titles', [])):
            processing = 'corrected_chi' if kind == 'chi' else 'corrected_mu'
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 1 or x.shape != y.shape:
            raise InputError(f'Athena 数据组 {label} 的 x/y 长度不一致。')
        header = [f'# Source.file: {filename}', f'# Source.sha256: {digest}', f'# Athena.group: {key}',
                  '# Column.1: k inverse_angstrom' if kind == 'chi' else '# Column.1: coordinate' if processing in ('r_space', 'q_space') else '# Column.1: energy eV',
                  '# Column.2: chi' if kind == 'chi' else '# Column.2: mu']
        if re.fullmatch('[A-Z][a-z]?', str(args.get('bkg_z', ''))):
            header.append(f"# Element.symbol: {args['bkg_z']}")
        edge = normalize_edge(args.get('fft_edge', ''))
        if edge in ('K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5'):
            header.append(f"# Element.edge: {edge}")
        raw = '\n'.join(header)+'\n'+'\n'.join(f'{a:.17g} {b:.17g}' for a,b in zip(x,y))+'\n'
        item=describe(raw, f'{filename} / {label}', kind)
        item['legacy_parameters']=args
        item['processing_state']=processing
        item['role']='reference' if flag('is_ref') else 'merged' if flag('is_merge') else 'spectrum'
        item['group_id']=key
        if processing not in ('mu', 'chi'):
            item['analysis_blockers'].append('processing_history_requires_review')
        if processing == 'chi':
            item['analysis_blockers'].append('chi_weighting_requires_verification')
        datasets.append(item)
    if not datasets:
        raise InputError('工程中未找到可导入的 μ(E) 或 χ(k) 数据。')
    warnings.append('已读取工程光谱；Athena 的处理参数和 Artemis 的拟合模型不会自动套用。')
    return datasets, warnings


def import_file(blob, filename):
    if len(blob) > MAX_UPLOAD:
        raise InputError('文件超过 50 MB。')
    digest = hashlib.sha256(blob).hexdigest()
    if blob.startswith(b'\x1f\x8b'):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(blob)) as stream:
                blob = stream.read(MAX_EXPANDED+1)
        except (OSError, EOFError) as exc:
            raise InputError('gzip 压缩文件损坏。') from exc
    if len(blob) > MAX_EXPANDED:
        raise InputError('解压后的内容超过 100 MB。')
    if blob.startswith(b'PK'):
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                entries = archive.infolist()
                if len(entries) > 10000 or sum(e.file_size for e in entries) > MAX_EXPANDED:
                    raise InputError('工程内容超过导入限制。')
                fits = [e for e in entries if e.filename.startswith('fits/') and e.filename.endswith('.fit')]
                if not fits:
                    raise InputError('未找到 Artemis 光谱；Web XAS 分析包请使用“打开分析包”。')
                datasets = []
                for entry in fits:
                    yamlname = entry.filename[:-4]+'.yaml'
                    if yamlname in archive.namelist():
                        docs = list(yaml.safe_load_all(archive.read(yamlname)))
                        if len(docs) < 3 or not isinstance(docs[0], dict):
                            raise InputError('Artemis 实验快照缺少元数据或原始数组。')
                        meta, x, y = docs[:3]
                        if not isinstance(x, list) or not isinstance(y, list) or len(x) != len(y):
                            raise InputError('Artemis 实验快照数组长度不匹配。')
                        chi = str(meta.get('is_chi', 0)) in ('1', '1.0') or meta.get('datatype') == 'chi'
                        kind = 'chi' if chi else 'mu'
                        raw = f'# Source.file: {filename}\n# Source.sha256: {digest}\n# Source.member: {yamlname}\n'
                        raw += '# Column.1: k inverse_angstrom\n# Column.2: chi\n' if chi else '# Column.1: energy eV\n# Column.2: mu\n'
                        raw += '\n'.join(f'{float(a):.17g} {float(b):.17g}' for a,b in zip(x,y))+'\n'
                        snapshot=describe(raw, f'{filename} / {meta.get("name", yamlname)}', kind)
                        snapshot.update(role='historical_snapshot', processing_state=kind, legacy_parameters=meta, member=yamlname)
                        snapshot['analysis_blockers'].append('historical_recipe_requires_review')
                        datasets.append(snapshot)
                    fit=describe(f'# Source.file: {filename}\n# Source.sha256: {digest}\n# Source.member: {entry.filename}\n'+archive.read(entry).decode('utf-8'), f'{filename} / {entry.filename}', 'chi')
                    fit.update(role='fit_history', processing_state='chi_fit_table', member=entry.filename)
                    fit['analysis_blockers'].append('derived_fit_table_not_experimental_input')
                    datasets.append(fit)
            return dict(datasets=datasets, warnings=['读取 Artemis 各次拟合保存的 χ(k)、fit 和 residual 列；不恢复拟合模型。'], sha256=digest)
        except (zipfile.BadZipFile, UnicodeError, RuntimeError, yaml.YAMLError, TypeError, ValueError) as exc:
            raise InputError('无法读取 Artemis 工程。') from exc
    if blob.startswith(b'%PDF-') or blob.startswith(b'\x89PNG\r\n\x1a\n'):
        item = dict(filename=filename, label=filename, raw_text='', kind='support', rows=0, columns=0,
                    labels=[], processing_state='not_experimental', role='supporting_material',
                    preview=dict(eligible=True, issues=[]), analysis_blockers=['supporting_material_not_spectrum'],
                    source_preview='二进制支持材料；源文件保留，未作为实验光谱。', source_sha256=digest)
        return dict(datasets=[item], warnings=['支持材料已保留；未作为实验光谱处理。'], sha256=digest)
    try:
        text = blob.decode('utf-8-sig')
    except UnicodeError:
        try:
            text = blob.decode('cp1252')
        except UnicodeError as exc:
            raise InputError('无法识别文件编码；请使用 UTF-8 或常见西文文本编码。') from exc
    if 'Athena project file' in text[:500] or (text.lstrip().startswith('{') and '_____order' in text):
        datasets, warnings = _athena(text, filename, digest)
    else:
        if filename.lower().endswith(('.inp', '.xyz', '.pdb', '.cif', '.ink', '.pdf', '.png', '.pl', '.org')):
            if len(blob.strip()) < 12:
                raise InputError('结构或说明文件不完整，未发现可核对内容。')
            structure = filename.lower().endswith(('.inp', '.xyz', '.pdb', '.cif', '.ink'))
            item = dict(filename=filename, label=filename, raw_text='', kind='support', rows=0, columns=0,
                        labels=[], processing_state='not_experimental', role='structural_model' if structure else 'supporting_material',
                        preview=dict(eligible=True, issues=[]), analysis_blockers=['supporting_material_not_spectrum'],
                        source_preview=text[:12000], source_sha256=digest)
            return dict(datasets=[item], warnings=['支持材料已保留；未作为实验光谱处理。'], sha256=digest)
        datasets, warnings = [describe(text, filename)], []
    return dict(datasets=datasets, warnings=warnings, sha256=digest)


def transform_chi(raw_text, kmin=2., kmax=12., kweight=2, dk=1.):
    """Transform an existing chi spectrum without fabricating mu or an edge."""
    from larch import Group
    from larch.xafs import xftf
    table, header = parse_ascii(raw_text)
    k, chi = table[:, 0], table[:, 1]
    if np.any(k < 0) or np.any(np.diff(k) <= 0):
        raise InputError('χ(k) 的 k 必须非负且严格递增。')
    if not k[0] <= kmin < kmax <= k[-1]:
        raise InputError(f'变换区间须位于 {k[0]:g}–{k[-1]:g} Å⁻¹ 内。')
    group = Group()
    xftf(k, chi, group=group, kmin=kmin, kmax=kmax, kweight=kweight, dk=dk, window='hanning')
    return dict(k=k.tolist(), chi=chi.tolist(), r=group.r.tolist(),
                chir_mag=group.chir_mag.tolist(), chir_re=group.chir_re.tolist(),
                chir_im=group.chir_im.tolist(), parameters=dict(kmin=kmin,kmax=kmax,kweight=kweight,dk=dk,window='hanning'),
                sha256=hashlib.sha256(raw_text.encode()).hexdigest())
