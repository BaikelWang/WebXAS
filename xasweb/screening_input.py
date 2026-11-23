"""Recorded, bounded input inference for exploratory uploads (never edits source)."""
import re
import numpy as np
from .ingest import parse_ascii
from .import_formats import column_labels
from .edges import normalize_edge


def identify_input(raw, context=None):
    from .automation import infer_mapping, signal_mapping, mapping_assumptions
    context = context or {}
    table, header = parse_ascii(raw, preview=True)
    text = '\n'.join(header).lower().replace('χ', 'chi').replace('å', 'angstrom')
    result = dict(status='unresolved', selection=None, confidence='unresolved', assumptions=[], evidence=[])
    if not np.isfinite(table).all() or len(table)<40:
        result['reason']='数据点不足或存在非有限值；原始输入保留。'
        return result
    mapping, reason = infer_mapping(raw)
    if mapping:
        result['assumptions'].extend(mapping_assumptions(reason))
        result.update(status='identified',selection={'kind':'mu',**mapping.model_dump()},
                      confidence='inferred' if reason=='fluorescence_preferred_over_transmission' else 'declared',evidence=[reason])
        return result
    if reason in ('ambiguous_column_declarations','invalid_column_declarations'):
        result['reason']='文件中的列声明冲突，不能据此自动计算。';return result
    if reason in ('ambiguous_signal_mode','declared_signal_mode_unavailable','reference_signal_only'):
        result['reason']='声明的测量模式冲突或缺少对应样品信号；未改用其他通道或参考信号。';return result
    labels=column_labels(header,table.shape[1])
    # Handle common beamline/CSV headers, including energy(eV) and k^2*chi.
    if not any(labels):
        for line in reversed(header):
            cells=re.split(r'[,;\t]+|\s+',line.strip().lstrip('#').strip())
            if len(cells)==table.shape[1] and any(re.match(r'(?i)^(energy|e\(|k\b|k\()',c) for c in cells):
                labels=cells;break
    labels=[x.lower().replace('χ','chi').replace('²','^2').replace('³','^3') for x in labels]
    find=lambda pattern:next((i for i,label in enumerate(labels) if re.fullmatch(pattern,label)),None)
    xcol=find(r'k(?:[\[(].*[\])])?')
    ecol=find(r'(?:energy|e)(?:[\[(].*[\])])?')
    ycol=find(r'(?:chi|chi\(k\)|k\^?[123]\*?chi|chi\*?k\^?[123])')
    unit='keV' if re.search(r'\bkev\b',text) else 'eV' if re.search(r'\bev\b',text) else None
    chi_named=xcol is not None or ycol is not None
    if xcol is None:xcol=ecol
    if xcol is None:
        axes=[i for i in range(table.shape[1]) if np.all(np.diff(table[:,i])>0)]
        # A leading monotonically increasing axis is a common two-column convention.
        if 0 in axes:xcol=0
        elif len(axes)==1:xcol=axes[0]
        else:
            result['reason']='无法可靠识别单调横坐标；请提供带列名的光谱文件。';return result
    if np.any(np.diff(table[:,xcol])<=0):
        result['reason']='横坐标不严格递增；不自动排序或删除测量点。';return result
    x=table[:,xcol]
    if ycol is None and table.shape[1]==2:ycol=1-xcol
    crosses=0 if ycol is None else np.count_nonzero(np.diff(np.sign(table[:,ycol])))
    edge_energy=None
    if context.get('element') and context.get('edge'):
        from xraydb import xray_edge
        edge=xray_edge(context['element'],normalize_edge(context['edge']))
        if edge:edge_energy=edge.energy
    # A broad k grid often contains E_edge/1000 numerically. Edge identity alone
    # must not turn an already recognizable oscillatory k spectrum into mu(E).
    edge_supports_kev=(edge_energy is not None and x[0]>0 and x[0]*1000<edge_energy<x[-1]*1000
                       and (x[-1]-x[0])*1000<.5*edge_energy)
    small_oscillation=(0<=x[0]<x[-1]<=40 and crosses>=4)
    if chi_named or (ecol is None and unit is None and small_oscillation and not edge_supports_kev):
        if ycol is None:
            result['reason']='识别到 k 轴，但无法唯一识别 χ(k) 信号列。';return result
        if re.search(r'(?:nm\^?-1|1/nm|nm⁻¹)',text):
            result['reason']='k 单位为 nm⁻¹，当前自动路径使用 Å⁻¹；请提供换算后的标注文件。';return result
        w=re.findall(r'chi\.weighting:\s*(\S+)',text)
        label_weight=re.search(r'k\^?([123])',labels[ycol])
        if w and any(v not in ('0','1','2','3','unweighted') for v in w):
            result['reason']='输入声明的 χ(k) 加权方式未知；不能解除未知加权。';return result
        weight=int(w[0]) if w and w[0]!='unweighted' else int(label_weight[1]) if label_weight else 0
        if len(set(w))>1 or (w and label_weight and int(label_weight[1])!=weight):
            result['reason']='输入 χ(k) 权重声明冲突。';return result
        declared_unit=bool(re.search(r'inverse_angstrom|1/angstrom|angstrom\^-1|a\^-1',text))
        if not declared_unit:result['assumptions'].append('k 单位未明确标注，按 Å⁻¹ 进行初筛；此单位假设随结果保留。')
        if not chi_named:result['assumptions'].append('依据横坐标范围及振荡特征，将两列输入识别为 χ(k)，属于初筛假设。')
        if not w and not label_weight and labels[ycol] not in ('chi','chi(k)'):
            result['assumptions'].append('输入权重未声明，初筛暂按未加权 χ(k)；若源谱已加权，此结果不适用。')
        result.update(status='identified',selection=dict(kind='chi',energy=xcol,signal=ycol,unit='inverse_angstrom',input_weight=weight),
                      confidence='inferred' if result['assumptions'] else 'declared',evidence=['k axis / chi label or oscillation heuristic'])
        return result
    if unit is None:
        if x[-1]>150 and x[-1]<=200000:unit='eV'
        elif edge_supports_kev:unit='keV'
        elif ecol is not None and x[-1]<150:unit='keV'
        elif ycol is not None and 0<x[0]<x[-1]<150 and crosses<4:
            # Distinguish a compact energy scan from k using edge identity; don't guess blindly.
            result.update(reason='小数能量轴需要吸收元素与吸收边辅助判断 eV / keV。',required=['element','edge']);return result
        else:
            result['reason']='无法区分能量与 k 轴；保留原谱，未套用默认列组合。';return result
        result['assumptions'].append(f'能量单位由范围与吸收边推断为 {unit}，未改动原始文件。')
    named, signal_source = signal_mapping({name:i for i,name in enumerate(labels) if name},xcol,unit,header)
    if named:
        result['assumptions'].extend(mapping_assumptions(signal_source))
        result.update(status='identified',selection={'kind':'mu',**named.model_dump()},
                      confidence='inferred' if result['assumptions'] else 'declared',
                      evidence=['column labels / monotonic axis / energy range',signal_source])
        return result
    if signal_source in ('ambiguous_signal_mode','declared_signal_mode_unavailable','reference_signal_only'):
        result['reason']='声明的测量模式冲突或缺少对应样品信号；未改用其他通道或参考信号。';return result
    mode='mu';ref=0
    if ycol is None:
        result['reason']='多列强度数据缺少可区分 I₀、Iₜ 或荧光信号的列名；未猜测探测器通道。';return result
    result['assumptions'].append('两列能量谱的第二列按 μ(E) 处理，保留此输入假设。')
    result.update(status='identified',selection=dict(kind='mu',energy=xcol,signal=ycol,reference=ref,mode=mode,energy_unit=unit),
                  confidence='inferred' if result['assumptions'] else 'declared',evidence=['column labels / monotonic axis / energy range'])
    return result
