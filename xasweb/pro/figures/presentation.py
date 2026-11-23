"""Brief display text derived only from a delivery's recorded evidence."""
from __future__ import annotations
import re


def brief_insight(insight, family, method=None):
    rows = insight.get('metrics', [])
    settings = method or insight.get('method', {}).get('figure_method', {})
    notices = []
    def find(text):
        return next((m for m in rows if text in m.get('label', '') and m.get('value') is not None), None)
    def selected(text, label):
        metric = find(text)
        return {**metric, 'label': label} if metric else None
    def number(metric):
        value = metric['value']
        return f'{value:.6g}' if isinstance(value, (int, float)) else str(value)
    def limits():
        return ' '.join(str(s) for s in insight.get('limitations', []))
    keys = []
    summary = ''
    if family == 'F01':
        lo, hi, count = find('横轴下界'), find('横轴上界'), find('采样点数')
        # A single statement must not attribute the first grid to every sample.
        multiple = len(insight.get('dataset_ids', [])) > 1
        if lo and hi and not multiple:
            summary = f'能量范围 {number(lo)}–{number(hi)} {lo.get("unit", "")}。'
        elif multiple:
            summary = f'{len(insight["dataset_ids"])} 个样品的原始谱与背景。'
        rejected = find('拒绝点数')
        if rejected and rejected['value']:
            notices.append(f'已排除 {number(rejected)} 个数据点。')
    elif family in ('F02', 'F03'):
        keys = [selected('工作流已存边位', '吸收边位置'), selected('工作流已存白线高度', '白线强度')] if family == 'F02' else [selected('导数最大值对应能量', '导数峰位')]
        if len(insight.get('dataset_ids', [])) > 1:
            # Keep per-sample ownership in the detailed record, not one generic card.
            keys = []
            summary = f'{len(insight["dataset_ids"])} 个样品的近边谱对比。'
        if any('未记录标准样' in str(m.get('basis', '')) for m in rows) or 'uncalibrated_energy' in limits() or '未校准' in limits() or '没有参考标准' in limits():
            notices.append('未做标准样能量校准。')
    elif family == 'F04':
        lo, hi = find('横轴下界'), find('横轴上界')
        if lo and hi and len(insight.get('dataset_ids', [])) <= 1:
            summary = f'k 范围 {number(lo)}–{number(hi)} Å⁻¹。'
        keys = [selected('显示 k 权重', 'k 权重')]
    elif family == 'F05':
        keys = [selected('最大 FT 幅值对应 R', 'FT 峰位')]
        notices.append('R 未做相位校正，峰位不等于键长。')
    elif family == 'F06':
        keys = [selected('WT 最大值对应 k', '峰值 k'), selected('WT 最大值对应 R', '峰值 R')]
        notices.append('R 未做相位校正；小波分布不能单独确定散射元素。')
    elif family == 'F07':
        for m in rows:
            label = m.get('label', '')
            if '未加权 RMS' not in label:
                continue
            part = 'Re' if re.search(r'(?:_|\b)(?:real|re)(?:_|\b)', label) else 'Im' if re.search(r'(?:_|\b)(?:imag|im)(?:_|\b)', label) else ''
            keys.append({**m, 'label': '未加权残差 RMS' + (f'（{part}）' if part else '')})
        scopes = insight.get('method', {}).get('fit_statistics_scope', [])
        statuses = [s['converged'] for s in scopes if isinstance(s.get('converged'), bool)]
        convergence = settings.get('converged')
        if statuses:
            convergence = all(statuses)
        if convergence is None:
            recorded = [re.match(r'^绑定(?:单谱|联合)运行(已收敛|未收敛)', text)
                        for text in insight.get('observations', [])]
            statuses = [m.group(1) == '已收敛' for m in recorded if m]
            if statuses:
                convergence = all(statuses)
        if convergence is not None:
            summary = '拟合已收敛。' if convergence else '拟合未收敛，请检查模型。'
        else:
            summary = '历史图件未记录收敛状态。'
        if any(s.get('scope') == 'joint_run' for s in scopes):
            notices.append('残差为本谱分量；整体统计量来自联合拟合。' if keys else '整体统计量来自联合拟合。')
    elif family == 'F08':
        keys = [selected('有效半路径长', '有效半路径长')]
        if settings.get('projection'):
            summary = f'{settings["projection"]} 方向的模型投影。'
        if settings.get('nleg', 0) > 2:
            notices.append('多重散射路径长不等于单根键长。')
    elif family == 'F09':
        keys = [{**m, 'label': m['label'].replace('参考组分系数', '参考系数')} for m in rows if '参考组分系数' in m.get('label', '')]
        notices.append('系数取决于参考谱与约束，不直接等同含量或价态。')
    elif family == 'F10':
        summary = '按记录状态比较；缺失结果不补值。'
        if settings.get('conditional_error_notice'):
            notices.append('误差为给定模型和噪声条件下的估计。')
    elif family == 'F11':
        keys = [selected('最大非对角绝对相关系数', '最大 |相关系数|')]
    keys = [m for m in keys if m is not None]
    if not summary and not keys:
        summary = '历史结果，详细数值见分析记录。' if not rows else '详细数值见分析记录。'
    return {'version': 1, 'summary': summary, 'metrics': keys, 'notices': notices}
