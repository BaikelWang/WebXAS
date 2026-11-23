"""Independent panels are re-rendered from their own arrays, never image crops."""
from __future__ import annotations

from .sources import FigureData, MAX_FIGURES
from ..projects import ProjectError

FAMILIES = tuple(f'F{i:02d}' for i in range(1, 14))


def validate_presentation(presentation='legacy', figure_families=None):
    if presentation not in ('legacy', 'individual'):
        raise ProjectError('invalid_delivery_presentation', '图件交付方式无效。')
    if figure_families is not None and (not isinstance(figure_families, list) or
            not figure_families or any(f not in FAMILIES for f in figure_families) or
            len(set(figure_families)) != len(figure_families)):
        raise ProjectError('invalid_figure_families', '请选择不重复的 F01–F13 图件类别。')


def independent_panels(items):
    output = []
    for item in items:
        if item.spec.family == 'F12' or item.spec.kind == 'composite':
            continue
        for index, original in enumerate(item.spec.panels):
            panel = dict(original)
            keys = {panel[k] for k in ('x', 'y', 'z', 'state_valid') if panel.get(k)}
            for key in ('ys', 'xkeys', 'valid_masks', 'error_keys', 'error_valid_masks'):
                keys.update(k for k in panel.get(key, []) if k)
            keys.update(panel.get('data_keys', ()))
            if item.spec.kind == 'geometry':
                keys.update(k for k in ('atom_index', 'occupancy') if k in item.arrays)
            arrays = {key: value for key, value in item.arrays.items() if key in keys}
            panel_id = f'{item.spec.id}-p{index+1:02d}'
            kind = 'line' if item.spec.kind == 'residual' else item.spec.kind
            suffix = ''
            suffix_en = ''
            if len(item.spec.panels) > 1:
                if item.spec.kind == 'geometry':
                    suffix = suffix_en = ' · ' + panel['projection']
                elif item.spec.kind == 'residual':
                    suffix = ' · 数据与模型' if index == 0 else ' · 残差'
                    suffix_en = ' · Data and model' if index == 0 else ' · Residual'
            spec = item.spec.model_copy(update={
                'id': panel_id, 'panel_id': panel_id, 'parent_figure_id': item.spec.id,
                'panel_index': index, 'kind': kind, 'panels': [panel],
                'title': item.spec.title + suffix, 'title_en': item.spec.title_en + suffix_en,
                'data_files': [f'data/{panel_id}.csv'] +
                    ([f'data/{panel_id}.npz'] if kind == 'heatmap' else []),
                'method': {**item.spec.method, 'individual_panel': True,
                           'source_panel_kind': item.spec.kind, 'source_panel_index': index,
                           **({'projection': panel['projection']} if kind == 'geometry' else {})},
            })
            output.append(FigureData(spec, arrays))
    if len(output) > MAX_FIGURES:
        raise ProjectError('delivery_scope_limit', '独立图件数量超过上限，请拆分工作流。')
    return output
