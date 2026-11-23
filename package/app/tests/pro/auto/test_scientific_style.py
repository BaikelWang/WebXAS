"""Actual text/font/artist checks across every renderer kind, without data edits."""
from __future__ import annotations

import copy
import hashlib
import io
import re
from unittest.mock import patch

import numpy as np
import pytest

from xasweb.pro.figures import FigureSpec, FigureStyle
from xasweb.pro.figures.sources import FigureData
from xasweb.pro.figures.render import make_figure, render_figure
from xasweb.pro.figures.scientific_style import (CJK, FONT_FAMILY, EnglishText,
    times_font_files)
from xasweb.pro.projects import ProjectError


def sample_item(kind):
    style=FigureStyle(language='zh',formats=['png','pdf','svg'])
    spec=FigureSpec(id='fixture-'+kind,family='F07' if kind=='residual' else 'F04',
        title='中文模块标题必须保留',title_en='Module title must stay outside the plot',kind=kind,style=style)
    k=np.linspace(1.,12.,31)
    panels=[{'x':'k','ys':['data','model'],'labels':['铜箔','铜箔'],
             'dataset_ids':['sample-a','sample-b'],'xlabel':'k (Å⁻¹)','ylabel':'k²χ(k) (Å⁻²)'}]
    arrays={'k':k,'data':np.sin(k),'model':np.sin(k)*.97,'residual':np.sin(k)*.03}
    if kind=='residual':
        panels=[{'x':'k','ys':['data','model'],'labels':['Data','Fit'],'xlabel':'k (Å⁻¹)','ylabel':'χ(k)'},
                {'x':'k','ys':['residual'],'labels':['Residual'],'xlabel':'k (Å⁻¹)','ylabel':'Residual χ(k)'}]
    elif kind=='heatmap':
        panels=[{'x':'index','y':'index','z':'matrix','ticks':['振幅','位移'],'xlabel':'参数','ylabel':'参数','colorlabel':'相关系数'}]
        arrays={'index':np.array([0.,1.]),'matrix':np.array([[1.,.3],[.3,1.]])}
    elif kind=='geometry':
        panels=[{'x':'x','y':'y','projection':'XY','atom_labels':['铜1','氧2'],'xlabel':'x (Å)','ylabel':'y (Å)'},
                {'x':'x','y':'z','projection':'XZ','atom_labels':['铜1','氧2'],'xlabel':'x (Å)','ylabel':'z (Å)'}]
        arrays={'x':np.array([0.,2.]),'y':np.array([0.,.5]),'z':np.array([0.,1.5])}
    elif kind=='trend':
        panels=[{'x':'k','ys':['data'],'labels':['铜箔'],'xlabel':'温度 (K)','ylabel':'数据',
                 'state_tick_labels':['初始','反应后'],'valid_masks':['valid'],'state_valid':'valid',
                 'failed_state_markers':True}]
        arrays={'k':np.array([0.,1.]),'data':np.array([1.,2.]),'valid':np.array([True,False])}
    item=FigureData(spec.model_copy(update={'panels':panels}),arrays)
    if kind=='composite':
        first=sample_item('line');second=sample_item('heatmap')
        first.spec=first.spec.model_copy(update={'family':'F04'});second.spec=second.spec.model_copy(update={'family':'F06'})
        item=FigureData(spec.model_copy(update={'panels':[{'child_id':first.spec.id},{'child_id':second.spec.id}]}),{},[first,second])
    return item


@pytest.mark.parametrize('kind',['line','residual','heatmap','geometry','trend','composite'])
def test_all_text_is_english_actual_times_and_no_titles_or_array_changes(kind):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    from matplotlib.text import Text
    item=sample_item(kind)
    before=copy.deepcopy(item.arrays);spec_before=item.spec.model_dump(mode='json')
    children_before=[copy.deepcopy(child.arrays) for child in item.children or []]
    figure,metadata=make_figure(item)
    try:
        assert figure._suptitle is None
        for axis in figure.axes:
            assert all(not axis.get_title(loc=where) for where in ('left','center','right'))
        texts=[t for t in figure.findobj(Text) if t.get_text()]
        assert texts
        for text in texts:
            assert not CJK.search(text.get_text())
            path=font_manager.findfont(text.get_fontproperties(),fallback_to_default=False)
            assert FT2Font(path).family_name==FONT_FAMILY
            assert text.get_fontsize()==item.spec.style.point_size
        assert metadata['font_audit']['all_text_resolved_to_times']
        assert not metadata['font_audit']['fallback_allowed']
        assert metadata['render_language']=='en' and metadata['figure_titles'] is False
        if kind!='residual':assert metadata['text_aliases']
        if kind=='line':
            assert len({row['rendered'] for row in metadata['text_aliases'] if row['original']=='铜箔'})==2
        if kind=='residual':
            assert figure.axes[0].lines[0].get_color()=='#111111'
            assert figure.axes[0].lines[1].get_color()=='#b62828'
            assert figure.axes[1].lines[0].get_linestyle()=='--'
        for name,array in before.items():np.testing.assert_array_equal(item.arrays[name],array)
        for child,previous in zip(item.children or [],children_before):
            for name,array in previous.items():np.testing.assert_array_equal(child.arrays[name],array)
        assert item.spec.model_dump(mode='json')==spec_before
    finally:plt.close(figure)


def test_times_font_files_are_real_system_fonts_and_missing_font_is_explicit():
    from matplotlib.ft2font import FT2Font
    paths=times_font_files()
    assert all(FT2Font(path).family_name==FONT_FAMILY for path in paths.values())
    assert all(re.search(r'times[^/\\]*\.ttf$',path,re.I) for path in paths.values())
    times_font_files.cache_clear()
    try:
        with patch('matplotlib.font_manager.findfont',side_effect=ValueError('font absent')):
            with pytest.raises(ProjectError) as error:times_font_files()
            assert error.value.code=='figure_font_missing'
    finally:times_font_files.cache_clear()


def test_math_glyphs_editable_svg_and_embedded_times_pdf():
    item=sample_item('line')
    panel={**item.spec.panels[0],'ylabel':r'$k^2\chi(k)$','labels':['Data','Fit'],'dataset_ids':[]}
    item.spec=item.spec.model_copy(update={'panels':[panel]})
    rendered,metadata=render_figure(item)
    svg=rendered[f'figures/{item.spec.id}.svg'].decode('utf-8')
    assert '<text' in svg and 'Times New Roman' in svg
    assert not CJK.search(svg)
    pdf=rendered[f'figures/{item.spec.id}.pdf']
    assert b'/FontFile2' in pdf  # subset TrueType font embedded by the PDF backend
    assert b'TimesNewRoman' in pdf or b'TimesNewRomanPS' in pdf
    assert metadata['font_audit']['font_files_redistributed'] is False
    assert not any(name.lower().endswith(('.ttf','.otf')) for name in rendered)


def test_aliases_are_stable_and_keep_original_identity():
    text=EnglishText()
    a=text('样品甲','Sample','dataset1');b=text('样品乙','Sample','dataset2')
    assert a!=b and text('样品甲','Sample','dataset1')==a
    assert text.aliases[0]['original']=='样品甲' and text.aliases[0]['source_identity']=='dataset1'
    assert text('温度 (K)','Axis')=='Temperature (K)'


def test_unsupported_glyph_fails_explicitly_and_closes_failed_figure():
    import matplotlib.pyplot as plt
    item=sample_item('line')
    item.spec=item.spec.model_copy(update={'panels':[{**item.spec.panels[0],'ylabel':'Signal 🧪'}]})
    before=set(plt.get_fignums())
    with pytest.warns(UserWarning,match='Glyph'):
        with pytest.raises(ProjectError) as error:make_figure(item)
    assert error.value.code=='figure_font_glyph_missing'
    assert set(plt.get_fignums())==before
