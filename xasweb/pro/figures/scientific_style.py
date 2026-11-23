"""English-only, title-free Times New Roman rendering; no font redistribution."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import re

from ..projects import ProjectError

FONT_FAMILY = 'Times New Roman'
STYLE_POLICY = 'english-times-title-free-v1'
CJK = re.compile('[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]')
PALETTES = {
    'okabe_ito': ['#111111', '#245b8a', '#b62828', '#426b59', '#6b5a85', '#8b6b39'],
    'muted': ['#111111', '#406b8c', '#a94747', '#527566', '#72627f', '#8b7954'],
}


@lru_cache(maxsize=8)
def font_files(family):
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    paths = {}
    for key, style, weight in [('regular','normal','normal'),('italic','italic','normal'),
                                ('bold','normal','bold'),('bold_italic','italic','bold')]:
        try:
            path=font_manager.findfont(font_manager.FontProperties(family=family,style=style,weight=weight),
                                       fallback_to_default=False)
            font=FT2Font(path)
            if font.family_name!=family or not Path(path).is_file():raise ValueError('foreign font')
            paths[key]=path
        except (ValueError,OSError,RuntimeError) as exc:
            raise ProjectError('figure_font_missing',
                f'系统缺少 {family} 字体。') from exc
    return paths


def times_font_files():
    return font_files(FONT_FAMILY)


times_font_files.cache_clear = font_files.cache_clear


@lru_cache(maxsize=1)
def nature_font():
    for family in ('Arial', 'Helvetica', 'Liberation Sans'):
        try:
            font_files(family)
            return family
        except ProjectError:
            continue
    raise ProjectError('figure_font_missing', '缺少适用的无衬线科研绘图字体。')


def rc_params(size, family=FONT_FAMILY):
    font_files(family)
    return {'font.family':[family],'font.serif':[family],'font.sans-serif':[family],
        'font.size':size,'axes.labelsize':size,'xtick.labelsize':size,'ytick.labelsize':size,
        'legend.fontsize':size,'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
        'pdf.use14corefonts':False,'text.usetex':False,'mathtext.fontset':'custom',
        'mathtext.rm':family,'mathtext.it':family+':italic',
        'mathtext.bf':family+':bold','mathtext.bfit':family+':bold:italic',
        'mathtext.sf':family,'mathtext.tt':family,'mathtext.cal':family,
        'mathtext.fallback':None,'axes.unicode_minus':True,'axes.linewidth':.8,
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
        'xtick.major.size':3.5,'ytick.major.size':3.5,'xtick.major.width':.75,'ytick.major.width':.75,
        'xtick.minor.size':2.,'ytick.minor.size':2.,'xtick.minor.width':.6,'ytick.minor.width':.6,
        'legend.frameon':False,'savefig.bbox':None,'figure.facecolor':'white','axes.facecolor':'white'}


class EnglishText:
    """Stable aliases retain the original in exported render/source metadata."""
    def __init__(self): self.aliases=[]

    def __call__(self, value, role='Label', identity=None):
        original=str(value)
        text=original.translate(str.maketrans({'（':'(', '）':')', '，':', ', '：':': ', '；':'; ', '　':' '}))
        known={'temperature_K':'Temperature (K)','potential_V':'Potential (V)','time_s':'Time (s)',
               '无结果 / No result':'No result','温度':'Temperature','电位':'Potential','时间':'Time',
               '能量':'Energy','参数':'Parameter','相关性':'Correlation','相关系数':'Correlation','归一化吸收':'Normalized absorption',
               '残差':'Residual','数据':'Data','模型':'Model','pre_edge':'Pre-edge',
               'post_edge':'Post-edge','bkg':'Background','mu':'Absorption'}
        if text in known and (not CJK.search(text) or role in ('Axis','Value')):text=known[text]
        elif role in ('Axis','Value'):
            for chinese,english in known.items():
                if CJK.search(chinese):text=text.replace(chinese,english)
        if CJK.search(text):
            # The digest uses the complete original, so distinct Chinese sample,
            # state and parameter names never collapse to an anonymous label.
            key=original if identity is None else original+'\0'+str(identity)
            prefix={'Parameter':'P','Atom':'A','Sample':'S'}.get(role,role)
            digest=hashlib.sha256(key.encode()).hexdigest()
            length=6
            text=f'{prefix}-{digest[:length]}'
            while any(row['rendered']==text and (row['original']!=original or row.get('source_identity')!=(None if identity is None else str(identity))) for row in self.aliases):
                length+=2
                text=f'{prefix}-{digest[:length]}'
        # Older licensed core fonts lack some Unicode superscript glyphs.
        # MathText uses the same audited font and a semantic exponent.
        if '$' not in text:
            superscript = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺','0123456789-+')
            text = re.sub('[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+', lambda m: '$^{'+m[0].translate(superscript)+'}$', text)
        if text!=original:
            row={'role':role,'original':original,'rendered':text}
            if identity is not None:row['source_identity']=str(identity)
            if row not in self.aliases:self.aliases.append(row)
        return text


def curve_style(name, index, panel, style):
    palette=PALETTES[style.palette]
    width=style.linewidth
    if panel.get('distinct_curves'):
        return palette[index%len(palette)], '-' if index<len(palette) else '--', width
    if panel.get('dataset_ids'):
        identities=sorted(set(panel['dataset_ids']))
        position=identities.index(panel['dataset_ids'][index])
        return palette[position%len(palette)], ('-' if position<len(palette) else '--'), width
    if name.startswith('residual') or name=='difference':return '#40658a','--',max(.65,width*.75)
    if name in ('pre_edge','post_edge','bkg','background'):
        return {'pre_edge':'.55','post_edge':'.35','bkg':'.65'}.get(name,'.5'),'--',max(.65,width*.75)
    if name.startswith('model') or name in ('fit','fitted'):
        return ('#245b8a','--',width) if name.endswith('_im') else ('#b62828','-',width)
    if name.startswith('contribution_'):return palette[1+index%(len(palette)-1)],'--',max(.7,width*.85)
    if name in ('data_im','chir_im'):return '#245b8a','--',width
    if name=='chir_re':return '#b62828','-',width
    return '#111111','-',width


def style_axis(axis, size):
    for spine in axis.spines.values():spine.set_visible(True);spine.set_linewidth(.8)
    axis.tick_params(which='both',direction='in',top=True,right=True,labelsize=size)
    axis.xaxis.labelpad=5
    axis.yaxis.labelpad=5


def audit_fonts(figure, family=FONT_FAMILY):
    """Verify every actual Text and math glyph uses Times, without fallback."""
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font
    from matplotlib.mathtext import MathTextParser
    from matplotlib.text import Text
    paths=font_files(family)
    parser=MathTextParser('path')
    for text in figure.findobj(Text):
        value=text.get_text()
        if not value:continue
        if CJK.search(value):raise ProjectError('figure_text_language','图件中残留未转换的中文标签。')
        actual=font_manager.findfont(text.get_fontproperties(),fallback_to_default=False)
        font=FT2Font(actual)
        if font.family_name!=family:
            raise ProjectError('figure_font_mismatch','图件文字实际解析为非指定字体。')
        if text.get_usetex():raise ProjectError('figure_font_mismatch','图件不可使用外部 TeX 字体替代指定字体。')
        if value.count('$')>=2 and text.get_parse_math():
            parsed=parser.parse(value,dpi=72,prop=text.get_fontproperties())
            if any(glyph[0].family_name!=family for glyph in parsed.glyphs):
                raise ProjectError('figure_font_mismatch','数学符号触发了非指定字体。')
        else:
            missing={char for char in value if not char.isspace() and ord(char) not in font.get_charmap()}
            if missing:raise ProjectError('figure_font_glyph_missing','图件字体缺少所需字形：'+''.join(sorted(missing)))
    return {'family':family,'files':{role:{'filename':Path(path).name,
            'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()} for role,path in paths.items()},
            'all_text_resolved_to_times':family==FONT_FAMILY,'all_text_resolved_to_declared_family':True,'fallback_allowed':False,'font_files_redistributed':False}
