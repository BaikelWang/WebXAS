"""Fixed-physical-size Matplotlib rendering from the portable FigureSpec."""
from __future__ import annotations

import io
import json
import math

import numpy as np

from ..projects import ProjectError
from .scientific_style import (PALETTES, FONT_FAMILY, STYLE_POLICY, EnglishText,
                               rc_params, curve_style, style_axis, audit_fonts)


def _draw(axis, item, panel, colorbar=True, fontsize=None):
    spec, a = item.spec, item.arrays
    style = spec.style
    text=axis.figure._webxas_text
    if spec.kind=='geometry':
        x,y=a[panel['x']],a[panel['y']]
        axis.plot(np.r_[x,x[0]],np.r_[y,y[0]],color='.45',lw=style.linewidth,ls='--')
        groups={}
        for i,(xx,yy) in enumerate(zip(x,y)):groups.setdefault((float(xx),float(yy)),[]).append(i)
        for (xx,yy),indices in groups.items():
            axis.scatter([xx],[yy],s=42 if style.layout!='presentation' else 90,
                         color=PALETTES[style.palette][indices[0]%len(PALETTES[style.palette])],edgecolors='white',linewidths=.6,zorder=3)
            label='/'.join(text(panel['atom_labels'][i],'Atom') for i in indices)
            dx=7 if xx>=float(np.mean(x)) else -7
            dy=7 if yy>=float(np.mean(y)) else -7
            axis.annotate(label,(xx,yy),xytext=(dx,dy),textcoords='offset points',
                          ha='left' if dx>0 else 'right',va='bottom' if dy>0 else 'top',
                          fontsize=fontsize or style.point_size)
        span=max(float(np.ptp(x)),float(np.ptp(y)),.1)
        cx=(float(x.min())+float(x.max()))/2;cy=(float(y.min())+float(y.max()))/2
        axis.set_box_aspect(1);axis.set_aspect('equal',adjustable='box')
        axis.set_xlim(cx-span*1.1,cx+span*1.1);axis.set_ylim(cy-span*1.1,cy+span*1.1)
        axis.set_autoscale_on(False)
    elif 'z' in panel:
        x, y, z = a[panel['x']], a[panel['y']], a[panel['z']]
        if z.shape != (len(y), len(x)):
            raise ProjectError('figure_array_shape', '热图矩阵必须与两个坐标等长。', spec.id)
        def edges(values):
            if len(values)==1:
                return np.array([values[0]-.5,values[0]+.5])
            middle=(values[1:]+values[:-1])/2
            return np.r_[values[0]-(middle[0]-values[0]),middle,values[-1]+(values[-1]-middle[-1])]
        # Matplotlib's center inference produces a zero-area cell for 1×1
        # correlation matrices. Explicit edges preserve that real coefficient.
        singleton=len(x)==1 or len(y)==1
        image = axis.pcolormesh(edges(x) if singleton else x, edges(y) if singleton else y, z,
                               shading='flat' if singleton else 'auto', rasterized=True,
                               cmap=spec.method.get('colormap','viridis'),
                               vmin=spec.method.get('color_min'), vmax=spec.method.get('color_max'))
        if len(x)>1: axis.set_xlim(float(x[0]),float(x[-1]))
        if len(y)>1: axis.set_ylim(float(y[0]),float(y[-1]))
        if colorbar:
            bar = axis.figure.colorbar(image, ax=axis, pad=.03, fraction=.045)
            # Keep colorbar labels compact at final size, with the numerical
            # transform and exact convention retained in methods / FigureSpec.
            label = panel.get('colorlabel','')
            if spec.family=='F06':label='|WT[k²χ(k)]|'
            bar.set_label(text(label,'Value'), fontsize=fontsize or style.point_size)
            style_axis(bar.ax,fontsize or style.point_size)
        if panel.get('ticks'):
            ticks = [text(tick,'Parameter') for tick in panel['ticks']]
            axis.set_xticks(np.arange(len(ticks)), ticks, rotation=45, ha='right')
            axis.set_yticks(np.arange(len(ticks)), ticks)
            axis.tick_params(axis='x',pad=10)
    else:
        missing_drawn=False
        fit_overlay = any(name.startswith('model') for name in panel['ys'])
        if fit_overlay and spec.family == 'F07' and spec.method.get('transform'):
            t = spec.method['transform']
            space = spec.method['space']
            axis.axvspan(t[f'{space}min'], t[f'{space}max'], color='#e4eaf0', zorder=0)
        for i, (name, label) in enumerate(zip(panel['ys'], panel['labels'])):
            values = a[name]
            # Display offset is applied only here; CSV always stores the actual
            # full-precision values and the FigureSpec declares the transform.
            color,linestyle,linewidth=curve_style(name,i,panel,style)
            identity=panel['dataset_ids'][i] if panel.get('dataset_ids') else next((s['dataset_id'] for s in spec.sources if s.get('dataset_id')),None)
            xkey=panel['xkeys'][i] if panel.get('xkeys') else panel['x']
            mask_name=panel.get('valid_masks',[None]*len(panel['ys']))[i]
            valid=np.asarray(a[mask_name],bool) if mask_name else np.ones(values.shape,bool)
            state_valid=np.asarray(a[panel['state_valid']],bool) if panel.get('state_valid') else np.ones(values.shape,bool)
            displayed=np.where(valid,values+i*style.offset,np.nan)
            xvalues=np.where(state_valid,a[xkey],np.nan)
            mode=panel.get('series_modes',['line']*len(panel['ys']))[i]
            observed_fit = fit_overlay and name in ('data', 'data_chi', 'data_re', 'data_im')
            axis.plot(xvalues, displayed,
                      label=text(label,'Sample',identity), color=color, lw=linewidth,
                      ls='none' if mode in ('points','markers') else linestyle,
                      marker='x' if mode=='points' else 'o' if mode in ('trend','markers') or observed_fit else None,
                      markersize=3.2 if observed_fit else 4, markerfacecolor='white' if observed_fit else color,
                      markevery=max(1, len(xvalues)//24) if observed_fit else None)
            if panel.get('error_keys'):
                errors=a[panel['error_keys'][i]]
                error_valid=a[panel['error_valid_masks'][i]]&valid&state_valid
                axis.errorbar(a[xkey][error_valid],values[error_valid]+i*style.offset,
                              yerr=errors[error_valid],fmt='none',ecolor=color,elinewidth=.8,capsize=2)
            if panel.get('failed_state_markers'):
                failed=state_valid&~valid
                if np.any(failed):
                    missing_drawn=True
                    axis.plot(a[xkey][failed],np.full(int(failed.sum()),.025),transform=axis.get_xaxis_transform(),
                          color='.4',marker='x',ls='none',markersize=4,clip_on=False,
                          label='No result')
        if style.legend and (len(panel['ys']) > 1 or missing_drawn):
            axis.legend(loc='best', fontsize=fontsize or style.point_size, handlelength=1.6)
        if (spec.method.get('quantity') in ('objective_residual','normalized_difference') or
                (spec.kind == 'residual' and not fit_overlay) or (spec.method.get('source_panel_kind') == 'residual' and
                spec.method.get('source_panel_index', 0) > 0)) and len(panel['ys']) <= 2:
            axis.axhline(0, color='.65', linewidth=.5, zorder=0)
    axis.set_xlabel(text(panel.get('xlabel',''),'Axis'), fontsize=fontsize or style.point_size)
    axis.set_ylabel(text(panel.get('ylabel',''),'Axis'), fontsize=fontsize or style.point_size)
    style_axis(axis,fontsize or style.point_size)
    if panel.get('state_tick_labels'):
        axis.set_xticks(np.arange(len(panel['state_tick_labels'])),[text(label,'State') for label in panel['state_tick_labels']],rotation=30,ha='right',rotation_mode='anchor')
    axis.ticklabel_format(axis='both', style='sci', scilimits=(-3,5), useMathText=False) if not panel.get('ticks') and not panel.get('state_tick_labels') else None
    if style.xlim is not None: axis.set_xlim(*style.xlim)
    if style.ylim is not None: axis.set_ylim(*style.ylim)


def make_figure(item):
    """Create a figure; caller owns closing it. Used by rendering and QA."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    style = item.spec.style
    width, height = style.size_mm
    if item.spec.kind == 'composite':
        cols = 1 if style.layout == 'single' else 2
        rows = math.ceil(len(item.children) / cols)
        height = height if style.layout == 'presentation' else max(height, rows * (60. if cols==1 else 72.))
    elif item.spec.kind=='geometry' and len(item.spec.panels)>1:
        rows,cols=(2,1) if style.layout=='single' else (1,2)
        height=145. if style.layout=='single' else style.size_mm[1]
    else:
        rows, cols = (2,1) if item.spec.kind == 'residual' else (1,1)
        if rows == 2 and style.layout == 'single': height = 96.
    with matplotlib.rc_context(rc_params(style.point_size, style.font_family)):
        fig, axes = plt.subplots(rows, cols, figsize=(width/25.4,height/25.4), squeeze=False,
                                 layout='constrained',
                                 gridspec_kw={'height_ratios':[3,1]} if item.spec.kind == 'residual' else None)
        # rc_context must also remain in effect during savefig (render_figure).
        fig._webxas_fonts = [style.font_family]
        fig._webxas_text = EnglishText()
        fig.get_layout_engine().set(w_pad=4/72,h_pad=4/72,wspace=.04,hspace=.05)
        if item.spec.kind == 'composite':
            for i, axis in enumerate(axes.flat):
                if i >= len(item.children): axis.set_visible(False); continue
                child = item.children[i]
                panel={**child.spec.panels[0],**{k:v for k,v in item.spec.panels[i].items() if k!='child_id'}}
                _draw(axis, child, panel, colorbar=False)
                # Small internal colorbars do not alter comparable plot bounds.
                if child.spec.kind == 'heatmap':
                    inset = axis.inset_axes([.63,.88,.32,.035])
                    fig.colorbar(axis.collections[0],cax=inset,orientation='horizontal')
                    from matplotlib.ticker import MaxNLocator
                    inset.xaxis.set_major_locator(MaxNLocator(nbins=2))
                    inset.tick_params(labelsize=style.point_size, length=2, colors='white',direction='in')
                    for spine in inset.spines.values(): spine.set_edgecolor('white')
        else:
            for axis, panel in zip(axes.flat, item.spec.panels): _draw(axis,item,panel)
            if item.spec.kind == 'residual':
                axes[0,0].set_xlabel(''); axes[0,0].tick_params(labelbottom=False)
        fig.canvas.draw()
        try:
            font_audit=audit_fonts(fig, style.font_family)
        except Exception:
            plt.close(fig)
            raise
        rects=[]
        for i,axis in enumerate(axes.flat):
            if not axis.get_visible(): continue
            box=axis.get_position()
            rects.append({'id':str(i), 'bbox_pt':[box.x0*width/25.4*72,box.y0*height/25.4*72,
                                                  box.x1*width/25.4*72,box.y1*height/25.4*72]})
        geometry={'schema_version':1,'figure_id':item.spec.id,'width_mm':width,'height_mm':height,
                  'axes':rects,'font_families':[style.font_family],'font_size_pt':style.point_size,
                  'font_audit':font_audit,'render_policy':'nature-final-v1' if style.profile == 'nature' else STYLE_POLICY,'render_language':'en',
                  'figure_titles':False,'text_aliases':fig._webxas_text.aliases,
                  'dpi':style.dpi,'rasterized_heatmap':item.spec.kind=='heatmap' or any(c.spec.kind=='heatmap' for c in item.children or []),
                  'alignment_exemptions':['Residual panels intentionally have 3:1 height ratio.'] if item.spec.kind=='residual' else []}
        return fig, geometry


def render_figure(item):
    import matplotlib
    import matplotlib.pyplot as plt
    fig, geometry = make_figure(item)
    entries = {}
    try:
        style=item.spec.style
        with matplotlib.rc_context(rc_params(style.point_size, style.font_family)):
            # Lock geometry at the intended physical size before changing backends.
            fig.set_layout_engine('none')
            for extension in ('png', *[ext for ext in style.formats if ext != 'png']):
                stream = io.BytesIO()
                if extension == 'tiff':
                    # PNG and TIFF use the same final Agg pixels, including
                    # per-pixel alpha. Keep that raster until both encoders finish.
                    raster.save(stream, format='TIFF', compression='tiff_lzw', dpi=(style.dpi, style.dpi))
                else:
                    options = {'dpi': style.dpi, 'transparent': style.transparent}
                    if extension in ('svg', 'png'):
                        options['metadata'] = {'Description': item.spec.id}
                    if extension == 'png':
                        options['pil_kwargs'] = {'compress_level': 1}
                    fig.savefig(stream, format=extension, **options)
                    if extension == 'png' and 'tiff' in style.formats:
                        from PIL import Image
                        pixels = fig.canvas.renderer.buffer_rgba()
                        raster = Image.frombuffer('RGBA', (pixels.shape[1], pixels.shape[0]), pixels, 'raw', 'RGBA', 0, 1)
                entries[f'figures/{item.spec.id}.{extension}'] = stream.getvalue()
        entries[f'specs/{item.spec.id}.geometry.json']=json.dumps(geometry,ensure_ascii=False).encode('utf-8')
        return entries, geometry
    finally:
        plt.close(fig)
