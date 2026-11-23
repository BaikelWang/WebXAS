"""Derived views with the exact transform, interpolation and window recorded."""
import numpy as np
from larch import Group
from larch.xafs import cauchy_wavelet, ftwindow
from .ingest import InputError


def spectral_views(arrays, parameters):
    if len(arrays.get('k',[]))<2 or len(arrays.get('chi',[]))<2:
        return {'status':'unavailable','reason':'当前阶段尚无 χ(k)。'}
    k=np.asarray(arrays['k'],dtype=float);chi=np.asarray(arrays['chi'],dtype=float)
    if len(k)!=len(chi) or not np.isfinite([k,chi]).all() or np.any(np.diff(k)<=0):
        raise InputError('Derived views require finite, increasing k.')
    lo=max(float(k[0]),float(parameters.get('kmin',2)))
    hi=min(float(k[-1]),float(parameters.get('kmax',12)))
    view=dict(status='completed',k=k.tolist(),weighted={str(w):(chi*k**w).tolist() for w in range(4)},
              r=arrays.get('r',[]),r_components={name:arrays.get('chir_'+name,[]) for name in ('mag','re','im')},
              parameters={'kmin':lo,'kmax':hi,'kweight':parameters.get('kweight',2)})
    if hi-lo<=1:
        view['wavelet']={'status':'unavailable','reason':'有效 k 区间过短。'};return view
    step=.05
    # Cauchy requires a grid starting at zero. Interpolate only measured data,
    # zero-pad elsewhere, and taper to the SAME usable experimental window.
    grid=np.arange(0,np.floor(min(float(k[-1]),hi+.5)/step)+1)*step
    uniform=np.interp(grid,k,chi,left=0,right=0)
    uniform*=ftwindow(grid,xmin=lo,xmax=hi,dx=1,window='hanning')
    group=Group();cauchy_wavelet(grid,uniform,group=group,kweight=2,rmax_out=6,nfft=2048)
    cols=np.where((grid>=lo)&(grid<=hi))[0][::2]
    rows=np.where(group.wcauchy_r>=.3)[0][::2]
    mag=group.wcauchy_mag[np.ix_(rows,cols)]
    if not np.isfinite(mag).all():raise InputError('Wavelet returned nonfinite values.')
    view['wavelet']=dict(status='completed',k=grid[cols].tolist(),r=group.wcauchy_r[rows].tolist(),magnitude=mag.tolist(),
        parameters=dict(method='Larch continuous Cauchy',kweight=2,kstep=step,nfft=2048,window='hanning',dk=1,kmin=lo,kmax=hi,rmax_out=6),
        notes=['线性插值到 Δk=0.05 Å⁻¹；测量范围外零填充并施加与有效区间一致的 Hanning 窗。',
               '显示 k²χ(k) 的小波模值；边缘区域受窗和零填充影响。R 未校正相移，小波图不独立证明散射元素。'],
        reference='Munoz, Argoul & Farges (2003), American Mineralogist 88, 694–700',
        method_url='https://xraypy.github.io/xraylarch/xafs_wavelets.html')
    return view
