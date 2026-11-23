"""Explicit professional processing, preserving all original input bytes."""
import io
import re
from uuid import uuid4
import numpy as np

from ..models import AnalysisRequest, Parameters
from ..science import analyze
from ..ingest import InputError, parse_ascii, construct_mu
from .contracts import ArrayRef, Dataset, NoiseSpec, ProcessingRevision, Project
from .projects import ProjectError, ProjectStore


def save_array(store, project_id, values):
    array=np.asarray(values)
    dtype='bool' if array.dtype.kind=='b' else 'complex128' if array.dtype.kind=='c' else 'float64'
    array=np.asarray(array,dtype=dtype)
    if not np.isfinite(array).all():
        raise ProjectError('nonfinite_array','处理数组包含非有限值。')
    stream=io.BytesIO();np.save(stream,array,allow_pickle=False)
    ref=store.put_asset(project_id,stream.getvalue(),'array.npy')
    return ArrayRef(**ref.model_dump(mode='json'),dtype=dtype,shape=list(array.shape))


def load_array(store,project_id,ref):
    array=np.load(io.BytesIO(store.read_asset(project_id,ref)),allow_pickle=False)
    if list(array.shape)!=ref.shape or array.dtype.name!=ref.dtype or not np.isfinite(array).all():
        raise ProjectError('array_contract','数值资源的形状、类型或有限值检查失败。')
    return array


def validate_noise(store,project_id,noise,processing_id,k_size=None):
    noise=NoiseSpec.model_validate(noise.model_dump(mode='json') if hasattr(noise,'model_dump') else noise)
    if noise.unit!='chi':
        raise ProjectError('noise_unit','EXAFS 噪声必须在已处理 chi(k) 单位下声明。')
    if noise.processing_revision_id not in (None,processing_id):
        raise ProjectError('noise_revision','噪声属于另一个处理版本。')
    if noise.source=='explicit_array':
        array=load_array(store,project_id,noise.array)
        if array.ndim!=1 or np.any(array<=0) or (k_size is not None and len(array)!=k_size):
            raise ProjectError('invalid_noise','噪声数组必须为正并与当前 k 网格等长。')
    value=noise.model_dump(mode='json');value['processing_revision_id']=processing_id
    return NoiseSpec.model_validate(value)


def process_dataset(store,project_id,dataset_id,parameters=None,mask=None,calibration_id=None,noise=None):
    project=store.read(project_id)
    dataset=next((x for x in project.datasets if x.id==dataset_id),None)
    if dataset is None:
        raise ProjectError('dataset_not_found','找不到待处理数据。',dataset_id)
    if dataset.kind=='support':
        raise ProjectError('not_spectrum','支持材料不能直接作为光谱处理。',dataset_id)
    store.read_asset(project_id,dataset.raw_file)
    raw=dataset.raw_text
    if raw is None:
        raw=store.read_asset(project_id,dataset.raw_file).decode('utf-8-sig')
    from ..import_formats import describe
    declaration=describe(raw,dataset.filename)
    state=declaration.get('processing_state','unknown')
    if state in ('normalized','difference','background','chi_fit_table'):
        raise ProjectError('derived_input','归一化谱、差谱、背景或历史拟合表不能作为原始 mu(E)/chi(k) 进入本流程。',dataset_id)
    if state in ('corrected_mu','corrected_chi') and not dataset.metadata.get('external_processing_note'):
        raise ProjectError('processing_provenance','外部已校正谱需要记录校正方法与适用条件。',dataset_id)
    for field,pattern in [('element',r'Element\.symbol:\s*([A-Z][a-z]?)\b'),('edge',r'Element\.edge:\s*(K|L[123]|M[1-5])\b')]:
        found=re.search(pattern,raw)
        if found and getattr(dataset,field) not in (None,found[1]) and not dataset.metadata.get('header_identity_override'):
            raise ProjectError('header_identity_conflict','元素/吸收边与文件头声明冲突，请修正或记录覆盖理由。',dataset_id)
    if dataset.kind=='chi' and 'chi_weighting_requires_verification' in dataset.metadata.get('import_analysis_blockers',[]) and dataset.metadata.get('input_kweight_confirmed') is not True:
        raise ProjectError('chi_weight_unconfirmed','请明确确认输入 chi(k) 的 k 权重。',dataset_id)
    table,header=parse_ascii(raw)
    selected=np.ones(len(table),dtype=bool) if mask is None else np.asarray(mask)
    if selected.dtype.kind!='b' or selected.shape!=(len(table),):
        raise ProjectError('invalid_mask','掩码必须为与原始点数等长的布尔数组。')
    if selected.sum()<40:
        raise ProjectError('insufficient_points','掩码后至少需要 40 个数据点。')
    mapping=dataset.mapping
    columns=[mapping.energy,mapping.signal]+([mapping.reference] if mapping.mode!='mu' else [])
    if len(set(columns))!=len(columns) or max(columns)>=table.shape[1]:
        raise ProjectError('invalid_mapping','列映射无效。',dataset_id)
    if not np.isfinite(table).all() or np.any(np.diff(table[:,mapping.energy])<=0):
        raise ProjectError('invalid_grid','原始横坐标必须严格递增，且数值必须有限；请显式修正输入。',dataset_id)
    filtered=table[selected]
    user_parameters=Parameters.model_validate(parameters.model_dump() if hasattr(parameters,'model_dump') else parameters or {})
    resolved=user_parameters.model_dump()
    calibration=None
    if calibration_id:
        calibration=next((x for x in project.calibrations if x.id==calibration_id),None)
        if calibration is None or calibration.kind!='energy':
            raise ProjectError('invalid_calibration','未找到可用能量校准。',calibration_id)
        if dataset.element!=calibration.element or dataset.edge!=calibration.edge:
            raise ProjectError('calibration_edge','校准元素和吸收边与数据不一致。',dataset_id)
        resolved['energy_shift']+=calibration.value
    warnings=[]
    if dataset.kind=='mu':
        text='\n'.join(header)+'\n'+'\n'.join(' '.join(format(x,'.17g') for x in row) for row in filtered)
        result=analyze(AnalysisRequest(filename=dataset.filename,raw_text=text,mapping=mapping,parameters=Parameters.model_validate(resolved)))
        arrays=result['arrays'];warnings=list(result['warnings'])
        calls=[{'operation':name,'arguments':value} for name,value in result['resolved_calls'].items()]
        metrics=result['metrics']
    else:
        if calibration is not None or user_parameters.energy_shift!=0:
            raise ProjectError('chi_energy_shift','chi(k) 输入不接受能量偏移；请从原始 mu(E) 重新提取。')
        from larch import Group
        from larch.xafs import xftf
        k=filtered[:,mapping.energy];chi=filtered[:,mapping.signal]
        if k[0]<0 or resolved['kmax']+resolved['dk']/2>k[-1]:
            raise ProjectError('chi_coverage','chi(k) 未覆盖完整拟合窗口。')
        weight=dataset.metadata.get('input_kweight',0)
        if isinstance(weight,bool) or weight not in (0,1,2,3):
            raise ProjectError('chi_weight','输入 k 权重声明无效。')
        weight=int(weight)
        if weight:
            if np.any(k<=0):
                raise ProjectError('chi_weight','带权 chi(k) 包含 k=0，无法无损去权。')
            chi=chi/k**weight
        settings={'kmin':resolved['kmin'],'kmax':resolved['kmax'],'kweight':resolved['kweight'],
                  'dk':resolved['dk'],'window':'hanning','kstep':0.05,'nfft':2048,'rmax_out':6}
        group=Group();xftf(k,chi,group=group,**settings)
        arrays={'k':k,'chi':chi,**{name:getattr(group,name) for name in ['r','chir_re','chir_im','chir_mag']}}
        calls=[{'operation':'xftf','arguments':settings}]
        metrics={}
    processing_id='pp_'+uuid4().hex
    chosen_noise=noise or NoiseSpec(source='engine_estimate',estimation_range=[15.0,30.0],assumptions=['white_noise_high_R'])
    chosen_noise=validate_noise(store,project_id,chosen_noise,processing_id,len(arrays['k']))
    if any(value in ('unknown','uncorrected') for value in dataset.correction_states.values()):
        warnings.append('实验修正状态未全部验证，定量解释仍受测量条件限制。')
    return ProcessingRevision(id=processing_id,dataset_id=dataset_id,input_revision=project.revision,
        parameters=user_parameters,calibration_id=calibration_id,
        arrays={name:save_array(store,project_id,value) for name,value in arrays.items()},
        mask=save_array(store,project_id,selected),noise=chosen_noise,
        engine_versions={'xraylarch':'2026.3.1'},resolved_calls=calls,warnings=warnings,
        metadata={'metrics':metrics,'effective_energy_shift_eV':resolved['energy_shift'],'source_sha256':dataset.raw_file.sha256,
                  'input_mapping':dataset.mapping.model_dump(mode='json'),'input_kind':dataset.kind,
                  'input_kweight':dataset.metadata.get('input_kweight',0)})


def commit_processing(store,project_id,processing,expected_revision):
    project=store.read(project_id)
    value=project.model_dump(mode='json');value['processing'].append(processing.model_dump(mode='json'))
    return store.save(Project.model_validate(value),expected_revision)


def merge_scan_arrays(energies,signals,variances=None):
    """Merge only on the measured intersection, returning variance and scatter."""
    if len(energies)<2 or len(energies)!=len(signals):
        raise ProjectError('scan_count','至少需要两个对应的扫描。')
    xs=[np.asarray(x,dtype=float) for x in energies];ys=[np.asarray(y,dtype=float) for y in signals]
    for x,y in zip(xs,ys):
        if x.ndim!=1 or x.shape!=y.shape or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x)<=0):
            raise ProjectError('scan_grid','扫描横坐标或信号无效。')
    lo=max(x[0] for x in xs);hi=min(x[-1] for x in xs)
    grid=xs[0][(xs[0]>=lo)&(xs[0]<=hi)]
    if len(grid)<2:
        raise ProjectError('no_overlap','扫描没有足够的共同测量范围。')
    values=np.vstack([np.interp(grid,x,y) for x,y in zip(xs,ys)])
    if variances is None:
        weights=np.ones_like(values)
        propagated=None
    else:
        if len(variances)!=len(xs):
            raise ProjectError('scan_variance','每条扫描均需声明方差。')
        vs=[]
        for x,var in zip(xs,variances):
            var=np.asarray(var,dtype=float)
            if var.ndim==0:var=np.full(x.shape,float(var))
            if var.shape!=x.shape or not np.isfinite(var).all() or np.any(var<=0):
                raise ProjectError('scan_variance','扫描方差必须有限、为正且与扫描等长。')
            # Variance of linearly interpolated independent samples uses squared weights.
            idx=np.clip(np.searchsorted(x,grid,side='right'),1,len(x)-1)
            fraction=(grid-x[idx-1])/(x[idx]-x[idx-1])
            vs.append((1-fraction)**2*var[idx-1]+fraction**2*var[idx])
        weights=1/np.vstack(vs);propagated=1/weights.sum(axis=0)
    mean=np.sum(weights*values,axis=0)/weights.sum(axis=0)
    return {'energy':grid,'mu':mean,'variance':propagated,'scatter':np.std(values,axis=0,ddof=1),
            'weights':weights,'method':'inverse_variance' if variances is not None else 'arithmetic_mean'}


def merge_project_scans(store,project_id,dataset_ids,expected_revision,energy_shifts=None,variances=None):
    project=store.read(project_id)
    if project.revision!=expected_revision:
        raise ProjectError('revision_conflict','项目已更新。')
    if len(set(dataset_ids))!=len(dataset_ids) or len(dataset_ids)<2:
        raise ProjectError('duplicate_scan','至少选择两个不同的重复扫描。')
    by_id={d.id:d for d in project.datasets}
    if any(i not in by_id for i in dataset_ids):
        raise ProjectError('dataset_not_found','某个扫描不存在。')
    scans=[by_id[i] for i in dataset_ids]
    if len({d.raw_file.sha256 for d in scans})!=len(scans):
        raise ProjectError('duplicate_scan','同一个原始测量的副本不能按独立重复扫描合并。')
    first=scans[0]
    if not first.sample_id or not first.element or not first.edge:
        raise ProjectError('scan_metadata','合并前需确认样品、元素、吸收边与状态元数据。')
    for dataset in scans:
        if dataset.kind!='mu' or (dataset.sample_id,dataset.element,dataset.edge,dataset.conditions,dataset.mapping.mode,dataset.correction_states)!=(first.sample_id,first.element,first.edge,first.conditions,first.mapping.mode,first.correction_states):
            raise ProjectError('incompatible_scans','只有同一样品、元素/边、状态、测量模式和修正状态的 mu(E) 扫描可合并。')
    shifts=energy_shifts or {}
    if set(shifts)-set(dataset_ids) or any(not np.isfinite(v) or abs(v)>1000 for v in shifts.values()):
        raise ProjectError('invalid_energy_shifts','扫描偏移必须有限且属于选中扫描。')
    xs=[];ys=[]
    for dataset in scans:
        store.read_asset(project_id,dataset.raw_file)
        text=dataset.raw_text or store.read_asset(project_id,dataset.raw_file).decode('utf-8-sig')
        x,y,_=construct_mu(text,dataset.mapping,shifts.get(dataset.id,0))
        xs.append(x);ys.append(y)
    merged=merge_scan_arrays(xs,ys,variances)
    raw='\n'.join(f'{x:.17g} {y:.17g}' for x,y in zip(merged['energy'],merged['mu']))+'\n'
    filename='merged-scans.dat'
    arrays={name:save_array(store,project_id,merged[name]) for name in ('energy','mu','scatter','weights')}
    if merged['variance'] is not None:arrays['variance']=save_array(store,project_id,merged['variance'])
    new=Dataset(id='ds_'+uuid4().hex,filename=filename,raw_file=store.put_asset(project_id,raw.encode(),filename),
        raw_text=raw,sample_id=first.sample_id,element=first.element,edge=first.edge,conditions=first.conditions,
        lineage=sorted(set(dataset_ids+[parent for dataset in scans for parent in dataset.lineage])),arrays=arrays,
        correction_states=first.correction_states,metadata={'merge_method':merged['method'],'energy_shifts_eV':shifts,
            'variance_basis':'pointwise independent original samples; interpolation can introduce covariance',
            'scan_count':len(scans),'source_hashes':[d.raw_file.sha256 for d in scans]})
    value=project.model_dump(mode='json');value['datasets'].append(new.model_dump(mode='json'))
    return store.save(Project.model_validate(value),expected_revision)


def run_job(payload,context):
    context.heartbeat('preprocessing',0.1)
    store=ProjectStore();project_id=payload['project_id']
    project=store.read(project_id)
    if project.revision!=payload['expected_revision']:
        raise ProjectError('revision_conflict','任务提交后项目已更新。',project_id)
    processing=process_dataset(store,project_id,payload['dataset_id'],payload.get('parameters'),
        payload.get('mask'),payload.get('calibration_id'),payload.get('noise'))
    if processing.input_revision!=payload['expected_revision']:
        raise ProjectError('revision_conflict','读取处理数据时项目已更新，请重新提交。',project_id)
    context.heartbeat('preprocessing',1.0)
    return {'processing':processing.model_dump(mode='json'),'project_id':project_id,'expected_revision':project.revision}
