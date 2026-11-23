"""Content-identified FEFF path assets for the professional workflow.

No first-peak selection or single-scattering filter is applied here. Execution
objects are transient; original bytes and calculation identity are portable.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
from uuid import uuid4

import numpy as np

from ..edges import normalize_edge
from ..ingest import InputError
from ..local_structure import atomic_input, sanitize_feff
from .contracts import AssetRef, GeometryAtom, PathAsset, Structure
from .projects import ProjectError, ProjectStore, atomic_write, canonical

MAX_PATH_BYTES = 5 * 1024 * 1024
MAX_PATH_COUNT = 512


def _error(code, message, object_id=None):
    raise ProjectError(code, message, object_id)


def _profile(value=None):
    value=dict(value or {})
    if set(value)-{'nleg','rmax','timeout_seconds'}:
        _error('invalid_feff_profile','专业 FEFF 配置包含不支持的字段。')
    nleg=value.get('nleg',4)
    radius=value.get('rmax',5.5)
    timeout=value.get('timeout_seconds',75.)
    if type(nleg) is not int or not 2 <= nleg <= 6:
        _error('invalid_feff_profile','NLEG 必须为 2 到 6 的整数。')
    if isinstance(radius,bool) or not isinstance(radius,(float,int)) or not math.isfinite(radius) or not .7 < radius <= 5.5:
        _error('invalid_feff_profile','当前专业 FEFF 生成半径支持 0.7 到 5.5 Å。')
    if isinstance(timeout,bool) or not isinstance(timeout,(float,int)) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        _error('invalid_feff_profile','FEFF 超时必须为 0 到 300 秒之间的有限数值。')
    return {'nleg':nleg,'rmax':float(radius),'timeout_seconds':float(timeout)}


def professional_input(text,source_format,element,edge,profile=None):
    """Reuse validated bounded geometry, replacing the QFS-only NLEG profile."""
    settings=_profile(profile)
    if source_format not in ('cif','xyz','feff'):
        _error('invalid_structure_format','结构格式必须为 cif、xyz 或 feff。')
    try:
        if source_format in ('cif','xyz'):
            # This deliberately preserves rejection of mixed occupancy and
            # unresolved inequivalent absorbing sites in the existing parser.
            text=atomic_input(text,source_format,element,normalize_edge(edge))
        text=sanitize_feff(text,element,normalize_edge(edge))
    except (InputError,ValueError,IndexError,KeyError) as exc:
        _error('invalid_structure',str(exc))
    text,count=re.subn(r'(?m)^NLEG\s+2\s*$',f"NLEG {settings['nleg']}",text)
    if count != 1:
        _error('invalid_structure','未找到规范化后的 FEFF 路径阶数配置。')
    text,count=re.subn(r'(?m)^RMAX\s+[0-9.]+\s*$',f"RMAX {settings['rmax']:.6f}",text)
    if count != 1:
        _error('invalid_structure','未找到规范化后的 FEFF 路径半径配置。')
    return text,settings


def prepare_structure(store,project_id,raw_bytes,filename,*,source_format,element,edge,absorber_site,metadata=None):
    """Validate a definite absorber geometry without changing the project revision."""
    if not isinstance(raw_bytes,bytes) or not raw_bytes or len(raw_bytes)>MAX_PATH_BYTES:
        _error('invalid_structure','结构文件为空或超过 5 MB。')
    try:
        text=raw_bytes.decode('utf-8-sig')
    except UnicodeDecodeError:
        _error('invalid_structure','结构文件必须使用 UTF-8 文本。')
    if not isinstance(absorber_site,str) or not absorber_site.strip():
        _error('absorber_site_required','必须明确吸收位点标识。')
    normalized,_=professional_input(text,source_format,element,edge)
    ref=store.put_asset(project_id,raw_bytes,filename)
    ident=hashlib.sha256(canonical({'source_sha256':ref.sha256,'format':source_format,
        'element':element,'edge':normalize_edge(edge),'site':absorber_site})).hexdigest()
    info=dict(metadata or {})
    info.update({'source_filename':filename,'validation':'bounded_single_absorber_geometry',
                 'normalized_input_sha256':hashlib.sha256(normalized.encode('ascii')).hexdigest(),
                 'site_selection':'source-explicit XYZ absorber_index or validated unique inequivalent absorber site'})
    return Structure(id='struct_'+ident[:32],source_file=ref,source_format=source_format,
                     absorber_element=element,edge=normalize_edge(edge),absorber_site=absorber_site,metadata=info)


def parse_path(store,project_id,raw_bytes,filename,*,element,edge,absorber_site,
               structure_id=None,feff_input=None,program_sha256=None,metadata=None):
    """Read and validate a real FEFF path; identity includes its generation context."""
    from larch.xafs import feffpath
    from xraydb import atomic_number,atomic_symbol
    if not isinstance(raw_bytes,bytes) or not raw_bytes or len(raw_bytes)>MAX_PATH_BYTES:
        _error('invalid_feff_path','FEFF 路径为空或超过 5 MB。')
    if not absorber_site or not isinstance(absorber_site,str):
        _error('absorber_site_required','路径必须具有明确吸收位点标识。')
    edge=normalize_edge(edge)
    try:
        raw_bytes.decode('utf-8-sig')
    except UnicodeDecodeError:
        _error('invalid_feff_path','FEFF 路径必须为 UTF-8 文本。')
    source=store.put_asset(project_id,raw_bytes,filename)
    path=store.folder(project_id).joinpath(*source.path.split('/'))
    try:
        parsed=feffpath(str(path))
        native=parsed._feffdat
        k=np.asarray(native.k,dtype=float)
        if len(k)<20 or not np.isfinite(k).all() or np.any(np.diff(k)<=0):
            _error('invalid_feff_path','FEFF 路径 k 网格不完整或非递增。')
        for name in ('real_phc','mag_feff','pha_feff','red_fact','lam','rep'):
            array=np.asarray(getattr(native,name),dtype=float)
            if array.shape!=k.shape or not np.isfinite(array).all():
                _error('invalid_feff_path','FEFF 振幅或相位数组不完整。')
        if not parsed.geom or parsed.geom[0][1] != atomic_number(element):
            _error('path_element_mismatch','FEFF 路径吸收元素与声明不符。')
        if normalize_edge(native.shell)!=edge:
            _error('path_edge_mismatch','FEFF 路径吸收边与声明不符。')
        nleg=int(parsed.nleg);degen=float(parsed.degen);reff=float(parsed.reff)
        if nleg<2 or nleg!=parsed.nleg or not math.isfinite(degen) or degen<=0 or not math.isfinite(reff) or reff<=0:
            _error('invalid_feff_path','路径阶数、简并度或有效半程长度无效。')
        # FEFF geometry labels may be tags such as "absorbing", not symbols.
        geometry=[GeometryAtom(element=atomic_symbol(int(atom[1])),xyz=[float(x) for x in atom[4:7]]) for atom in parsed.geom]
        if len(geometry)!=nleg:
            _error('invalid_feff_path','路径几何原子序列与 NLEG 不一致。')
    except ProjectError:
        raise
    except Exception as exc:
        _error('invalid_feff_path','无法读取 FEFF 路径：'+str(exc)[:400])
    input_ref=AssetRef.model_validate(feff_input) if isinstance(feff_input,dict) else feff_input
    if input_ref is not None:
        store.read_asset(project_id,input_ref)
    calculation={'path_sha256':source.sha256,'feff_input_sha256':input_ref.sha256 if input_ref else None,
                 'program_sha256':program_sha256,'element':element,'edge':edge,'site':absorber_site,
                 'structure_id':structure_id}
    ident=hashlib.sha256(canonical(calculation)).hexdigest()
    info=dict(metadata or {})
    info.update({'source_filename':filename,'feff_header':str(native.title),
                 'geometry_labels':[str(atom[0]) for atom in parsed.geom],
                 'calculation_identity':calculation,'identity_sha256':ident,
                 'amplitude_meaning':'FEFF path degeneracy retained; no CN inferred',
                 'distance_meaning':'effective half path length; not a single bond for MS',
                 'scattering_type':'single' if nleg==2 else 'multiple'})
    return PathAsset(id='path_'+ident[:32],source_file=source,feff_input=input_ref,
        feff_program_sha256=program_sha256,structure_id=structure_id,
        absorber_element=element,edge=edge,absorber_site=absorber_site,nleg=nleg,reff=reff,degen=degen,
        geometry=geometry,k_min=float(k[0]),k_max=float(k[-1]),
        engine_versions={'xraylarch':importlib.metadata.version('xraylarch'),'feff_file':str(native.version)},metadata=info)


def import_paths(store,project_id,files,**identity):
    files=list(files)
    if not files or len(files)>MAX_PATH_COUNT:
        _error('path_count','每次需导入 1 到 512 条路径。')
    result=[]
    for filename,raw in files:
        path=parse_path(store,project_id,raw,filename,**identity)
        if path.id in {p.id for p in result}:
            _error('duplicate_path','同一条路径在本次导入中重复。',path.id)
        result.append(path)
    return result


def feff_executable():
    import larch
    import platform
    systems={'Windows':'win64','Linux':'linux64','Darwin':'darwin64'}
    system=systems.get(platform.system())
    if not system:
        _error('feff_unavailable','当前平台未配置 FEFF6L。')
    executable=Path(larch.__file__).parent/'bin'/system/('feff6l.exe' if os.name=='nt' else 'feff6l')
    if not executable.is_file():
        _error('feff_unavailable','未找到 FEFF6L 可执行程序。')
    return executable.resolve()


def _cache_read(folder,identity=None):
    try:
        record=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        if identity is not None and record.get('identity')!=identity:
            return None
        entries=record['paths']
        if not entries or len(entries)>MAX_PATH_COUNT:
            return None
        if set(entries)!={p.name for p in folder.glob('feff[0-9][0-9][0-9][0-9].dat')}:
            return None
        values=[]
        for name,digest in sorted(entries.items()):
            if not re.fullmatch(r'feff\d{4}\.dat',name):
                return None
            raw=(folder/name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=digest:
                return None
            values.append((name,raw))
        return values
    except (OSError,ValueError,KeyError,TypeError):
        return None


def generate_paths(store,project_id,structure,work_dir,profile=None,context=None):
    """Run bounded FEFF6L in a job-owned directory, returning assets only."""
    if isinstance(structure,dict):
        structure=Structure.model_validate(structure)
    settings=_profile(profile)
    raw=store.read_asset(project_id,structure.source_file)
    text,settings=professional_input(raw.decode('utf-8-sig'),structure.source_format,
        structure.absorber_element,structure.edge,settings)
    input_bytes=text.encode('ascii')
    executable=feff_executable()
    executable_hash=hashlib.sha256(executable.read_bytes()).hexdigest()
    identity={'input_sha256':hashlib.sha256(input_bytes).hexdigest(),'program_sha256':executable_hash,
              'element':structure.absorber_element,'edge':structure.edge,'absorber_site':structure.absorber_site,
              'structure_sha256':structure.source_file.sha256,'nleg':settings['nleg'],'rmax':settings['rmax']}
    digest=hashlib.sha256(canonical(identity)).hexdigest()
    cache=store.folder(project_id)/'feff-cache'/digest
    input_ref=store.put_asset(project_id,input_bytes,'feff.inp')
    cached=_cache_read(cache,identity)
    logs=[]
    if cached is not None:
        try:
            for ref in json.loads((cache/'complete.json').read_text(encoding='utf-8')).get('logs',[]):
                ref=AssetRef.model_validate(ref)
                store.read_asset(project_id,ref)
                logs.append(ref)
        except (ProjectError,ValueError,TypeError):
            cached=None;logs=[]
    cache_hit=cached is not None
    if context:context.heartbeat('feff_prepared',.15)
    if cached is None:
        target=Path(work_dir)/('feff-'+uuid4().hex)
        target.mkdir(parents=True,exist_ok=False)
        (target/'feff.inp').write_bytes(input_bytes)
        if context:context.heartbeat('feff_running',.35)
        try:
            run=subprocess.run([str(executable)],cwd=target,capture_output=True,
                timeout=settings['timeout_seconds'],creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except subprocess.TimeoutExpired as exc:
            (target/'stdout.log').write_bytes(exc.stdout or b'')
            (target/'stderr.log').write_bytes(exc.stderr or b'')
            _error('feff_timeout','FEFF 计算达到超时；输入和日志保留在任务目录。')
        (target/'stdout.log').write_bytes(run.stdout)
        (target/'stderr.log').write_bytes(run.stderr)
        logs=[store.put_asset(project_id,run.stdout,'feff-stdout.log'),store.put_asset(project_id,run.stderr,'feff-stderr.log')]
        files=sorted(target.glob('feff[0-9][0-9][0-9][0-9].dat'))
        if run.returncode or not files:
            _error('feff_failed','FEFF 未生成可用路径；请检查任务日志。')
        if len(files)>MAX_PATH_COUNT:
            _error('path_count','FEFF 生成超过 512 条路径，请减小半径或阶数。')
        cached=[(p.name,p.read_bytes()) for p in files]
    if context:context.heartbeat('feff_paths_parsing',.8)
    paths=import_paths(store,project_id,cached,element=structure.absorber_element,edge=structure.edge,
        absorber_site=structure.absorber_site,structure_id=structure.id,feff_input=input_ref,
        program_sha256=executable_hash,metadata={'generation_identity':identity,'cache_key':digest,
                                               'generation_logs':[r.model_dump(mode='json') for r in logs]})
    # Publish cache only after all generated paths passed the same parser.
    if not cache_hit:
        cache.mkdir(parents=True,exist_ok=True)
        for name,payload in cached:atomic_write(cache/name,payload)
        atomic_write(cache/'complete.json',canonical({'identity':identity,'logs':[r.model_dump(mode='json') for r in logs],
            'paths':{name:hashlib.sha256(payload).hexdigest() for name,payload in cached}}))
    if context:context.heartbeat('feff_paths_ready',1.)
    return {'paths':paths,'feff_input':input_ref,'logs':logs,'identity':identity,'cache_key':digest}


def run_job(payload,context):
    store=ProjectStore()
    project_id=payload['project_id']
    project=store.read(project_id)
    if project.revision!=payload['expected_revision']:
        _error('revision_conflict','任务提交后项目已更新。',project_id)
    structure=next((s for s in project.structures if s.id==payload['structure_id']),None)
    if structure is None:
        _error('structure_not_found','项目中没有所选结构。',payload['structure_id'])
    output=generate_paths(store,project_id,structure,context.work_dir,payload.get('profile'),context)
    return {'project_id':project_id,'expected_revision':project.revision,
            'paths':[p.model_dump(mode='json') for p in output['paths']],
            'feff_input':output['feff_input'].model_dump(mode='json'),
            'logs':[r.model_dump(mode='json') for r in output['logs']],
            'identity':output['identity'],'cache_key':output['cache_key']}
