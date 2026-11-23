from pathlib import Path
from unittest.mock import patch
import hashlib
import subprocess

import pytest

from xasweb.pro.paths import (generate_paths,import_paths,parse_path,prepare_structure,
                            professional_input,run_job)
from xasweb.pro.projects import ProjectError,ProjectStore

BENCH=Path(__file__).resolve().parents[4]/'benchmarks'
CERU=BENCH/'inputs'/'parseq_xas'/'parseq_XAS'/'data'/'CeRu2'/'feff'/'CeRu2'
SIMPLE_INPUT=b'''TITLE finite Cu chain
HOLE 1 1.0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0 0 0 0 Cu
2.55 0 0 1 Cu
5.10 0 0 1 Cu
END
'''


@pytest.fixture
def setup(tmp_path):
    store=ProjectStore(tmp_path/'projects')
    project=store.create('Paths')
    return store,project


def test_real_ss_ms_and_high_shell_are_preserved(setup):
    store,p=setup
    paths=import_paths(store,p.id,[('feff0001.dat',(CERU/'feff0001.dat').read_bytes()),
                                  ('feff0003.dat',(CERU/'feff0003.dat').read_bytes()),
                                  ('feff0005.dat',(CERU/'feff0005.dat').read_bytes())],
                       element='Ce',edge='L3',absorber_site='Ce_unique')
    assert [x.nleg for x in paths]==[2,3,2]
    assert paths[1].degen==36 and paths[1].reff==pytest.approx(4.4588)
    assert paths[2].reff>4.8
    assert all(len(x.geometry)==x.nleg and x.k_max>=19 and x.k_min==0 for x in paths)
    assert paths[1].metadata['distance_meaning'].endswith('not a single bond for MS')
    assert store.read_asset(p.id,paths[1].source_file)==(CERU/'feff0003.dat').read_bytes()


def test_same_name_content_and_calculation_identity_do_not_collide(setup):
    store,p=setup
    kwargs=dict(element='Ce',edge='L3',absorber_site='Ce_unique')
    raw=(CERU/'feff0001.dat').read_bytes()
    first=parse_path(store,p.id,raw,'feff0001.dat',**kwargs)
    same=parse_path(store,p.id,raw,'renamed.dat',**kwargs)
    other=parse_path(store,p.id,(CERU/'feff0003.dat').read_bytes(),'feff0001.dat',**kwargs)
    new_program=parse_path(store,p.id,raw,'feff0001.dat',program_sha256='a'*64,**kwargs)
    new_input=parse_path(store,p.id,raw,'feff0001.dat',feff_input=store.put_asset(p.id,b'TITLE other','feff.inp'),**kwargs)
    assert first.id==same.id
    assert len({first.id,other.id,new_program.id,new_input.id})==4
    assert first.source_file.path!=other.source_file.path


@pytest.mark.parametrize('element,edge,code',[('Cu','L3','path_element_mismatch'),('Ce','K','path_edge_mismatch')])
def test_declared_absorber_and_edge_are_checked(setup,element,edge,code):
    store,p=setup
    with pytest.raises(ProjectError) as err:
        parse_path(store,p.id,(CERU/'feff0003.dat').read_bytes(),'feff0003.dat',element=element,edge=edge,absorber_site='site')
    assert err.value.code==code


def test_duplicate_and_incomplete_paths_rejected(setup):
    store,p=setup
    raw=(CERU/'feff0001.dat').read_bytes()
    with pytest.raises(ProjectError,match='重复'):
        import_paths(store,p.id,[('a.dat',raw),('b.dat',raw)],element='Ce',edge='L3',absorber_site='site')
    with pytest.raises(ProjectError) as err:
        parse_path(store,p.id,b'not a path','x.dat',element='Ce',edge='L3',absorber_site='site')
    assert err.value.code=='invalid_feff_path'


def test_professional_profile_retains_geometry_and_allows_ms(setup):
    store,p=setup
    text,settings=professional_input(SIMPLE_INPUT.decode(),'feff','Cu','K',{'nleg':4,'rmax':5.5})
    assert '\nNLEG 4\n' in text and '\nNLEG 2\n' not in text
    assert '\nRMAX 5.500000\n' in text
    structure=prepare_structure(store,p.id,SIMPLE_INPUT,'chain.inp',source_format='feff',element='Cu',edge='K',absorber_site='center0')
    assert store.read_asset(p.id,structure.source_file)==SIMPLE_INPUT
    bad=SIMPLE_INPUT.replace(b'2.55 0 0 1 Cu',b'2.55 0 0 0 Cu')
    with pytest.raises(ProjectError,match='一个吸收中心'):
        professional_input(bad.decode(),'feff','Cu','K')
    with pytest.raises(ProjectError):
        professional_input(SIMPLE_INPUT.decode(),'feff','Cu','K',{'nleg':1})


def test_generate_real_feff_and_cache_identity(setup,tmp_path):
    store,p=setup
    structure=prepare_structure(store,p.id,SIMPLE_INPUT,'chain.inp',source_format='feff',element='Cu',edge='K',absorber_site='center0')
    output=generate_paths(store,p.id,structure,tmp_path/'job')
    assert output['paths'] and any(x.nleg>2 for x in output['paths'])
    assert any(x.reff>4.5 for x in output['paths'])
    assert output['identity']['nleg']==4
    assert len(output['logs'])==2
    assert output['identity']['program_sha256']==output['paths'][0].feff_program_sha256
    with patch('xasweb.pro.paths.subprocess.run',side_effect=AssertionError('cache should avoid FEFF')):
        cached=generate_paths(store,p.id,structure,tmp_path/'second-job')
    assert [x.id for x in cached['paths']]==[x.id for x in output['paths']]
    assert cached['cache_key']==output['cache_key']
    assert [x.sha256 for x in cached['logs']]==[x.sha256 for x in output['logs']]
    assert all(len(x.metadata['generation_logs'])==2 for x in cached['paths'])
    changed=generate_paths(store,p.id,structure,tmp_path/'third-job',{'nleg':2,'rmax':5.5})
    assert changed['cache_key']!=output['cache_key']
    assert all(x.nleg==2 for x in changed['paths'])


def test_timeout_and_job_revision_do_not_commit_project(setup,tmp_path):
    store,p=setup
    structure=prepare_structure(store,p.id,SIMPLE_INPUT,'chain.inp',source_format='feff',element='Cu',edge='K',absorber_site='center0')
    with patch('xasweb.pro.paths.subprocess.run',side_effect=subprocess.TimeoutExpired('feff',1,output=b'partial',stderr=b'log')):
        with pytest.raises(ProjectError) as err:
            generate_paths(store,p.id,structure,tmp_path/'timeout')
    assert err.value.code=='feff_timeout'
    assert store.read(p.id).revision==p.revision
    logs=list((tmp_path/'timeout').rglob('stdout.log'))
    assert len(logs)==1 and logs[0].read_bytes()==b'partial'
    class Context:
        work_dir=tmp_path/'job'
        def heartbeat(self,*args):pass
    with patch('xasweb.pro.paths.ProjectStore',return_value=store):
        with pytest.raises(ProjectError) as err:
            run_job({'project_id':p.id,'expected_revision':99,'structure_id':structure.id},Context())
    assert err.value.code=='revision_conflict'


def test_cross_project_assets_and_corrupted_cache(setup,tmp_path):
    store,p=setup
    second=store.create('Other')
    kwargs=dict(source_format='feff',element='Cu',edge='K',absorber_site='center0')
    structure=prepare_structure(store,p.id,SIMPLE_INPUT,'chain.inp',**kwargs)
    out=generate_paths(store,p.id,structure,tmp_path/'first')
    asset=out['paths'][0].source_file
    with pytest.raises(ProjectError) as err:
        store.read_asset(second.id,asset)
    assert err.value.code=='missing_asset'
    cache=store.folder(p.id)/'feff-cache'/out['cache_key']
    sample=next(cache.glob('feff*.dat'))
    sample.write_bytes(b'corrupt')
    again=generate_paths(store,p.id,structure,tmp_path/'again')
    assert [x.id for x in again['paths']]==[x.id for x in out['paths']]
    assert hashlib.sha256(sample.read_bytes()).hexdigest()!=hashlib.sha256(b'corrupt').hexdigest()
