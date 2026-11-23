"""Create schema examples only: hashes and spectra here are placeholders."""
from pathlib import Path

from xasweb.pro.contracts import (
    AssetRef, Dataset, DatasetBinding, ModelRevision, NoiseSpec, Parameter,
    ParameterGraph, PathAsset, PathBinding, ProcessingRevision, Project,
)


def build(kind):
    elements=["Cu"] if kind=="single" else (["Cu","Zn"] if kind=="cross_edge" else ["Cu","Cu"])
    datasets=[];processing=[];paths=[];bindings=[];parameters=[]
    parameters.append(Parameter(id="shared_R",unit="angstrom",kind="free",value=2.6,minimum=2.4,maximum=2.8,physical_basis="Contract example of a declared common atom-pair distance; not an experimental claim."))
    for index,element in enumerate(elements):
        did=f"dataset_{index}"
        pid=f"path_{index}"
        pp=f"processed_{index}"
        binding=f"binding_{index}"
        digest=str(index+1)*64
        raw=AssetRef(path=f"raw/contract_{index}.txt",sha256=digest,byte_size=1)
        datasets.append(Dataset(id=did,raw_file=raw,filename=f"contract_{index}.txt",kind="chi",element=element,edge="K",sample_id="sample_example",conditions={"temperature_K":100+index*100},metadata={"contract_example":True,"not_experimental_data":True}))
        processing.append(ProcessingRevision(id=pp,dataset_id=did))
        paths.append(PathAsset(id=pid,source_file=AssetRef(path=f"paths/contract_{index}.dat",sha256=digest,byte_size=1),absorber_element=element,edge="K",absorber_site="site0",nleg=2,reff=2.55+index*0.02,degen=12,k_max=20,metadata={"contract_example":True}))
        for name,unit,value,vary in [("s02","1",0.9,False),("e0","eV",0.,True),("sigma2","angstrom2",0.003,True),("reff","angstrom",2.55+index*0.02,False)]:
            parameters.append(Parameter(id=f"{name}_{index}",unit=unit,kind="free" if vary else "fixed",value=value,scope="dataset",owner_id=did))
        bindings.append(DatasetBinding(dataset_id=did,processing_revision_id=pp,paths=[PathBinding(id=binding,path_id=pid,expressions={"s02":f"s02_{index}","e0":f"e0_{index}","deltar":f"shared_R - reff_{index}","sigma2":f"sigma2_{index}"})],noise=NoiseSpec(source="engine_estimate",assumptions=["White-noise estimate must be checked when real spectra are supplied."])))
    model=ModelRevision(id="example_model",title=f"{kind} contract example",bindings=bindings,parameters=ParameterGraph(nodes=parameters),notes="Schema contract only. This project contains no executable FEFF paths or measured spectra.")
    return Project(title=f"{kind} schema example",datasets=datasets,processing=processing,paths=paths,models=[model],recomputable=False,metadata={"contract_example":True,"not_experimental_data":True,"placeholder_assets":True})


if __name__=="__main__":
    for name in ("single","joint","cross_edge"):
        Path(__file__).with_name(name+".json").write_text(build(name).model_dump_json(indent=2)+"\n",encoding="utf-8")
