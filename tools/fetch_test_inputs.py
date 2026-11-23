"""Acquire pinned upstream inputs for tests; verify every download before use."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LARCH = 'https://raw.githubusercontent.com/xraypy/xraylarch/860d8a690c81eefb0e61dee4ca3703ef4b67e93d/examples/xafsdata/'
PARSEQ = 'https://raw.githubusercontent.com/kklmn/ParSeq-XAS/b60f1ac63f20d1ee792223f2ec2e06aadb0cd124/parseq_XAS/data/CeRu2/feff/CeRu2/'
ZNSE = (LARCH+'znse_zn_xafs.001','9a60a114bf1da7da96490c09deb2cd58eadf0b2d02fa14e7e42db6eb2fadf4b8')
FEFF = {'feff0001.dat':'8f8c08104b7c129d5fa78d168e3a9a860e808f36afb77a274c27532b693f7dcd',
        'feff0003.dat':'2dff2d683da2ede3cec73a2be2a0d769e8c80381540f9d0d0fa762a49b4c1e36',
        'feff0005.dat':'edbd31799f5754571cd2167d9bf72fd71051c69e7c57dab6df5f42419546b88d'}


def fetch(url, digest, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / url.split('/')[-1]
    if path.exists():
        data = path.read_bytes()
    else:
        with urlopen(url, timeout=90) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Input checksum mismatch for {url}')
    if not path.exists():
        path.write_bytes(data)
    return data


def prepare(cache):
    fetch(*ZNSE, cache)
    dest = ROOT/'benchmarks/inputs'
    pa = dest/'parseq_xas/parseq_XAS/data/CeRu2/feff/CeRu2'
    pa.mkdir(parents=True, exist_ok=True)
    for name, digest in FEFF.items():
        (pa/name).write_bytes(fetch(PARSEQ+name, digest, cache))
    foil_data = [
        ('Cu','cu_50k.xmu','50K','f8fc32296f8045ab39ac738416f8d0121672d17d291d61658fcd1531b193f882'),
        ('Fe','fe.060','300K','92f698464e3f27390ed554ce65ebb3aa7734cce0844fc1cd7fd7c8d04aae85b4'),
    ]
    for element, name, temp, digest in foil_data:
        text = fetch(LARCH+name,digest,cache).decode('ascii')
        lines = []
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                lines.append([float(value) for value in fields])
            except ValueError:
                continue
        data = np.asarray(lines)
        mu = data[:,1] if element=='Cu' else np.log(data[:,1]/data[:,2])
        tag = element+'_Kedge'
        folder = dest/'delivered_demo'/tag
        folder.mkdir(parents=True, exist_ok=True)
        np.savetxt(folder/(tag+'_metal_foil_'+temp+'.dat'),np.column_stack((data[:,0],mu)),fmt='%.10g')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'.public-fixtures')
    prepare(parser.parse_args().cache)
