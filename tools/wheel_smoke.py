"""Start an installed WebXAS wheel, import measured ZnSe and reopen its saved project."""
import argparse
from hashlib import sha256
from io import BytesIO
from importlib.metadata import version
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile

import httpx
from fetch_test_inputs import fetch, ZNSE, ROOT


def check_server(client, raw):
    health = client.get('/api/health').raise_for_status().json()
    assert health['application_version'] == version('xasweb-pilot')
    for route in ('/', '/auto', '/pro', '/knowledge', '/methods'):
        assert client.get(route).raise_for_status().text
    manifest = client.get('/validation/manifest.json').raise_for_status().json()
    assert len(manifest['figures']) == 8
    for record in manifest['figures']:
        for name in record['files'].values():
            assert client.get('/validation/'+name).raise_for_status().content
    kb = client.get('/api/knowledge').raise_for_status().json()
    assert kb['total'] >= 4
    assert client.post('/api/pro/projects/demo').raise_for_status().json()['datasets']
    upload = {'file': ('znse_zn_xafs.001',raw,'text/plain')}
    assert client.post('/api/import',files=upload).raise_for_status().json()['datasets']
    project = client.post('/api/pro/projects',json={'title':'Measured ZnSe'}).raise_for_status().json()
    route = '/api/pro/projects/'+project['id']
    project = client.post(route+'/datasets',params={'expected_revision':project['revision']},files=upload).raise_for_status().json()
    dataset = project['datasets'][0]
    assert dataset['metadata']['original_upload']['sha256'] == sha256(raw).hexdigest()
    archive = client.post(route+'/export',params={'expected_revision':project['revision']}).raise_for_status().content
    with ZipFile(BytesIO(archive)) as z:
        assert z.testzip() is None
        assert any(z.read(n)==raw for n in z.namelist())
    reopened = client.post('/api/pro/projects/import',files={'file':('znse.zip',archive,'application/zip')}).raise_for_status().json()
    assert reopened['datasets']==project['datasets'] and reopened['title']==project['title']
    return {'version':health['application_version'],'engine':health['version'],'znse_sha256':sha256(raw).hexdigest(),
            'gallery_records':len(manifest['figures']),'zip_bytes':len(archive),'reopened_datasets':len(reopened['datasets'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT/'.public-fixtures')
    raw = fetch(*ZNSE, parser.parse_args().cache)
    with tempfile.TemporaryDirectory() as directory:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port = sock.getsockname()[1]
        env = dict(os.environ)
        env.pop('PYTHONPATH',None)
        env['XASWEB_RUNTIME_ROOT'] = directory
        env['PYTHONUTF8'] = '1'
        with tempfile.TemporaryFile() as output:
            server = subprocess.Popen([sys.executable,'-m','uvicorn','xasweb.app:app',
                '--host','127.0.0.1','--port',str(port)],cwd=directory,env=env,stdout=output,stderr=output)
            try:
                with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=45,trust_env=False) as client:
                    for _ in range(180):
                        if server.poll() is not None:
                            output.seek(0)
                            raise RuntimeError(output.read().decode(errors='replace')[-3000:])
                        try:
                            if client.get('/api/health').status_code==200:
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(.5)
                    print(json.dumps(check_server(client,raw),indent=2))
            finally:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
if __name__ == '__main__':
    main()
