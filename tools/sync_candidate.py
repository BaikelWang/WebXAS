"""Update reviewed source paths from a candidate's package/app directory.

Review new files and add them to release-files.txt before importing a candidate.
Public gallery content, notices, sample project, documentation and fixtures are maintained separately.
"""
import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def change(text, old, new):
    if old not in text:
        raise ValueError(f"Candidate layout changed: {old!r}")
    return text.replace(old, new)


def public_source(rel, code, version):
    if rel == 'xasweb/__init__.py':
        code = re.sub(r'__version__ = "[^"]+"', f'__version__ = "{version}"', code)
        code = change(code, 'Path(__file__).resolve().parents[1] / ".runtime"', 'Path(os.environ.get("XASWEB_RUNTIME_ROOT", Path.cwd() / ".runtime")).resolve()')
        code = code.replace('in this source checkout', 'in the working runtime directory')
    if rel in ('xasweb/pro/projects.py','xasweb/pro/jobs.py','xasweb/local_structure.py'):
        code = change(code, 'from pathlib import Path', 'from xasweb import _runtime\nfrom pathlib import Path')
    if rel == 'xasweb/pro/projects.py':
        code = change(code, "Path(__file__).resolve().parents[2] / '.runtime/projects'", "_runtime / 'projects'")
    if rel == 'xasweb/pro/jobs.py':
        code = change(code, 'Path(__file__).resolve().parents[3] / ".runtime" / "jobs"', "_runtime / 'jobs'")
    if rel == 'xasweb/local_structure.py':
        for name in ('path-library', 'feff-screening-cache'):
            code = change(code, f"Path(__file__).resolve().parents[1]/'.runtime/{name}'", f"_runtime / '{name}'")
    if rel == 'xasweb/pro/api.py':
        code = change(code, "Path(__file__).resolve().parents[2]/'data/pro-demo.zip'", "Path(__file__).resolve().parents[1]/'data/pro-demo.zip'")
    if rel.startswith('xasweb/static/') and rel.endswith('.html'):
        code = re.sub(r'(?<=\?v=)\d+\.\d+\.\d+(?:rc\d+)?', version, code)
    if rel == 'xasweb/app.py':
        code = change(code, 'os.environ.get("WEBXAS_VALIDATION_ROOT")', 'os.environ.get("WEBXAS_VALIDATION_ROOT", static.parent / "gallery")')
    return '\n'.join(line.rstrip() for line in code.rstrip().splitlines()) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('application', type=Path)
    parser.add_argument('version')
    args = parser.parse_args()
    if not re.fullmatch(r'\d+\.\d+\.\d+(rc\d+)?', args.version):
        parser.error('Expected a PEP 440 release or release candidate.')
    for rel in (ROOT/'tools/release-files.txt').read_text().splitlines():
        destination = ROOT / ('package/app' if rel.startswith('tests/') else '') / rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError(rel)
        code = public_source(rel, (args.application/rel).read_text(encoding='utf-8'), args.version)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(code, encoding='utf-8', newline='\n')
    meta = ROOT/'pyproject.toml'
    meta.write_text(re.sub(r'(?m)^version = "[^"]+"', f'version = "{args.version}"', meta.read_text()), encoding='utf-8')

if __name__ == '__main__':
    main()
