"""Evidence-bound assessments shared by import, processing, transforms and export.

Production reads source facts and original arrays only. Expected outcomes and
reference computation arrays are deliberately absent from this module.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import io
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .ingest import InputError, parse_ascii
from .models import AnalysisRequest, Mapping

BASE = Path(__file__).resolve().parents[1] / 'data/examples_benchmark'
_catalog = None
_token_key = secrets.token_bytes(32)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


class Benchmark:
    def __init__(self):
        release_blob = (BASE/'release_manifest.json').read_bytes()
        self.release_hash = digest(release_blob)
        self.release = json.loads(release_blob)
        if self.release.get('status') != 'accepted_frozen' or not self.release.get('pipeline_implementation_allowed'):
            raise InputError('Benchmark has not passed independent acceptance.')
        self.verified = {}
        self.objects = {o['id']:o for o in self.read('objects.json')['objects']}
        self.files = self.read('inventory.json')['files']
        self.relationships = self.read('relationships.json')['relationships']
        self.version = self.release['version']
        self.file_by_hash = {}
        for f in self.files:
            self.file_by_hash.setdefault(f['sha256'], []).append(f)
        self.datasets = {}
        self.raw_ids = {}
        self.original_text_ids = {}
        self.unsafe_arrays = {}
        for obj in self.objects.values():
            if not obj['arrays']:
                continue
            data = self.read(obj['arrays']['path'])['data']
            raw = self._text(obj, data)
            self.raw_ids.setdefault(digest(raw.encode()), []).append(obj['id'])
            if obj['kind'] == 'standalone_spectrum' and not obj['path'].endswith('.gz'):
                self.original_text_ids.setdefault(obj['source_evidence'][0]['sha256'], []).append(obj['id'])
            # Prevent a known normalized/corrected/derived spectrum being laundered
            # through the generic mu endpoint by removing its project headers.
            if obj['processing_state'] not in ('mu', 'raw_channels') or obj.get('correction_history') or obj.get('metadata_conflicts'):
                self.unsafe_arrays.setdefault(digest(canonical(data)), []).append(obj['id'])
                self.unsafe_arrays.setdefault(digest(canonical([row[:2] for row in data])), []).append(obj['id'])

    def read(self, relative):
        path = BASE/relative
        if relative not in self.release['files']:
            raise InputError('File is not included in the frozen benchmark.')
        blob = path.read_bytes()
        if digest(blob) != self.release['files'][relative]:
            raise InputError('Frozen benchmark checksum mismatch: '+relative)
        return json.loads(blob)

    def assert_current(self):
        if digest((BASE/'release_manifest.json').read_bytes()) != self.release_hash:
            raise InputError('Benchmark release changed; restart and revalidate the deployment.')
        for rel in ('objects.json', 'inventory.json', 'relationships.json'):
            path = BASE/rel
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
            if self.verified.get(rel) != stamp:
                self.read(rel)
                self.verified[rel] = stamp

    @staticmethod
    def _text(obj, data):
        out = io.StringIO()
        # Explicit source identifiers do not supply absent experimental facts.
        out.write('# Benchmark.object: '+obj['id']+'\n')
        if obj['axis_unit']:
            axis = 'k' if obj['processing_state'] == 'chi' else 'energy'
            out.write('# Column.1: '+axis+' '+obj['axis_unit']+'\n')
        if obj['processing_state'] == 'mu':
            out.write('# Column.2: mu\n')
        elif obj['processing_state'] == 'chi':
            out.write('# Column.2: chi\n# Chi.weighting: unknown\n')
        out.write('# Processing.state: '+obj['processing_state']+'\n')
        np.savetxt(out, np.asarray(data), fmt='%.17g')
        return out.getvalue()

    def dataset(self, object_id):
        if object_id not in self.objects:
            raise InputError('Unknown benchmark object.')
        obj = self.objects[object_id]
        data = self.read(obj['arrays']['path'])['data'] if obj['arrays'] else []
        raw = self._text(obj, data) if data else ''
        mapping = None
        if obj['axis_unit'] == 'eV':
            if obj['signal_formula'] in ('stored_mu', 'stored_y'):
                mapping = Mapping().model_dump()
            elif obj['signal_formula'] == 'ln(column_2 / column_3)':
                mapping = Mapping(signal=2, reference=1, mode='transmission').model_dump()
            elif obj['signal_formula'] == 'column_9 / column_4':
                mapping = Mapping(signal=8, reference=3, mode='fluorescence').model_dump()
        assessment = source_assessment(obj)
        display_header = [line for line in raw.splitlines() if line.startswith('#')]
        for key, field in (('bkg_z','Element.symbol'), ('fft_edge','Element.edge')):
            value = obj['metadata'].get(key)
            if value and str(value).strip() not in ('None','0',''):
                display_header.append('# '+field+': '+str(value))
        return {'object_id':obj['id'], 'filename':obj['path'], 'label':obj['label'],
                'raw_text':raw, 'kind':'chi' if obj['processing_state']=='chi' else ('mu' if data else 'supporting'),
                'processing_state':obj['processing_state'], 'role':obj['role'],
                'legacy_parameters':obj['metadata'],
                'rows':len(data), 'columns':len(data[0]) if data else 0,
                'labels':obj['arrays']['columns'] if data else [],
                'assessment':assessment, 'chi_data':data if obj['processing_state']=='chi' else None,
                'simple_input':{'mapping':mapping, 'mapping_reason':'accepted_source_rule' if mapping else 'source_rule_or_review_required',
                                'chi_mapping':None, 'table':data, 'header':display_header}}


def get_benchmark():
    global _catalog
    if _catalog is None:
        _catalog = Benchmark()
    _catalog.assert_current()
    return _catalog


def dataset_for_object(object_id):
    return get_benchmark().dataset(object_id)


def source_assessment(obj):
    reason = None
    if obj['kind'] not in ('athena_object','artemis_snapshot','standalone_spectrum'):
        reason = 'not_experimental_mu_object'
    elif obj['example'] == 'Fe foil':
        reason = 'intentional_miscalibration_requires_reference_rule'
    elif obj.get('conflicts') or obj.get('metadata_conflicts'):
        reason = 'source_metadata_conflict'
    elif obj.get('correction_history'):
        reason = 'self_absorption_already_applied'
    elif obj['processing_state'] not in ('mu','raw_channels'):
        reason = 'processing_state_not_unprocessed_mu'
    elif obj['axis_unit'] != 'eV':
        reason = 'axis_unit_unresolved'
    return {'schema_version':1, 'object_id':obj['id'], 'benchmark_version':'1.0.0',
            'provenance':{'file_id':obj['file_id'], 'path':obj['path'], 'member':obj['member'],
                          'group_id':obj.get('group_id'), 'evaluation_group':obj.get('evaluation_group', 'example:'+obj['example']),
                          'evidence':obj['source_evidence']},
            'role':obj['role'], 'processing_state':obj['processing_state'],
            'confirmed_stage':obj['confirmed_stage'], 'candidate_ceiling':obj['candidate_ceiling'],
            'stage':obj['confirmed_stage'], 'preview_eligible':bool(obj['arrays']),
            'analysis_eligible':reason is None,
            'permitted_operations':['preview'] if obj['arrays'] else ['inspect_metadata'],
            'allowed_operations':['preview'] if obj['arrays'] else ['inspect_metadata'],
            'completed_stages':[], 'completed_operations':[], 'adjustments':[],
            'stop_reasons':[{'code':reason,'message':reason.replace('_',' ')}] if reason else [],
            'prerequisites':obj['prerequisites'], 'saved_recipe':{'status':'historical_only',
                'issues':obj['quality'].get('saved_recipe_issues', []), 'automatically_applied':False},
            'claim_scope':'Evidence-backed computational processing; no phase or calibration certification.'}


def import_benchmark(blob, filename):
    bench = get_benchmark()
    files = bench.file_by_hash.get(digest(blob))
    if not files:
        return None
    file_ids = {f['id'] for f in files}
    datasets = [bench.dataset(o['id']) for o in bench.objects.values() if o['file_id'] in file_ids]
    return {'filename':filename, 'sha256':digest(blob), 'format':'accepted_benchmark_source',
            'benchmark_version':bench.version, 'datasets':datasets,
            'warnings':['Saved recipes are historical evidence. Processing uses reviewed source rules.']}


def attach_import_assessment(dataset):
    """Bind importer observations to exact text without trusting later client fields."""
    fields = {key:dataset.get(key) for key in ('processing_state','role','analysis_blockers')}
    fields['raw_sha256'] = digest(dataset['raw_text'].encode())
    encoded = base64.urlsafe_b64encode(canonical(fields)).decode()
    dataset['assessment_token'] = encoded+'.'+hmac.new(_token_key, encoded.encode(), 'sha256').hexdigest()
    dataset['assessment'] = generic_assessment(dataset['raw_text'], fields)
    if fields.get('processing_state') == 'chi':
        from types import SimpleNamespace
        try:
            table, _ = parse_ascii(dataset['raw_text'])
            dataset['assessment'] = assess_chi(SimpleNamespace(raw_text=dataset['raw_text'], object_id=None,
                assessment_token=dataset['assessment_token'], kmin=max(2, float(table[0, 0]) + .5),
                kmax=min(12, float(table[-1, 0]) - .5), kweight=2, dk=1))
        except InputError as exc:
            dataset['assessment']['stop_reasons'] = [{'code':'chi_requires_review','message':str(exc)}]
    return dataset


def token_context(request):
    token = getattr(request, 'assessment_token', None)
    if not token:
        return {}
    try:
        payload, signature = token.rsplit('.',1)
        if not hmac.compare_digest(signature, hmac.new(_token_key, payload.encode(),'sha256').hexdigest()):
            raise ValueError('signature')
        fields = json.loads(base64.urlsafe_b64decode(payload))
        if fields['raw_sha256'] != digest(request.raw_text.encode()):
            raise ValueError('source')
        return fields
    except (ValueError, KeyError, TypeError) as exc:
        raise InputError('Import assessment no longer matches this input. Reimport before processing.') from exc


def resolve_object(request):
    """Production processes uploaded text only; historical object identities are retired."""
    if getattr(request, 'object_id', None):
        raise InputError('Built-in object identities are no longer supported. Upload the original spectrum.')
    return None


def generic_assessment(raw, context=None):
    import re
    context = context or {}
    stops = list(context.get('analysis_blockers') or [])
    state = context.get('processing_state') or 'user_declared_mu'
    for line in raw.splitlines():
        match = re.search(r'Processing\.state:\s*(\S+)', line, re.I)
        if match:
            state = match[1].lower()
        if re.search(r'self.absorption.*correct|correct.*self.absorption', line, re.I):
            stops.append('self_absorption_already_applied')
    if state in ('normalized', 'norm', 'corrected', 'corrected_mu','corrected_chi',
                 'chi', 'fit', 'residual', 'derived_fit','difference','background','r_space','q_space','chi_fit_table'):
        stops.append('processing_state_requires_separate_workflow')
    preview = False
    try:
        data, header = parse_ascii(raw, preview=True)
        from .automation import declared_columns
        _, _, column_error = declared_columns(header, data.shape[1])
        if column_error:
            stops.append('conflicting_column_declarations' if column_error == 'ambiguous_column_declarations' else column_error)
        preview = bool(len(data))
        if not np.isfinite(data).all(): stops.append('nonfinite_input')

    except InputError:
        stops.append('unreadable_numeric_input')
    return {'schema_version':1, 'object_id':None, 'role':context.get('role') or 'user_declared_spectrum',
            'processing_state':state, 'confirmed_stage':'S1', 'candidate_ceiling':'conditional', 'stage':'S1',
            'preview_eligible':preview, 'analysis_eligible':not stops,
            'provenance':{'source_sha256':digest(raw.encode()), 'assertion_status':'user_declaration_or_literal_header'},
            'permitted_operations':['preview'], 'allowed_operations':['preview'],
            'completed_stages':[], 'completed_operations':[], 'adjustments':[],
            'stop_reasons':[{'code':s, 'message':s.replace('_',' ')} for s in dict.fromkeys(stops)]}


def assess_request(request):
    obj = resolve_object(request)
    if obj:
        return source_assessment(obj)
    return generic_assessment(request.raw_text, token_context(request))


def _envelope(request, processed, assessment):
    from .science import software_identity
    rec, arrays = processed['record'], processed['arrays']
    assessment.update(stage=rec['achieved_stage'], completed_operations=rec['completed_operations'],
        completed_stages=[s for s in ('S2','S3','S4') if int(s[1])<=int(rec['achieved_stage'][1])],
        adjustments=rec['adjustments'], stop_reasons=[{'code':s,'message':s.replace('_',' ')} for s in rec['stop_reasons']],
        permitted_operations=list(dict.fromkeys(assessment['permitted_operations']+rec['completed_operations'])),
        analysis_eligible=bool(rec['completed_operations']))
    assessment['allowed_operations'] = assessment['permitted_operations']
    normalized_request = AnalysisRequest.model_validate(request.model_dump())
    recipe = normalized_request.model_dump(exclude={'raw_text', 'assessment_token'})
    identity = software_identity()
    source = {'sha256':digest(request.raw_text.encode()), 'filename':request.filename,
              'provenance':assessment['provenance']}
    run_id = digest(canonical({'recipe':recipe, 'source':source, 'software':identity, 'assessment':assessment}))
    e, k, r = arrays.get('energy',[]), arrays.get('k',[]), arrays.get('r',[])
    p = rec['parameters']; checks = rec['checks']
    return {'schema_version':2, 'run_id':run_id, 'created_utc':datetime.now(timezone.utc).isoformat(),
            'request':recipe, 'source':source, 'software':identity, 'arrays':arrays,
            'assessment':assessment, 'processing_record':rec, 'provenance':assessment['provenance'],
            'resolved_calls':p, 'warnings':[s['message'] for s in assessment['stop_reasons']],
            'metrics':{'e0_eV':p.get('e0'), 'edge_step':checks.get('larch_edge_step'),
                'k_available':checks.get('measured_k_max'), 'k_grid_max':k[-1] if k else None,
                'energy_min':e[0] if e else None, 'energy_max':e[-1] if e else None,
                'r_grid_spacing':r[1]-r[0] if len(r)>1 else None},
            'array_units':{'energy':'eV','mu':'relative absorbance','norm':'dimensionless','chi':'dimensionless',
                           'k':'angstrom^-1','r':'angstrom, phase uncorrected'}}


def analyze_assessed(request):
    obj = resolve_object(request)
    if obj:
        from .evidence_processing import process_object
        bench = get_benchmark()
        data = bench.read(obj['arrays']['path'])['data'] if obj['arrays'] else None
        return _envelope(request, process_object(obj, bench.objects, data), source_assessment(obj))
    assessment = assess_request(request)
    if assessment['stop_reasons']:
        raise InputError('; '.join(s['message'] for s in assessment['stop_reasons']))
    from .automation import infer_mapping
    inferred, mapping_reason = infer_mapping(request.raw_text)
    if mapping_reason in ('ambiguous_column_declarations','invalid_column_declarations'):
        raise InputError('Conflicting column declarations require review.')
    if 'mapping' not in request.model_fields_set and inferred is None:
        raise InputError('Explicit columns and energy units are required; default numbers do not establish metadata.')
    from .ingest import construct_mu
    selected_e, selected_mu, _ = construct_mu(request.raw_text, request.mapping, 0)

    if request.processing_profile == 'adaptive':
        from .evidence_processing import process_declared_mu
        return _envelope(request, process_declared_mu(selected_e, selected_mu, {}), assessment)
    from .science import analyze
    # Keep the established successful-response fields for ordinary manual inputs.
    try:
        result = analyze(request)
    except InputError as exc:
        from .ingest import construct_mu
        from .evidence_processing import process_declared_mu
        e, mu, _ = construct_mu(request.raw_text, request.mapping, 0)
        processed = process_declared_mu(e, mu, request.parameters)
        if not processed['arrays']:
            raise exc
        return _envelope(request, processed, assessment)
    assessment.update(stage='S4', completed_stages=['S2','S3','S4'],
        completed_operations=['construct_or_retain_mu','normalize','extract_exafs','fourier_transform'],
        permitted_operations=['preview','construct_or_retain_mu','normalize','extract_exafs','fourier_transform'])
    assessment['allowed_operations'] = assessment['permitted_operations']
    result['assessment'] = assessment
    result['provenance'] = assessment['provenance']
    return result


def adaptive_partial(request, reason, metadata=None):
    manual = request.model_copy(update={'processing_profile':'adaptive'})
    if metadata and metadata.get('detected_edge_eV') is not None:
        from .ingest import construct_mu
        from .evidence_processing import process_declared_mu
        # Retain the validated local edge when only later EXAFS coverage failed.
        e0 = metadata['detected_edge_eV']
        manual = manual.model_copy(update={'parameters': manual.parameters.model_copy(update={'e0':e0})})
        energy, mu, _ = construct_mu(manual.raw_text, manual.mapping)
        processed = process_declared_mu(energy, mu, {'e0':e0})
        result = _envelope(manual, processed, assess_request(manual))
    else:
        result = analyze_assessed(manual)
    return {'status':'partial' if result['arrays'] else 'needs_review', 'result':result,
            'request':{**result['request'],'raw_text':request.raw_text}, 'reason':reason,
            'assessment':result['assessment'], 'automation':{'method':'adaptive_stage_profile',
                'scientific_metadata':metadata or {},
                'selected_run_id':result['run_id'],'cost':{'evaluations':1},'candidate_count':1}}


def assess_chi(request):
    """Existing chi requires an explicit definition and unit; never normalize it."""
    import re
    obj = resolve_object(request)
    if obj:
        # The accepted v1 dataset has no admitted chi weighting definitions.
        # Preserve its source state; the FT endpoint cannot bypass this boundary.
        raise InputError('chi_definition_or_weighting_requires_review: the source has no accepted unweighted chi definition.')
    context = token_context(request)
    if context.get('role') in ('fit','residual','derived_fit') or context.get('analysis_blockers'):
        raise InputError('Derived or previously corrected data require a reviewed chi definition before FT.')
    table, header = parse_ascii(request.raw_text)
    text = '\n'.join(header)
    if not re.search(r'Column\.1:\s*k\s+(inverse_angstrom|1/angstrom|angstrom\^-1|a\^-1)\b', text, re.I):
        raise InputError('chi_axis_unit_unresolved: declare k in inverse_angstrom.')
    if not re.search(r'Column\.2:\s*chi\s*$', text, re.I|re.M):
        raise InputError('chi_definition_unresolved: column 2 must explicitly be unweighted chi.')
    weights = re.findall(r'Chi\.weighting:\s*(\S+)', text, re.I)
    if any(v.lower() not in ('0','unweighted') for v in weights):
        raise InputError('chi_weighting_unresolved: preweighted or unknown chi cannot be transformed automatically.')
    if not np.isfinite(table).all() or np.any(np.diff(table[:,0]) <= 0):
        raise InputError('chi_grid_requires_review: k must be finite and strictly increasing.')
    if request.kmin < request.dk/2 or request.kmax-request.kmin <= request.dk or request.kmax+request.dk/2 > table[-1,0]:
        raise InputError('The complete FT taper must stay inside measured k coverage.')
    return {'schema_version':1,'object_id':None,'role':'user_declared_chi', 'processing_state':'chi',
            'confirmed_stage':'S1','candidate_ceiling':'S4','stage':'S1','preview_eligible':True,'analysis_eligible':True,
            'permitted_operations':['preview','fourier_transform'],'allowed_operations':['preview','fourier_transform'],
            'completed_stages':[],'completed_operations':[],'adjustments':[],'stop_reasons':[],
            'provenance':{'source_sha256':digest(request.raw_text.encode()),'evidence':'literal Column.1 and Column.2 declarations',
                          'chi_definition':'unweighted chi; explicitly labeled column, not verified experimental history'}}


def automatic_assessed(request):
    obj = resolve_object(request)
    context = token_context(request)
    assessment = source_assessment(obj) if obj else generic_assessment(request.raw_text, context)
    if not obj:
        if assessment['stop_reasons']:
            return {'status':'needs_review','result':None,'request':None,'assessment':assessment,
                    'reason':'; '.join(s['message'] for s in assessment['stop_reasons'])}
        return None
    manual = AnalysisRequest(object_id=obj['id'], raw_text=request.raw_text, filename=request.filename,
                             sample=request.sample, notes=request.notes)
    result = analyze_assessed(manual)
    return {'status':'completed' if result['assessment']['stage']=='S4' else 'partial' if result['arrays'] else 'needs_review',
            'result':result, 'request':{**result['request'],'raw_text':request.raw_text},
            'assessment':result['assessment'], 'automation':{'method':'accepted_source_rules',
            'selected_run_id':result['run_id'],'benchmark_version':get_benchmark().version,
            'cost':{'evaluations':1}, 'candidate_count':1}}
