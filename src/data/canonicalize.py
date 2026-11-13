"""[B] CoPriva/CAMBER 원본을 읽고 연구에 사용할 항목을 정리한다.

N_total은 원본 항목 수, N_candidate는 추출한 후보 수다.
N_valid는 필수 정보가 갖춰진 후보 수이며, 검토를 마쳤거나 gold가 있다는 뜻은 아니다.
제외한 후보도 제외 사유와 함께 남긴다."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import urllib.request
import zipfile


def digest(content: bytes) -> str:
    return 'sha256:' + hashlib.sha256(content).hexdigest()


def download_source(destination: Path, spec: dict, *, offline: bool = False) -> None:
    """원본 파일을 내려받고 upstream Git tree의 hash와 비교한다."""
    destination.mkdir(parents=True, exist_ok=True)
    tree_path = destination / 'upstream_tree.json'
    repo, revision = spec['repository'], spec['revision']
    if offline:
        tree_bytes = tree_path.read_bytes()
    else:
        with urllib.request.urlopen(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1', timeout=90) as response:
            tree_bytes = response.read()
    tree = json.loads(tree_bytes)
    if tree.get('sha') != revision or tree.get('truncated'):
        raise ValueError('Incomplete or unexpected upstream tree')
    files = [item for item in tree['tree'] if item['type'] == 'blob' and
             (item['path'] in spec['files'] or
              (spec.get('data_prefix') and item['path'].startswith(spec['data_prefix']) and item['path'].endswith('.json')))]
    if not files or not set(spec['files']) <= {item['path'] for item in files}:
        raise ValueError('Configured upstream files missing')
    for item in files:
        path = PurePosixPath(item['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in item['path'] or ':' in item['path']:
            raise ValueError('Unsafe upstream path')

    def matches(item):
        path = destination / item['path']
        if not path.is_file():
            return False
        content = path.read_bytes()
        return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == item['sha']

    missing = [item for item in files if not matches(item)]
    if missing and offline:
        raise ValueError(f"Missing or modified raw files: {[item['path'] for item in missing]}")
    if missing:
        with urllib.request.urlopen(f'https://codeload.github.com/{repo}/zip/{revision}', timeout=180) as response:
            archive = response.read()
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            for item in missing:
                content = zipped.read(f"{repo.split('/')[-1]}-{revision}/{item['path']}")
                blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
                if blob != item['sha']:
                    raise ValueError(f"Upstream integrity mismatch: {item['path']}")
                path = destination / item['path']
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    if not offline:
        tree_path.write_bytes(tree_bytes)


def text(value):
    return value if isinstance(value, str) and value.strip() else None


def base_seed(source, component, record_id, parent, record, relative, file_hash, spec):
    group = f'{source}:{parent}' if parent else None
    return {
        'schema_version': 'seed-1.0.0', 'source_dataset': source,
        'source_component': component, 'source_record_id': record_id,
        'source_seed_id': parent, 'seed_id': f'{source}:{record_id}',
        'group_id': group, 'variant_group_id': group, 'data_kind': 'real_derived',
        'data_type': None, 'subject': None, 'sender': None, 'recipient': None,
        'purpose': None, 'policy_context': None, 'consent': None,
        'source_appropriateness_label': None, 'source_record': record,
        'field_provenance': {},
        'source_provenance': {'repository': spec['repository'], 'revision': spec['revision'],
                              'file': relative, 'file_hash': file_hash,
                              'record_hash': digest(json.dumps(record, sort_keys=True, ensure_ascii=False).encode())},
        'gold_decision': None, 'gold_state': None, 'evidence_trace': None,
        'gold_status': 'missing_external_annotation', 'human_review_status': 'pending',
        'review_reasons': [], 'exclusion_reasons': [],
    }


def screen(seed):
    """필수 정보가 있는지 확인한다. 원본 label로 후보를 고르지 않는다."""
    for field in ('data_type', 'subject', 'sender', 'recipient', 'purpose', 'policy_context'):
        if not seed[field]:
            seed['exclusion_reasons'].append(f'missing_{field}')
    if seed['group_id'] is None:
        seed['exclusion_reasons'].append('missing_upstream_group')
    seed['exclusion_reasons'] = sorted(set(seed['exclusion_reasons']))
    seed['status'] = 'excluded' if seed['exclusion_reasons'] else 'eligible_pending_review'
    seed['review_reasons'] = sorted(set(seed['review_reasons'] + ['verify_factual_preservation_before_generation']))
    seed['generation_ready'] = False
    return seed


def copriva_adapter(document, context_id, index, relative, file_hash, spec):
    record = document['specific_query_list'][index]
    seed = base_seed('CoPriva', 'specific_query', f'{context_id}:specific_query_list:{index}',
                     context_id, record, relative, file_hash, spec)
    seed['request_text'] = text(record.get('query'))
    seed['policy_context'] = record.get('policy_targets') or None
    seed['field_provenance']['policy_context'] = 'specific_query_list[].policy_targets (topics, not permission rules)'
    seed['context_reference'] = {'file': relative, 'pointer': '/meeting_transcripts'}
    seed['review_reasons'] = ['policy_targets_are_not_personal_data_types', 'requires_explicit_scope_annotation']
    return screen(seed)


def camber_adapter(record, component, relative, file_hash, spec):
    original_id, parent = text(record.get('example_id')), text(record.get('example_name'))
    record_id = f'{component}:{original_id}' if original_id else f'{component}:missing_id:{digest(json.dumps(record, sort_keys=True).encode())}'
    seed = base_seed('CAMBER', component, record_id, f'{component}:{parent}' if parent else None,
                     record, relative, file_hash, spec)
    seed['source_appropriateness_label'] = record.get('appropriateness_label')
    seed['upstream_example_id'], seed['upstream_example_name'] = original_id, parent
    if not original_id:
        seed['exclusion_reasons'].append('missing_source_record_id')
    try:
        scenario = ast.literal_eval(record['scenario'])
        if not isinstance(scenario, dict):
            raise ValueError('scenario must be a dictionary')
    except (ValueError, SyntaxError, TypeError, KeyError, RecursionError):
        seed['exclusion_reasons'].append('invalid_scenario')
        return screen(seed)
    if component == 'privacylensplus':
        mapping = {'data_type': 'data_type', 'subject': 'data_subject', 'sender': 'data_sender', 'recipient': 'data_recipient'}
        context_field = 'transmission_principle'
        seed['review_reasons'].append('transmission_principle_is_not_explicit_purpose_or_permission')
    elif component == 'confaideplus':
        mapping = {'data_type': 'detail', 'subject': 'subject_agent', 'sender': 'aware_agent', 'recipient': 'oblivious_agent', 'purpose': 'reveal_reason'}
        context_field = 'story'
        seed['review_reasons'].append('check_story_question_against_structured_agents')
    else:
        raise ValueError(f'Unsupported CAMBER component: {component}')
    seed['policy_context'] = text(scenario.get(context_field))
    seed['field_provenance']['policy_context'] = f'scenario.{context_field}'
    for target, source in mapping.items():
        seed[target] = text(scenario.get(source))
        if seed[target] is not None:
            seed['field_provenance'][target] = f'scenario.{source}'
    seed['review_reasons'].extend(['verify_personal_information_suitability', 'policy_and_consent_require_human_annotation'])
    return screen(seed)


def load_source(raw, source, spec):
    records = []
    if source == 'CoPriva':
        for path in sorted((raw / source / 'data').glob('*.json')):
            doc = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(doc.get('specific_query_list'), list):
                raise ValueError(f'Unexpected CoPriva structure: {path}')
            file_hash = digest(path.read_bytes())
            for index in range(len(doc['specific_query_list'])):
                records.append(copriva_adapter(doc, path.stem, index, f'data/{path.name}', file_hash, spec))
    elif source == 'CAMBER':
        for relative in spec['files']:
            if not relative.endswith('.csv'):
                continue
            path = raw / source / relative
            file_hash = digest(path.read_bytes())
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream, delimiter='\t')
                expected = {'example_id', 'example_name', 'expansion_type', 'expansion_dimension', 'appropriateness_label', 'scenario'}
                if set(reader.fieldnames or []) != expected:
                    raise ValueError(f'Unexpected CAMBER columns: {path}')
                for row in reader:
                    records.append(camber_adapter(row, path.stem, relative, file_hash, spec))
    else:
        raise ValueError(f'Unsupported source: {source}')
    if not records:
        raise ValueError(f'No records for {source}')
    ids = [record['seed_id'] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate source record IDs in {source}')
    return records


def select_candidates(records, count, random_seed):
    """특정 유형에 몰리지 않도록 component, expansion, context family별로 후보를 고른다.

    선정 비율은 추출한 후보를 기준으로 계산한다. 제외된 후보를 다른 항목으로 채우지 않는다."""
    strata = defaultdict(list)
    for record in records:
        key = (record['source_component'], record['source_record'].get('expansion_type', '')) if record['source_dataset'] == 'CAMBER' else (record['source_seed_id'].split('_')[0].rstrip('0123456789abcd'),)
        strata[key].append(record)
    rng = random.Random(random_seed)
    for key in sorted(strata):
        rng.shuffle(strata[key])
    candidates = []
    while len(candidates) < min(count, len(records)):
        for key in sorted(strata):
            if strata[key] and len(candidates) < count:
                candidates.append(strata[key].pop())
    return candidates


def run_pilot(raw, canonical_path, inventory_path, setup):
    if setup['screening_version'] != 'structural-v1':
        raise ValueError('Unsupported screening policy')
    count = setup['candidates_per_source']
    if not isinstance(count, int) or not 50 <= count <= 100:
        raise ValueError('G1 requires 50-100 candidates per source')
    candidates, summaries = [], []
    for source, spec in setup['sources'].items():
        records = load_source(raw, source, spec)
        if len(records) < count:
            raise ValueError(f'Insufficient pilot population for {source}')
        selected = select_candidates(records, count, setup['random_seed'])
        valid = sum(seed['status'] == 'eligible_pending_review' for seed in selected)
        summaries.append({'source_dataset': source, 'N_total': len(records), 'N_candidate': len(selected),
                          'N_valid': valid, 'conversion_yield': valid / len(selected),
                          'exclusion_reasons': dict(Counter(reason for seed in selected for reason in seed['exclusion_reasons'])),
                          'human_review_required': len(selected), 'human_reviewed': 0,
                          'annotation_agreement': None, 'gold_verified': 0})
        candidates.extend(selected)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(''.join(json.dumps(seed, ensure_ascii=False, sort_keys=True) + '\n' for seed in candidates), encoding='utf-8')
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['row_type', 'source_dataset', 'source_component', 'source_record_id', 'group_id', 'status', 'N_total', 'N_candidate', 'N_valid', 'conversion_yield', 'exclusion_reasons', 'review_reasons', 'human_review_required', 'human_reviewed', 'annotation_agreement', 'gold_verified', 'source_label', 'data_type', 'screening_version', 'sampling_seed', 'revision']
    with inventory_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({**summary, 'row_type': 'summary', 'exclusion_reasons': json.dumps(summary['exclusion_reasons'], sort_keys=True), 'screening_version': setup['screening_version'], 'sampling_seed': setup['random_seed'], 'revision': setup['sources'][summary['source_dataset']]['revision']})
        for seed in candidates:
            writer.writerow({**{key: seed[key] for key in ('source_dataset', 'source_component', 'source_record_id', 'group_id', 'status', 'data_type')}, 'row_type': 'candidate', 'exclusion_reasons': json.dumps(seed['exclusion_reasons']), 'review_reasons': json.dumps(seed['review_reasons']), 'human_review_required': True, 'human_reviewed': False, 'gold_verified': False, 'source_label': seed['source_appropriateness_label'], 'screening_version': setup['screening_version'], 'sampling_seed': setup['random_seed'], 'revision': seed['source_provenance']['revision']})
    return summaries
