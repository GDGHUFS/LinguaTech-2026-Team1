"""Failure labels are checker results on actual Target responses only."""
from collections import Counter
from .contracts import (Blocked, digest, paths, read_json, read_jsonl, write_json,
                        write_jsonl, gate)
from .features import previous_record, valid_vector
from .representation import SampleCache
from .checker import OfficialChecker
from .splitting import audit_split


def failure_label(results):
    if not results or any(r['result'] == 'UNKNOWN' for r in results):
        return None
    if any(r['result'] == 'FAIL' for r in results):
        return 1
    if all(r['result'] == 'PASS' for r in results):
        return 0
    return None


def label_one(row, response, previous, checker):
    record = {**row, **(response or {}), 'constraint_results': [],
              'failure_label': None, 'eligible': False,
              'previous_representation_key': None, 'exclusion_reason': None}
    if not response or response.get('status') != 'OK':
        record['exclusion_reason'] = 'TARGET_NOT_GENERATED' if not response else response['status']
        return record
    if response.get('truncated'):
        record['exclusion_reason'] = 'OUTPUT_TRUNCATED'
        return record
    results = [checker.evaluate(c, response['target_response']) for c in row['active_constraints']]
    record['constraint_results'] = results
    record['failure_label'] = failure_label(results)
    if record['failure_label'] is None:
        record['exclusion_reason'] = 'NO_CONSTRAINTS' if not results else 'UNKNOWN_CHECKER'
    elif not response.get('representation_key'):
        record['exclusion_reason'] = 'CURRENT_REPRESENTATION_MISSING'
    elif not previous_record(row, previous) or not previous.get('representation_key'):
        record['exclusion_reason'] = 'IMMEDIATE_PREVIOUS_REPRESENTATION_MISSING'
    else:
        record['previous_representation_key'] = previous['representation_key']
        record['eligible'] = True
    return record


def coverage(rows):
    result = {}
    for split in ('train', 'validation', 'id_test', 'ood'):
        selected = [r for r in rows if r.get('eligible') and r['split'] == split]
        turns = {str(y): sum(r['failure_label'] == y for r in selected) for y in (0, 1)}
        groups = {str(y): len({r['group_id'] for r in selected if r['failure_label'] == y}) for y in (0, 1)}
        nturns, ngroups = (30, 10) if split == 'train' else (10, 5)
        result[split] = {'turns': turns, 'groups': groups,
                         'status': 'PASS' if min(turns.values()) >= nturns and min(groups.values()) >= ngroups else 'BLOCKED'}
    result['status'] = 'PASS' if all(result[s]['status'] == 'PASS' for s in ('train', 'validation', 'id_test')) else 'BLOCKED'
    return result


def label(cfg):
    data, output = paths(cfg)
    rows = read_jsonl(data / 'normalized/turns.jsonl')
    manifest = read_json(data / 'manifests/split_manifest.json')
    audit_split(rows, manifest)
    run = read_json(data / 'manifests/collection_lock.json')
    cache = SampleCache(data)
    checker = OfficialChecker(data / 'raw/official_code')
    records, previous = [], {}
    for row in rows:
        response = cache.load(row['sample_id'], run['fingerprint'])
        record = label_one(row, response, previous.get(row['trajectory_id']), checker)
        record['split'] = manifest['membership'][row['group_id']]
        if record['split'] == 'ood' and row['composition_signature'] != manifest['ood_signature']:
            record.update(eligible=False, exclusion_reason='OOD_NON_SELECTED_COMPOSITION')
        if record['eligible']:
            if not valid_vector(cache.vector(record['representation_key'])) or not valid_vector(cache.vector(record['previous_representation_key'])):
                raise Blocked('Invalid cached representation')
        records.append(record)
        # Previous label is irrelevant; truncated/UNKNOWN turns still provide hidden history.
        previous[row['trajectory_id']] = {**row, **(response or {})}
    write_jsonl(data / 'labeled/turns.jsonl', records)
    report = coverage(records)
    report['labeled_hash'] = digest(records)
    report['exclusions'] = dict(Counter(r['exclusion_reason'] for r in records if not r['eligible']))
    report['all_original_turns_collected'] = all(r.get('status') is not None for r in records)
    report['cohort_hash'] = digest([r['sample_id'] for r in records if r['eligible']])
    write_json(output / 'reports/label_coverage.json', report)
    gate(cfg, 'G3', report['status'], coverage=report)
    return report
