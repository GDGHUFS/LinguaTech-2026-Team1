"""Keep item failures local while preserving machine-readable diagnostics."""
import json
import sys
import re


def model_json(raw):
    """Accept one complete JSON fence without repairing or inventing any content."""
    if isinstance(raw, str):
        fence = re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*', raw, re.DOTALL)
        if fence:
            raw = fence[1]
    return json.loads(raw)


def diagnostic(row, stage, status, code=None, field=None):
    """Stable stage/type fields for streaming JSONL and CSV reports."""
    code = code or ''
    related = {'invalid_initial_rule_coverage':'initial_rule_turns',
        'invalid_initial_rule_turn':'initial_rule_turns', 'invalid_operation_coverage':'operation_turns',
        'invalid_operation_order':'operation_turns', 'invalid_turn_count':'turns', 'invalid_turn':'turns',
        'invalid_output_schema':'dialogue', 'request_mismatch':'final_request',
        'invalid_semantic_schema':'semantic_validation'}
    if field is None:
        field = related.get(code)
        if code.startswith(('semantic_check_unresolved:', 'semantic_check_evidence_missing:')):
            field = code.split(':', 1)[1]
        elif 'synthetic_policy' in code and 'preparation' not in code:
            field = 'purpose_provenance'
    return dict(source_record_id=row.get('source_record_id'), group_id=row.get('group_id'),
        seed_id=row.get('seed_id'), sample_id=row.get('sample_id'), stage=stage, status=status,
        failure_code=code or None, field=field,
        json_parse_error='json_parse' in code,
        schema_error='schema' in code or code in related and code != 'request_mismatch',
        source_evidence_mismatch='source_evidence' in code or 'source_conflict' in code,
        synthetic_policy_consistency_error=('synthetic_policy' in code and 'preparation' not in code) or 'synthetic_purpose' in code or 'policy_conditions' in code,
        rule_engine_error='oracle' in code or 'policy_trace' in code or 'policy_annotation' in code or stage == 'rule_engine' and status == 'failed',
        natural_language_error=stage == 'generation' and status == 'failed',
        semantic_validation_error=stage == 'semantic_validation' and status == 'failed')


def warning(stage, identity, reason):
    # Exception text and raw model output may contain credentials or source data.
    print(json.dumps(dict(level='WARNING', stage=stage, item=identity, reason=reason),
                     ensure_ascii=False), file=sys.stderr, flush=True)


def read_items(path, failures, required=()):
    rows = []
    with path.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or any(
                        not isinstance(row.get(k), str) or not row[k].strip() for k in required):
                    raise ValueError('Invalid record identity')
            except Exception as exc:
                finding = dict(line=number, status='excluded', reason='input:'+type(exc).__name__)
                failures.append(finding)
                warning('input', number, finding['reason'])
                continue
            rows.append(row)
    return rows
