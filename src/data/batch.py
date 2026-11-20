"""Keep item failures local while preserving machine-readable diagnostics."""
import json
import sys


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
