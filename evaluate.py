"""Freeze validation thresholds, then evaluate saved scores under that protocol.

python evaluate.py --records data/splits --predictions outputs/predictions \
  --split-manifest data/splits/split_manifest.json --protocol configs/eval.yaml \
  --freeze --out outputs/eval_protocol.json
Repeat without --freeze and use the saved protocol to evaluate test predictions.
"""
import argparse
import json
from pathlib import Path

import yaml
from src.evaluation.protocol import evaluate_predictions, freeze_protocol


def read_rows(path):
    path = Path(path)
    files = sorted(path.glob('*.jsonl')) if path.is_dir() else [path]
    return [json.loads(line) for file in files for line in file.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--split-manifest', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--data-config', default='configs/data.yaml')
    parser.add_argument('--execution-mode', choices=('pilot','final'), default='final')
    parser.add_argument('--qa-manifest')
    parser.add_argument('--freeze-manifest')
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        parser.error('Output exists; use a new revision/path')
    records, predictions = read_rows(args.records), read_rows(args.predictions)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding='utf-8-sig'))
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding='utf-8-sig'))
    try:
        from src.data.source_policy import enforce_source_policy
        if args.test_only:
            data_config = yaml.safe_load(Path(args.data_config).read_text(encoding='utf-8'))
        else:
            from src.data.approvals import enforce_execution
            data_config = enforce_execution(records, mode=args.execution_mode, qa_path=args.qa_manifest, freeze_path=args.freeze_manifest, data_config=args.data_config, config_paths=[args.data_config,'configs/model.yaml','configs/train.yaml',args.protocol] if args.freeze else [args.data_config,'configs/model.yaml','configs/train.yaml'])
        if not args.test_only and manifest.get('source_selection') != data_config.get('source_selection'):
            raise ValueError('Manifest source selection differs from config')
        if args.freeze and not args.test_only:
            protocol['execution_approval'] = dict(qa_path=args.qa_manifest,freeze_path=args.freeze_manifest,data_config=args.data_config,config_paths=[args.data_config,'configs/model.yaml','configs/train.yaml',args.protocol])
        result = freeze_protocol(predictions,records,manifest,protocol,test_only=args.test_only) if args.freeze else evaluate_predictions(predictions,manifest,protocol,records=records,test_only=args.test_only)
    except (ValueError, KeyError, TypeError) as exc:
        result = dict(status='ERROR', reason=str(exc))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result.get('status','FROZEN'), 'output':str(out)}))
    return 1 if result.get('status') == 'ERROR' else 0


if __name__ == '__main__':
    raise SystemExit(main())
