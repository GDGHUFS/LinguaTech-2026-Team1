"""Validate Qwen/oracle data and freeze records/config hashes automatically."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import yaml
    from evaluate import read_rows
    from src.data.approvals import enforce_execution, research_rows, validate_freeze, split_membership
    from src.data.splitting import check_leakage
    from src.evaluation.protocol import _manifest
    from src.data.serving import validate_generation_cohort
    from src.utils.reproducibility import content_hash, file_hash
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-config', default='configs/data.yaml')
    parser.add_argument('--model-config', default='configs/model.yaml')
    parser.add_argument('--split-manifest', type=Path, help='Defaults to records directory/split_manifest.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Existing freeze preserved; choose a new output')
    config = yaml.safe_load(Path(args.data_config).read_text(encoding='utf-8'))
    if config.get('generation_mode') != 'camber_automatic':
        parser.error('Automatic freeze requires generation_mode: camber_automatic')
    rows = read_rows(args.records)
    manifest_path = args.split_manifest or (args.records if args.records.is_dir() else args.records.parent)/'split_manifest.json'
    snapshot = json.loads(manifest_path.read_text(encoding='utf-8'))
    check_leakage(rows, snapshot['ood_compositions'], snapshot['ood_templates'], snapshot.get('fewshot_template_families', ()))
    _manifest(snapshot, rows)
    enforce_execution(rows, mode='pilot', data_config=args.data_config)
    research_rows(rows)
    plans = {content_hash(r.get('holdout_plan')) for r in rows}
    if len(plans) != 1:
        parser.error('Inconsistent generation holdout plans')
    paths = [args.data_config, args.model_config, 'configs/train.yaml',
             'configs/baselines.yaml', 'configs/eval.yaml', 'configs/model_registry.yaml']
    manifest = dict(schema_version='automatic-freeze-1',
                    source_selection=config['source_selection'], holdout_plan=rows[0]['holdout_plan'],
                    records={r['sample_id']: content_hash(r) for r in rows},
                    split_manifest=snapshot, split_membership=split_membership(rows),
                    generation_model=validate_generation_cohort(rows), model_config_path=args.model_config,
                    config_hashes={p: file_hash(p) for p in paths})
    validate_freeze(rows, manifest, config, paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'FROZEN', 'records': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
