"""Export B checkpoint scores and optional frozen last-layer CLS embeddings."""
import argparse
import json
from pathlib import Path

from evaluate import read_rows
from src.utils.reproducibility import file_hash
from src.data.progress import Journal, signature, atomic_text, append_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--records', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--embeddings-out')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--diagnostic-only', action='store_true')
    parser.add_argument('--data-config', default='configs/data.yaml')
    parser.add_argument('--execution-mode', choices=('pilot','final'), default='final')
    parser.add_argument('--qa-manifest')
    parser.add_argument('--freeze-manifest')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    path = Path(args.checkpoint)
    if not path.is_file():
        print(json.dumps(dict(status='SKIPPED',reason='checkpoint_missing')))
        return 0
    outputs = [Path(p) for p in (args.out,args.embeddings_out) if p]
    # Evaluation reads every top-level *.jsonl; keep journals outside that set.
    progress_path = Path(args.out).parent/'.progress'/Path(args.out).name
    if len(set(p.resolve() for p in outputs)) != len(outputs) or (any(p.exists() for p in outputs) and not (args.resume and progress_path.exists())):
        parser.error('Use distinct new output files')
    try:
        import torch
        from src.models.dataset import load_tokenizer
        from src.models.predictor import ModernBertPredictor
        from src.evaluation.export import predict_records
    except ImportError as exc:
        print(json.dumps(dict(status='SKIPPED',reason='model_runtime_missing',dependency=exc.name)))
        return 0
    payload = torch.load(path,map_location='cpu',weights_only=False)
    metadata = payload.get('metadata') or {}
    if type(metadata.get('seed')) is not int or not metadata.get('model_config'):
        parser.error('A training checkpoint with seed/tokenizer/config provenance is required')
    if metadata.get('data_kind') not in ('real_derived', 'real_derived_unlabeled', 'model_generated'):
        parser.error('Synthetic/dry-run checkpoints are test-only and cannot be exported as research artifacts')
    model = ModernBertPredictor.from_checkpoint(path,device=args.device)
    mode = {'C_plus':'C+'}.get(metadata['mode'],metadata['mode'])
    if mode != model.variant:
        parser.error('Checkpoint architecture and run metadata disagree')
    if (mode == 'B_oracle') != args.diagnostic_only:
        parser.error('B_oracle export requires --diagnostic-only and is excluded from fair comparison')
    if args.embeddings_out and mode not in ('A','C','C+'):
        parser.error('Independent state representation probes are defined only for A/C/C+')
    config = metadata['model_config']['encoder']
    revision = config.get('tokenizer_revision') or metadata.get('model_revision') or config.get('revision')
    if not revision:
        parser.error('Pinned tokenizer/model revision required')
    tokenizer = load_tokenizer(config.get('tokenizer_id',config['model_id']),revision=revision)
    from src.evaluation.protocol import _valid
    records=[r for r in read_rows(args.records) if _valid(r)]
    from src.data.source_policy import enforce_source_policy
    from src.data.approvals import enforce_execution, research_rows
    research_rows(records)
    enforce_execution(records, mode=args.execution_mode, qa_path=args.qa_manifest, freeze_path=args.freeze_manifest, data_config=args.data_config, config_paths=[args.data_config,'configs/model.yaml','configs/train.yaml'])
    if not records:
        parser.error('No common valid-Target records; no predictions written')
    if len({r['sample_id'] for r in records}) != len(records):
        parser.error('Duplicate export sample ID')
    checkpoint_hash = file_hash(path)
    journal = Journal(progress_path, signature(dict(args={k:v for k,v in vars(args).items() if k != 'resume'},
        records=records, checkpoint_hash=checkpoint_hash),
        [Path(__file__), Path('src/evaluation/export.py'), args.data_config, args.freeze_manifest]), resume=args.resume)
    for output in outputs:
        atomic_text(output, '')
    for record in records:
        pair = journal.run('predict:'+record['sample_id'], lambda: list(predict_records(
            model,tokenizer,[record],model_name=mode,seed=metadata['seed'],max_length=config['max_length'],
            checkpoint_hash=checkpoint_hash,diagnostic_only=args.diagnostic_only))[0], cache_failures=False)
        for index, output in enumerate(outputs):
            append_json(output, pair[index])
    print(json.dumps(dict(status='OK',samples=len(records))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
