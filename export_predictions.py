"""Export B checkpoint scores and optional frozen last-layer CLS embeddings."""
import argparse
import json
from pathlib import Path

from evaluate import read_rows
from src.utils.reproducibility import file_hash


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
    args = parser.parse_args(argv)
    path = Path(args.checkpoint)
    if not path.is_file():
        print(json.dumps(dict(status='SKIPPED',reason='checkpoint_missing')))
        return 0
    outputs = [Path(p) for p in (args.out,args.embeddings_out) if p]
    if len(set(p.resolve() for p in outputs)) != len(outputs) or any(p.exists() for p in outputs):
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
    pairs = list(predict_records(model,tokenizer,records,model_name=mode,seed=metadata['seed'],
                                max_length=config['max_length'],checkpoint_hash=file_hash(path),diagnostic_only=args.diagnostic_only))
    # 추론이 모두 성공한 뒤 출력을 시작한다. 파일 저장 도중의 실패까지 원자적으로 처리하지는 않는다.
    for output,index in zip(outputs,range(2)):
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(''.join(json.dumps(pair[index],allow_nan=False)+'\n' for pair in pairs),encoding='utf-8')
    print(json.dumps(dict(status='OK',samples=len(pairs))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
