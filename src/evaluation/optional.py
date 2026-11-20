"""Independent optional runner; JSON outputs never enter core tables.

python -m src.evaluation.optional --kind mmd --input mmd_inputs.json --out mmd.json
MMD keys: x,y,groups_x,groups_y,sigma,permutations,seed. Sigma must be fixed
before test. H3 keys: records,budget,seed,random_repetitions,access_timing.
"""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('mmd','failure-guided'),required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--test-only',action='store_true')
    parser.add_argument('--approval-records')
    parser.add_argument('--qa-manifest')
    parser.add_argument('--freeze-manifest')
    args=parser.parse_args(argv)
    out=Path(args.out)
    if out.exists():
        parser.error('Use a new output path/revision')
    if not args.test_only:
        if not args.approval_records: parser.error('Production optional analysis requires approved records')
        from evaluate import read_rows
        from src.data.approvals import enforce_execution, research_rows
        records=read_rows(args.approval_records)
        research_rows(records)
        enforce_execution(records,mode='final',qa_path=args.qa_manifest,freeze_path=args.freeze_manifest,config_paths=['configs/data.yaml','configs/model.yaml','configs/train.yaml'])
    payload=json.loads(Path(args.input).read_text(encoding='utf-8'))
    try:
        if args.kind=='mmd':
            from .ood_analysis import grouped_mmd
            result=grouped_mmd(**payload)
        else:
            from .failure_guided import failure_guided
            result=failure_guided(**payload)
    except ImportError as exc:
        result=dict(status='SKIPPED',reason='optional_runtime_missing',dependency=exc.name)
    result['test_only']=args.test_only
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
