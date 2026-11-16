"""Summarize actual saved pilot records; missing measurements remain null."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from evaluate import read_rows
    from src.utils.reproducibility import runtime_metadata
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/data.yaml')
    parser.add_argument('--data')
    parser.add_argument('--out',default='outputs/reports/pilot/current_audit.json')
    args=parser.parse_args(argv)
    out=Path(args.out)
    if out.exists():
        parser.error('Use a new report path; previous measurements are preserved')
    import yaml
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    rows=read_rows(args.data or config['paths']['labeled'])
    from src.data.source_policy import enforce_source_policy
    enforce_source_policy(rows,args.config)
    valid=[r for r in rows if r.get('parse_status')=='VALID']
    observed=[r for r in valid if type(r.get('failure_label_unsafe')) is int]
    attempted=[r for r in rows if r.get('target_attempts')]
    report=dict(source_selection=config['source_selection'],source_policy_revision=config['source_policy_revision'],status='observed_saved_artifacts_only',records=len(rows),
        source_counts=dict(Counter(r.get('source_dataset') for r in rows)),
        capability_counts=dict(Counter(r.get('capability_id') for r in rows)),
        parse_counts=dict(Counter(r.get('parse_status') for r in rows)),
        target_attempted=len(attempted),target_valid=len(valid),
        valid_rate=len(valid)/len(attempted) if attempted else None,
        unsafe_prevalence=sum(r['failure_label_unsafe'] for r in observed)/len(observed) if observed else None,
        training_vram=None,token_diagnostics=None,human_alignment=None,
        runtime=runtime_metadata(),existing_results_revalidated=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],records=len(rows))))


if __name__=='__main__':
    main()
