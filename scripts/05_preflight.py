"""Read-only execution readiness. Never calls APIs, trains, or evaluates scores."""
import argparse
from collections import Counter
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def inspect_readiness(root):
    """설정·로컬 산출물·설치 환경으로 실행 준비 상태를 검사한다. 외부 호출, 학습, 평가 점수 계산은 하지 않으며 검증 완료를 대신하지 않는다."""
    import yaml
    root=Path(root).resolve()
    checks={}
    def record(name,status,reason,**details):
        checks[name]=dict(status=status,reason=reason,**details)
    configs={}
    try:
        for name in ('data','model','train','eval','baselines','baseline_p2','model_registry'):
            configs[name]=yaml.safe_load((root/'configs'/f'{name}.yaml').read_text(encoding='utf-8'))
        record('configs','PASS','all seven configuration files parse')
    except (OSError,ValueError,yaml.YAMLError) as exc:
        record('configs','BLOCKED',type(exc).__name__)
        return dict(status='BLOCKED',mode='read_only_preflight_no_experiment',checks=checks)
    from src.utils.experiment import seed_metadata
    shared = all(configs[name].get('seeds') == [42] for name in ('train','eval','baselines'))
    record('experiment_seeds','PASS' if shared else 'BLOCKED',
           'registered experiment uses random seed 42 only', **seed_metadata([42]))
    try:
        from src.data.fewshots import load_fewshots
        from src.data.splitting import DEFAULT_TEMPLATES
        examples = load_fewshots(root/'prompts/naturalization_shots.json', DEFAULT_TEMPLATES)
        record('naturalization_shots','PASS','offline schema/oracle/known holdout-expression checks only; Qwen semantic validation NOT RUN',
               shot_count=len(examples['shots']))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        record('naturalization_shots','BLOCKED',str(exc))
    versions={}
    for name in ('pydantic','PyYAML','pytest','numpy','scikit-learn','torch','transformers','tokenizers','accelerate'):
        try:
            versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name]=None
    missing=[name for name,value in versions.items() if value is None]
    record('runtime','BLOCKED' if missing else 'PASS','missing dependencies' if missing else 'required distributions installed',versions=versions,missing=missing)
    try:
        import torch
        record('cuda','PASS' if torch.cuda.is_available() else 'BLOCKED',
            'CUDA available; no GPU workload executed' if torch.cuda.is_available() else 'CUDA unavailable in this runtime',
            devices=torch.cuda.device_count())
    except ImportError:
        record('cuda','NOT RUN','torch not installed')
    cache=Path(os.environ.get('HF_HUB_CACHE',str(Path.home()/'.cache'/'huggingface'/'hub')))
    for name,spec in (('encoder',configs['model']['encoder']),('target',configs['model']['target'])):
        revision=spec.get('revision')
        pinned=isinstance(revision,str) and re.fullmatch('[0-9a-f]{40}',revision) is not None
        snapshot=cache/('models--'+spec['model_id'].replace('/','--'))/'snapshots'/(revision or 'UNPINNED')
        weights=pinned and snapshot.is_dir() and any(p.suffix in ('.safetensors','.bin') for p in snapshot.iterdir())
        record(name+'_weights','PASS' if weights else 'BLOCKED',
            'pinned cached files found; model execution NOT RUN' if weights else 'immutable revision or local weights missing',
            model_id=spec['model_id'],revision=revision,local_weights_found=bool(weights))
    from src.data.canonicalize import active_setup,download_source,load_source,select_group_candidates,screening_summary
    setup = active_setup(configs['data'])
    record('source_selection','PASS','explicit active sources; optional sources preserved but not loaded',
           active=configs['data'].get('source_selection'), optional=configs['data'].get('optional_sources'))
    for source,spec in setup['sources'].items():
        raw=root/configs['data']['paths']['raw']
        if not (raw/source/'upstream_tree.json').is_file():
            record('source_'+source,'NOT RUN','pinned local original files unavailable')
            continue
        try:
            download_source(raw/source,spec,offline=True)
            rows=load_source(raw,source,spec)
            candidates=select_group_candidates(rows,setup['candidates_per_source'],setup['random_seed'])
            record('source_'+source,'PASS','offline pinned source integrity and structural scan only',
                inventory_records=len(rows),candidate_statuses=dict(Counter(r['status'] for r in candidates)),
                full_inventory=screening_summary(rows), sampled_inventory=screening_summary(candidates))
        except (OSError,ValueError,KeyError) as exc:
            record('source_'+source,'BLOCKED','offline source integrity failed: '+type(exc).__name__)
    try:
        from src.data.source_policy import enforce_source_policy
        from evaluate import read_rows
        canonical=root/configs['data']['paths']['annotated']/configs['data']['canonical_filename']
        source_rows=read_rows(canonical)
        annotation_path=configs['data'].get('approvals',{}).get('annotation_revisions')
        if annotation_path:
            from src.data.annotations import apply_revisions
            source_rows=apply_revisions(source_rows,read_rows(annotation_path))
        review_path=configs['data'].get('approvals',{}).get('source_reviews')
        if review_path:
            from src.data.source_policy import apply_review
            reviews=read_rows(review_path)
            by_id={r['seed_id']:r for r in reviews}
            if len(by_id)!=len(reviews) or set(by_id)-{r['seed_id'] for r in source_rows}:
                raise ValueError('Duplicate/unknown source review IDs')
            source_rows=[apply_review(r,by_id[r['seed_id']]) if r['seed_id'] in by_id else r for r in source_rows]
        approved=[r for r in source_rows if r.get('generation_ready')]
        if not approved: raise ValueError('No approved independent source groups')
        enforce_source_policy(approved,root/'configs/data.yaml')
        record('source_approval','PASS','exact-record approvals verified',approved_groups=len({r['group_id'] for r in approved}))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        record('source_approval','BLOCKED',str(exc),approved_groups=0)
    if configs['data'].get('generation_mode') == 'camber_automatic':
        try:
            from src.data.source_policy import selected_source
            canonical=root/configs['data']['paths']['annotated']/configs['data']['canonical_filename']
            source_rows=read_rows(canonical)
            if not source_rows or any(not selected_source(r, configs['data']) or not r.get('group_id') for r in source_rows):
                raise ValueError('Selected canonical group IDs required')
            from src.data.automatic import source_scenario, SourceExcluded
            inputs=[]
            for row in source_rows:
                try:
                    source_scenario(row)
                    inputs.append(row)
                except SourceExcluded:
                    pass
            if not inputs:
                raise ValueError('No original CAMBER records ready for automatic normalization')
            record('source_approval','PASS','original CAMBER inputs ready; Qwen extraction NOT RUN; no human approval required',
                   candidate_groups=len({r['group_id'] for r in inputs}))
        except (OSError,ValueError,KeyError,TypeError) as exc:
            record('source_approval','BLOCKED',str(exc))
    from evaluate import read_rows
    generated=root/configs['data']['paths']['generated']/'generated.jsonl'
    rows=read_rows(generated) if generated.is_file() else []
    ready=[r for r in rows if r.get('validation_status')=='semantic_pass_oracle_verified' and r.get('oracle_relation_status')=='passed']
    record('generated_dialogues','PASS' if ready else 'BLOCKED','validated saved dialogue availability',
        records=len(rows),ready_records=len(ready),generation_statuses=dict(Counter(r.get('generation_status') for r in rows)))
    try:
        from src.data.approvals import enforce_execution
        gate=configs['data'].get('approvals',{})
        enforce_execution(ready,mode='pilot',qa_path=gate.get('qa_manifest'),data_config=root/'configs/data.yaml')
        record('automatic_validation','PASS','Qwen semantics and oracle replay verified')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        record('automatic_validation','BLOCKED',str(exc))
    folder=root/configs['data']['paths']['splits']
    manifest_path=folder/'split_manifest.json'
    if manifest_path.is_file():
        try:
            rows=read_rows(folder)
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            from src.data.source_policy import selected_source
            if any(not selected_source(r, configs['data']) for r in rows) or manifest.get('source_selection') != configs['data'].get('source_selection'):
                raise ValueError('Manifest/input source selection differs from current config')
            from src.data.splitting import template_validation_scope
            record('OOD_template_scope','NOT RUN','structural checks alone do not establish semantic generalization',
                   **template_validation_scope(rows, manifest.get('ood_templates', ())))
            from src.evaluation.protocol import _manifest,_valid
            from src.data.splitting import check_leakage
            _manifest(manifest,rows)
            check_leakage(rows,manifest.get('ood_compositions',()),manifest.get('ood_templates',()),manifest.get('fewshot_template_families',()))
            valid=[r for r in rows if _valid(r)]
            record('split_structure','PASS','provisional saved manifest/group structure only; no test metrics evaluated',records=len(rows))
            record('target_labels','PASS' if valid else 'BLOCKED','observed label availability only; provenance must pass training gates',
                valid_target_records=len(valid),parse_statuses=dict(Counter(r.get('parse_status') for r in rows)))
            required=('train','validation','id_test','ood_composition','ood_template')
            coverage={p:sorted({r['failure_label_unsafe'] for r in valid if r['split']==p}) for p in required}
            record('class_coverage','PASS' if all(v==[0,1] for v in coverage.values()) else 'BLOCKED','both classes needed in common partitions',coverage=coverage)
            from src.data.approvals import enforce_execution
            enforce_execution(rows,mode='final',qa_path=gate.get('qa_manifest'),freeze_path=gate.get('freeze_manifest'),data_config=root/'configs/data.yaml',config_paths=['configs/data.yaml','configs/model.yaml','configs/train.yaml'])
            record('final_split_freeze','PASS','approved QA and frozen records/config verified')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            record('split_structure','BLOCKED',type(exc).__name__+': '+str(exc))
            record('final_split_freeze','BLOCKED',str(exc))
    else:
        record('split_structure','NOT RUN','split manifest missing')
        record('target_labels','BLOCKED','observed labeled split missing')
    from src.evaluation.registry import load_registry
    registry,_=load_registry(root/'configs/model_registry.yaml')
    for name,row in registry['models'].items():
        if row['eval_inclusion']!='fair':
            continue
        artifacts=root/row['artifact']
        candidates=list(artifacts.rglob('*.pt'))+list(artifacts.rglob('fit.pkl')) if artifacts.is_dir() else []
        record('artifact_'+name,'NOT RUN' if not candidates else 'BLOCKED',
            'checkpoint/fit artifact missing' if not candidates else 'artifacts present; genuine checkpoint validation requires a separate authorized run',count=len(candidates))
    activations=root/configs['baselines']['activations']
    record('P1_activations','BLOCKED' if not activations.is_file() else 'NOT RUN',
        'same-Target pre-generation activations missing' if not activations.is_file() else 'present; extraction validation not run by preflight')
    record('P2_disabled','PASS' if configs['baseline_p2']['enabled'] is False else 'BLOCKED','optional baseline must be explicitly activated')
    record('actual_training','NOT RUN','external calls and training intentionally not executed')
    record('actual_test_evaluation','NOT RUN','real test evaluation intentionally not executed')
    generation=configs['model'].get('generation',{})
    record('generation_protocol','PASS' if generation.get('model_id') and generation.get('prompt_version') else 'BLOCKED','exact Qwen serving model and prompt version required')
    stage_readiness = {
        'generation': 'READY' if all(checks.get(k,{}).get('status')=='PASS' for k in ('source_approval','generation_protocol','naturalization_shots')) else 'BLOCKED',
        'target_pilot': 'READY' if checks['automatic_validation']['status']=='PASS' and checks['target_weights']['status']=='PASS' else 'BLOCKED',
        'final_training': 'READY' if all(checks.get(k,{}).get('status')=='PASS' for k in ('automatic_validation','final_split_freeze','class_coverage','runtime','encoder_weights')) else 'BLOCKED'}
    return dict(stage_readiness=stage_readiness,status='BLOCKED' if any(c['status'] in ('BLOCKED','NOT RUN') for c in checks.values()) else 'READY',
        mode='read_only_preflight_no_experiment',checks=checks)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path)
    parser.add_argument('--strict',action='store_true')
    args=parser.parse_args(argv)
    if args.out and args.out.exists():
        parser.error('Existing readiness report is preserved; use a new path')
    result=inspect_readiness(args.root)
    encoded=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(encoded,encoding='utf-8')
    print(encoded)
    return 1 if args.strict and result['status']!='READY' else 0


if __name__=='__main__':
    raise SystemExit(main())
