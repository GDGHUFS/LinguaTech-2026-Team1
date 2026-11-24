"""Seven stage entry points with explicit readiness, never fabricated artifacts."""
import argparse
import json
from .contracts import (Blocked, config, paths, read_json, read_jsonl,
                        write_json, gate)


def main(stage):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment.yaml')
    if stage == 1:
        parser.add_argument('--fetch', action='store_true', help='Download pinned official evidence before audit')
    if stage == 3:
        parser.add_argument('--pilot', action='store_true')
        parser.add_argument('--resume', action='store_true')
    if stage in (5,6):
        parser.add_argument('--resume', action='store_true', help='Resume the same checkpointed run without changing the experiment')
    if stage == 7:
        parser.add_argument('--input', help='JSON file of user-supplied synthetic privacy scenario')
    args = parser.parse_args()
    cfg = config(args.config)
    data, output = paths(cfg)
    try:
        if stage == 1:
            if args.fetch:
                from .setup import fetch_sources
                fetch_sources(cfg)
            from .source_audit import audit
            result = audit(cfg)
            print(json.dumps(result['census'], indent=2))
        elif stage == 2:
            from .splitting import make_split
            lock = read_json(data / 'manifests/revision_lock.json')
            result = make_split(read_jsonl(data / 'normalized/turns.jsonl'), lock['manifest_hash'])
            write_json(data / 'manifests/split_manifest.json', result, immutable=True)
            gate(cfg, 'G1', 'PASS', counts=result['counts'], ood=result['ood_status'])
            print(json.dumps(result['counts'], indent=2))
        elif stage == 3:
            from .target import collect
            print(json.dumps(collect(cfg, args.pilot, args.resume)))
        elif stage == 4:
            from .labeling import label
            print(json.dumps(label(cfg), indent=2))
        elif stage == 5:
            from .training import train
            print(json.dumps(train(cfg,resume=args.resume), indent=2))
        elif stage == 6:
            from .evaluation import evaluate
            print(json.dumps(evaluate(cfg,resume=args.resume), indent=2))
        elif stage == 7:
            from .privacy_demo import PrivacyDemo
            demo = PrivacyDemo(cfg)
            if demo.status != 'READY':
                raise Blocked(demo.status)
            if not args.input:
                raise Blocked('Provide --input with synthetic identifiers and original user-supplied turns')
            scenario = read_json(args.input)
            if scenario.get('synthetic_identifiers_confirmed') is not True:
                raise Blocked('Demo input must explicitly confirm synthetic identifiers')
            results = []
            for turn in scenario['turns']:
                results.append(demo.step(turn['query'], turn['forbidden_identifiers'], turn['policy_description'],
                    on_risk=lambda r: print(json.dumps({'pre_generation': r}, ensure_ascii=False), flush=True)))
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
            gate(cfg, 'G6', 'PASS' if len(results) >= 2 and all(r['status'] == 'OK' for r in results) else 'BLOCKED',
                 turns=len(results), checkpoint_loaded=True)
    except (Blocked, FileNotFoundError, ImportError) as exc:
        gate_name = {1:'G0', 2:'G1', 3:'G2', 4:'G3', 5:'G4', 6:'G5', 7:'G6'}[stage]
        gate(cfg, gate_name, 'BLOCKED', reason=str(exc))
        print(json.dumps({'status': 'BLOCKED', 'stage': stage, 'reason': str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
