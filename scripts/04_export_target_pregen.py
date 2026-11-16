"""Export actual frozen label-collection Target input activations without generation."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.labeling import LocalTarget, target_prompt, sha
from src.evaluation.baselines_pre import partitions
from src.evaluation.baselines_target_probe import validate_activations


def main(argv=None):
    import yaml
    from evaluate import read_rows
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baselines.yaml')
    parser.add_argument('--split-manifest')
    args = parser.parse_args(argv)
    c = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    out = Path(c['activations'])
    if out.exists():
        parser.error('Use a new activation path; existing artifacts are preserved')
    manifest = json.loads(Path(args.split_manifest or c['manifest']).read_text(encoding='utf-8'))
    _, _, rows = partitions(read_rows(c['records']), manifest)
    from src.data.source_policy import enforce_source_policy
    from src.data.approvals import enforce_execution, research_rows
    research_rows(rows)
    enforce_execution(rows, mode=c.get('execution_mode','pilot'), qa_path=c.get('qa_manifest'),freeze_path=c.get('freeze_manifest'),data_config=c.get('data_config','configs/data.yaml'),config_paths=[c.get('data_config','configs/data.yaml'),'configs/model.yaml','configs/train.yaml'])
    runner = LocalTarget(c['target_revision'])
    if any(any((row.get('target_execution') or {}).get(k) != runner.metadata[k] for k in ('model_id','model_revision','chat_template_hash')) for row in rows):
        raise ValueError('P1 must use the exact Target used to collect labels')
    template = Path(c['target_prompt']).read_text(encoding='utf-8-sig')
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    activations = []
    for row in rows:
        prompt = target_prompt(row, template)
        if sha(prompt) != row.get('prompt_hash'):
            raise ValueError('Frozen label prompt mismatch; recollect labels or restore original template')
        started = time.perf_counter()
        feature, extraction = runner.pre_generation(prompt)
        activations.append(dict(sample_id=row['sample_id'], **runner.metadata, **extraction,
            prompt_hash=sha(prompt), hidden_state=feature, encoder_frozen=True,
            representation='target_pregen_middle_final_nonpadding', latency_seconds=time.perf_counter()-started,
            peak_vram_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None))
    validate_activations(rows, activations)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(a, allow_nan=False)+'\n' for a in activations), encoding='utf-8')


if __name__ == '__main__':
    main()
