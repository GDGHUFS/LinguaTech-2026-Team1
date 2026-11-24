"""Explicitly enabled optional P2 training; core pipeline never imports P2."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baseline_p2.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable-p2', action='store_true')
    parser.add_argument('--checkpoint', help='Optional P2 inference instead of training')
    parser.add_argument('--records')
    parser.add_argument('--predictions-out')
    parser.add_argument('--execution-mode',choices=('pilot','final'),default='final')
    parser.add_argument('--qa-manifest')
    parser.add_argument('--freeze-manifest')
    args = parser.parse_args(argv)
    if args.seed != 42:
        parser.error("This experiment registers only random seed 42")
    c = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if not (args.enable_p2 or c['enabled']):
        print(json.dumps(dict(model='P2', status='SKIPPED', reason='disabled_by_default')))
        return 0
    if c.get('teacher_forcing_ratio') != 0 or c.get('operation_mapping') != 'predicted_selective_write_revoke_or_carry':
        raise ValueError('P2 currently supports predicted rollout only, with the registered operation mapping')
    if set(c['weights']) != {'operation','state','failure'} or any(v < 0 for v in c['weights'].values()):
        raise ValueError('Invalid P2 loss weights')
    import torch
    from transformers import AutoModel
    from train import read_training_records, seed_everything, validation_metrics, is_better, validate_config
    from src.models.dataset import DialogueDataset, DialogueCollator, load_tokenizer
    from src.evaluation.baselines_sequential_dst import SequentialDST, offline_labels, training_loss
    from src.data.splitting import check_leakage
    from src.utils.reproducibility import content_hash
    if args.checkpoint:
        from evaluate import read_rows
        from src.evaluation.export import predict_records
        from src.evaluation.protocol import _valid
        from src.utils.reproducibility import file_hash
        path=Path(args.checkpoint)
        if not path.is_file():
            print(json.dumps(dict(model='P2',status='SKIPPED',reason='checkpoint_missing')))
            return 0
        if not args.records or not args.predictions_out:
            parser.error('P2 inference requires --records and --predictions-out')
        output=Path(args.predictions_out)
        if output.exists():
            parser.error('Existing optional predictions are preserved')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        metadata=payload['metadata']
        mc=metadata['encoder']
        model=SequentialDST(AutoModel.from_pretrained(mc['model_id'],revision=mc['revision']),metadata['config']['memory_slots'])
        model.load_state_dict(payload['model'])
        model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
        tokenizer=load_tokenizer(mc['model_id'],revision=mc['revision'])
        rows=[r for r in read_rows(args.records) if _valid(r)]
        from src.data.source_policy import enforce_source_policy
        from src.data.approvals import enforce_execution
        enforce_execution(rows,mode=args.execution_mode,qa_path=args.qa_manifest,freeze_path=args.freeze_manifest,data_config=c.get('data_config','configs/data.yaml'),config_paths=[c.get('data_config','configs/data.yaml'),'configs/model.yaml','configs/train.yaml'])
        if not rows:
            parser.error('No common valid-Target samples')
        predictions=[p for p,_ in predict_records(model,tokenizer,rows,model_name='P2',seed=metadata['seed'],max_length=mc['max_length'],checkpoint_hash=file_hash(path))]
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(''.join(json.dumps(p,allow_nan=False)+'\n' for p in predictions),encoding='utf-8')
        print(json.dumps(dict(model='P2',status='EXPORTED',samples=len(predictions))))
        return 0
    tc = yaml.safe_load(Path(c['training_config']).read_text(encoding='utf-8'))
    # 기본 모델의 10 epoch·early stopping 설정과 별개로 P2는 자체 설정을 따른다.
    tc = {**tc, 'max_epochs': c.get('max_epochs', 3)}
    mc = yaml.safe_load(Path(c['model_config']).read_text(encoding='utf-8'))['encoder']
    validate_config(tc)
    import re
    if not re.fullmatch('[0-9a-f]{40}', mc.get('revision') or ''):
        raise ValueError('Pin ModernBERT before optional training')
    train, val = read_training_records(c['train'],'train'), read_training_records(c['validation'],'validation')
    from src.data.source_policy import enforce_source_policy
    from src.data.approvals import enforce_execution
    enforce_execution(train + val,mode=args.execution_mode,qa_path=args.qa_manifest,freeze_path=args.freeze_manifest,data_config=c.get('data_config','configs/data.yaml'),config_paths=[c.get('data_config','configs/data.yaml'),'configs/model.yaml','configs/train.yaml'])
    check_leakage([dict(r,split='train') for r in train]+[dict(r,split='validation') for r in val], compositions=(), templates=())
    out = Path(c['output'])/f'seed_{args.seed}'
    if out.exists():
        raise ValueError('Existing optional artifacts are preserved; use a new output path')
    seed_everything(args.seed)
    tokenizer = load_tokenizer(mc['model_id'], revision=mc['revision'])
    datasets = [DialogueDataset(rows,tokenizer,mc['max_length']) for rows in (train,val)]
    if any(not item['request_mask'] or item['diagnostics']['truncated'] for d in datasets for item in d):
        raise ValueError('P2 requires intact sequential history; resolve common context policy first')
    model = SequentialDST(AutoModel.from_pretrained(mc['model_id'],revision=mc['revision']),c['memory_slots'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    import math
    micro, accumulation = tc['per_device_batch_size'], tc['gradient_accumulation_steps']
    batches = math.ceil(len(train)/micro)
    budget = min(math.ceil(batches/accumulation)*tc['max_epochs'], tc.get('max_optimizer_steps') or 10**12)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:max(0.,1-s/budget))
    collator = DialogueCollator(tokenizer)
    features = ('input_ids','attention_mask','turn_spans','turn_mask','request_span','request_mask')
    best, step = None, 0
    out.mkdir(parents=True)
    from src.utils.experiment import seed_metadata
    metadata = dict(experiment=seed_metadata(tc["seeds"]),config=c,training=tc,encoder=mc,seed=args.seed,train_hash=content_hash(train),validation_hash=content_hash(val),
                    teacher_forcing_ratio=0,optimizer_step_budget=budget,parameters=sum(p.numel() for p in model.parameters()))
    import time
    from src.utils.reproducibility import runtime_metadata
    metadata['runtime']=runtime_metadata()
    started=time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(tc['max_epochs']):
        order = torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        for start in range(0,batches,accumulation):
            indices = order[start*micro:min((start+accumulation)*micro,len(train))]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0,len(indices),micro):
                ix = indices[offset:offset+micro]
                items = [datasets[0][i] for i in ix]
                batch = collator(items)
                labels = [offline_labels(train[i],item['diagnostics']['turn_ids']) for i,item in zip(ix,items)]
                width = batch['turn_spans'].shape[1]
                pad = lambda values: torch.tensor([v+[-100]*(width-len(v)) for v in values],device=device)
                output = model(**{k:batch[k].to(device) for k in features})
                loss = training_loss(output,batch['failure_labels'].to(device),pad([l[0] for l in labels]),
                    {k:pad([l[1][k] for l in labels]) for k in labels[0][1]},c['weights'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite P2 loss')
                (loss*len(ix)/len(indices)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),tc['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if step == budget or start+accumulation >= batches:
                model.eval()
                scores, targets = [], []
                with torch.inference_mode():
                    for i in range(len(val)):
                        batch = collator([datasets[1][i]])
                        scores.append(float(model(**{k:batch[k].to(device) for k in features})['failure_logits'].sigmoid()[0]))
                        targets.append(val[i]['failure_label_unsafe'])
                quality = validation_metrics(targets,scores)
                if is_better(quality,best):
                    best = quality
                    metadata.update(runtime_seconds=time.perf_counter()-started,device=device,
                        peak_vram_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
                    torch.save(dict(model=model.state_dict(),metadata=metadata,validation=quality,step=step),out/'best.pt')
            if step == budget:
                print(json.dumps(dict(model='P2',status='TRAINED',validation=best,optimizer_steps=step)))
                return 0


if __name__ == '__main__':
    raise SystemExit(main())
