"""One seed, one fit per model; validation AP/Brier checkpoint selection."""
import copy
import hashlib
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .contracts import (Blocked, digest, file_hash, paths, read_json, read_jsonl,
                        write_json, write_jsonl, gate)
from .checkpointing import save_torch, save_joblib
from .features import temporal_features
from .representation import SampleCache
from .probes import PROBES
from .metrics import metrics, select_threshold
from .baselines import build_baselines, baseline_input


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)


def feature_tensors(rows, cache):
    if not rows:
        raise Blocked('Empty cohort')
    current = torch.stack([cache.vector(r['representation_key']) for r in rows])
    previous = torch.stack([cache.vector(r['previous_representation_key']) for r in rows])
    return temporal_features(current, previous)


def labels(rows):
    return torch.tensor([r['failure_label'] for r in rows], dtype=torch.float32)


def predict_probe(model, features):
    model.eval()
    with torch.inference_mode():
        return model(*features).sigmoid().cpu().numpy()


def fit_probe(name, train_features, train_y, validation_features, validation_y, cfg,
              checkpoint_dir=None, resume=False, provenance=None):
    if len(torch.unique(validation_y)) != 2 or len(torch.unique(train_y)) != 2:
        raise Blocked('Train and validation require both classes')
    seed_all()
    model = PROBES[name](train_features[0].shape[-1]).float()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=0.)
    dataset = TensorDataset(*train_features, train_y)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                        generator=generator, num_workers=0)
    best, best_state, wait, history = None, None, 0, []
    provenance = provenance or {}
    signature = digest({'name':name, 'config':cfg, 'provenance':provenance,
        'tensors':[{'shape':list(t.shape), 'sha256':hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
                   for t in (*train_features,train_y,*validation_features,validation_y)]})
    last_path = Path(checkpoint_dir) / (name + '.last.pt') if checkpoint_dir is not None else None
    best_path = Path(checkpoint_dir) / (name + '.best.pt') if checkpoint_dir is not None else None
    start_epoch, elapsed_before, finished = 1, 0., False
    if last_path is not None and last_path.exists():
        if not resume:
            raise Blocked(f'{name} has a saved run; use --resume')
        state = torch.load(last_path,map_location='cpu',weights_only=True)
        if state['signature'] != signature:
            raise Blocked('Training checkpoint configuration/cohort/features mismatch')
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer_state'])
        generator.set_state(state['generator_state'])
        torch.set_rng_state(state['torch_rng_state'])
        random.setstate(state['python_rng_state'])
        ns = state['numpy_rng_state']
        np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
        best, best_state, best_epoch = tuple(state['best_rank']),state['best_state'],state['best_epoch']
        wait, history = state['stale_epochs'],state['history']
        start_epoch, elapsed_before, finished = state['epoch'] + 1,state['elapsed_seconds'],state['finished']
    elif best_path is not None and best_path.exists():
        raise Blocked('Best checkpoint exists without resumable last checkpoint')
    started = time.perf_counter()
    epochs = range(start_epoch, cfg['max_epochs'] + 1) if not finished else []
    for epoch in epochs:
        model.train()
        losses = []
        for u, delta, distance, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(u, delta, distance), y)
            loss = loss + cfg['l2_lambda'] * model.head.weight.square().sum()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        probabilities = predict_probe(model, validation_features)
        result = metrics(validation_y.numpy(), probabilities, .5)
        rank = (result['AP'], -result['Brier'])
        history.append({'epoch': epoch, 'mean_batch_loss': float(np.mean(losses)),
                        'validation_ap': result['AP'], 'validation_brier': result['Brier']})
        if best is None or rank > best:
            best, best_state, best_epoch, wait = rank, copy.deepcopy(model.state_dict()), epoch, 0
            improved = True
        else:
            wait += 1
            improved = False
        finished = wait >= cfg['patience'] or epoch == cfg['max_epochs']
        elapsed = elapsed_before + time.perf_counter() - started
        if last_path is not None:
            ns = np.random.get_state()
            state = {'schema':'conshift-training-v2','signature':signature,'epoch':epoch,
                'state_dict':copy.deepcopy(model.state_dict()),'optimizer_state':optimizer.state_dict(),
                'generator_state':generator.get_state(),'torch_rng_state':torch.get_rng_state(),
                'python_rng_state':random.getstate(),
                'numpy_rng_state':(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
                'best_rank':best,'best_state':best_state,'best_epoch':best_epoch,
                'stale_epochs':wait,'history':history,'elapsed_seconds':elapsed,'finished':finished}
            save_torch(last_path,state)
            if improved:
                save_torch(best_path,{'state_dict':best_state,'metadata':{
                    **provenance,'model_name':name,'hidden_dim':train_features[0].shape[-1],
                    'threshold':select_threshold(validation_y.numpy(),probabilities),
                    'threshold_source':'validation','epoch':epoch,'validation_ap':best[0],
                    'validation_brier':-best[1],'training_config':cfg,'run_signature':signature,
                    'parameter_count':sum(p.numel() for p in model.parameters())}})
            write_jsonl(Path(checkpoint_dir).parent / 'logs' / (name + '_training.jsonl'),history)
        if finished:
            break
    model.load_state_dict(best_state)
    model.eval()
    threshold = select_threshold(validation_y.numpy(), predict_probe(model, validation_features))
    metadata = {**provenance, 'model_name': name, 'hidden_dim': train_features[0].shape[-1],
                'threshold': threshold, 'threshold_source': 'validation', 'epoch': best_epoch,
                'validation_ap': best[0], 'validation_brier': -best[1],
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'training_seconds': elapsed_before + time.perf_counter() - started,
                'training_config': cfg,'run_signature':signature,
                'epochs_completed':len(history),'early_stopped':len(history) < cfg['max_epochs']}
    if best_path is not None:
        save_torch(best_path,{'state_dict':model.state_dict(),'metadata':metadata})
    return model, metadata, history


def load_probe(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    meta = checkpoint['metadata']
    if meta['threshold_source'] != 'validation':
        raise Blocked('Checkpoint lacks fixed validation threshold')
    model = PROBES[meta['model_name']](meta['hidden_dim'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval(), meta


def train(cfg, resume=False):
    data, output = paths(cfg)
    coverage = read_json(output / 'reports/label_coverage.json')
    if coverage['status'] != 'PASS' or not coverage['all_original_turns_collected']:
        raise Blocked('G3 final collection/coverage gate did not pass')
    labeled = read_jsonl(data / 'labeled/turns.jsonl')
    if digest(labeled) != coverage['labeled_hash']:
        raise Blocked('Labeled data changed after coverage audit')
    records = [r for r in labeled if r['eligible']]
    if digest([r['sample_id'] for r in records]) != coverage['cohort_hash']:
        raise Blocked('Cohort changed after coverage audit')
    train_rows = [r for r in records if r['split'] == 'train']
    val_rows = [r for r in records if r['split'] == 'validation']
    checkpoint_dir = output / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if any(p.name != '.gitkeep' for p in checkpoint_dir.iterdir()) and not resume:
        raise Blocked('Training artifacts already exist; use --resume to continue the same run')
    cache = SampleCache(data)
    run = read_json(data / 'manifests/collection_lock.json')
    for r in records:
        cache.load(r['sample_id'], run['fingerprint'], r['prompt_hash'], r['prompt_token_ids_hash'])
    train_features, val_features = feature_tensors(train_rows, cache), feature_tensors(val_rows, cache)
    provenance = {'cohort_hash': coverage['cohort_hash'], 'collection_fingerprint': run['fingerprint'],
                  'labeled_hash': coverage['labeled_hash'],
                  'representation_protocol': {'dtype':run['dtype'], 'template_hash':run['template'],
                      'code_hash':run['code'], 'versions':run['versions']},
                  'split_hash': read_json(data / 'manifests/split_manifest.json')['manifest_hash'],
                  'target_config': cfg['target'],
                  'train_sample_hash': digest([r['sample_id'] for r in train_rows]),
                  'validation_sample_hash': digest([r['sample_id'] for r in val_rows])}
    summary = {}
    training_lock = {'provenance':provenance,'training_config':cfg['training']}
    write_json(checkpoint_dir / 'training_run.json',training_lock,immutable=True)
    for name in PROBES:
        deployed = checkpoint_dir / (name + '.pt')
        if resume and deployed.exists():
            model, meta = load_probe(deployed)
            if (meta['model_name'] != name or any(meta.get(k) != v for k,v in provenance.items())
                    or meta['training_config'] != cfg['training']):
                raise Blocked('Completed probe checkpoint differs from the resumed run')
        else:
            model, meta, history = fit_probe(name, train_features, labels(train_rows), val_features, labels(val_rows),
                cfg['training'],checkpoint_dir=checkpoint_dir,resume=resume,provenance=provenance)
            save_torch(deployed,{'state_dict':model.state_dict(),'metadata':meta})
            write_jsonl(output / 'logs' / (name + '_training.jsonl'),history)
        summary[name] = meta
        write_json(output / 'reports/training_progress.json',{'completed_models':list(summary),'run':training_lock})
    for name, model in build_baselines().items():
        artifact, metadata_path = checkpoint_dir / (name + '.joblib'),checkpoint_dir / (name + '.json')
        if resume and artifact.exists() and metadata_path.exists():
            meta = read_json(metadata_path)
            if any(meta.get(k) != v for k,v in provenance.items()) or meta['artifact_hash'] != file_hash(artifact):
                raise Blocked('Completed baseline checkpoint differs from the resumed run')
            summary[name] = meta
            continue
        started = time.perf_counter()
        model.fit(baseline_input(name, train_rows), labels(train_rows).numpy())
        probabilities = model.predict_proba(baseline_input(name, val_rows))[:, 1]
        meta = {**provenance, 'model_name': name, 'threshold_source': 'validation',
                'threshold': select_threshold(labels(val_rows).numpy(), probabilities),
                'training_seconds': time.perf_counter() - started,
                'parameter_count': int(model[-1].coef_.size + model[-1].intercept_.size)}
        save_joblib(artifact,model)
        meta['artifact_hash'] = file_hash(artifact)
        write_json(metadata_path,meta)
        summary[name] = meta
        write_json(output / 'reports/training_progress.json',{'completed_models':list(summary),'run':training_lock})
    quantiles = {'turn': np.quantile([r['turn_index'] for r in train_rows], [.33, .67]).tolist(),
                 'distance': np.quantile(train_features[2].numpy().ravel(), [.33, .67]).tolist(),
                 'fit_split': 'train', **provenance}
    write_json(checkpoint_dir / 'analysis_cutpoints.json', quantiles)
    write_json(output / 'reports/training_manifest.json', {'models': summary, **provenance})
    gate(cfg, 'G4', 'PASS', same_cohort=True, target_gradients=False,
         train_only_fitting=True, checkpoint_selection='validation AP/Brier/earliest epoch')
    return summary
